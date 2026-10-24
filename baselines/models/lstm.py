import math

import torch
from omegaconf import DictConfig
from torch import nn

from baselines.data.dataset import Events
from baselines.models.dropout import SequenceDropout
from baselines.models.embeddings import EventContentEmbedding

# Per-layer hidden and cell states, `[B, D]` each.
LSTMState = list[tuple[torch.Tensor, torch.Tensor]]


@torch.compile(dynamic=True)
def _lstm_step(
    inputs: torch.Tensor,
    hidden: torch.Tensor,
    cell: torch.Tensor,
    input_weights: torch.Tensor,
    hidden_weights: torch.Tensor,
    bias: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """One LSTM step from gate-wise masked inputs.

    Args:
        inputs: `[B, 4, I]`, the input as each gate reads it.
        hidden: `[B, 4, H]`, the previous hidden state as each gate reads it.
        cell: `[B, H]`, the previous cell state.
        input_weights: `[4, H, I]`.
        hidden_weights: `[4, H, H]`.
        bias: `[4, H]`.

    Returns:
        The next hidden and cell states, `[B, H]` each.
    """
    gates = (
        torch.einsum('bgi,ghi->bgh', inputs, input_weights)
        + torch.einsum('bgk,ghk->bgh', hidden, hidden_weights)
        + bias
    )  # [B, 4, H]
    input_gate, forget_gate, candidate, output_gate = gates.unbind(dim=1)
    cell = forget_gate.sigmoid() * cell + input_gate.sigmoid() * candidate.tanh()
    return output_gate.sigmoid() * cell.tanh(), cell


class VariationalLSTMCell(nn.Module):
    """LSTM cell with one dropout mask per sequence on each gate's input and hidden state.

    The recurrent path is masked as Gal and Ghahramani do, which cuDNN's `nn.LSTM` cannot.
    """

    def __init__(self, *, input_size: int, hidden_size: int, dropout: float) -> None:
        super().__init__()
        bound = 1.0 / math.sqrt(hidden_size)
        # Initialized as `nn.LSTM`: every weight and bias uniform in [-bound, bound].
        self.input_weights = nn.Parameter(
            torch.empty(4, hidden_size, input_size).uniform_(-bound, bound)
        )
        self.hidden_weights = nn.Parameter(
            torch.empty(4, hidden_size, hidden_size).uniform_(-bound, bound)
        )
        self.bias = nn.Parameter(torch.empty(4, hidden_size).uniform_(-bound, bound))
        self.input_dropout = SequenceDropout(p=dropout, shape=(4, input_size))
        self.hidden_dropout = SequenceDropout(p=dropout, shape=(4, hidden_size))

    def forward(
        self, inputs: torch.Tensor, state: tuple[torch.Tensor, torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Advance `state` by one input `[B, I]`."""
        hidden, cell = state
        return _lstm_step(
            inputs=self.input_dropout(inputs.unsqueeze(dim=1).expand(-1, 4, -1)),  # -> [B, 4, I]
            hidden=self.hidden_dropout(hidden.unsqueeze(dim=1).expand(-1, 4, -1)),  # -> [B, 4, H]
            cell=cell,
            input_weights=self.input_weights,
            hidden_weights=self.hidden_weights,
            bias=self.bias,
        )


class LSTMBackbone(nn.Module):
    """Stacked LSTM encoder and decoder, the encoder's final per-layer state starting the
    decoder's. No attention over prefix events and no position table: order is the recurrence."""

    def __init__(
        self, config: DictConfig, content: EventContentEmbedding, *, d_model: int, dropout: float
    ) -> None:
        """
        Args:
            config: The `backbone` section: layers per stack.
            content: The event embedding shared by encoder and decoder.
            d_model: The width of every state.
            dropout: The variational dropout rate of every site.
        """
        super().__init__()
        self.content = content
        self.encoder_dropout = SequenceDropout(p=dropout, shape=(d_model,))
        self.encoder_embedding_norm = nn.LayerNorm(normalized_shape=d_model)
        self.encoder_cells = nn.ModuleList(
            VariationalLSTMCell(input_size=d_model, hidden_size=d_model, dropout=dropout)
            for _ in range(config.num_layers)
        )
        self.decoder_dropout = SequenceDropout(p=dropout, shape=(d_model,))
        self.decoder_embedding_norm = nn.LayerNorm(normalized_shape=d_model)
        self.decoder_cells = nn.ModuleList(
            VariationalLSTMCell(input_size=d_model, hidden_size=d_model, dropout=dropout)
            for _ in range(config.num_layers)
        )

    def encode(self, prefix: Events) -> LSTMState:
        """Read prefix events `[B, P]` and return each layer's state at the row's last event."""
        embedded = self.encoder_embedding_norm(self.encoder_dropout(self.content(prefix)))
        real = prefix.pad_mask().logical_not()  # [B, P]
        zeros = embedded.new_zeros(size=(embedded.size(dim=0), embedded.size(dim=2)))
        states: LSTMState = [(zeros, zeros) for _ in self.encoder_cells]
        for position in range(embedded.size(dim=1)):
            # A row past its last event keeps the state it reached there.
            keep = real[:, position].unsqueeze(dim=1)  # [B, 1]
            states = [
                (torch.where(keep, hidden, old_hidden), torch.where(keep, cell, old_cell))
                for (hidden, cell), (old_hidden, old_cell) in zip(
                    self._advance(self.encoder_cells, embedded[:, position], states),
                    states,
                    strict=True,
                )
            ]
        return states

    def decode(self, inputs: Events, prefix: LSTMState) -> torch.Tensor:
        """Read every teacher-forced decoder input `[B, T]` in order.

        Returns:
            `[B, T, D]`, the top layer's state after each input.
        """
        embedded = self.decoder_embedding_norm(self.decoder_dropout(self.content(inputs)))
        states = prefix
        outputs = []
        for position in range(embedded.size(dim=1)):
            states = self._advance(self.decoder_cells, embedded[:, position], states)
            outputs.append(states[-1][0])
        return torch.stack(outputs, dim=1)  # T x [B, D] -> [B, T, D]

    def start(self, prefix: LSTMState, *, max_steps: int) -> LSTMState:
        """Prepare a generation: the decoder starts from the encoder's final states."""
        return prefix

    def step(
        self, inputs: Events, state: LSTMState, *, position: int
    ) -> tuple[torch.Tensor, LSTMState]:
        """Read the decoder input `[B, 1]` of the next suffix slot.

        Returns:
            `[B, D]` and the advanced state.
        """
        embedded = self.decoder_embedding_norm(self.decoder_dropout(self.content(inputs)))
        states = self._advance(self.decoder_cells, embedded[:, 0], state)
        return states[-1][0], states

    @staticmethod
    def _advance(cells: nn.ModuleList, inputs: torch.Tensor, states: LSTMState) -> LSTMState:
        """Push one input `[B, D]` up the stack, each layer reading the one below."""
        advanced: LSTMState = []
        for cell, state in zip(cells, states, strict=True):
            hidden, cell_state = cell(inputs, state)
            advanced.append((hidden, cell_state))
            inputs = hidden
        return advanced
