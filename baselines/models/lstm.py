import math

import torch
from omegaconf import DictConfig
from torch import nn

from baselines.data.dataset import Events
from baselines.models.dropout import DropoutScheme, DropoutSite
from baselines.models.embeddings import EmbeddingInput, EventContentEmbedding

# Per-layer hidden and cell states, `[B, D]` each.
LSTMState = list[tuple[torch.Tensor, torch.Tensor]]


@torch.compile(dynamic=True)
def _lstm_step(
    projected_inputs: torch.Tensor,
    hidden: torch.Tensor,
    cell: torch.Tensor,
    hidden_weights: torch.Tensor,
    bias: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """One LSTM step from an already projected input and a gate-wise masked hidden state.

    Args:
        projected_inputs: `[B, 4, H]`, the input's contribution to each gate.
        hidden: `[B, 4, H]`, the previous hidden state as each gate reads it.
        cell: `[B, H]`, the previous cell state.
        hidden_weights: `[4, H, H]`.
        bias: `[4, H]`.

    Returns:
        The next hidden and cell states, `[B, H]` each.
    """
    gates = projected_inputs + torch.einsum('bgk,ghk->bgh', hidden, hidden_weights) + bias
    input_gate, forget_gate, candidate, output_gate = gates.unbind(dim=1)  # 4 x [B, H]
    cell = forget_gate.sigmoid() * cell + input_gate.sigmoid() * candidate.tanh()
    return output_gate.sigmoid() * cell.tanh(), cell


class GateDropoutLSTMCell(nn.Module):
    """LSTM layer with dropout on each gate's input and hidden state.

    Under `SequenceDropout` the recurrent path is masked as Gal and Ghahramani do, which cuDNN's
    `nn.LSTM` cannot. Every position's input projection runs as one product before the
    recurrence.
    """

    def __init__(self, *, input_size: int, hidden_size: int, dropout: DropoutSite) -> None:
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
        self.input_dropout = dropout(shape=(4, input_size))
        self.hidden_dropout = dropout(shape=(4, hidden_size))

    def forward(
        self,
        inputs: torch.Tensor,
        state: tuple[torch.Tensor, torch.Tensor],
        *,
        keep: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        """Run the layer over inputs `[B, T, I]` from `state`.

        Args:
            inputs: One input per position.
            state: The hidden and cell states before the first position, `[B, H]` each.
            keep: `[B, T]`, False where a row keeps its state instead of advancing, or None to
                advance every row at every position.

        Returns:
            The hidden state after each position, `[B, T, H]`, and the final state.
        """
        masked = self.input_dropout(inputs.unsqueeze(dim=2).expand(-1, -1, 4, -1))  # [B, T, 4, I]
        projected = torch.einsum('btgi,ghi->btgh', masked, self.input_weights)  # [B, T, 4, H]
        hidden, cell = state
        outputs = []
        for position in range(inputs.size(dim=1)):
            next_hidden, next_cell = _lstm_step(
                projected_inputs=projected[:, position],
                hidden=self.hidden_dropout(hidden.unsqueeze(dim=1).expand(-1, 4, -1)),  # [B, 4, H]
                cell=cell,
                hidden_weights=self.hidden_weights,
                bias=self.bias,
            )
            if keep is not None:
                advance = keep[:, position].unsqueeze(dim=1)  # [B, 1]
                next_hidden = torch.where(advance, next_hidden, hidden)
                next_cell = torch.where(advance, next_cell, cell)
            hidden, cell = next_hidden, next_cell
            outputs.append(hidden)
        return torch.stack(outputs, dim=1), (hidden, cell)  # T x [B, H] -> [B, T, H]


class LSTMBackbone(nn.Module):
    """Stacked LSTM encoder and decoder, the encoder's final per-layer state starting the
    decoder's. No attention over prefix events and no position table: order is the recurrence."""

    def __init__(
        self,
        config: DictConfig,
        content: EventContentEmbedding,
        *,
        d_model: int,
        dropout: DropoutScheme,
    ) -> None:
        """
        Args:
            config: The `backbone` section: layers per stack.
            content: The event embedding shared by encoder and decoder.
            d_model: The width of every state.
            dropout: The dropout of every site; without attention, its attention rate is unused.
        """
        super().__init__()
        self.encoder_input = EmbeddingInput(content, d_model=d_model, dropout=dropout.site)
        self.encoder_cells = nn.ModuleList(
            GateDropoutLSTMCell(input_size=d_model, hidden_size=d_model, dropout=dropout.site)
            for _ in range(config.num_layers)
        )
        self.decoder_input = EmbeddingInput(content, d_model=d_model, dropout=dropout.site)
        self.decoder_cells = nn.ModuleList(
            GateDropoutLSTMCell(input_size=d_model, hidden_size=d_model, dropout=dropout.site)
            for _ in range(config.num_layers)
        )

    def encode(self, prefix: Events) -> LSTMState:
        """Read prefix events `[B, P]` and return each layer's state at the row's last event."""
        embedded = self.encoder_input(prefix)
        zeros = embedded.new_zeros(size=(embedded.size(dim=0), embedded.size(dim=2)))
        _, states = _run_stack(
            self.encoder_cells,
            inputs=embedded,
            states=[(zeros, zeros) for _ in self.encoder_cells],
            keep=prefix.pad_mask().logical_not(),
        )
        return states

    def decode(self, inputs: Events, prefix: LSTMState) -> torch.Tensor:
        """Read every teacher-forced decoder input `[B, T]` in order.

        Returns:
            `[B, T, D]`, the top layer's state after each input.
        """
        outputs, _ = _run_stack(
            self.decoder_cells, inputs=self.decoder_input(inputs), states=prefix
        )
        return outputs

    def start(self, prefix: Events, *, max_steps: int) -> LSTMState:
        """Encode prefix events `[B, P]` for a generation: the decoder starts from the encoder's
        final states."""
        return self.encode(prefix)

    def step(
        self, inputs: Events, state: LSTMState, *, position: int
    ) -> tuple[torch.Tensor, LSTMState]:
        """Read the decoder input `[B, 1]` of the next suffix slot.

        Returns:
            `[B, D]` and the advanced state.
        """
        outputs, state = _run_stack(
            self.decoder_cells, inputs=self.decoder_input(inputs), states=state
        )
        return outputs[:, 0], state

    def select(self, state: LSTMState, index: torch.Tensor) -> LSTMState:
        """Keep the generation rows at `index` `[B']`."""
        return [(hidden[index], cell[index]) for hidden, cell in state]


def _run_stack(
    cells: nn.ModuleList,
    *,
    inputs: torch.Tensor,
    states: LSTMState,
    keep: torch.Tensor | None = None,
) -> tuple[torch.Tensor, LSTMState]:
    """Run inputs `[B, T, D]` up the stack a layer at a time, each layer reading the one below.

    Returns:
        The top layer's outputs `[B, T, D]` and every layer's final state.
    """
    final: LSTMState = []
    for cell, state in zip(cells, states, strict=True):
        inputs, state = cell(inputs, state, keep=keep)
        final.append(state)
    return inputs, final
