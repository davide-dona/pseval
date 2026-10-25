from dataclasses import dataclass

import torch
import torch.nn.functional as F
from omegaconf import DictConfig
from torch import nn

from baselines.data.dataset import Events
from baselines.models.dropout import DropoutScheme, DropoutSite
from baselines.models.embeddings import EmbeddingInput, EventContentEmbedding, position_table


@dataclass(frozen=True)
class ProjectedKeysValues:
    """One attention input, already projected and split across heads.

    The encoded prefix does not change while a suffix is being written, and a suffix position
    already written does not change when the next one is added, so both are projected once.
    """

    keys: torch.Tensor  # [B, H, S, Dh]
    values: torch.Tensor  # [B, H, S, Dh]

    def select(self, index: torch.Tensor) -> 'ProjectedKeysValues':
        """Keep the rows at `index` `[B']`."""
        return ProjectedKeysValues(keys=self.keys[index], values=self.values[index])


class MultiHeadAttention(nn.Module):
    """Multi-head attention with cached key/value projections."""

    def __init__(self, *, d_model: int, num_heads: int, dropout: float) -> None:
        """
        Args:
            d_model: The width attended at.
            num_heads: The heads `d_model` is split into.
            dropout: The attention-weight dropout rate in training mode.
        """
        super().__init__()
        self.dropout = dropout
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.query_projection = nn.Linear(in_features=d_model, out_features=d_model)
        self.key_projection = nn.Linear(in_features=d_model, out_features=d_model)
        self.value_projection = nn.Linear(in_features=d_model, out_features=d_model)
        self.output_projection = nn.Linear(in_features=d_model, out_features=d_model)
        # Matches `nn.MultiheadAttention`: Xavier uniform q/k/v weights, zero biases throughout,
        # output weights at their default.
        for projection in (self.query_projection, self.key_projection, self.value_projection):
            nn.init.xavier_uniform_(tensor=projection.weight)
            nn.init.zeros_(tensor=projection.bias)
        nn.init.zeros_(tensor=self.output_projection.bias)

    def project(self, source: torch.Tensor) -> ProjectedKeysValues:
        """Project a sequence `[B, S, D]` to the keys and values attention reads it as."""
        return ProjectedKeysValues(
            keys=self._split_heads(self.key_projection(source)),
            values=self._split_heads(self.value_projection(source)),
        )

    def forward(
        self,
        query: torch.Tensor,
        keys_values: ProjectedKeysValues,
        *,
        key_padding_mask: torch.Tensor | None = None,
        causal: bool = False,
    ) -> torch.Tensor:
        """Attend from `query` `[B, T, D]` over `keys_values`.

        Args:
            query: The attending positions.
            keys_values: What to attend over, from `project`.
            key_padding_mask: `[B, S]`, True where a source position holds padding.
            causal: Whether a position attends only over itself and what precedes it. Valid only
                when the query covers every key, which is the un-cached pass.

        Returns:
            `[B, T, D]`.
        """
        batch_size, query_len, d_model = query.shape
        attention_mask = None
        if key_padding_mask is not None:
            attention_mask = ~key_padding_mask[:, None, None, :]  # [B, S] -> [B, 1, 1, S]
        attended = F.scaled_dot_product_attention(
            query=self._split_heads(self.query_projection(query)),
            key=keys_values.keys,
            value=keys_values.values,
            attn_mask=attention_mask,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=causal,
        )  # [B, H, T, Dh]
        merged = attended.transpose(dim0=1, dim1=2).reshape(
            (batch_size, query_len, d_model)
        )  # [B, H, T, Dh] -> [B, T, D]
        return self.output_projection(merged)

    def _split_heads(self, projected: torch.Tensor) -> torch.Tensor:
        batch_size, length, _ = projected.shape
        return projected.view((batch_size, length, self.num_heads, self.head_dim)).transpose(
            dim0=1, dim1=2
        )  # [B, T, D] -> [B, H, T, Dh]


class FeedForward(nn.Module):
    """Position-wise two-layer perceptron with dropout on its hidden features."""

    def __init__(self, *, d_model: int, feedforward_dim: int, dropout: DropoutSite) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(in_features=d_model, out_features=feedforward_dim),
            nn.ReLU(),
            dropout(shape=(feedforward_dim,)),
            nn.Linear(in_features=feedforward_dim, out_features=d_model),
        )

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.layers(hidden)


class EncoderLayer(nn.Module):
    """Pre-norm bidirectional self-attention over prefix events."""

    def __init__(self, config: DictConfig, *, d_model: int, dropout: DropoutScheme) -> None:
        super().__init__()
        self.attention = MultiHeadAttention(
            d_model=d_model, num_heads=config.num_heads, dropout=dropout.attention
        )
        self.feedforward = FeedForward(
            d_model=d_model, feedforward_dim=config.feedforward_dim, dropout=dropout.site
        )
        self.attention_norm = nn.LayerNorm(normalized_shape=d_model)
        self.feedforward_norm = nn.LayerNorm(normalized_shape=d_model)
        self.attention_dropout = dropout.site(shape=(d_model,))
        self.feedforward_dropout = dropout.site(shape=(d_model,))

    def forward(self, hidden: torch.Tensor, *, pad_mask: torch.Tensor) -> torch.Tensor:
        """Run one layer over prefix events `[B, P, D]`, padding marked by `pad_mask` `[B, P]`."""
        normed = self.attention_norm(hidden)
        attended = self.attention(
            query=normed, keys_values=self.attention.project(normed), key_padding_mask=pad_mask
        )
        hidden = hidden + self.attention_dropout(attended)
        return hidden + self.feedforward_dropout(self.feedforward(self.feedforward_norm(hidden)))


@dataclass(frozen=True)
class LayerCache:
    """One decoder layer's generation cache: the projected prefix, and the suffix keys and values
    preallocated to the most steps and written one position at a time, in place, so a step's cost
    does not grow with the steps taken."""

    prefix_kv: ProjectedKeysValues  # [B, H, P, Dh]
    suffix_kv: ProjectedKeysValues  # [B, H, max_steps, Dh]

    def select(self, index: torch.Tensor) -> 'LayerCache':
        """Keep the rows at `index` `[B']`."""
        return LayerCache(
            prefix_kv=self.prefix_kv.select(index), suffix_kv=self.suffix_kv.select(index)
        )


class DecoderLayer(nn.Module):
    """Pre-norm causal self-attention over the suffix and cross-attention over the prefix."""

    def __init__(self, config: DictConfig, *, d_model: int, dropout: DropoutScheme) -> None:
        super().__init__()
        self.self_attention = MultiHeadAttention(
            d_model=d_model, num_heads=config.num_heads, dropout=dropout.attention
        )
        self.cross_attention = MultiHeadAttention(
            d_model=d_model, num_heads=config.num_heads, dropout=dropout.attention
        )
        self.feedforward = FeedForward(
            d_model=d_model, feedforward_dim=config.feedforward_dim, dropout=dropout.site
        )
        self.self_attention_norm = nn.LayerNorm(normalized_shape=d_model)
        self.cross_attention_norm = nn.LayerNorm(normalized_shape=d_model)
        self.feedforward_norm = nn.LayerNorm(normalized_shape=d_model)
        self.self_attention_dropout = dropout.site(shape=(d_model,))
        self.cross_attention_dropout = dropout.site(shape=(d_model,))
        self.feedforward_dropout = dropout.site(shape=(d_model,))

    def forward(self, hidden: torch.Tensor, *, prefix: 'EncodedPrefix') -> torch.Tensor:
        """Run one layer over every suffix position `[B, T, D]` under a causal mask."""
        normed = self.self_attention_norm(hidden)
        attended = self.self_attention(
            query=normed, keys_values=self.self_attention.project(normed), causal=True
        )
        hidden = hidden + self.self_attention_dropout(attended)
        return self._attend_prefix(
            hidden, prefix_kv=self.cross_attention.project(prefix.hidden), pad_mask=prefix.pad_mask
        )

    def step(
        self, hidden: torch.Tensor, *, cache: LayerCache, pad_mask: torch.Tensor, position: int
    ) -> torch.Tensor:
        """Run one layer over suffix slot `position` `[B, 1, D]`, writing its keys and values
        into `cache`, the prefix's padding marked by `pad_mask` `[B, P]`."""
        normed = self.self_attention_norm(hidden)
        step_kv = self.self_attention.project(normed)  # [B, H, 1, Dh]
        cache.suffix_kv.keys[:, :, position : position + 1] = step_kv.keys
        cache.suffix_kv.values[:, :, position : position + 1] = step_kv.values
        filled = ProjectedKeysValues(
            keys=cache.suffix_kv.keys[:, :, : position + 1],
            values=cache.suffix_kv.values[:, :, : position + 1],
        )
        attended = self.self_attention(query=normed, keys_values=filled)
        hidden = hidden + self.self_attention_dropout(attended)
        return self._attend_prefix(hidden, prefix_kv=cache.prefix_kv, pad_mask=pad_mask)

    def init_cache(self, prefix: 'EncodedPrefix', *, max_steps: int) -> LayerCache:
        """The cache for generation: the prefix projected once and an empty suffix cache of
        capacity `max_steps`."""
        batch_size = prefix.hidden.size(dim=0)
        shape = (batch_size, self.self_attention.num_heads, max_steps, self.self_attention.head_dim)
        return LayerCache(
            prefix_kv=self.cross_attention.project(prefix.hidden),
            suffix_kv=ProjectedKeysValues(
                keys=prefix.hidden.new_zeros(size=shape), values=prefix.hidden.new_zeros(size=shape)
            ),
        )

    def _attend_prefix(
        self, hidden: torch.Tensor, *, prefix_kv: ProjectedKeysValues, pad_mask: torch.Tensor
    ) -> torch.Tensor:
        """Cross-attend over the projected prefix, then the feedforward sublayer."""
        attended = self.cross_attention(
            query=self.cross_attention_norm(hidden),
            keys_values=prefix_kv,
            key_padding_mask=pad_mask,
        )
        hidden = hidden + self.cross_attention_dropout(attended)
        return hidden + self.feedforward_dropout(self.feedforward(self.feedforward_norm(hidden)))


@dataclass(frozen=True)
class EncodedPrefix:
    """Encoded prefix events and their padding."""

    hidden: torch.Tensor  # [B, P, D]
    pad_mask: torch.Tensor  # [B, P], True at padding

    @property
    def lengths(self) -> torch.Tensor:
        """`[B]` events per prefix, the trace position the suffix starts at."""
        return self.pad_mask.logical_not().sum(dim=1)


@dataclass(frozen=True)
class TransformerState:
    """What a generation carries between steps."""

    pad_mask: torch.Tensor  # [B, P], True at prefix padding
    starts: torch.Tensor  # [B], the trace position of suffix slot 0
    caches: list[LayerCache]

    def select(self, index: torch.Tensor) -> 'TransformerState':
        """Keep the rows at `index` `[B']`."""
        return TransformerState(
            pad_mask=self.pad_mask[index],
            starts=self.starts[index],
            caches=[cache.select(index) for cache in self.caches],
        )


class TransformerBackbone(nn.Module):
    """Prefix encoder and causal suffix decoder over trace positions, as in SuTraN.

    Positions are trace positions: suffix slot `j` sits at position `prefix_length + j`.
    """

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
            config: The `backbone` section: layers, heads, and feedforward width.
            content: The event embedding shared by encoder and decoder, whose length cap sizes
                the position table.
            d_model: The width the stacks run at.
            dropout: The dropout of every site and of the attention weights.
        """
        super().__init__()
        self.register_buffer(
            name='positional_encoding',
            tensor=position_table(max_trace_length=content.max_trace_length, d_model=d_model),
        )
        self.encoder_input = EmbeddingInput(content, d_model=d_model, dropout=dropout.site)
        self.encoder_layers = nn.ModuleList(
            EncoderLayer(config, d_model=d_model, dropout=dropout) for _ in range(config.num_layers)
        )
        self.encoder_norm = nn.LayerNorm(normalized_shape=d_model)
        self.decoder_input = EmbeddingInput(content, d_model=d_model, dropout=dropout.site)
        self.decoder_layers = nn.ModuleList(
            DecoderLayer(config, d_model=d_model, dropout=dropout) for _ in range(config.num_layers)
        )
        # Pre-norm leaves the last layer's residual stream unnormalized, so each stack closes
        # with a norm of its own.
        self.decoder_norm = nn.LayerNorm(normalized_shape=d_model)

    def encode(self, prefix: Events) -> EncodedPrefix:
        """Encode prefix events `[B, P]` with bidirectional attention over unpadded events."""
        pad_mask = prefix.pad_mask()
        hidden = self.encoder_input(
            prefix, self.positional_encoding[: prefix.activities.size(dim=1)]
        )
        for layer in self.encoder_layers:
            hidden = layer(hidden, pad_mask=pad_mask)
        return EncodedPrefix(hidden=self.encoder_norm(hidden), pad_mask=pad_mask)

    def decode(self, inputs: Events, prefix: EncodedPrefix) -> torch.Tensor:
        """Read every teacher-forced decoder input `[B, T]` at once under a causal mask.

        No suffix padding mask: under the causal mask a padded position is visible only to later
        positions, themselves padded and dropped from the loss. Masking it would leave a row with
        nothing to attend to, whose softmax is a NaN.

        Returns:
            `[B, T, D]`, position `j` having read inputs `0..j` only.
        """
        hidden = self.decoder_input(
            inputs, self._positions(prefix.lengths, width=inputs.activities.size(dim=1))
        )
        for layer in self.decoder_layers:
            hidden = layer(hidden, prefix=prefix)
        return self.decoder_norm(hidden)

    def start(self, prefix: Events, *, max_steps: int) -> TransformerState:
        """Encode prefix events `[B, P]` for a generation of at most `max_steps` steps."""
        encoded = self.encode(prefix)
        return TransformerState(
            pad_mask=encoded.pad_mask,
            starts=encoded.lengths,
            caches=[
                layer.init_cache(encoded, max_steps=max_steps) for layer in self.decoder_layers
            ],
        )

    def step(
        self, inputs: Events, state: TransformerState, *, position: int
    ) -> tuple[torch.Tensor, TransformerState]:
        """Read the decoder input `[B, 1]` of suffix slot `position`, caching it in `state`.

        Returns:
            `[B, D]` and the state.
        """
        hidden = self.decoder_input(inputs, self._positions(state.starts + position, width=1))
        for layer, cache in zip(self.decoder_layers, state.caches, strict=True):
            hidden = layer.step(hidden, cache=cache, pad_mask=state.pad_mask, position=position)
        return self.decoder_norm(hidden)[:, 0], state

    def select(self, state: TransformerState, index: torch.Tensor) -> TransformerState:
        """Keep the generation rows at `index` `[B']`."""
        return state.select(index)

    def _positions(self, start: torch.Tensor, *, width: int) -> torch.Tensor:
        """`[B, width, D]` encodings of each row's positions from `start` `[B]` on."""
        positions = torch.arange(end=width, device=start.device)
        return self.positional_encoding[start.unsqueeze(dim=1) + positions]  # [B, 1] + [W]
