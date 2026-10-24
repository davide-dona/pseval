from dataclasses import dataclass

import torch
import torch.nn.functional as F
from omegaconf import DictConfig
from torch import nn

from baselines.data.dataset import Events
from baselines.models.dropout import SequenceDropout
from baselines.models.embeddings import EventContentEmbedding, position_table


@dataclass(frozen=True)
class ProjectedKeysValues:
    """One attention input, already projected and split across heads.

    The encoded prefix does not change while a suffix is being written, and a suffix position
    already written does not change when the next one is added, so both are projected once.
    """

    keys: torch.Tensor  # [B, H, S, Dh]
    values: torch.Tensor  # [B, H, S, Dh]


class MultiHeadAttention(nn.Module):
    """Multi-head attention with cached key/value projections and no attention-weight dropout,
    which has no analog shared across positions."""

    def __init__(self, *, d_model: int, num_heads: int) -> None:
        super().__init__()
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
    """Position-wise two-layer perceptron with variational dropout on its hidden features."""

    def __init__(self, *, d_model: int, feedforward_dim: int, dropout: float) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(in_features=d_model, out_features=feedforward_dim),
            nn.ReLU(),
            SequenceDropout(p=dropout, shape=(feedforward_dim,)),
            nn.Linear(in_features=feedforward_dim, out_features=d_model),
        )

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.layers(hidden)


class EncoderLayer(nn.Module):
    """Pre-norm bidirectional self-attention over prefix events."""

    def __init__(self, config: DictConfig, *, d_model: int, dropout: float) -> None:
        super().__init__()
        self.attention = MultiHeadAttention(d_model=d_model, num_heads=config.num_heads)
        self.feedforward = FeedForward(
            d_model=d_model, feedforward_dim=config.feedforward_dim, dropout=dropout
        )
        self.attention_norm = nn.LayerNorm(normalized_shape=d_model)
        self.feedforward_norm = nn.LayerNorm(normalized_shape=d_model)
        self.attention_dropout = SequenceDropout(p=dropout, shape=(d_model,))
        self.feedforward_dropout = SequenceDropout(p=dropout, shape=(d_model,))

    def forward(self, hidden: torch.Tensor, *, pad_mask: torch.Tensor) -> torch.Tensor:
        """Run one layer over prefix events `[B, P, D]`, padding marked by `pad_mask` `[B, P]`."""
        normed = self.attention_norm(hidden)
        attended = self.attention(
            query=normed, keys_values=self.attention.project(normed), key_padding_mask=pad_mask
        )
        hidden = hidden + self.attention_dropout(attended)
        return hidden + self.feedforward_dropout(self.feedforward(self.feedforward_norm(hidden)))


@dataclass(frozen=True)
class SuffixCache:
    """The suffix self-attention's key/value cache, preallocated to its final length and filled
    one decoding step at a time, in place, so a step's cost does not grow with the steps taken."""

    keys: torch.Tensor  # [B, H, max_steps, Dh]
    values: torch.Tensor  # [B, H, max_steps, Dh]
    length: int = 0

    def write(self, step: ProjectedKeysValues) -> 'SuffixCache':
        """Write one step's projection `[B, H, 1, Dh]` into the next free position."""
        self.keys[:, :, self.length : self.length + 1] = step.keys
        self.values[:, :, self.length : self.length + 1] = step.values
        return SuffixCache(keys=self.keys, values=self.values, length=self.length + 1)

    def filled(self) -> ProjectedKeysValues:
        """The positions written so far, as a view over the buffer."""
        return ProjectedKeysValues(
            keys=self.keys[:, :, : self.length], values=self.values[:, :, : self.length]
        )


@dataclass(frozen=True)
class LayerCache:
    """One decoder layer's cache: the projected prefix and the suffix positions already read."""

    prefix_kv: ProjectedKeysValues
    suffix_kv: SuffixCache


class DecoderLayer(nn.Module):
    """Pre-norm causal self-attention over the suffix and cross-attention over the prefix."""

    def __init__(self, config: DictConfig, *, d_model: int, dropout: float) -> None:
        super().__init__()
        self.self_attention = MultiHeadAttention(d_model=d_model, num_heads=config.num_heads)
        self.cross_attention = MultiHeadAttention(d_model=d_model, num_heads=config.num_heads)
        self.feedforward = FeedForward(
            d_model=d_model, feedforward_dim=config.feedforward_dim, dropout=dropout
        )
        self.self_attention_norm = nn.LayerNorm(normalized_shape=d_model)
        self.cross_attention_norm = nn.LayerNorm(normalized_shape=d_model)
        self.feedforward_norm = nn.LayerNorm(normalized_shape=d_model)
        self.self_attention_dropout = SequenceDropout(p=dropout, shape=(d_model,))
        self.cross_attention_dropout = SequenceDropout(p=dropout, shape=(d_model,))
        self.feedforward_dropout = SequenceDropout(p=dropout, shape=(d_model,))

    def forward(
        self,
        hidden: torch.Tensor,
        *,
        prefix: 'EncodedPrefix',
        cache: LayerCache | None,
    ) -> tuple[torch.Tensor, LayerCache]:
        """Run one layer over suffix positions `[B, T, D]`: the whole suffix without a cache, the
        one position that follows it with one.

        Returns:
            The layer's output for the positions read, and the cache to hand the next call.
        """
        normed = self.self_attention_norm(hidden)
        step_kv = self.self_attention.project(normed)
        if cache is not None:
            suffix_cache = cache.suffix_kv.write(step_kv)
            suffix_kv = suffix_cache.filled()
        else:
            suffix_cache = SuffixCache(
                keys=step_kv.keys, values=step_kv.values, length=step_kv.keys.size(dim=2)
            )
            suffix_kv = step_kv
        attended = self.self_attention(query=normed, keys_values=suffix_kv, causal=cache is None)
        hidden = hidden + self.self_attention_dropout(attended)

        prefix_kv = (
            cache.prefix_kv if cache is not None else self.cross_attention.project(prefix.hidden)
        )
        attended = self.cross_attention(
            query=self.cross_attention_norm(hidden),
            keys_values=prefix_kv,
            key_padding_mask=prefix.pad_mask,
        )
        hidden = hidden + self.cross_attention_dropout(attended)
        hidden = hidden + self.feedforward_dropout(self.feedforward(self.feedforward_norm(hidden)))
        return hidden, LayerCache(prefix_kv=prefix_kv, suffix_kv=suffix_cache)

    def init_cache(self, prefix: 'EncodedPrefix', *, max_steps: int) -> LayerCache:
        """The cache for generation: the prefix projected once and an empty suffix cache of
        capacity `max_steps`."""
        batch_size = prefix.hidden.size(dim=0)
        shape = (batch_size, self.self_attention.num_heads, max_steps, self.self_attention.head_dim)
        return LayerCache(
            prefix_kv=self.cross_attention.project(prefix.hidden),
            suffix_kv=SuffixCache(
                keys=prefix.hidden.new_zeros(size=shape),
                values=prefix.hidden.new_zeros(size=shape),
            ),
        )


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

    prefix: EncodedPrefix
    caches: list[LayerCache]


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
        dropout: float,
        max_trace_length: int,
    ) -> None:
        """
        Args:
            config: The `backbone` section: layers, heads, and feedforward width.
            content: The event embedding shared by encoder and decoder.
            d_model: The width the stacks run at.
            dropout: The variational dropout rate of every site.
            max_trace_length: The longest case, which sizes the position table.
        """
        super().__init__()
        self.content = content
        self.register_buffer(
            name='positional_encoding',
            tensor=position_table(max_trace_length=max_trace_length, d_model=d_model),
        )
        # The embeddings are shared but these norms are not: the two stacks read one embedding
        # space at whatever scale each of them settles on.
        self.encoder_dropout = SequenceDropout(p=dropout, shape=(d_model,))
        self.encoder_embedding_norm = nn.LayerNorm(normalized_shape=d_model)
        self.encoder_layers = nn.ModuleList(
            EncoderLayer(config, d_model=d_model, dropout=dropout) for _ in range(config.num_layers)
        )
        self.encoder_norm = nn.LayerNorm(normalized_shape=d_model)
        self.decoder_dropout = SequenceDropout(p=dropout, shape=(d_model,))
        self.decoder_embedding_norm = nn.LayerNorm(normalized_shape=d_model)
        self.decoder_layers = nn.ModuleList(
            DecoderLayer(config, d_model=d_model, dropout=dropout) for _ in range(config.num_layers)
        )
        # Pre-norm leaves the last layer's residual stream unnormalized, so each stack closes
        # with a norm of its own.
        self.decoder_norm = nn.LayerNorm(normalized_shape=d_model)

    def encode(self, prefix: Events) -> EncodedPrefix:
        """Encode prefix events `[B, P]` with bidirectional attention over unpadded events."""
        pad_mask = prefix.pad_mask()
        hidden = self.encoder_embedding_norm(self.encoder_dropout(self._embed(prefix)))
        for layer in self.encoder_layers:
            hidden = layer(hidden, pad_mask=pad_mask)
        return EncodedPrefix(hidden=self.encoder_norm(hidden), pad_mask=pad_mask)

    def decode(self, inputs: Events, prefix: EncodedPrefix) -> torch.Tensor:
        """Read every teacher-forced decoder input `[B, T]` at once under a causal mask.

        Returns:
            `[B, T, D]`, position `j` having read inputs `0..j` only.
        """
        hidden, _ = self._run_layers(inputs, prefix=prefix, start=prefix.lengths, caches=None)
        return hidden

    def start(self, prefix: EncodedPrefix, *, max_steps: int) -> TransformerState:
        """Prepare a generation of at most `max_steps` steps."""
        return TransformerState(
            prefix=prefix,
            caches=[layer.init_cache(prefix, max_steps=max_steps) for layer in self.decoder_layers],
        )

    def step(
        self, inputs: Events, state: TransformerState, *, position: int
    ) -> tuple[torch.Tensor, TransformerState]:
        """Read the decoder input `[B, 1]` of suffix slot `position`.

        Returns:
            `[B, D]` and the state holding this position too.
        """
        hidden, caches = self._run_layers(
            inputs, prefix=state.prefix, start=state.prefix.lengths + position, caches=state.caches
        )
        return hidden[:, 0], TransformerState(prefix=state.prefix, caches=caches)

    def _embed(self, events: Events, start: torch.Tensor | None = None) -> torch.Tensor:
        """Embed events `[B, T]` as `[B, T, D]`, each row's first at position `start` `[B]`, or
        at position 0 when None."""
        content = self.content(events)
        positions = torch.arange(end=content.size(dim=1), device=content.device)  # [T]
        if start is not None:
            positions = start.unsqueeze(dim=1) + positions  # [B, 1] + [T] -> [B, T]
        return content + self.positional_encoding[positions]

    def _run_layers(
        self,
        inputs: Events,
        *,
        prefix: EncodedPrefix,
        start: torch.Tensor,
        caches: list[LayerCache] | None,
    ) -> tuple[torch.Tensor, list[LayerCache]]:
        """Embed decoder inputs and push them through the stack.

        No suffix padding mask: under the causal mask a padded position is visible only to later
        positions, themselves padded and dropped from the loss. Masking it would leave a row with
        nothing to attend to, whose softmax is a NaN.
        """
        hidden = self.decoder_embedding_norm(self.decoder_dropout(self._embed(inputs, start)))
        layer_caches = caches if caches is not None else [None] * len(self.decoder_layers)
        new_caches: list[LayerCache] = []
        for layer, cache in zip(self.decoder_layers, layer_caches, strict=True):
            hidden, layer_cache = layer(hidden, prefix=prefix, cache=cache)
            new_caches.append(layer_cache)
        return self.decoder_norm(hidden), new_caches
