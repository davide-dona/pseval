from dataclasses import dataclass
from functools import cached_property

import numpy as np
import pandas as pd
import torch
from omegaconf import DictConfig
from pandas.api.types import is_numeric_dtype

from baselines.data.features import derive_features
from pseval import Log
from pseval.data import INTER_EVENT_MINUTES_KEY, RESOURCE_KEY

# Special tokens. Each follows the vocabulary of whichever channel carries it.
PAD_TOKEN = 'PAD'  # Padding, aligning sequences to one width in a batch
EOT_TOKEN = 'EOT'  # End Of Trace
SOS_TOKEN = 'SOS'  # Start Of Suffix, the first decoder input
UNK_TOKEN = 'UNK'  # A value the train split did not hold

# The special tokens of the resource and categorical feature channels, in index order.
RESOURCE_TOKENS = (PAD_TOKEN, UNK_TOKEN)
FEATURE_TOKENS = (UNK_TOKEN,)

# The value a missing categorical entry is read as, so a gap is a value of its own.
MISSING_FEATURE = '<MISSING>'


@dataclass(frozen=True)
class ActivityColumn:
    """The activity channel: each benchmark vocabulary name at its position, then PAD, EOT, SOS.

    The benchmark vocabulary ends with `UNK`, so an activity unseen in train is a modeled class
    like any other, and every index below `pad_index` decodes to a name pseval accepts.
    """

    vocabulary: tuple[str, ...]

    @property
    def pad_index(self) -> int:
        return len(self.vocabulary)

    @property
    def eot_index(self) -> int:
        return len(self.vocabulary) + 1

    @property
    def sos_index(self) -> int:
        return len(self.vocabulary) + 2

    @property
    def num_rows(self) -> int:
        """Rows of the activity embedding table and classes of the activity head."""
        return len(self.vocabulary) + 3

    @cached_property
    def unk_index(self) -> int:
        return self.vocabulary.index(UNK_TOKEN)

    @cached_property
    def to_index(self) -> dict[str, int]:
        return {name: index for index, name in enumerate(self.vocabulary)}

    def encode(self, names: pd.Series) -> np.ndarray:
        """`[len(names)]` int64 vocabulary positions, `UNK` for a name outside the vocabulary."""
        return names.astype(str).map(self.to_index).fillna(self.unk_index).to_numpy(np.int64)

    def decode(self, indices: np.ndarray) -> tuple[str, ...]:
        """The names of vocabulary positions, each below `pad_index`."""
        return tuple(self.vocabulary[int(index)] for index in indices)


@dataclass(frozen=True)
class CategoricalColumn:
    """One categorical channel of the log, and the vocabulary it is embedded through.

    Its rows start at `offset` in the table it is embedded with: the special tokens first, then
    the vocabulary.
    """

    column: str
    vocab: tuple[str, ...]
    special_tokens: tuple[str, ...]
    offset: int = 0

    @classmethod
    def fit(
        cls,
        train: pd.DataFrame,
        *,
        column: str,
        special_tokens: tuple[str, ...],
        offset: int = 0,
    ) -> 'CategoricalColumn':
        """Fit one channel's vocabulary on the train split.

        Args:
            train: The train events.
            column: Which column of them this channel reads.
            special_tokens: The tokens preceding the vocabulary, in the order they are indexed.
            offset: Where this channel's block starts in the table it is embedded with.
        """
        return cls(
            column=column,
            vocab=tuple(sorted(_text(train[column]).unique().tolist())),
            special_tokens=special_tokens,
            offset=offset,
        )

    @property
    def num_rows(self) -> int:
        """Rows this channel owns in its table, its special tokens included."""
        return len(self.vocab) + len(self.special_tokens)

    @cached_property
    def to_index(self) -> dict[str, int]:
        """Each train value to its row. No value maps to a special token, so an unseen value
        falls through to `unk_index`."""
        start = self.offset + len(self.special_tokens)
        return {value: start + i for i, value in enumerate(self.vocab)}

    @property
    def pad_index(self) -> int:
        return self._index_of(PAD_TOKEN)

    @property
    def unk_index(self) -> int:
        return self._index_of(UNK_TOKEN)

    def encode(self, events: pd.DataFrame) -> np.ndarray:
        """`[len(events)]` int64 rows of this channel's values, `UNK` for one unseen in train."""
        return (
            _text(events[self.column])
            .map(self.to_index)
            .fillna(self.unk_index)
            .to_numpy(dtype=np.int64)
        )

    def _index_of(self, token: str) -> int:
        """The row of one special token.

        Raises:
            ValueError: If this channel does not carry that token.
        """
        if token not in self.special_tokens:
            raise ValueError(f'column "{self.column}" carries no {token} token')
        return self.offset + self.special_tokens.index(token)


@dataclass(frozen=True)
class NumericColumn:
    """One numeric channel of the log, and the train-split statistics it is standardized with.

    - log: Whether the values pass through a log1p before the standardization.
    - mean: The mean of the train values, after the log1p if there is one.
    - std: The standard deviation of the same, 1.0 for a channel that never varies.
    """

    column: str
    log: bool
    mean: float
    std: float

    @classmethod
    def fit(cls, train: pd.DataFrame, *, column: str, log: bool) -> 'NumericColumn':
        """Fit one channel's standardization on the train split, skipping missing values.

        Raises:
            ValueError: If the column holds no finite value, or a negative one on a log-scaled
                channel, where log1p is undefined below -1 and compresses the rest towards it.
        """
        values = train[column].to_numpy(dtype=np.float64)
        finite = values[np.isfinite(values)]
        if finite.size == 0:
            raise ValueError(f'column "{column}" holds no finite value on the train split')
        if log and finite.min() < 0:
            raise ValueError(
                f'column "{column}" is log-scaled but holds negative values '
                f'(min {finite.min()}), which log1p cannot represent'
            )
        scaled = np.log1p(finite) if log else finite
        # A channel whose train values are all identical has no deviation to divide by. 1.0
        # leaves every one of them standardizing to 0.0 and denormalizing back to the mean.
        std = float(scaled.std())
        return cls(column=column, log=log, mean=float(scaled.mean()), std=std if std > 0 else 1.0)

    def normalize(self, values: np.ndarray) -> np.ndarray:
        """Standardize raw values against the train statistics, as float32, unbounded."""
        scaled = np.log1p(values) if self.log else values
        return ((scaled - self.mean) / self.std).astype(np.float32)

    def denormalize(self, normalized: np.ndarray) -> np.ndarray:
        """Read standardized values back as the raw float64 quantity they came from."""
        scaled = np.asarray(normalized, dtype=np.float64) * self.std + self.mean
        return np.expm1(scaled) if self.log else scaled

    def encode(self, events: pd.DataFrame) -> np.ndarray:
        """`[len(events)]` float32 standardized values, a missing one pinned to exactly 0.0."""
        raw = events[self.column].to_numpy(dtype=np.float64)
        finite = np.nan_to_num(raw, nan=0.0, posinf=0.0, neginf=0.0)
        return self.normalize(finite) * self.present(events)

    def present(self, events: pd.DataFrame) -> np.ndarray:
        """`[len(events)]` float32, 0.0 where the log has no value, which 0.0 alone cannot say."""
        return np.isfinite(events[self.column].to_numpy(dtype=np.float64)).astype(np.float32)


@dataclass(frozen=True)
class Codec:
    """What a model's inputs and outputs are encoded through, fitted on the train split."""

    activity: ActivityColumn
    resource: CategoricalColumn
    # The time channel every event carries as model input. Always log-scaled.
    input_inter_event_time: NumericColumn
    # The time models predict and draws are decoded through.
    inter_event_time: NumericColumn
    # The configured event features, sorted by dtype into the two kinds.
    categorical_features: tuple[CategoricalColumn, ...]
    numeric_features: tuple[NumericColumn, ...]
    # The longest case the benchmark keeps, which sizes every sequence tensor.
    max_trace_length: int

    @classmethod
    def fit(
        cls,
        train: Log,
        *,
        vocabulary: tuple[str, ...],
        max_trace_length: int,
        data_config: DictConfig,
    ) -> 'Codec':
        """Fit the codec on the train split.

        Args:
            train: The benchmark's train split.
            vocabulary: The benchmark vocabulary, every name a draw may hold.
            max_trace_length: The benchmark's length cap, its longest case kept.
            data_config: The `data` section: the feature columns, which of them are log-scaled,
                and whether the target inter-event time is.
        """
        events = derive_features(train.events, train.case_offsets)
        log_scaled = set(data_config.log_scaled_features)
        categorical: list[CategoricalColumn] = []
        numeric: list[NumericColumn] = []
        # Row 0 of the shared feature table is the PAD every feature uses.
        offset = 1
        for column in data_config.event_features:
            if is_numeric_dtype(events[column]):
                numeric.append(NumericColumn.fit(events, column=column, log=column in log_scaled))
            else:
                feature = CategoricalColumn.fit(
                    events, column=column, special_tokens=FEATURE_TOKENS, offset=offset
                )
                categorical.append(feature)
                offset += feature.num_rows
        return cls(
            activity=ActivityColumn(vocabulary=vocabulary),
            resource=CategoricalColumn.fit(
                events, column=RESOURCE_KEY, special_tokens=RESOURCE_TOKENS
            ),
            input_inter_event_time=NumericColumn.fit(
                events, column=INTER_EVENT_MINUTES_KEY, log=True
            ),
            inter_event_time=NumericColumn.fit(
                events,
                column=INTER_EVENT_MINUTES_KEY,
                log=data_config.log_scaled_target_inter_event_time,
            ),
            categorical_features=tuple(categorical),
            numeric_features=tuple(numeric),
            max_trace_length=max_trace_length,
        )

    @property
    def num_feature_categories(self) -> int:
        """Rows of the table every categorical feature shares: one PAD, then a block each."""
        return 1 + sum(feature.num_rows for feature in self.categorical_features)

    def encode_categorical_features(self, events: pd.DataFrame) -> torch.Tensor:
        """`[len(events), num_categorical]` int64 shared-table rows."""
        return torch.from_numpy(
            _columns([feature.encode(events) for feature in self.categorical_features], len(events))
        ).long()

    def encode_numeric_features(self, events: pd.DataFrame) -> tuple[torch.Tensor, torch.Tensor]:
        """`[len(events), num_numeric]` float32 standardized values and presence flags."""
        values = [feature.encode(events) for feature in self.numeric_features]
        present = [feature.present(events) for feature in self.numeric_features]
        return (
            torch.from_numpy(_columns(values, len(events))).float(),
            torch.from_numpy(_columns(present, len(events))).float(),
        )


def _columns(columns: list[np.ndarray], rows: int) -> np.ndarray:
    """Stack per-column arrays `[rows]` into `[rows, len(columns)]`, empty when there are none."""
    return np.stack(columns, axis=1) if columns else np.zeros((rows, 0))


def _text(values: pd.Series) -> pd.Series:
    """A categorical column as text, a missing entry read as `MISSING_FEATURE`."""
    return values.astype(object).where(values.notna(), MISSING_FEATURE).astype(str)
