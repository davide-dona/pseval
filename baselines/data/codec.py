from dataclasses import dataclass
from functools import cached_property

import numpy as np
import pandas as pd
from omegaconf import DictConfig
from pandas.api.types import is_numeric_dtype

import pseval
from baselines.data.features import derive_features
from pseval.data import INTER_EVENT_MINUTES_KEY, RESOURCE_KEY

# Special tokens. Each precedes the vocabulary of whichever channel carries it.
PAD_TOKEN = 'PAD'  # Padding, aligning sequences to one width in a batch
EOT_TOKEN = 'EOT'  # End Of Trace
SOS_TOKEN = 'SOS'  # Start Of Suffix, the first decoder input
UNK_TOKEN = 'UNK'  # A value the train split did not hold

# The special tokens of each channel, in index order. PAD is row 0 of every table.
ACTIVITY_TOKENS = (PAD_TOKEN, EOT_TOKEN, SOS_TOKEN)
RESOURCE_TOKENS = (PAD_TOKEN, UNK_TOKEN)
FEATURE_TOKENS = (UNK_TOKEN,)

# The value a missing categorical entry is read as, so a gap is a value of its own.
MISSING_FEATURE = '<MISSING>'


@dataclass(frozen=True)
class ActivityColumn:
    """The activity channel: PAD, EOT, and SOS, then each benchmark vocabulary name in order.

    The benchmark vocabulary ends with its `UNK` entry, so an activity unseen in train is a modeled
    class like any other, and every row from `first_name` on decodes to a name pseval accepts.
    """

    vocabulary: tuple[str, ...]

    pad_index = ACTIVITY_TOKENS.index(PAD_TOKEN)
    eot_index = ACTIVITY_TOKENS.index(EOT_TOKEN)
    sos_index = ACTIVITY_TOKENS.index(SOS_TOKEN)
    first_name = len(ACTIVITY_TOKENS)

    @property
    def num_rows(self) -> int:
        """Rows of the activity embedding table and classes of the activity head."""
        return self.first_name + len(self.vocabulary)

    @cached_property
    def _rows(self) -> dict[str, int]:
        return {name: self.first_name + i for i, name in enumerate(self.vocabulary)}

    @cached_property
    def _names(self) -> np.ndarray:
        return np.array(self.vocabulary, dtype=object)

    def encode(self, names: pd.Series) -> np.ndarray:
        """`[len(names)]` int64 rows, the vocabulary's last entry, `UNK`, for a name outside it."""
        return names.astype(str).map(self._rows).fillna(self.num_rows - 1).to_numpy(np.int64)

    def decode(self, rows: np.ndarray) -> np.ndarray:
        """The names of rows `[...]`, each from `first_name` on, as an object array `[...]`."""
        return self._names[rows - self.first_name]


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
    def unk_index(self) -> int:
        return self.offset + self.special_tokens.index(UNK_TOKEN)

    def encode(self, events: pd.DataFrame) -> np.ndarray:
        """`[len(events)]` int64 rows of this channel's values, `UNK` for one unseen in train."""
        return (
            _text(events[self.column])
            .map(self.to_index)
            .fillna(self.unk_index)
            .to_numpy(dtype=np.int64)
        )


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

    def standardize[T](self, scaled: T) -> T:
        """Standardize values already through the log1p, if any: arrays or tensors alike."""
        return (scaled - self.mean) / self.std

    def unstandardize[T](self, standardized: T) -> T:
        """Read standardized values back on the log1p scale, if any: arrays or tensors alike."""
        return standardized * self.std + self.mean

    def normalize(self, values: np.ndarray) -> np.ndarray:
        """Standardize raw values against the train statistics, as float32, unbounded."""
        return self.standardize(np.log1p(values) if self.log else values).astype(np.float32)

    def denormalize(self, normalized: np.ndarray) -> np.ndarray:
        """Read standardized values back as the raw float64 quantity they came from."""
        scaled = self.unstandardize(np.asarray(normalized, dtype=np.float64))
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
    # The benchmark's length cap, the most events a draw may hold.
    max_trace_length: int

    @classmethod
    def fit(cls, benchmark: pseval.Benchmark, data_config: DictConfig) -> 'Codec':
        """Fit the codec on the benchmark's train split, with its vocabulary and length cap.

        Args:
            benchmark: The benchmark.
            data_config: The `data` section: the feature columns, which of them are log-scaled,
                and whether the target inter-event time is.
        """
        train = benchmark.train
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
            activity=ActivityColumn(vocabulary=benchmark.vocabulary),
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
            max_trace_length=benchmark.max_suffix_length,
        )

    @property
    def num_feature_categories(self) -> int:
        """Rows of the table every categorical feature shares: one PAD, then a block each."""
        return 1 + sum(feature.num_rows for feature in self.categorical_features)

    def encode_categorical_features(self, events: pd.DataFrame) -> np.ndarray:
        """`[len(events), num_categorical]` int64 shared-table rows."""
        return _columns(
            [feature.encode(events) for feature in self.categorical_features],
            rows=len(events),
            dtype=np.int64,
        )

    def encode_numeric_features(self, events: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        """`[len(events), num_numeric]` float32 standardized values and presence flags."""
        return (
            _columns(
                [feature.encode(events) for feature in self.numeric_features],
                rows=len(events),
                dtype=np.float32,
            ),
            _columns(
                [feature.present(events) for feature in self.numeric_features],
                rows=len(events),
                dtype=np.float32,
            ),
        )


def _columns(columns: list[np.ndarray], *, rows: int, dtype: type) -> np.ndarray:
    """Stack per-column arrays `[rows]` into `[rows, len(columns)]`, empty when there are none."""
    return np.stack(columns, axis=1) if columns else np.zeros((rows, 0), dtype=dtype)


def _text(values: pd.Series) -> pd.Series:
    """A categorical column as text, a missing entry read as `MISSING_FEATURE`."""
    return values.astype(object).where(values.notna(), MISSING_FEATURE).astype(str)
