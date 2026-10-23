from collections.abc import Sequence
from dataclasses import dataclass
from functools import cached_property

import numpy as np
import pandas as pd

# Canonical names of the structural columns every split carries, following the pm4py naming. A
# dataset's other columns keep the names of its original log.
CASE_KEY = 'case:concept:name'
ACTIVITY_KEY = 'concept:name'
RESOURCE_KEY = 'org:resource'
TIMESTAMP_KEY = 'time:timestamp'


# A prefix: the case it is cut from and how many of its events it holds.
type PrefixKey = tuple[str, int]


def inter_event_minutes(events: pd.DataFrame) -> np.ndarray:
    """Return the minutes since the previous event of the same case, zero at a case's first event.

    Args:
        events: Events with the canonical case and timestamp columns, each case's rows in order.

    Returns:
        `[len(events)]` float64 minutes.
    """
    delta = events.groupby(CASE_KEY, sort=False)[TIMESTAMP_KEY].diff()
    return (delta.dt.total_seconds() / 60.0).fillna(0.0).to_numpy(np.float64)


class Log:
    """One split of a benchmark, in log units, one row per event.

    Each case is a run of consecutive rows ordered by timestamp. The columns are the canonical
    case, activity, resource, and timestamp columns and every other column of the original log,
    unchanged: no derived feature and no encoding. A cut `(case_id, k)` splits a case into its
    first `k` events, the prefix, and the rest, the suffix; both hold at least one event.
    """

    def __init__(self, events: pd.DataFrame, *, prefix_starts: np.ndarray | None = None) -> None:
        """
        Args:
            events: The split's events, each case's rows consecutive and in time order.
            prefix_starts: `[cases]` the first cut of each case, in case order. Every cut of
                every case when None.

        Raises:
            ValueError: If a structural column is missing or null, a case is not contiguous,
                or its timestamps decrease.
        """
        missing = {CASE_KEY, ACTIVITY_KEY, RESOURCE_KEY, TIMESTAMP_KEY} - set(events.columns)
        if missing:
            raise ValueError(f'A log needs the columns {sorted(missing)}')
        if events[[CASE_KEY, ACTIVITY_KEY, TIMESTAMP_KEY]].isna().any().any():
            raise ValueError('A log has a null case, activity, or timestamp')
        events = events.reset_index(drop=True)
        cases = events[CASE_KEY].astype(str)
        changes = np.flatnonzero(cases.ne(cases.shift()).to_numpy())
        if cases.iloc[changes].duplicated().any():
            raise ValueError("A log's cases must each be one run of consecutive rows")
        if (inter_event_minutes(events) < 0).any():
            raise ValueError("A log's events must be in time order within each case")
        self._events = events
        self._case_ids = cases.iloc[changes].to_numpy(dtype=object)
        self._offsets = np.append(changes, len(events)).astype(np.int64)
        lengths = np.diff(self._offsets)
        starts = np.ones(len(lengths), np.int64) if prefix_starts is None else prefix_starts
        if starts.shape != lengths.shape or (starts < 1).any():
            raise ValueError('A log needs one positive first cut per case')
        self._prefix_starts = starts.astype(np.int64)

    def __len__(self) -> int:
        """How many events the log holds."""
        return len(self._events)

    @property
    def events(self) -> pd.DataFrame:
        """A copy of the events, one row per event."""
        return self._events.copy()

    @property
    def case_ids(self) -> tuple[str, ...]:
        """Every case, in log order."""
        return tuple(self._case_ids)

    def inter_event_minutes(self) -> np.ndarray:
        """`[len(self)]` float64 minutes since the previous event of the same case, zero first.

        Draws are spelled in these units, so a model trained on them predicts the scored quantity.
        """
        return self._minutes.copy()

    @property
    def case_offsets(self) -> np.ndarray:
        """`[cases + 1]` int64 rows where each case starts, then the past-the-end row.

        Case `i` is rows `case_offsets[i]:case_offsets[i + 1]` of `events`.
        """
        return self._offsets.copy()

    def cuts(self) -> list[PrefixKey]:
        """Every cut of the split, case by case in log order, each case's by ascending length."""
        lengths = np.diff(self._offsets)
        return [
            (case_id, k)
            for case_id, start, length in zip(
                self._case_ids, self._prefix_starts, lengths, strict=True
            )
            for k in range(int(start), int(length))
        ]

    def prefixes(self, keys: Sequence[PrefixKey]) -> 'PrefixBatch':
        """Gather the observed events of each prefix.

        Args:
            keys: Cuts of this log.

        Returns:
            The prefixes in the order of `keys`.

        Raises:
            ValueError: If a key is not a cut of this log.
        """
        rows = [self._rows(key) for key in keys]
        index = np.concatenate([np.zeros(0, np.int64), *(np.arange(*row) for row in rows)])
        lengths = np.fromiter((stop - start for start, stop in rows), np.int64, len(rows))
        return PrefixBatch(
            keys=tuple(keys),
            events=self._events.iloc[index].reset_index(drop=True),
            offsets=np.concatenate([np.zeros(1, np.int64), np.cumsum(lengths)]),
        )

    def prefix_trace(self, key: PrefixKey) -> tuple[np.ndarray, np.ndarray]:
        """The activity names `[k]` and inter-event minutes `[k]` of one prefix.

        Raises:
            ValueError: If the key is not a cut of this log.
        """
        start, stop = self._rows(key)
        return self._activities[start:stop].copy(), self._minutes[start:stop].copy()

    def suffix_trace(self, key: PrefixKey) -> tuple[np.ndarray, np.ndarray]:
        """The activity names `[L]` and inter-event minutes `[L]` of one prefix's suffix.

        Raises:
            ValueError: If the key is not a cut of this log.
        """
        _, stop = self._rows(key)
        end = self._offsets[self._case_index[key[0]] + 1]
        return self._activities[stop:end].copy(), self._minutes[stop:end].copy()

    def _rows(self, key: PrefixKey) -> tuple[int, int]:
        """The `[start, stop)` rows of one prefix.

        Raises:
            ValueError: If the key is not a cut of this log.
        """
        case_id, k = key
        case = self._case_index.get(case_id)
        if case is None:
            raise ValueError(f'No case {case_id!r} in this log')
        start = int(self._offsets[case])
        length = int(self._offsets[case + 1]) - start
        if not self._prefix_starts[case] <= k < length:
            raise ValueError(f'{key} is not a cut of this log')
        return start, start + int(k)

    @cached_property
    def _case_index(self) -> dict[str, int]:
        return {case_id: index for index, case_id in enumerate(self._case_ids)}

    @cached_property
    def _activities(self) -> np.ndarray:
        return self._events[ACTIVITY_KEY].astype(str).to_numpy(dtype=object)

    @cached_property
    def _minutes(self) -> np.ndarray:
        return inter_event_minutes(self._events)


@dataclass(frozen=True, eq=False)
class PrefixBatch:
    """Observed prefixes handed to a model, in log units.

    Prefix `i` is `keys[i]` and its events are rows `offsets[i]:offsets[i + 1]` of `events`, with
    the columns of the split it is cut from. Nothing past a prefix's last event is included.
    """

    keys: tuple[PrefixKey, ...]
    events: pd.DataFrame
    offsets: np.ndarray  # [N + 1]

    def __post_init__(self) -> None:
        lengths = np.diff(self.offsets)
        if self.offsets.shape != (len(self.keys) + 1,) or self.offsets[-1] != len(self.events):
            raise ValueError('Prefix offsets do not match their keys and events')
        if any(int(length) != k for length, (_, k) in zip(lengths, self.keys, strict=True)):
            raise ValueError('Prefix events do not match their prefix lengths')

    def __len__(self) -> int:
        return len(self.keys)

    def __getitem__(self, index: int) -> pd.DataFrame:
        """The events of prefix `index`."""
        return self.events.iloc[self.offsets[index] : self.offsets[index + 1]]


@dataclass(frozen=True, eq=False)
class SuffixDraws:
    """A model's sampled suffixes for one prefix, in log units.

    Draw `i` is the activity names `activities[i]`, the inter-event minutes `minutes[i]` before
    each of them, measured from the previous event and from the last prefix event for the first,
    and whether the model closed it at the length cap without ending it, `truncated[i]`.
    """

    activities: Sequence[Sequence[str]]
    minutes: Sequence[np.ndarray]
    truncated: Sequence[bool]

    def __post_init__(self) -> None:
        if not len(self.activities) == len(self.minutes) == len(self.truncated):
            raise ValueError('Suffix draws disagree on how many draws there are')
        for names, minutes in zip(self.activities, self.minutes, strict=True):
            if np.shape(minutes) != (len(names),):
                raise ValueError('A draw needs one inter-event time per activity')

    def __len__(self) -> int:
        return len(self.activities)
