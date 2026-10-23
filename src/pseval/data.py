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
# Minutes since the previous event of the same case, zero at a case's first event.
INTER_EVENT_MINUTES_KEY = 'inter_event_minutes'


# A prefix: the case it is cut from and how many of its events it holds.
type PrefixKey = tuple[str, int]


class Log:
    """One split of a benchmark, in log units, one row per event.

    A cut `(case_id, k)` splits a case into its first `k` events, the prefix, and the rest, the
    suffix; both hold at least one event.
    """

    def __init__(self, events: pd.DataFrame, *, prefix_starts: np.ndarray | None = None) -> None:
        """
        Args:
            events: The split's events, each case's rows consecutive and in time order.
            prefix_starts: `[cases]` the first cut of each case, in case order. Every cut of
                every case when None.
        """
        cases = events[CASE_KEY].astype(str).to_numpy(dtype=object)
        # A case starts at the first row and wherever the case differs from the row before
        starts = np.flatnonzero(np.append(True, cases[1:] != cases[:-1]))
        self._events = events
        # `[cases]` id of each case, in log order.
        self._case_ids = cases[starts]
        # `[cases + 1]` row where each case starts, then the past-the-end row.
        self._offsets = np.append(starts, len(events)).astype(np.int64)
        # `[cases]` length of each case's shortest cut.
        self._prefix_starts = (
            np.ones(len(starts), np.int64) if prefix_starts is None else prefix_starts
        )

    def __len__(self) -> int:
        """How many events the log holds."""
        return len(self._events)

    @property
    def events(self) -> pd.DataFrame:
        """A copy of the events, one row per event."""
        return self._events.copy()

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
        """Gather the observed events of each prefix, in the order of `keys`, cuts of this log."""
        rows = [self._rows(key) for key in keys]
        # The leading empty array keeps the concatenation valid when `keys` is empty.
        index = np.concatenate([np.zeros(0, np.int64), *(np.arange(*row) for row in rows)])
        lengths = np.fromiter((stop - start for start, stop in rows), np.int64, len(rows))
        return PrefixBatch(
            keys=tuple(keys),
            events=self._events.iloc[index].reset_index(drop=True),
            offsets=np.concatenate([np.zeros(1, np.int64), np.cumsum(lengths)]),
        )

    def prefix_activities(self, key: PrefixKey) -> np.ndarray:
        """The activity names `[k]` of one prefix."""
        start, stop = self._rows(key)
        return self._activities[start:stop].copy()

    def suffix_trace(self, key: PrefixKey) -> tuple[np.ndarray, np.ndarray]:
        """The activity names `[L]` and inter-event minutes `[L]` of one prefix's suffix."""
        _, stop = self._rows(key)
        end = self._offsets[self._case_index[key[0]] + 1]
        return self._activities[stop:end].copy(), self._minutes[stop:end].copy()

    def _rows(self, key: PrefixKey) -> tuple[int, int]:
        """The `[start, stop)` rows of one prefix."""
        case_id, k = key
        start = int(self._offsets[self._case_index[case_id]])
        return start, start + int(k)

    @cached_property
    def _case_index(self) -> dict[str, int]:
        return {case_id: index for index, case_id in enumerate(self._case_ids)}

    @cached_property
    def _activities(self) -> np.ndarray:
        return self._events[ACTIVITY_KEY].astype(str).to_numpy(dtype=object)

    @cached_property
    def _minutes(self) -> np.ndarray:
        return self._events[INTER_EVENT_MINUTES_KEY].to_numpy(np.float64)


@dataclass(frozen=True, eq=False)
class PrefixBatch:
    """Observed prefixes handed to a model, in log units.

    Prefix `i` is `keys[i]` and its events are rows `offsets[i]:offsets[i + 1]` of `events`, with
    the columns of the split it is cut from. Nothing past a prefix's last event is included.
    """

    keys: tuple[PrefixKey, ...]
    events: pd.DataFrame
    offsets: np.ndarray  # [N + 1]

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
