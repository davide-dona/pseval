import numpy as np

from baselines.data.codec import ActivityColumn
from pseval import Draw, Forecast, Log, PrefixBatch, SuffixModel
from pseval.data import ACTIVITY_KEY

# Marks the start of a trace in a context, below every activity row.
_START = np.int32(-1)


class Retrieval(SuffixModel):
    """Draw real train suffixes whose preceding activities match the prefix's latest ones.

    A context is the run of activities right before a train cut, read backwards and closed by a
    start-of-trace marker. A prefix is matched to the longest context shared with at least one
    train cut: the whole prefix when a train case began with exactly these activities, otherwise
    its last k activities for the largest k some cut shares, and every cut when not even the last
    activity was seen. Each draw is the observed suffix, activities and inter-event minutes, of a
    cut drawn uniformly from the matched ones: the empirical conditional distribution of
    continuations under back-off.
    """

    def __init__(self, train: Log, *, vocabulary: tuple[str, ...]) -> None:
        """Index every cut of the train split by its context.

        Args:
            train: The train split.
            vocabulary: The benchmark vocabulary. A prefix activity outside it is read as its
                `UNK` entry, which no train context holds.
        """
        self._activity = ActivityColumn(vocabulary=vocabulary)
        events = train.events
        self._names = events[ACTIVITY_KEY].astype(str).to_numpy(dtype=object)
        self._minutes = train.inter_event_minutes()
        codes = self._activity.encode(events[ACTIVITY_KEY]).astype(np.int32)
        offsets = train.case_offsets
        cuts = train.cuts()
        case = train.case_indices(cuts)
        # [N] The first row of every cut's case, and the first and past-the-end rows of its suffix.
        case_starts = offsets[case]
        self._starts = case_starts + np.fromiter(
            (cut.prefix_len for cut in cuts), np.int64, len(cuts)
        )
        self._stops = offsets[case + 1]
        # Depth `d` holds the activity `d + 1` events before the cut, `_START` from the trace start
        # onwards, one int32 column at a time.
        columns = []
        for depth in range(int((self._starts - case_starts).max()) + 1):
            positions = self._starts - 1 - depth
            columns.append(
                np.where(positions >= case_starts, codes[np.maximum(positions, 0)], _START)
            )
        # Sorted so the cuts sharing any context prefix are one contiguous range. `lexsort` sorts
        # by its last key first, so depth 0 goes last.
        self._order = np.lexsort(columns[::-1])
        # Column-major, so each depth reads as one contiguous column.
        self._contexts = np.empty((len(cuts), len(columns)), dtype=np.int32, order='F')
        for depth, column in enumerate(columns):
            self._contexts[:, depth] = column[self._order]

    def sample(self, prefixes: PrefixBatch, num_samples: int, *, seed: int) -> list[Forecast]:
        """Draw `num_samples` matched train suffixes per prefix, independently with replacement."""
        rng = np.random.default_rng(seed)
        forecasts = []
        for row in range(len(prefixes)):
            names = prefixes[row][ACTIVITY_KEY]
            lower, upper = self._match(self._activity.encode(names).tolist())
            drawn = self._order[lower + rng.integers(upper - lower, size=num_samples)]
            forecasts.append(
                Forecast(
                    draws=tuple(
                        Draw(
                            activities=self._names[self._starts[cut] : self._stops[cut]],
                            inter_event_time_minutes=self._minutes[
                                self._starts[cut] : self._stops[cut]
                            ],
                            truncated=False,
                        )
                        for cut in drawn
                    )
                )
            )
        return forecasts

    def _match(self, prefix: list[int]) -> tuple[int, int]:
        """Narrow the sorted contexts to the longest run shared with a prefix's activity codes.

        Returns:
            The `[lower, upper)` range of sorted cuts sharing the longest matched context, every
            cut when not even the last activity matches.
        """
        lower, upper = 0, len(self._contexts)
        query = [*reversed(prefix), _START]
        for depth, token in enumerate(query[: self._contexts.shape[1]]):
            column = self._contexts[lower:upper, depth]
            first = lower + int(np.searchsorted(column, token, side='left'))
            last = lower + int(np.searchsorted(column, token, side='right'))
            if first == last:
                break
            lower, upper = first, last
        return lower, upper
