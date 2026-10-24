import numpy as np

from pseval import Draw, Forecast, Log, PrefixBatch, SuffixModel
from pseval.data import ACTIVITY_KEY

# Marks the start of a trace in a context, below every activity index.
_START = -1


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
        self._index = {name: code for code, name in enumerate(vocabulary)}
        self._unknown = self._index['UNK']
        self._names = train.events[ACTIVITY_KEY].astype(str).to_numpy(dtype=object)
        self._minutes = train.inter_event_minutes()
        codes = np.fromiter((self._index[name] for name in self._names), np.int32, len(self._names))
        offsets = train.case_offsets
        lengths = np.diff(offsets)
        # [N] The case and length of every cut, each case's cuts by ascending length.
        case = np.repeat(np.arange(len(lengths)), lengths - 1)
        cut = np.concatenate([np.zeros(0, np.int64), *(np.arange(1, n) for n in lengths)])
        # [N] The first and past-the-end rows of every cut's suffix.
        self._starts = offsets[case] + cut
        self._stops = offsets[case + 1]
        depth = np.arange(int(cut.max()) + 1)
        positions = self._starts[:, None] - 1 - depth[None, :]  # [N, W]
        # Position `d` holds the activity `d + 1` events before the cut, `_START` from the trace
        # start onwards.
        contexts = np.where(
            positions >= offsets[case][:, None], codes[np.maximum(positions, 0)], _START
        ).astype(np.int32)
        # Sorted so the cuts sharing any context prefix are one contiguous range. `lexsort` sorts
        # by its last key first, so depth 0 goes last.
        self._order = np.lexsort(contexts.T[::-1])
        # Column-major, so each depth reads as one contiguous column.
        self._contexts = np.asfortranarray(contexts[self._order])

    def sample(self, prefixes: PrefixBatch, num_samples: int, *, seed: int) -> list[Forecast]:
        """Draw `num_samples` matched train suffixes per prefix, independently with replacement."""
        rng = np.random.default_rng(seed)
        forecasts = []
        for row in range(len(prefixes)):
            names = prefixes[row][ACTIVITY_KEY].astype(str)
            lower, upper = self._match([self._index.get(name, self._unknown) for name in names])
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
