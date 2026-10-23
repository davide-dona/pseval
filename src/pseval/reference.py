import numpy as np

from pseval.data import ACTIVITY_KEY, Log, PrefixBatch, SuffixDraws
from pseval.protocol import SuffixModel


class Unconditional(SuffixModel):
    """Draw real train suffixes regardless of the prefix.

    Each draw is the observed suffix, activities and inter-event minutes, of a train cut drawn
    uniformly from all of them, so the forecast is the marginal suffix distribution over train
    prefixes. It is the reference of `skill`.
    """

    def __init__(self) -> None:
        self._activities = np.zeros(0, dtype=object)
        self._minutes = np.zeros(0)
        # [N] The first and the past-the-end event row of each train cut's suffix.
        self._starts = np.zeros(0, np.int64)
        self._stops = np.zeros(0, np.int64)

    def fit(self, train: Log, validation: Log) -> None:
        """Collect the suffix of every train cut; validation is not read."""
        self._activities = train.events[ACTIVITY_KEY].astype(str).to_numpy(dtype=object)
        self._minutes = train.inter_event_minutes()
        offsets = train.case_offsets
        case_of_event = np.repeat(np.arange(len(offsets) - 1), np.diff(offsets))
        first = np.zeros(len(train), dtype=bool)
        first[offsets[:-1]] = True
        # A cut's suffix starts at any event but its case's first and runs to the case's end.
        self._starts = np.flatnonzero(~first)
        self._stops = offsets[1:][case_of_event[~first]]
        if not len(self._starts):
            raise ValueError('The unconditional model needs at least one train cut')

    def sample(self, prefixes: PrefixBatch, num_samples: int, *, seed: int) -> list[SuffixDraws]:
        """Draw `num_samples` train suffixes per prefix, independently and with replacement."""
        if not len(self._starts):
            raise RuntimeError('The unconditional model must be fitted before it samples')
        drawn = np.random.default_rng(seed).integers(
            len(self._starts), size=(len(prefixes), num_samples)
        )
        return [
            SuffixDraws(
                activities=[
                    self._activities[self._starts[cut] : self._stops[cut]].tolist() for cut in row
                ],
                minutes=[self._minutes[self._starts[cut] : self._stops[cut]] for cut in row],
                truncated=[False] * num_samples,
            )
            for row in drawn
        ]
