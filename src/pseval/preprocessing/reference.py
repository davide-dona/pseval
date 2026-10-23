import numpy as np

from pseval.data import ACTIVITY_KEY, Log, PrefixBatch, SuffixDraws
from pseval.protocol import SuffixModel


class Unconditional(SuffixModel):
    """The unconditional reference model, whose scores every bundle stores for `skill`.

    It draws real train suffixes regardless of the prefix: each draw is the observed suffix,
    activities and inter-event minutes, of a train cut drawn uniformly from all of them, so the
    forecast is the marginal suffix distribution over train prefixes.
    """

    def __init__(self, train: Log) -> None:
        """Collect the suffix of every train cut."""
        self._activities = train.events[ACTIVITY_KEY].astype(str).to_numpy(dtype=object)
        self._minutes = train.inter_event_minutes()
        offsets = train.case_offsets
        case_of_event = np.repeat(np.arange(len(offsets) - 1), np.diff(offsets))
        first = np.zeros(len(train), dtype=bool)
        first[offsets[:-1]] = True
        # [N] The first and the past-the-end event row of each train cut's suffix. A cut's suffix
        # starts at any event but its case's first and runs to the case's end.
        self._starts = np.flatnonzero(~first)
        self._stops = offsets[1:][case_of_event[~first]]

    def sample(self, prefixes: PrefixBatch, num_samples: int, *, seed: int) -> list[SuffixDraws]:
        """Draw `num_samples` train suffixes per prefix, independently and with replacement."""
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
