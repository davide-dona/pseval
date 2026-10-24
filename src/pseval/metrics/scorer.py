import json
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from functools import cached_property
from itertools import pairwise
from pathlib import Path
from typing import Self

import numpy as np
import pandas as pd
from declarray import Checker, DeclareModel, mine

from pseval.data import ACTIVITY_KEY, CASE_KEY, Forecast, Log, PrefixKey, Suffix
from pseval.metrics.columns import METRIC_KEYS, PrefixScores
from pseval.metrics.inputs import MetricInputs, Pairing
from pseval.metrics.signature import SignatureKernel

# The vocabulary entry an activity unseen in the train split is scored as.
UNK = 'UNK'


@dataclass(frozen=True, eq=False)
class Scorer:
    """The scorer fitted on a benchmark's train split, and the scoring it drives.

    `vocabulary` names every activity a draw may hold, `max_suffix_length` caps every draw, and
    `kernel` is the signature kernel calibrated on train. Both Declare models are mined on train,
    on activity names: `hard` holds the rules full conformance checks, and `soft` every rule
    witnessed in some train trace, whose non-vacuous satisfaction the support score forecasts.
    """

    vocabulary: tuple[str, ...]
    max_suffix_length: int
    hard: DeclareModel
    soft: DeclareModel
    kernel: SignatureKernel

    @classmethod
    def fit(
        cls,
        train: Log,
        *,
        max_suffix_length: int,
        declare_hard_min_support: float,
        declare_hard_min_confidence: float,
        declare_max_cardinality: int,
    ) -> Self:
        """Fit the vocabulary, mine both Declare models, and calibrate the signature kernel.

        Args:
            train: The train split.
            max_suffix_length: The length cap of every draw.
            declare_hard_min_support: Least non-vacuous support a hard constraint reaches.
            declare_hard_min_confidence: Least confidence a hard constraint reaches.
            declare_max_cardinality: Largest occurrence count of the cardinality templates of
                both models.

        Raises:
            ValueError: If the train split holds an activity named `UNK`.
        """
        events = train.events
        activities = events[ACTIVITY_KEY].astype(str)
        if (activities == UNK).any():
            raise ValueError(f'The train split holds an activity named {UNK}')
        vocabulary = (*sorted(activities.unique().tolist()), UNK)
        traces = activities.groupby(events[CASE_KEY], sort=False).agg(list).tolist()
        return cls(
            vocabulary=vocabulary,
            max_suffix_length=max_suffix_length,
            hard=mine(
                traces,
                min_support=declare_hard_min_support,
                min_confidence=declare_hard_min_confidence,
                max_cardinality=declare_max_cardinality,
            ),
            # A support of one trace in the split keeps exactly the rules some trace witnesses.
            soft=mine(
                traces,
                min_support=1.0 / len(traces),
                min_confidence=0.0,
                max_cardinality=declare_max_cardinality,
            ),
            kernel=_fit_kernel(train, alphabet=len(vocabulary)),
        )

    def save(self, *, path: Path, hard: Path, soft: Path) -> None:
        """Write the fitted scorer as a benchmark stores it.

        Args:
            path: Destination of the vocabulary, length cap, and kernel calibration.
            hard: Destination of the hard Declare model, on activity names.
            soft: Destination of the soft Declare model, on activity names.
        """
        self.hard.save(hard)
        self.soft.save(soft)
        payload = {
            'vocabulary': list(self.vocabulary),
            'max_suffix_length': self.max_suffix_length,
            'signature_kernel': self.kernel.as_dict(),
        }
        path.write_text(json.dumps(payload, indent=4) + '\n')

    @classmethod
    def load(cls, *, path: Path, hard: Path, soft: Path) -> Self:
        """Read a fitted scorer written by `save`."""
        payload = json.loads(path.read_bytes())
        return cls(
            vocabulary=tuple(payload['vocabulary']),
            max_suffix_length=int(payload['max_suffix_length']),
            hard=DeclareModel.load(hard),
            soft=DeclareModel.load(soft),
            kernel=SignatureKernel.from_dict(payload['signature_kernel']),
        )

    @cached_property
    def axes(self) -> dict[str, int]:
        """Each vocabulary name to its index, its signature kernel axis."""
        return {name: index for index, name in enumerate(self.vocabulary)}

    @cached_property
    def hard_checker(self) -> Checker:
        """A checker of activity name traces against the hard Declare model."""
        return Checker(self.hard)

    @cached_property
    def soft_checker(self) -> Checker:
        """A checker of activity name traces against the soft Declare model."""
        return Checker(self.soft)

    def require_forecast(self, key: PrefixKey, forecast: Forecast, *, num_samples: int) -> None:
        """Require one prefix's forecast to honor the protocol contract.

        Args:
            key: The prefix.
            forecast: Its forecast.
            num_samples: How many draws the prefix was asked for.

        Raises:
            ValueError: On `num_samples` below two, a wrong draw count, an activity outside the
                vocabulary, a negative or non-finite time, or a draw longer than the cap.
        """
        if num_samples < 2:
            raise ValueError('Every metric needs at least two draws per prefix')
        if len(forecast) != num_samples:
            raise ValueError(f'{key} has {len(forecast)} draws instead of {num_samples}')
        if forecast.lengths.max() > self.max_suffix_length:
            raise ValueError(f'{key} has a draw longer than the cap of {self.max_suffix_length}')
        outside = {name for names in forecast.activities for name in names}.difference(self.axes)
        if outside:
            raise ValueError(f'{key} has draws outside the vocabulary: {sorted(outside)!r}')
        minutes = np.concatenate(
            [np.zeros(0), *(draw.inter_event_time_minutes for draw in forecast)]
        )
        if not np.isfinite(minutes).all() or (minutes < 0).any():
            raise ValueError(f'{key} has a negative or non-finite draw time')

    def score(
        self, log: Log, forecasts: Sequence[tuple[PrefixKey, Forecast]], *, num_samples: int
    ) -> pd.DataFrame:
        """Score a batch of prefixes against their observed suffixes.

        Observed activities outside the vocabulary, unseen in train, are scored as `UNK`.

        Args:
            log: The split the prefixes were cut from.
            forecasts: Each prefix and its forecast.
            num_samples: How many draws each prefix was asked for.

        Returns:
            One row per prefix in input order, one float64 column per `METRIC_KEYS` entry.

        Raises:
            ValueError: If a forecast breaks the protocol contract, as `require_forecast` checks.
        """
        for key, forecast in forecasts:
            self.require_forecast(key, forecast, num_samples=num_samples)
        batch = MetricInputs.of_batch(
            [self._pairing(log, key, forecast) for key, forecast in forecasts],
            hard_checker=self.hard_checker,
            soft_checker=self.soft_checker,
            kernel=self.kernel,
            axes=self.axes,
        )
        return pd.DataFrame(
            [PrefixScores.of(inputs) for inputs in batch],
            columns=list(METRIC_KEYS),
            dtype=np.float64,
        )

    def _pairing(self, log: Log, key: PrefixKey, forecast: Forecast) -> Pairing:
        """Join one prefix's forecast with its prefix and observed suffix from `log`."""
        activities, minutes = log.suffix_trace(key)
        return Pairing(
            case_id=key[0],
            prefix_activities=self._known(log.prefix_activities(key)),
            forecast=forecast,
            observed=Suffix(activities=self._known(activities), inter_event_time_minutes=minutes),
        )

    def _known(self, names: Iterable[str]) -> tuple[str, ...]:
        """The names, those outside the vocabulary as `UNK`."""
        return tuple(name if name in self.axes else UNK for name in names)


def _fit_kernel(train: Log, *, alphabet: int) -> SignatureKernel:
    """Calibrate the signature kernel on the longest suffix of every training case.

    A case's longest suffix follows its first event, since every prefix holds at least one event,
    so its inter-event times are all measured from a real preceding event. The train activities,
    sorted, take the first axes of the `alphabet`, in vocabulary order.
    """
    events = train.events
    axes = events[ACTIVITY_KEY].astype(str).astype('category').cat.codes.to_numpy(np.int64)
    minutes = train.inter_event_minutes()
    suffixes = [(axes[a + 1 : b], minutes[a + 1 : b]) for a, b in pairwise(train.case_offsets)]
    return SignatureKernel.fit(suffixes, alphabet=alphabet)
