import json
from collections.abc import Sequence
from dataclasses import dataclass
from functools import cached_property
from itertools import pairwise
from pathlib import Path
from typing import Self

import numpy as np
import pandas as pd
from declarray import Checker, DeclareModel, mine

from pseval.data import ACTIVITY_KEY, CASE_KEY, Log
from pseval.generation.records import Generation
from pseval.metrics.columns import METRIC_KEYS, PrefixScores
from pseval.metrics.inputs import MetricInputs
from pseval.metrics.signature import SignatureKernel

# The vocabulary entry an activity unseen in the train split is scored as.
UNK = 'UNK'


@dataclass(frozen=True, eq=False)
class Scorer:
    """The scorer fitted on a benchmark's train split, and the scoring it drives.

    `vocabulary` names every activity a draw may hold, `max_suffix_length` caps every draw,
    `declare` is the Declare model mined on train, on activity names, and `kernel` the signature
    kernel calibrated on train.
    """

    vocabulary: tuple[str, ...]
    max_suffix_length: int
    declare: DeclareModel
    kernel: SignatureKernel

    @classmethod
    def fit(
        cls,
        train: Log,
        *,
        max_suffix_length: int,
        declare_min_support: float,
        declare_min_confidence: float,
        declare_max_cardinality: int,
    ) -> Self:
        """Fit the vocabulary, mine the Declare model, and calibrate the signature kernel.

        Args:
            train: The train split.
            max_suffix_length: The length cap of every draw.
            declare_min_support: Least non-vacuous support a mined constraint reaches.
            declare_min_confidence: Least confidence a mined constraint reaches.
            declare_max_cardinality: Largest occurrence count of the cardinality templates.

        Raises:
            ValueError: If the train split holds an activity named `UNK`.
        """
        events = train.events
        activities = events[ACTIVITY_KEY].astype(str)
        if (activities == UNK).any():
            raise ValueError(f'The train split holds an activity named {UNK}')
        vocabulary = (*sorted(activities.unique().tolist()), UNK)
        return cls(
            vocabulary=vocabulary,
            max_suffix_length=max_suffix_length,
            declare=mine(
                activities.groupby(events[CASE_KEY], sort=False).agg(list),
                min_support=declare_min_support,
                min_confidence=declare_min_confidence,
                max_cardinality=declare_max_cardinality,
            ),
            kernel=_fit_kernel(train, alphabet=len(vocabulary)),
        )

    def save(self, *, path: Path, declare: Path) -> None:
        """Write the fitted scorer as a benchmark stores it.

        Args:
            path: Destination of the vocabulary, length cap, and kernel calibration.
            declare: Destination of the Declare model, on activity names.
        """
        self.declare.save(declare)
        payload = {
            'vocabulary': list(self.vocabulary),
            'max_suffix_length': self.max_suffix_length,
            'signature_kernel': self.kernel.as_dict(),
        }
        path.write_text(json.dumps(payload, indent=4) + '\n')

    @classmethod
    def load(cls, *, path: Path, declare: Path) -> Self:
        """Read a fitted scorer written by `save`."""
        payload = json.loads(path.read_bytes())
        return cls(
            vocabulary=tuple(payload['vocabulary']),
            max_suffix_length=int(payload['max_suffix_length']),
            declare=DeclareModel.load(declare),
            kernel=SignatureKernel.from_dict(payload['signature_kernel']),
        )

    @cached_property
    def axes(self) -> dict[str, int]:
        """Each vocabulary name to its index, its signature kernel axis."""
        return {name: index for index, name in enumerate(self.vocabulary)}

    @cached_property
    def checker(self) -> Checker:
        """A Declare checker of activity name traces."""
        return Checker(self.declare)

    def require_draws(self, generation: Generation, *, num_samples: int) -> None:
        """Require one prefix's draws to honor the protocol contract.

        Args:
            generation: The prefix and its draws.
            num_samples: How many draws the prefix was asked for.

        Raises:
            ValueError: On `num_samples` below two, a wrong draw count, an activity outside the
                vocabulary, a negative or non-finite time, or a draw longer than the cap.
        """
        draws = generation.draws
        key = (generation.case_id, generation.prefix_len)
        if num_samples < 2:
            raise ValueError('Every metric needs at least two draws per prefix')
        if len(draws) != num_samples:
            raise ValueError(f'{key} has {len(draws)} draws instead of {num_samples}')
        if draws.lengths.max() > self.max_suffix_length:
            raise ValueError(f'{key} has a draw longer than the cap of {self.max_suffix_length}')
        outside = {name for names in draws.activities for name in names}.difference(self.axes)
        if outside:
            raise ValueError(f'{key} has draws outside the vocabulary: {sorted(outside)!r}')
        minutes = draws.inter_event_time_minutes
        if not np.isfinite(minutes).all() or (minutes < 0).any():
            raise ValueError(f'{key} has a negative or non-finite draw time')

    def score(self, generations: Sequence[Generation]) -> pd.DataFrame:
        """Score a batch of prefixes.

        Args:
            generations: Prefixes with their draws and observed suffixes.

        Returns:
            One row per prefix in input order, one float64 column per `METRIC_KEYS` entry.
        """
        batch = MetricInputs.of_batch(
            generations, checker=self.checker, kernel=self.kernel, axes=self.axes
        )
        return pd.DataFrame(
            [PrefixScores.of(inputs) for inputs in batch],
            columns=list(METRIC_KEYS),
            dtype=np.float64,
        )


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
