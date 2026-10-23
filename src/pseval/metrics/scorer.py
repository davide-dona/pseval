import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from functools import cached_property
from itertools import pairwise
from pathlib import Path
from typing import Self

import numpy as np
import pandas as pd
import torch
from declarray import Checker, Constraint, DeclareModel, mine

from pseval.data import ACTIVITY_KEY, CASE_KEY, Log
from pseval.metrics.columns import METRIC_KEYS, PrefixScores
from pseval.metrics.prepared import PreparedPrefix
from pseval.metrics.signature.kernel import SignatureKernel
from pseval.records import Generation
from pseval.vocabulary import UNK, ActivityCodes


@dataclass(frozen=True, eq=False)
class Scorer:
    """The protocol fitted on a benchmark's train split, and the scoring it drives.

    `codes` spell the vocabulary every draw is written on, `max_suffix_length` caps every draw,
    `declare` is the Declare model mined on train, on activity names, and `kernel` the signature
    kernel calibrated on train.
    """

    codes: ActivityCodes
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
        device: torch.device | str,
    ) -> Self:
        """Fit the vocabulary, mine the Declare model, and calibrate the signature kernel.

        Args:
            train: The train split.
            max_suffix_length: The length cap of every draw.
            declare_min_support: Least non-vacuous support a mined constraint reaches.
            declare_min_confidence: Least confidence a mined constraint reaches.
            declare_max_cardinality: Largest occurrence count of the cardinality templates.
            device: Where the signature kernel's bandwidth median is computed.

        Raises:
            ValueError: If the train split holds an activity named `UNK`.
        """
        events = train.events
        activities = events[ACTIVITY_KEY].astype(str)
        if (activities == UNK).any():
            raise ValueError(f'The train split holds an activity named {UNK}')
        codes = ActivityCodes([*sorted(activities.unique().tolist()), UNK])
        return cls(
            codes=codes,
            max_suffix_length=max_suffix_length,
            declare=mine(
                activities.groupby(events[CASE_KEY], sort=False).agg(list),
                min_support=declare_min_support,
                min_confidence=declare_min_confidence,
                max_cardinality=declare_max_cardinality,
            ),
            kernel=_fit_kernel(train, device=device),
        )

    def save(self, *, protocol: Path, declare: Path) -> None:
        """Write the fitted protocol as a bundle stores it.

        Args:
            protocol: Destination of the vocabulary, length cap, and kernel calibration.
            declare: Destination of the Declare model, on activity names.
        """
        self.declare.save(declare)
        payload = {
            'vocabulary': list(self.codes.vocabulary),
            'max_suffix_length': self.max_suffix_length,
            'signature_kernel': self.kernel.as_dict(),
        }
        protocol.write_text(json.dumps(payload, indent=4) + '\n')

    @classmethod
    def load(cls, *, protocol: Path, declare: Path) -> Self:
        """Read a fitted protocol written by `save`, its kernel solving on the CPU.

        Raises:
            ValueError: If the protocol file is malformed.
        """
        payload = json.loads(protocol.read_bytes())
        if not isinstance(payload, dict) or set(payload) != {
            'vocabulary',
            'max_suffix_length',
            'signature_kernel',
        }:
            raise ValueError(f'{protocol} is not a valid protocol file')
        return cls(
            codes=ActivityCodes(payload['vocabulary']),
            max_suffix_length=int(payload['max_suffix_length']),
            declare=DeclareModel.load(declare),
            kernel=SignatureKernel.from_dict(payload['signature_kernel'], device='cpu'),
        )

    @cached_property
    def checker(self) -> Checker:
        """A Declare checker of traces spelled by `codes`.

        Raises:
            ValueError: If the Declare model names an activity outside the vocabulary.
        """
        return _code_checker(self.declare, self.codes.codes)

    def on(self, device: torch.device | str) -> Self:
        """Return this scorer solving the signature kernel on `device`."""
        return replace(self, kernel=self.kernel.on(torch.device(device)))

    def require_draws(self, generation: Generation, *, num_samples: int) -> None:
        """Require one prefix's draws to honor the protocol contract.

        Args:
            generation: The prefix, its draws spelled by `codes`.
            num_samples: How many draws the prefix was asked for.

        Raises:
            ValueError: On a wrong draw count, an activity code outside the vocabulary, a
                negative or non-finite time, or a draw longer than the cap.
        """
        samples = generation.samples
        key = (generation.case_id, generation.prefix_len)
        if len(samples) != num_samples:
            raise ValueError(f'{key} has {len(samples)} draws instead of {num_samples}')
        if len(samples) and samples.lengths.max() > self.max_suffix_length:
            raise ValueError(f'{key} has a draw longer than the cap of {self.max_suffix_length}')
        if not set(''.join(samples.suffixes)).issubset(self.codes.codes.values()):
            raise ValueError(f'{key} has a draw outside the vocabulary')
        minutes = samples.inter_event_time_minutes
        if not np.isfinite(minutes).all() or (minutes < 0).any():
            raise ValueError(f'{key} has a negative or non-finite draw time')

    def score(self, generations: Sequence[Generation]) -> pd.DataFrame:
        """Score a batch of prefixes.

        Args:
            generations: Prefixes with their draws and observed suffixes, spelled by `codes`.

        Returns:
            One row per prefix in input order, one float64 column per `METRIC_KEYS` entry.

        Raises:
            ValueError: If a prefix has fewer than ten draws, which best-of-10 needs.
        """
        prepared = PreparedPrefix.of_batch(generations, checker=self.checker, kernel=self.kernel)
        return pd.DataFrame(
            [PrefixScores.of(prefix) for prefix in prepared],
            columns=list(METRIC_KEYS),
            dtype=np.float64,
        )


def _fit_kernel(train: Log, *, device: torch.device | str) -> SignatureKernel:
    """Calibrate the signature kernel on the longest suffix of every training case.

    A case's longest suffix follows its first event, since every prefix holds at least one event,
    so its inter-event times are all measured from a real preceding event.
    """
    events = train.events
    axes = events[ACTIVITY_KEY].astype(str).astype('category').cat.codes.to_numpy(np.int64)
    minutes = train.inter_event_minutes()
    suffixes = [(axes[a + 1 : b], minutes[a + 1 : b]) for a, b in pairwise(train.case_offsets)]
    alphabet = int(axes.max()) + 1 if len(axes) else 0
    return SignatureKernel.fit(suffixes, alphabet=alphabet, device=device)


def _code_checker(model: DeclareModel, codes: Mapping[str, str]) -> Checker:
    """A checker of traces spelled one character per activity.

    A trace without a constraint's activation satisfies it.

    Raises:
        ValueError: If `codes` lacks an activity the model names, which means the two belong to
            different benchmarks.
    """
    try:
        constraints = {
            Constraint(
                constraint.template,
                tuple(codes[name] for name in constraint.activities),
                constraint.n,
            ): measures
            for constraint, measures in model.constraints.items()
        }
    except KeyError as error:
        raise ValueError(
            f'The vocabulary lacks the activity {error.args[0]!r} of the Declare model.'
        ) from error
    return Checker(DeclareModel(settings=model.settings, constraints=constraints))
