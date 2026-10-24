import zlib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Self

import numpy as np
from declarray import Checker

from pseval.data import Forecast, Suffix
from pseval.metrics.signature import SignatureKernel

# Seeds the uniform that breaks PIT ties, together with the prefix key.
PIT_SEED = 42


@dataclass(frozen=True, eq=False)
class Pairing:
    """One prefix's forecast and the observed suffix it is scored against."""

    case_id: str
    prefix_activities: tuple[str, ...]
    forecast: Forecast
    observed: Suffix

    @property
    def prefix_len(self) -> int:
        """How many events the prefix holds."""
        return len(self.prefix_activities)


@dataclass(frozen=True, slots=True)
class MetricInputs:
    """One prefix's pairing and the values needed to compute its metrics."""

    # The prefix, its forecast, and its observed suffix.
    pairing: Pairing
    # Uniform in [0, 1) breaking PIT ties, seeded by the prefix key alone.
    pit_noise: float
    # [K + 1, K + 1] normalized signature kernel between the draws and, last, the observed suffix.
    signature_gram: np.ndarray
    # [K] Whether the prefix completed by each draw satisfies the hard Declare model.
    draw_full_conformance: np.ndarray
    # Whether the prefix completed by its observed suffix satisfies the hard Declare model.
    observed_full_conformance: float
    # [K + 1, R] Whether the prefix completed by each draw and, last, by the observed suffix
    # witnesses each soft Declare rule.
    soft_witnesses: np.ndarray
    # [K + 1, A] Whether the prefix completed by each draw and, last, by the observed suffix
    # holds each vocabulary activity, in axis order.
    activity_presence: np.ndarray

    @classmethod
    def of_batch(
        cls,
        pairings: Sequence[Pairing],
        *,
        hard_checker: Checker,
        soft_checker: Checker,
        kernel: SignatureKernel,
        axes: Mapping[str, int],
    ) -> list[Self]:
        """Compute the inputs for a batch of prefixes.

        Args:
            pairings: Prefixes, their forecasts, and their observed suffixes.
            hard_checker: Checks full conformance of each prefix completed by each suffix.
            soft_checker: Checks which soft rules each prefix completed by each suffix witnesses.
            kernel: Signature kernel comparing the draws and observed suffix of each prefix.
            axes: Each activity name to its signature kernel axis.

        Returns:
            Metric inputs in input order. Empty input gives an empty list.
        """
        if not pairings:
            return []
        traces = _completed_traces(pairings)
        conformances = _per_prefix(hard_checker.check(traces).full, pairings)
        witnesses = _per_prefix(soft_checker.witnesses(traces), pairings)
        grams = kernel.grams(
            [
                [kernel.embed(suffix, axes=axes) for suffix in _draws_then_observed(pairing)]
                for pairing in pairings
            ]
        )
        return [
            cls(
                pairing=pairing,
                pit_noise=_pit_noise(pairing),
                signature_gram=gram,
                draw_full_conformance=conformance[:-1],
                observed_full_conformance=float(conformance[-1]),
                soft_witnesses=soft_witnesses,
                activity_presence=_activity_presence(pairing, axes=axes),
            )
            for pairing, conformance, soft_witnesses, gram in zip(
                pairings, conformances, witnesses, grams, strict=True
            )
        ]


def _draws_then_observed(pairing: Pairing) -> list[Suffix]:
    """One prefix's draws, then its observed suffix."""
    return [*pairing.forecast, pairing.observed]


def _completed_traces(pairings: Sequence[Pairing]) -> list[tuple[str, ...]]:
    """Each prefix completed by each of its draws, then by its observed suffix, over the batch."""
    return [
        pairing.prefix_activities + suffix.activities
        for pairing in pairings
        for suffix in _draws_then_observed(pairing)
    ]


def _per_prefix(values: np.ndarray, pairings: Sequence[Pairing]) -> list[np.ndarray]:
    """Split rows over the batch's completed traces into `[K + 1, ...]` rows per prefix."""
    sizes = [len(pairing.forecast) + 1 for pairing in pairings]
    return np.split(values, np.cumsum(sizes)[:-1])


def _activity_presence(pairing: Pairing, *, axes: Mapping[str, int]) -> np.ndarray:
    """`[K + 1, A]` whether the prefix completed by each draw, then by its observed suffix,
    holds each vocabulary activity."""
    presence = np.zeros((len(pairing.forecast) + 1, len(axes)), bool)
    for row, suffix in enumerate(_draws_then_observed(pairing)):
        presence[row, [axes[name] for name in suffix.activities]] = True
    presence[:, [axes[name] for name in pairing.prefix_activities]] = True
    return presence


def _pit_noise(pairing: Pairing) -> float:
    """The uniform breaking one prefix's PIT ties, seeded by its key."""
    seed = (PIT_SEED, zlib.crc32(pairing.case_id.encode()), pairing.prefix_len)
    return float(np.random.default_rng(seed).random())
