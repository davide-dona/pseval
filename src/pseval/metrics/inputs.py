import zlib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Self

import numpy as np
from declarray import Checker

from pseval.generation.records import DecodedEvents, Draws
from pseval.metrics.signature import SignatureKernel

# Seeds the uniform that breaks PIT ties, together with the prefix key.
PIT_SEED = 42


@dataclass(frozen=True, eq=False)
class Forecast:
    """One prefix's drawn suffixes and the observed suffix they are scored against."""

    case_id: str
    prefix_activities: tuple[str, ...]
    draws: Draws
    observed: DecodedEvents

    @property
    def prefix_len(self) -> int:
        """How many events the prefix holds."""
        return len(self.prefix_activities)


@dataclass(frozen=True, slots=True)
class MetricInputs:
    """One prefix's forecast and the values needed to compute its metrics."""

    # The prefix, its draws, and its observed suffix.
    forecast: Forecast
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
        forecasts: Sequence[Forecast],
        *,
        hard_checker: Checker,
        soft_checker: Checker,
        kernel: SignatureKernel,
        axes: Mapping[str, int],
    ) -> list[Self]:
        """Compute the inputs for a batch of prefixes.

        Args:
            forecasts: Decoded prefixes, drawn suffixes, and observed suffixes.
            hard_checker: Checks full conformance of each prefix completed by each suffix.
            soft_checker: Checks which soft rules each prefix completed by each suffix witnesses.
            kernel: Signature kernel comparing the draws and observed suffix of each prefix.
            axes: Each activity name to its signature kernel axis.

        Returns:
            Metric inputs in input order. Empty input gives an empty list.
        """
        if not forecasts:
            return []
        traces = _completed_traces(forecasts)
        conformances = _per_prefix(hard_checker.check(traces).full, forecasts)
        witnesses = _per_prefix(soft_checker.witnesses(traces), forecasts)
        grams = kernel.grams(
            [
                [kernel.embed(events, axes=axes) for events in _draws_then_observed(forecast)]
                for forecast in forecasts
            ]
        )
        return [
            cls(
                forecast=forecast,
                pit_noise=_pit_noise(forecast),
                signature_gram=gram,
                draw_full_conformance=conformance[:-1],
                observed_full_conformance=float(conformance[-1]),
                soft_witnesses=soft_witnesses,
                activity_presence=_activity_presence(forecast, axes=axes),
            )
            for forecast, conformance, soft_witnesses, gram in zip(
                forecasts, conformances, witnesses, grams, strict=True
            )
        ]


def _draws_then_observed(forecast: Forecast) -> list[DecodedEvents]:
    """One prefix's draws, then its observed suffix."""
    return [*forecast.draws, forecast.observed]


def _completed_traces(forecasts: Sequence[Forecast]) -> list[tuple[str, ...]]:
    """Each prefix completed by each of its draws, then by its observed suffix, over the batch."""
    return [
        forecast.prefix_activities + events.activities
        for forecast in forecasts
        for events in _draws_then_observed(forecast)
    ]


def _per_prefix(values: np.ndarray, forecasts: Sequence[Forecast]) -> list[np.ndarray]:
    """Split rows over the batch's completed traces into `[K + 1, ...]` rows per prefix."""
    sizes = [len(forecast.draws) + 1 for forecast in forecasts]
    return np.split(values, np.cumsum(sizes)[:-1])


def _activity_presence(forecast: Forecast, *, axes: Mapping[str, int]) -> np.ndarray:
    """`[K + 1, A]` whether the prefix completed by each draw, then by its observed suffix,
    holds each vocabulary activity."""
    presence = np.zeros((len(forecast.draws) + 1, len(axes)), bool)
    for row, events in enumerate(_draws_then_observed(forecast)):
        presence[row, [axes[name] for name in events.activities]] = True
    presence[:, [axes[name] for name in forecast.prefix_activities]] = True
    return presence


def _pit_noise(forecast: Forecast) -> float:
    """The uniform breaking one prefix's PIT ties, seeded by its key."""
    seed = (PIT_SEED, zlib.crc32(forecast.case_id.encode()), forecast.prefix_len)
    return float(np.random.default_rng(seed).random())
