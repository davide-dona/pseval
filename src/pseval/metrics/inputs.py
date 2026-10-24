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
    # [K] Whether the prefix completed by each draw satisfies the Declare model.
    draw_full_conformance: np.ndarray
    # Whether the prefix completed by its observed suffix satisfies the Declare model.
    observed_full_conformance: float

    @classmethod
    def of_batch(
        cls,
        forecasts: Sequence[Forecast],
        *,
        checker: Checker,
        kernel: SignatureKernel,
        axes: Mapping[str, int],
    ) -> list[Self]:
        """Compute the inputs for a batch of prefixes.

        Args:
            forecasts: Decoded prefixes, drawn suffixes, and observed suffixes.
            checker: Declare checker applied to each prefix completed by each of its suffixes.
            kernel: Signature kernel comparing the draws and observed suffix of each prefix.
            axes: Each activity name to its signature kernel axis.

        Returns:
            Metric inputs in input order. Empty input gives an empty list.
        """
        if not forecasts:
            return []
        conformances = _full_conformances(forecasts, checker=checker)
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
            )
            for forecast, conformance, gram in zip(forecasts, conformances, grams, strict=True)
        ]


def _draws_then_observed(forecast: Forecast) -> list[DecodedEvents]:
    """One prefix's draws, then its observed suffix."""
    return [*forecast.draws, forecast.observed]


def _full_conformances(forecasts: Sequence[Forecast], *, checker: Checker) -> list[np.ndarray]:
    """`[K + 1]` per prefix: whether the prefix completed by each draw, then by its observed
    suffix, satisfies the Declare model, from one checker call over the batch."""
    full = checker.check(
        [
            forecast.prefix_activities + events.activities
            for forecast in forecasts
            for events in _draws_then_observed(forecast)
        ]
    ).full
    sizes = [len(forecast.draws) + 1 for forecast in forecasts]
    return np.split(full, np.cumsum(sizes)[:-1])


def _pit_noise(forecast: Forecast) -> float:
    """The uniform breaking one prefix's PIT ties, seeded by its key."""
    seed = (PIT_SEED, zlib.crc32(forecast.case_id.encode()), forecast.prefix_len)
    return float(np.random.default_rng(seed).random())
