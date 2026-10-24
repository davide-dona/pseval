import zlib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Self

import numpy as np
from declarray import Checker

from pseval.generation.records import DecodedEvents, Generation
from pseval.metrics.signature import SignatureKernel

# Seeds the uniform that breaks PIT ties, together with the prefix key.
PIT_SEED = 42


@dataclass(frozen=True, slots=True)
class MetricInputs:
    """One prefix's generation and the values needed to compute its metrics."""

    # The prefix, its draws, and its observed suffix.
    generation: Generation
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
        generations: Sequence[Generation],
        *,
        checker: Checker,
        kernel: SignatureKernel,
        axes: Mapping[str, int],
    ) -> list[Self]:
        """Compute the inputs for a batch of prefixes.

        Args:
            generations: Decoded prefixes, drawn suffixes, and observed suffixes.
            checker: Declare checker applied to each prefix completed by each of its suffixes.
            kernel: Signature kernel comparing the draws and observed suffix of each prefix.
            axes: Each activity name to its signature kernel axis.

        Returns:
            Metric inputs in input order. Empty input gives an empty list.
        """
        if not generations:
            return []
        conformances = _full_conformances(generations, checker=checker)
        grams = kernel.grams(
            [
                [kernel.embed(events, axes=axes) for events in _draws_then_observed(generation)]
                for generation in generations
            ]
        )
        return [
            cls(
                generation=generation,
                pit_noise=_pit_noise(generation),
                signature_gram=gram,
                draw_full_conformance=conformance[:-1],
                observed_full_conformance=float(conformance[-1]),
            )
            for generation, conformance, gram in zip(generations, conformances, grams, strict=True)
        ]


def _draws_then_observed(generation: Generation) -> list[DecodedEvents]:
    """One prefix's draws, then its observed suffix."""
    return [*generation.draws, generation.observed]


def _full_conformances(generations: Sequence[Generation], *, checker: Checker) -> list[np.ndarray]:
    """`[K + 1]` per prefix: whether the prefix completed by each draw, then by its observed
    suffix, satisfies the Declare model, from one checker call over the batch."""
    full = checker.check(
        [
            generation.prefix_activities + events.activities
            for generation in generations
            for events in _draws_then_observed(generation)
        ]
    ).full
    sizes = [len(generation.draws) + 1 for generation in generations]
    return np.split(full, np.cumsum(sizes)[:-1])


def _pit_noise(generation: Generation) -> float:
    """The uniform breaking one prefix's PIT ties, seeded by its key."""
    seed = (PIT_SEED, zlib.crc32(generation.case_id.encode()), generation.prefix_len)
    return float(np.random.default_rng(seed).random())
