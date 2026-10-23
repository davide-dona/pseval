import zlib
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Self

import numpy as np
from declarray import Checker

from pseval.generation.records import Generation
from pseval.metrics.signature import SignatureKernel

# Seeds the uniform that breaks PIT ties, together with the prefix key.
PIT_SEED = 42


@dataclass(frozen=True, slots=True)
class PreparedPrefix:
    """Decoded values and constraint checks shared by every metric for one prefix.

    Sample full conformance, suffix lengths, and remaining times have shape [K], one entry per
    draw, and are compared with the observed suffix's. Full conformance is 1.0 when the full
    trace satisfies every constraint. Times are in minutes. The signature
    Gram matrix has shape [K + 1, K + 1]: the normalized signature kernel between every pair of
    draws, with the observed suffix last. The PIT noise is one uniform on [0, 1] drawn from the
    prefix key alone, so it does not depend on the worker or batch scoring the prefix.
    """

    generation: Generation
    suffix_lengths: np.ndarray
    remaining_times: np.ndarray
    true_suffix_length: float
    true_remaining_time: float
    pit_noise: float
    signature_gram: np.ndarray
    sample_full_conformance: np.ndarray
    observed_full_conformance: float

    @classmethod
    def of_batch(
        cls, generations: Sequence[Generation], *, checker: Checker, kernel: SignatureKernel
    ) -> list[Self]:
        """Prepare the shared draw and observation arrays for a batch of prefixes.

        Every full trace of the batch is checked in one call, and every signature kernel of the
        batch is solved in one call, since both have a fixed cost per call that outweighs their
        work per item.

        Args:
            generations: Decoded prefixes, sampled suffixes, and observed continuations.
            checker: Declare checker applied to each prefix plus each of its suffixes.
            kernel: Signature kernel comparing the draws and observed suffix of each prefix.

        Returns:
            Shared metric inputs in input order, with numeric arrays stored in float64. Empty
            input gives an empty list.
        """
        if not generations:
            return []
        # Each prefix contributes its sampled traces followed by its observed one.
        full = checker.check(
            [
                generation.prefix_activities + suffix
                for generation in generations
                for suffix in (*generation.samples.suffixes, generation.truth.activities)
            ]
        ).full
        grams = kernel.grams(_suffix_paths(generations, kernel=kernel))
        prepared = []
        start = 0
        for generation, gram in zip(generations, grams, strict=True):
            stop = start + len(generation.samples) + 1
            prepared.append(cls._of(generation, full=full[start:stop], gram=gram))
            start = stop
        return prepared

    @classmethod
    def _of(cls, generation: Generation, *, full: np.ndarray, gram: np.ndarray) -> Self:
        """Prepare one prefix from the conformance of its sampled traces, then its observed one."""
        samples = generation.samples
        truth = generation.truth
        noise = np.random.default_rng(
            (PIT_SEED, zlib.crc32(generation.case_id.encode()), generation.prefix_len)
        ).random()
        return cls(
            generation=generation,
            suffix_lengths=samples.lengths.astype(np.float64),
            remaining_times=samples.remaining_time_minutes,
            true_suffix_length=float(len(truth)),
            true_remaining_time=truth.remaining_time_minutes,
            pit_noise=float(noise),
            signature_gram=gram,
            sample_full_conformance=full[:-1],
            observed_full_conformance=float(full[-1]),
        )


def _suffix_paths(
    generations: Sequence[Generation], *, kernel: SignatureKernel
) -> list[list[np.ndarray]]:
    """Embed each prefix's draws, in draw order, followed by its observed suffix."""
    codes = {
        code
        for generation in generations
        for suffix in (*generation.samples.suffixes, generation.truth.activities)
        for code in suffix
    }
    offset = min(map(ord, codes), default=0)
    alphabet = max(map(ord, codes), default=offset - 1) - offset + 1
    groups = []
    for generation in generations:
        samples, truth = generation.samples, generation.truth
        times = samples.inter_event_time_minutes
        offsets = samples.offsets
        groups.append(
            [
                *(
                    kernel.embed(
                        suffix,
                        times[offsets[draw] : offsets[draw + 1]],
                        offset=offset,
                        alphabet=alphabet,
                    )
                    for draw, suffix in enumerate(samples.suffixes)
                ),
                kernel.embed(
                    truth.activities,
                    truth.inter_event_time_minutes,
                    offset=offset,
                    alphabet=alphabet,
                ),
            ]
        )
    return groups
