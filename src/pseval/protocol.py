from typing import Protocol, runtime_checkable

import numpy as np

from pseval.data import PrefixBatch, SuffixDraws


@runtime_checkable
class SuffixModel(Protocol):
    """What pseval asks of a trained model."""

    def sample(self, prefixes: PrefixBatch, num_samples: int, *, seed: int) -> list[SuffixDraws]:
        """Draw suffixes for a batch of observed prefixes.

        The draws must depend on nothing but `prefixes` and `seed`.

        Args:
            prefixes: The observed prefixes, holding no event past each prefix's last.
            num_samples: Draws per prefix.
            seed: Seed of the batch's draws.

        Returns:
            For each prefix in order, `num_samples` draws spelled on the benchmark vocabulary, in
            inter-event minutes, each at most the benchmark's `max_suffix_length` events long.
        """
        ...


def batch_seed(seed: int, batch: int) -> int:
    """The seed handed to `sample` for one batch, independent across batches."""
    return int(np.random.SeedSequence((seed, batch)).generate_state(1)[0])
