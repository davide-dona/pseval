from collections.abc import Sequence
from typing import Protocol, runtime_checkable

import numpy as np

from pseval.data import PrefixBatch, SuffixDraws
from pseval.generation.records import Draws


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
            For each prefix in order, `num_samples` draws named from the benchmark vocabulary, in
            inter-event minutes, each at most the benchmark's `max_suffix_length` events long.
        """
        ...


def batch_seed(seed: int, batch: int) -> int:
    """The seed handed to `sample` for one batch, independent across batches."""
    return int(np.random.SeedSequence((seed, batch)).generate_state(1)[0])


def draw(model: SuffixModel, prefixes: PrefixBatch, num_samples: int, *, seed: int) -> list[Draws]:
    """Sample a batch's draws from a model.

    Args:
        model: The model to sample.
        prefixes: The observed prefixes.
        num_samples: Draws per prefix.
        seed: Seed of the batch's draws.

    Returns:
        One prefix's draws per prefix, in order.

    Raises:
        ValueError: If the model does not return `SuffixDraws` for every prefix.
    """
    draws = model.sample(prefixes, num_samples, seed=seed)
    if not isinstance(draws, Sequence) or len(draws) != len(prefixes):
        raise ValueError(f'Expected draws for {len(prefixes)} prefixes')
    if not all(isinstance(prefix_draws, SuffixDraws) for prefix_draws in draws):
        raise ValueError('A model must return SuffixDraws for every prefix')
    return [Draws.of(prefix_draws) for prefix_draws in draws]
