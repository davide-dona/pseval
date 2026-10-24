from collections.abc import Sequence
from typing import Protocol, runtime_checkable

import numpy as np

from pseval.data import Forecast, PrefixBatch


@runtime_checkable
class SuffixModel(Protocol):
    """What pseval asks of a trained model."""

    def sample(self, prefixes: PrefixBatch, num_samples: int, *, seed: int) -> list[Forecast]:
        """Draw suffixes for a batch of observed prefixes.

        The draws must depend on nothing but `prefixes` and `seed`.

        Args:
            prefixes: The observed prefixes, holding no event past each prefix's last.
            num_samples: Draws per prefix.
            seed: Seed of the batch's draws.

        Returns:
            For each prefix in order, a `Forecast` of `num_samples` draws named from the benchmark
            vocabulary, in inter-event minutes, each at most the benchmark's `max_suffix_length`
            events long.
        """
        ...


def batch_seed(seed: int, batch: int) -> int:
    """The seed handed to `sample` for one batch, independent across batches."""
    return int(np.random.SeedSequence((seed, batch)).generate_state(1)[0])


def forecast(
    model: SuffixModel, prefixes: PrefixBatch, num_samples: int, *, seed: int
) -> list[Forecast]:
    """Sample a batch's forecasts from a model.

    Args:
        model: The model to sample.
        prefixes: The observed prefixes.
        num_samples: Draws per prefix.
        seed: Seed of the batch's draws.

    Returns:
        One forecast per prefix, in order.

    Raises:
        ValueError: If the model does not return a `Forecast` for every prefix.
    """
    forecasts = model.sample(prefixes, num_samples, seed=seed)
    if not isinstance(forecasts, Sequence) or len(forecasts) != len(prefixes):
        raise ValueError(f'Expected forecasts for {len(prefixes)} prefixes')
    if not all(isinstance(prefix_forecast, Forecast) for prefix_forecast in forecasts):
        raise ValueError('A model must return a Forecast for every prefix')
    return list(forecasts)
