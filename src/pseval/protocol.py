from typing import Protocol, runtime_checkable

import numpy as np

from pseval.data import PrefixBatch, SuffixDraws


@runtime_checkable
class SuffixModel(Protocol):
    """What pseval asks of a trained model.

    How a model trains and selects itself, on `Benchmark.train` and `Benchmark.validation`, is its
    author's choice. `sample` receives observed prefixes only and returns, for each prefix in
    order, `num_samples` draws spelled on the benchmark vocabulary, in inter-event minutes, no
    longer than the benchmark's `max_suffix_length`. It must depend on nothing but the prefixes
    and `seed`.
    """

    def sample(
        self, prefixes: PrefixBatch, num_samples: int, *, seed: int
    ) -> list[SuffixDraws]: ...


def batch_seed(seed: int, batch: int) -> int:
    """The seed handed to `sample` for one batch, independent across batches."""
    return int(np.random.SeedSequence((seed, batch)).generate_state(1)[0])
