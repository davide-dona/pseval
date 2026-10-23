from typing import Protocol, runtime_checkable

import numpy as np

from pseval.data import Log, PrefixBatch, SuffixDraws

# Draws per test prefix.
NUM_SAMPLES = 100
# Seeds every protocol draw: the generation seeds, the reference, and the validation subset.
SEED = 42


@runtime_checkable
class SuffixModel(Protocol):
    """What pseval asks of a model.

    `fit` receives the train and validation splits only; how a model trains and selects itself
    is its author's choice. `sample` receives observed prefixes only and returns, for each prefix
    in order, `num_samples` draws spelled on the benchmark vocabulary, in inter-event minutes, no
    longer than the benchmark's `max_suffix_length`. It must depend on nothing but the prefixes
    and `seed`.
    """

    def fit(self, train: Log, validation: Log) -> None: ...

    def sample(
        self, prefixes: PrefixBatch, num_samples: int, *, seed: int
    ) -> list[SuffixDraws]: ...


def batch_seed(seed: int, batch: int) -> int:
    """The seed handed to `sample` for one batch, independent across batches."""
    return int(np.random.SeedSequence((seed, batch)).generate_state(1)[0])
