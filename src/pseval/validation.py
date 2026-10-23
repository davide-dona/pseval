import numpy as np
import pandas as pd
import torch

from pseval.benchmark.bundle import Benchmark
from pseval.generation.join import join_draws
from pseval.protocol import SEED, SuffixModel, batch_seed
from pseval.scoring.summary import means


def validate(
    model: SuffixModel,
    benchmark: Benchmark,
    *,
    num_samples: int = 32,
    prefixes: int = 2000,
    seed: int = SEED,
    batch_size: int = 256,
    device: torch.device | str = 'cpu',
) -> dict[str, float]:
    """Score a model on a fixed, seeded subset of validation prefixes.

    Args:
        model: The model to score, its `sample` called on validation prefixes only.
        benchmark: The benchmark whose validation split to read.
        num_samples: Draws per prefix.
        prefixes: Size of the subset, the whole split when it is smaller.
        seed: Seed of the subset and of the draws.
        batch_size: Prefixes per `sample` call.
        device: Where the signature kernel is solved.

    Returns:
        Every metric's equal-weight mean over the subset, summed in subset order, in
        `METRIC_KEYS` order. Which one selects the model is its author's choice.

    Raises:
        ValueError: If the draws break the protocol contract, or `num_samples` is below the draws
            a metric needs, such as the ten of best-of-10.
    """
    log = benchmark.validation
    cuts = log.cuts()
    rng = np.random.default_rng(seed)
    chosen = np.sort(rng.choice(len(cuts), size=min(prefixes, len(cuts)), replace=False))
    subset = [cuts[index] for index in chosen]
    if not subset:
        raise ValueError('The validation split has no prefix')
    scorer = benchmark.scorer(device)
    scores = []
    # Score the prefixes in batches, to avoid memory blowup on large logs
    for batch, start in enumerate(range(0, len(subset), batch_size)):
        batch_keys = subset[start : start + batch_size]
        # Draw the suffixes from the model, then score them
        draws = model.sample(log.prefixes(batch_keys), num_samples, seed=batch_seed(seed, batch))
        scores.append(
            scorer.score(join_draws(log, batch_keys, draws, scorer=scorer, num_samples=num_samples))
        )
    return means(pd.concat(scores, ignore_index=True)).to_dict()
