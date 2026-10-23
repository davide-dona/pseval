import pandas as pd
import torch

from pseval.benchmark.bundle import Benchmark
from pseval.generation.join import join_draws
from pseval.metrics import MIN_DRAWS
from pseval.protocol import SuffixModel, batch_seed
from pseval.scoring.summary import means


def validate(
    model: SuffixModel,
    benchmark: Benchmark,
    *,
    num_samples: int = 32,
    seed: int = 42,
    batch_size: int = 256,
    device: torch.device | str = 'cpu',
) -> dict[str, float]:
    """Score a model on every validation prefix.

    Args:
        model: The model to score, its `sample` called on validation prefixes only.
        benchmark: The benchmark whose validation split to read.
        num_samples: Draws per prefix, at least the ten best-of-10 needs.
        seed: Seed of the draws.
        batch_size: Prefixes per `sample` call.
        device: Where the signature kernel is solved.

    Returns:
        Every metric's equal-weight mean over the split, in `METRIC_KEYS` order. Which one
        selects the model is its author's choice.

    Raises:
        ValueError: If `num_samples` is below ten or the draws break the protocol contract.
    """
    # Prepare the validation data
    if num_samples < MIN_DRAWS:
        raise ValueError(f'Validation needs at least {MIN_DRAWS} draws per prefix')
    log = benchmark.validation
    cuts = log.cuts()

    scorer = benchmark.scorer(device)
    scores = []
    # Score the prefixes in batches, to avoid memory blowup on large logs
    for batch, start in enumerate(range(0, len(cuts), batch_size)):
        batch_keys = cuts[start : start + batch_size]
        # Draw the suffixes from the model, then score them
        draws = model.sample(log.prefixes(batch_keys), num_samples, seed=batch_seed(seed, batch))
        scores.append(
            scorer.score(join_draws(log, batch_keys, draws, scorer=scorer, num_samples=num_samples))
        )
    return means(pd.concat(scores, ignore_index=True)).to_dict()
