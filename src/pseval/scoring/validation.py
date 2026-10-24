import pandas as pd

from pseval.benchmark import Benchmark
from pseval.model import SuffixModel, batch_seed, forecast
from pseval.scoring.summary import means


def validate(
    model: SuffixModel,
    benchmark: Benchmark,
    *,
    num_samples: int = 32,
    seed: int = 42,
    batch_size: int = 256,
) -> dict[str, float]:
    """Score a model on every validation prefix.

    Args:
        model: The model to score, its `sample` called on validation prefixes only.
        benchmark: The benchmark whose validation split to read.
        num_samples: Draws per prefix, at least two.
        seed: Seed of the draws.
        batch_size: Prefixes per `sample` call.

    Returns:
        Every metric's equal-weight mean over the split, in `METRIC_KEYS` order. Which one
        selects the model is its author's choice.

    Raises:
        ValueError: If the draws break the protocol contract, which also asks for at least two
            draws per prefix.
    """
    # Prepare the validation data
    log = benchmark.validation
    cuts = log.cuts()

    scorer = benchmark.scorer
    scores = []
    # Score the prefixes in batches, to avoid memory blowup on large logs
    for batch, start in enumerate(range(0, len(cuts), batch_size)):
        batch_cuts = cuts[start : start + batch_size]
        # Draw the suffixes from the model, then score them
        forecasts = forecast(
            model, log.prefixes(batch_cuts), num_samples, seed=batch_seed(seed, batch)
        )
        scores.append(
            scorer.score(
                log, list(zip(batch_cuts, forecasts, strict=True)), num_samples=num_samples
            )
        )
    return means(pd.concat(scores, ignore_index=True)).to_dict()
