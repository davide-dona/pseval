from collections.abc import Sequence

import pandas as pd

from pseval.benchmark import Benchmark
from pseval.data import Cut
from pseval.model import SuffixModel, batch_seed, forecast
from pseval.scoring.summary import means


def validate(
    model: SuffixModel,
    benchmark: Benchmark,
    *,
    cuts: Sequence[Cut] | None = None,
    num_samples: int = 32,
    seed: int = 42,
    batch_size: int = 256,
) -> dict[str, float]:
    """Score a model on validation prefixes.

    Args:
        model: The model to score, its `sample` called on validation prefixes only.
        benchmark: The benchmark whose validation split to read.
        cuts: The validation cuts to score, such as a fixed subset drawn once per training run.
            Every cut of the split when None.
        num_samples: Draws per prefix, at least two.
        seed: Seed of the draws.
        batch_size: Prefixes per `sample` call.

    Returns:
        Every metric's equal-weight mean over the scored prefixes, in `METRIC_KEYS` order.
        Which one selects the model is its author's choice.

    Raises:
        ValueError: If the draws break the protocol contract, which also asks for at least two
            draws per prefix, or if `cuts` is empty or holds a cut outside the split.
    """
    # Prepare the validation data
    log = benchmark.validation
    every_cut = log.cuts()
    if cuts is None:
        cuts = every_cut
    elif not cuts or not set(cuts) <= set(every_cut):
        raise ValueError('Validation cuts must be a nonempty subset of the validation split')

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
