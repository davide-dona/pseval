import numpy as np
import pandas as pd

from pseval.benchmark import Benchmark
from pseval.metrics import METRIC_FAMILIES
from pseval.model import SuffixModel, batch_seed, forecast
from pseval.scoring.summary import means


def validate(
    model: SuffixModel,
    benchmark: Benchmark,
    *,
    max_prefixes: int | None = None,
    num_samples: int = 32,
    seed: int = 42,
    batch_size: int = 256,
) -> dict[str, dict[str, float]]:
    """Score a model on validation prefixes.

    Args:
        model: The model to score, its `sample` called on validation prefixes only.
        benchmark: The benchmark whose validation split to read.
        max_prefixes: How many validation prefixes to score, drawn uniformly from `seed`, so
            every call with the same seed scores the same ones. Every prefix when None or when the
            split holds no more.
        num_samples: Draws per prefix, at least two.
        seed: Seed of the prefix subset and of the draws.
        batch_size: Prefixes per `sample` call.

    Returns:
        Every metric's equal-weight mean over the scored prefixes, grouped by family, with
        families and metrics in `METRIC_FAMILIES` order. Which one selects the model is its
        author's choice.

    Raises:
        ValueError: If `max_prefixes` is below one, or if the draws break the protocol contract,
            which also asks for at least two draws per prefix.
    """
    if max_prefixes is not None and max_prefixes < 1:
        raise ValueError('max_prefixes must be at least one')
    # Prepare the validation data, sorted by prefix length as generation does
    log = benchmark.validation
    cuts = log.cuts()
    if max_prefixes is not None and max_prefixes < len(cuts):
        drawn = np.random.default_rng(seed).choice(len(cuts), size=max_prefixes, replace=False)
        cuts = [cuts[index] for index in drawn]
    cuts = sorted(cuts, key=lambda cut: cut.prefix_len)

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
    overall = means(pd.concat(scores, ignore_index=True))
    return {family: {key: overall[key] for key in keys} for family, keys in METRIC_FAMILIES.items()}
