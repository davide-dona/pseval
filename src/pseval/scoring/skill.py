from pseval.benchmark.bundle import Benchmark
from pseval.metrics import PROPER_SCORES
from pseval.scoring.store import Scores, read_scores


def skill(scores: Scores, benchmark: Benchmark) -> dict[str, float]:
    """Return the skill of every proper score against the unconditional reference.

    The skill is `1 - mean / reference mean`: zero for the reference, one for a perfect score,
    negative when worse than the reference. Ranking by skill equals ranking by the mean.

    Args:
        scores: A model's scores on `benchmark`.
        benchmark: The benchmark holding the reference scores.

    Returns:
        The skill per proper score, in `PROPER_SCORES` order.

    Raises:
        ValueError: If the scores belong to another benchmark or cover other prefixes.
    """
    reference = read_scores(benchmark._reference_scores())
    for scored in (scores, reference):
        scored.provenance.require_benchmark(benchmark.dataset, benchmark.fingerprint)
    keys = ['case_id', 'prefix_len']
    if not (
        scores.frame[keys]
        .sort_values(keys)
        .reset_index(drop=True)
        .equals(reference.frame[keys].sort_values(keys).reset_index(drop=True))
    ):
        raise ValueError('The scores and the reference cover different prefixes')
    model_means = scores.summary().overall
    reference_means = reference.summary().overall
    return {
        key: 1.0 - float(model_means[key]) / float(reference_means[key]) for key in PROPER_SCORES
    }
