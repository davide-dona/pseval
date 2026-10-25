import pandas as pd

from pseval.benchmark import Benchmark
from pseval.metrics import PROPER_SCORES
from pseval.scoring.store import Scores, ScoresReader
from pseval.scoring.summary import Summary, summarize


def _bucket_skill(model: pd.DataFrame, reference: pd.DataFrame) -> pd.DataFrame:
    keys = list(PROPER_SCORES)
    return pd.concat([model[['prefixes']], 1.0 - model[keys] / reference[keys]], axis=1)


def skill(scores: Scores, benchmark: Benchmark) -> Summary:
    """Return the skill of every proper score against the unconditional reference, overall and
    by length.

    The skill is `1 - mean / reference mean`: zero for the reference, one for a perfect score,
    negative when worse than the reference. Ranking by skill equals ranking by the mean. Every
    proper score is a fair estimator, so the skill compares models drawn with any draw count.

    Args:
        scores: A model's scores on `benchmark`.
        benchmark: The benchmark holding the reference scores.

    Returns:
        The skill per proper score, in `PROPER_SCORES` order, overall and in each length bucket.

    Raises:
        ValueError: If the scores belong to another benchmark or cover other prefixes.
    """
    with ScoresReader(benchmark._reference_scores()) as reader:
        reference = reader.read()
    for scored in (scores, reference):
        scored.record.require_input('benchmark', benchmark.digest)
    keys = ['case_id', 'prefix_len']
    if not (
        scores.frame[keys]
        .sort_values(keys)
        .reset_index(drop=True)
        .equals(reference.frame[keys].sort_values(keys).reset_index(drop=True))
    ):
        raise ValueError('The scores and the reference cover different prefixes')
    model_means = summarize(scores.frame, list(PROPER_SCORES))
    reference_means = summarize(reference.frame, list(PROPER_SCORES))
    return Summary(
        prefixes=model_means.prefixes,
        overall=1.0 - model_means.overall / reference_means.overall,
        by_prefix_length=_bucket_skill(
            model_means.by_prefix_length, reference_means.by_prefix_length
        ),
        by_suffix_length=_bucket_skill(
            model_means.by_suffix_length, reference_means.by_suffix_length
        ),
    )
