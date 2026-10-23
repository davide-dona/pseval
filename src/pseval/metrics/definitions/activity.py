import numpy as np

from pseval.metrics.functions.scoring_rules import best_of_k, energy_score, mean_between_draws
from pseval.metrics.functions.sequences import (
    bigram_distances,
    dld_distances,
    exact_distances,
    sequence_similarity,
)
from pseval.metrics.prepared import PreparedPrefix


def activity(prefix: PreparedPrefix) -> dict[str, float]:
    """The activity columns of one prefix.

    Raises:
        ValueError: With fewer than ten draws, which best-of-10 needs.
    """
    samples, truth = prefix.generation.samples, prefix.generation.truth
    choices = (*samples.suffixes, truth.activities)
    similarities = np.array(
        [sequence_similarity(suffix, truth.activities) for suffix in samples.suffixes],
        dtype=np.float64,
    )  # [U]
    dld = dld_distances(samples.suffixes, choices)  # [U, U + 1]
    if len(samples) < 2:
        spread = gap = 0.0
    else:
        spread = mean_between_draws(samples.counts, dld[:, :-1])
        gap = spread - samples.mean(dld[:, -1])
    return {
        'dls_sample_mean': samples.mean(similarities),
        'activity_dld_spread': spread,
        'activity_spread_error_gap': gap,
        'unique_suffix_ratio': len(samples.suffixes) / len(samples) if len(samples) else 0.0,
        'best_of_5_dls': best_of_k(similarities[samples.taken], 5),
        'best_of_10_dls': best_of_k(similarities[samples.taken], 10),
        'energy_score_dls': energy_score(samples.counts, dld),
        'energy_score_exact': energy_score(
            samples.counts, exact_distances(samples.suffixes, choices)
        ),
        'energy_score_bigram': energy_score(
            samples.counts, bigram_distances(samples.suffixes, choices)
        ),
    }
