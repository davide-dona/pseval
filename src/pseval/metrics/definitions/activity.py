from pseval.metrics.functions.scoring_rules import best_of_k, energy_score, mean_between_draws
from pseval.metrics.functions.sequences import (
    bigram_distances,
    damerau_levenshtein_distances,
    exact_distances,
)
from pseval.metrics.prepared import PreparedPrefix


def activity(prefix: PreparedPrefix) -> dict[str, float]:
    """The activity columns of one prefix."""
    samples, truth = prefix.generation.samples, prefix.generation.truth
    # The sampled suffixes, then the observed one.
    targets = (*samples.suffixes, truth.activities)
    dl_distances = damerau_levenshtein_distances(samples.suffixes, targets)  # [K, K + 1]
    similarities = 1.0 - dl_distances[:, -1]  # [K]
    spread = mean_between_draws(dl_distances[:, :-1])
    return {
        'dls_sample_mean': float(similarities.mean()),
        'activity_dld_spread': spread,
        'activity_spread_error_gap': spread - float(dl_distances[:, -1].mean()),
        'unique_suffix_ratio': len(set(samples.suffixes)) / len(samples),
        'best_of_5_dls': best_of_k(similarities, 5),
        'best_of_10_dls': best_of_k(similarities, 10),
        'energy_score_dls': energy_score(dl_distances),
        'energy_score_exact': energy_score(exact_distances(samples.suffixes, targets)),
        'energy_score_bigram': energy_score(bigram_distances(samples.suffixes, targets)),
    }
