from pseval.metrics.functions.scoring_rules import energy_score, mean_between_draws
from pseval.metrics.functions.sequences import (
    bigram_distances,
    damerau_levenshtein_distances,
    exact_distances,
)
from pseval.metrics.prepared import PreparedPrefix


def activity(prefix: PreparedPrefix) -> dict[str, float]:
    """The activity columns of one prefix."""
    suffixes, observed = prefix.generation.samples.suffixes, (prefix.generation.truth.activities,)
    dld_between = damerau_levenshtein_distances(suffixes, suffixes)  # [K, K]
    dld_to_truth = damerau_levenshtein_distances(suffixes, observed)[:, 0]  # [K]
    spread = mean_between_draws(dld_between)
    return {
        'dls_sample_mean': 1.0 - float(dld_to_truth.mean()),
        'activity_dld_spread': spread,
        'activity_spread_error_gap': spread - float(dld_to_truth.mean()),
        'unique_suffix_ratio': len(set(suffixes)) / len(suffixes),
        'energy_score_dls': energy_score(dld_between, dld_to_truth),
        'energy_score_exact': energy_score(
            exact_distances(suffixes, suffixes), exact_distances(suffixes, observed)[:, 0]
        ),
        'energy_score_bigram': energy_score(
            bigram_distances(suffixes, suffixes), bigram_distances(suffixes, observed)[:, 0]
        ),
    }
