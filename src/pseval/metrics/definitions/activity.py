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
    # The distinct sampled suffixes, then the observed one.
    targets = (*samples.suffixes, truth.activities)
    dl_distances = damerau_levenshtein_distances(samples.suffixes, targets)  # [U, U + 1]
    similarities = 1.0 - dl_distances[:, -1]  # [U]
    spread = mean_between_draws(samples.counts, dl_distances[:, :-1])
    return {
        'dls_sample_mean': samples.mean(similarities),
        'activity_dld_spread': spread,
        'activity_spread_error_gap': spread - samples.mean(dl_distances[:, -1]),
        'unique_suffix_ratio': len(samples.suffixes) / len(samples),
        'best_of_5_dls': best_of_k(similarities[samples.taken], 5),
        'best_of_10_dls': best_of_k(similarities[samples.taken], 10),
        'energy_score_dls': energy_score(samples.counts, dl_distances),
        'energy_score_exact': energy_score(
            samples.counts, exact_distances(samples.suffixes, targets)
        ),
        'energy_score_bigram': energy_score(
            samples.counts, bigram_distances(samples.suffixes, targets)
        ),
    }
