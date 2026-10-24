from pseval.metrics.functions.scoring_rules import energy_score, mean_between_draws
from pseval.metrics.functions.sequences import (
    bigram_distances,
    damerau_levenshtein_distances,
    exact_distances,
)
from pseval.metrics.inputs import MetricInputs


def activity(inputs: MetricInputs) -> dict[str, float]:
    """The activity columns of one inputs."""
    pairing = inputs.pairing
    draws, observed = pairing.forecast.activities, (pairing.observed.activities,)
    dld_between = damerau_levenshtein_distances(draws, draws)  # [K, K]
    dld_to_observed = damerau_levenshtein_distances(draws, observed)[:, 0]  # [K]
    spread = mean_between_draws(dld_between)
    return {
        'dls_sample_mean': 1.0 - float(dld_to_observed.mean()),
        'activity_dld_spread': spread,
        'activity_spread_error_gap': spread - float(dld_to_observed.mean()),
        'unique_suffix_ratio': len(set(draws)) / len(draws),
        'energy_score_dls': energy_score(dld_between, dld_to_observed),
        'energy_score_exact': energy_score(
            exact_distances(draws, draws), exact_distances(draws, observed)[:, 0]
        ),
        'energy_score_bigram': energy_score(
            bigram_distances(draws, draws), bigram_distances(draws, observed)[:, 0]
        ),
    }
