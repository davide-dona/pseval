import numpy as np

from pseval.metrics.functions.scoring_rules import crps, crps_spread, mae, randomized_pit
from pseval.metrics.inputs import MetricInputs


def suffix_length(inputs: MetricInputs) -> dict[str, float]:
    """The suffix length columns of one inputs."""
    draws = inputs.forecast.draws
    lengths = draws.lengths.astype(np.float64)  # [K]
    observed_length = float(len(inputs.forecast.observed))
    return {
        'suffix_length_crps': crps(lengths, observed_length),
        'suffix_length_crps_error': mae(lengths, observed_length),
        'suffix_length_crps_spread': crps_spread(lengths),
        'suffix_length_signed_error': float(lengths.mean() - observed_length),
        'suffix_length_truth_quantile': randomized_pit(
            lengths, observed_length, noise=inputs.pit_noise
        ),
        'truncated_draw_rate': float(draws.truncated.mean()),
    }
