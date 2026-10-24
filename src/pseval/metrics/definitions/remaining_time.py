import numpy as np

from pseval.metrics.functions.scoring_rules import crps, crps_spread, mae, randomized_pit
from pseval.metrics.inputs import MetricInputs

MINUTES_PER_DAY = 1440.0


def remaining_time(inputs: MetricInputs) -> dict[str, float]:
    """The remaining-time columns of one inputs."""
    generation = inputs.generation
    days = generation.draws.remaining_time_minutes / MINUTES_PER_DAY  # [K]
    observed_days = generation.observed.remaining_time_minutes / MINUTES_PER_DAY
    return {
        'remaining_time_crps_days': crps(days, observed_days),
        'remaining_time_crps_error_days': mae(days, observed_days),
        'remaining_time_crps_spread_days': crps_spread(days),
        'remaining_time_log_error': float(np.log1p(days).mean() - np.log1p(observed_days)),
        'remaining_time_truth_quantile': randomized_pit(
            days, observed_days, noise=inputs.pit_noise
        ),
    }
