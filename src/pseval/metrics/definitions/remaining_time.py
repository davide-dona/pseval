import numpy as np

from pseval.metrics.functions.scoring_rules import crps, crps_spread, mae, randomized_pit
from pseval.metrics.prepared import PreparedPrefix

MINUTES_PER_DAY = 1440.0


def remaining_time(prefix: PreparedPrefix) -> dict[str, float]:
    """The remaining-time columns of one prefix."""
    draws, truth = prefix.remaining_times, prefix.true_remaining_time
    if draws.size:
        log_error = float(
            np.log1p(draws / MINUTES_PER_DAY).mean() - np.log1p(truth / MINUTES_PER_DAY)
        )
    else:
        log_error = 0.0
    return {
        'remaining_time_crps_days': crps(draws, truth) / MINUTES_PER_DAY,
        'remaining_time_log_error': log_error,
        'remaining_time_crps_error_days': mae(draws, truth) / MINUTES_PER_DAY,
        'remaining_time_crps_spread_days': crps_spread(draws) / MINUTES_PER_DAY,
        'remaining_time_truth_quantile': randomized_pit(draws, truth, noise=prefix.pit_noise),
    }
