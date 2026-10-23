import numpy as np

from pseval.metrics.functions.scoring_rules import crps, crps_spread, mae, randomized_pit
from pseval.metrics.prepared import PreparedPrefix

MINUTES_PER_DAY = 1440.0


def remaining_time(prefix: PreparedPrefix) -> dict[str, float]:
    """The remaining-time columns of one prefix."""
    days = prefix.remaining_times / MINUTES_PER_DAY  # [K]
    true_days = prefix.true_remaining_time / MINUTES_PER_DAY
    return {
        'remaining_time_crps_days': crps(days, true_days),
        'remaining_time_crps_error_days': mae(days, true_days),
        'remaining_time_crps_spread_days': crps_spread(days),
        'remaining_time_log_error': float(np.log1p(days).mean() - np.log1p(true_days)),
        'remaining_time_truth_quantile': randomized_pit(days, true_days, noise=prefix.pit_noise),
    }
