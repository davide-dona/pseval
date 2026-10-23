import numpy as np

from pseval.metrics.functions.scoring_rules import crps, crps_spread, mae, randomized_pit
from pseval.metrics.prepared import PreparedPrefix
from pseval.metrics.registry import METRICS, Direction, MetricGroup

MINUTES_PER_DAY = 1440.0


@METRICS.register(
    'remaining_time_crps_days',
    label='Remaining-time CRPS',
    group=MetricGroup.REMAINING_TIME,
    unit='days',
    bounds=(0.0, None),
    direction=Direction.LOWER,
)
def remaining_time_crps_days(context: PreparedPrefix) -> float:
    """Remaining-time CRPS in days; zero without draws."""
    return crps(context.remaining_times, context.true_remaining_time) / MINUTES_PER_DAY


@METRICS.register(
    'remaining_time_log_error',
    label='Remaining-time log error',
    group=MetricGroup.REMAINING_TIME,
    unit='log days',
    diagnostic=True,
    direction=Direction.ZERO,
)
def remaining_time_log_error(context: PreparedPrefix) -> float:
    """Mean `log1p` sampled remaining days minus the observed one; zero without draws."""
    if not context.remaining_times.size:
        return 0.0
    sampled = np.log1p(context.remaining_times / MINUTES_PER_DAY).mean()
    return float(sampled - np.log1p(context.true_remaining_time / MINUTES_PER_DAY))


@METRICS.register(
    'remaining_time_crps_error_days',
    label='Remaining-time CRPS error',
    group=MetricGroup.REMAINING_TIME,
    unit='days',
    bounds=(0.0, None),
    diagnostic=True,
)
def remaining_time_crps_error_days(context: PreparedPrefix) -> float:
    """CRPS error term of remaining time, the mean absolute error of the draws, in days."""
    return mae(context.remaining_times, context.true_remaining_time) / MINUTES_PER_DAY


@METRICS.register(
    'remaining_time_crps_spread_days',
    label='Remaining-time CRPS spread',
    group=MetricGroup.REMAINING_TIME,
    unit='days',
    bounds=(0.0, None),
    diagnostic=True,
)
def remaining_time_crps_spread_days(context: PreparedPrefix) -> float:
    """CRPS spread term of remaining time, half the mean distance between draws, in days."""
    return crps_spread(context.remaining_times) / MINUTES_PER_DAY


@METRICS.register(
    'remaining_time_truth_quantile',
    label='Remaining-time truth quantile',
    group=MetricGroup.REMAINING_TIME,
    bounds=(0.0, 1.0),
    diagnostic=True,
)
def remaining_time_truth_quantile(context: PreparedPrefix) -> float:
    """Randomized PIT of the observed remaining time among the draws."""
    return randomized_pit(
        context.remaining_times, context.true_remaining_time, noise=context.pit_noise
    )
