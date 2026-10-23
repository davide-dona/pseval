from pseval.metrics.functions.scoring_rules import crps, crps_spread, mae, randomized_pit
from pseval.metrics.prepared import PreparedPrefix
from pseval.metrics.registry import METRICS, Direction, MetricGroup


@METRICS.register(
    'suffix_length_crps_error',
    label='Suffix length CRPS error',
    group=MetricGroup.SUFFIX_LENGTH,
    unit='events',
    bounds=(0.0, None),
    diagnostic=True,
)
def suffix_length_crps_error(context: PreparedPrefix) -> float:
    """CRPS error term of suffix length, the mean absolute error of the draws."""
    return mae(context.suffix_lengths, context.true_suffix_length)


@METRICS.register(
    'suffix_length_signed_error',
    label='Suffix length signed error',
    group=MetricGroup.SUFFIX_LENGTH,
    unit='events',
    diagnostic=True,
    direction=Direction.ZERO,
)
def suffix_length_signed_error(context: PreparedPrefix) -> float:
    """Mean sampled suffix length minus the observed one; zero without draws."""
    if not context.suffix_lengths.size:
        return 0.0
    return float(context.suffix_lengths.mean() - context.true_suffix_length)


@METRICS.register(
    'truncated_draw_rate',
    label='Truncated draw rate',
    group=MetricGroup.SUFFIX_LENGTH,
    bounds=(0.0, 1.0),
    diagnostic=True,
    direction=Direction.LOWER,
)
def truncated_draw_rate(context: PreparedPrefix) -> float:
    """Share of draws closed at the length cap without ending; zero without draws."""
    truncated = context.generation.samples.truncated
    return float(truncated.mean()) if truncated.size else 0.0


@METRICS.register(
    'suffix_length_crps',
    label='Suffix length CRPS',
    group=MetricGroup.SUFFIX_LENGTH,
    unit='events',
    bounds=(0.0, None),
    direction=Direction.LOWER,
)
def suffix_length_crps(context: PreparedPrefix) -> float:
    """Suffix-length CRPS; zero without draws."""
    return crps(context.suffix_lengths, context.true_suffix_length)


@METRICS.register(
    'suffix_length_crps_spread',
    label='Suffix length CRPS spread',
    group=MetricGroup.SUFFIX_LENGTH,
    unit='events',
    bounds=(0.0, None),
    diagnostic=True,
)
def suffix_length_crps_spread(context: PreparedPrefix) -> float:
    """CRPS spread term of suffix length, half the mean distance between draws."""
    return crps_spread(context.suffix_lengths)


@METRICS.register(
    'suffix_length_truth_quantile',
    label='Suffix length truth quantile',
    group=MetricGroup.SUFFIX_LENGTH,
    bounds=(0.0, 1.0),
    diagnostic=True,
)
def suffix_length_truth_quantile(context: PreparedPrefix) -> float:
    """Randomized PIT of the observed suffix length among the draws."""
    return randomized_pit(
        context.suffix_lengths, context.true_suffix_length, noise=context.pit_noise
    )
