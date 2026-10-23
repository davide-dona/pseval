from pseval.metrics.prepared import PreparedPrefix
from pseval.metrics.registry import METRICS, Direction, MetricGroup, Owner


@METRICS.register(
    'conformance_sample_mean',
    label='Conformance (sample mean)',
    group=MetricGroup.CONFORMANCE,
    bounds=(0.0, 1.0),
    diagnostic=True,
    direction=Direction.HIGHER,
)
def conformance_sample_mean(context: PreparedPrefix) -> float:
    """Draw-weighted mean share of satisfied constraints; zero without draws."""
    return context.generation.samples.mean(context.sample_conformance_share)


@METRICS.register(
    'conformance_share_bias',
    label='Conformance share bias',
    group=MetricGroup.CONFORMANCE,
    bounds=(-1.0, 1.0),
    diagnostic=True,
    direction=Direction.ZERO,
)
def conformance_share_bias(context: PreparedPrefix) -> float:
    """Draw-weighted satisfied-constraint share minus the observed trace's."""
    return conformance_sample_mean(context) - context.observed_conformance_share


@METRICS.register(
    'conformance_observed',
    label='Conformance observed',
    group=MetricGroup.CONFORMANCE,
    bounds=(0.0, 1.0),
    owner=Owner.LOG,
    diagnostic=True,
)
def conformance_observed(context: PreparedPrefix) -> float:
    """Satisfied-constraint share of the observed trace."""
    return context.observed_conformance_share


@METRICS.register(
    'full_conformance_sample_rate',
    label='Full conformance sample rate',
    group=MetricGroup.CONFORMANCE,
    bounds=(0.0, 1.0),
    direction=Direction.HIGHER,
)
def full_conformance_sample_rate(context: PreparedPrefix) -> float:
    """Draw-weighted rate of fully conformant traces; zero without draws."""
    return context.generation.samples.mean(context.sample_full_conformance)


@METRICS.register(
    'full_conformance_observed',
    label='Full conformance observed',
    group=MetricGroup.CONFORMANCE,
    bounds=(0.0, 1.0),
    owner=Owner.LOG,
)
def full_conformance_observed(context: PreparedPrefix) -> float:
    """1.0 when the observed trace satisfies every constraint."""
    return context.observed_full_conformance
