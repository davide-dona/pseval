from pseval.metrics.functions.scoring_rules import kernel_score
from pseval.metrics.prepared import PreparedPrefix
from pseval.metrics.registry import METRICS, Direction, MetricGroup


@METRICS.register(
    'signature_kernel_score',
    label='Signature kernel score',
    group=MetricGroup.JOINT,
    direction=Direction.LOWER,
)
def signature_kernel_score(context: PreparedPrefix) -> float:
    """Fair kernel score of the joint activity and time draws, under the cosine-normalized
    signature kernel."""
    return kernel_score(context.signature_gram)
