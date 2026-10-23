from pseval.metrics.functions.scoring_rules import kernel_score
from pseval.metrics.prepared import PreparedPrefix


def joint(prefix: PreparedPrefix) -> dict[str, float]:
    """The joint activity and time columns of one prefix."""
    return {'signature_kernel_score': kernel_score(prefix.signature_gram)}
