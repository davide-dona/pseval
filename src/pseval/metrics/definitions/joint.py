from pseval.metrics.functions.scoring_rules import kernel_score
from pseval.metrics.inputs import MetricInputs


def joint(inputs: MetricInputs) -> dict[str, float]:
    """The joint activity and time columns of one inputs."""
    gram = inputs.signature_gram  # [K + 1, K + 1], the observed suffix last
    return {'signature_kernel_score': kernel_score(gram[:-1, :-1], gram[:-1, -1])}
