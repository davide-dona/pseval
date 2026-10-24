from pseval.metrics.functions.scoring_rules import fair_brier
from pseval.metrics.inputs import MetricInputs


def support(inputs: MetricInputs) -> dict[str, float]:
    """The support columns of one inputs."""
    witnesses, presence = inputs.soft_witnesses, inputs.activity_presence
    return {
        'declare_support_brier': float(fair_brier(witnesses[:-1], witnesses[-1]).mean()),
        'declare_activation_brier': float(fair_brier(presence[:-1], presence[-1]).mean()),
    }
