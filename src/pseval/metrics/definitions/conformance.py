from pseval.metrics.inputs import MetricInputs


def conformance(inputs: MetricInputs) -> dict[str, float]:
    """The conformance columns of one inputs."""
    return {
        'full_conformance_sample_rate': float(inputs.draw_full_conformance.mean()),
        'full_conformance_observed': inputs.observed_full_conformance,
    }
