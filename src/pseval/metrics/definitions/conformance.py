from pseval.metrics.prepared import PreparedPrefix


def conformance(prefix: PreparedPrefix) -> dict[str, float]:
    """The conformance columns of one prefix."""
    return {
        'full_conformance_sample_rate': float(prefix.sample_full_conformance.mean()),
        'full_conformance_observed': prefix.observed_full_conformance,
    }
