from pseval.metrics.prepared import PreparedPrefix


def conformance(prefix: PreparedPrefix) -> dict[str, float]:
    """The conformance columns of one prefix."""
    samples = prefix.generation.samples
    return {
        'full_conformance_sample_rate': samples.mean(prefix.sample_full_conformance),
        'full_conformance_observed': prefix.observed_full_conformance,
    }
