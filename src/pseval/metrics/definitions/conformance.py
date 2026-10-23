from pseval.metrics.prepared import PreparedPrefix


def conformance(prefix: PreparedPrefix) -> dict[str, float]:
    """The conformance columns of one prefix."""
    samples = prefix.generation.samples
    sample_mean = samples.mean(prefix.sample_conformance_share)
    return {
        'conformance_sample_mean': sample_mean,
        'conformance_share_bias': sample_mean - prefix.observed_conformance_share,
        'conformance_observed': prefix.observed_conformance_share,
        'full_conformance_sample_rate': samples.mean(prefix.sample_full_conformance),
        'full_conformance_observed': prefix.observed_full_conformance,
    }
