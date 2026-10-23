from dataclasses import dataclass, fields
from typing import Self

from pseval.metrics.definitions.activity import activity
from pseval.metrics.definitions.conformance import conformance
from pseval.metrics.definitions.joint import joint
from pseval.metrics.definitions.remaining_time import remaining_time
from pseval.metrics.definitions.suffix_length import suffix_length
from pseval.metrics.inputs import MetricInputs


@dataclass(frozen=True, slots=True)
class PrefixScores:
    """Every metric of one prefix. Field order is the scores column order.

    DLS is the normalized Damerau-Levenshtein similarity and DLD its distance. Scoring rules are
    fair, and truth quantiles are randomized PITs of the observed value among the draws.
    """

    signature_kernel_score: float

    dls_sample_mean: float
    activity_dld_spread: float
    # Zero for a calibrated forecast, negative when overconfident.
    activity_spread_error_gap: float
    unique_suffix_ratio: float
    energy_score_dls: float
    energy_score_exact: float
    # On the multiset Jaccard distance between padded bigrams.
    energy_score_bigram: float

    suffix_length_crps: float
    suffix_length_crps_error: float
    suffix_length_crps_spread: float
    suffix_length_signed_error: float
    suffix_length_truth_quantile: float
    truncated_draw_rate: float

    remaining_time_crps_days: float
    remaining_time_crps_error_days: float
    remaining_time_crps_spread_days: float
    # On `log1p` days.
    remaining_time_log_error: float
    remaining_time_truth_quantile: float

    full_conformance_sample_rate: float
    full_conformance_observed: float

    @classmethod
    def of(cls, inputs: MetricInputs) -> Self:
        """Score one prefix."""
        return cls(
            **joint(inputs),
            **activity(inputs),
            **suffix_length(inputs),
            **remaining_time(inputs),
            **conformance(inputs),
        )


METRIC_KEYS = tuple(field.name for field in fields(PrefixScores))

# The proper scoring rules, lower is better. Models are ranked on them.
PROPER_SCORES = (
    'signature_kernel_score',
    'energy_score_dls',
    'energy_score_exact',
    'energy_score_bigram',
    'suffix_length_crps',
    'remaining_time_crps_days',
)
