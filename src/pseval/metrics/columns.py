from dataclasses import Field, dataclass, field, fields
from typing import Self

from pseval.metrics.definitions.activity import activity
from pseval.metrics.definitions.conformance import conformance
from pseval.metrics.definitions.joint import joint
from pseval.metrics.definitions.remaining_time import remaining_time
from pseval.metrics.definitions.suffix_length import suffix_length
from pseval.metrics.definitions.support import support
from pseval.metrics.inputs import MetricInputs


def _family(name: str) -> Field:
    """Declare a `PrefixScores` field as a metric of the family `name`."""
    return field(metadata={'family': name})


@dataclass(frozen=True, slots=True)
class PrefixScores:
    """Every metric of one prefix. Field order is the scores column order, and each field names
    its family, the definition module that computes it.

    DLS is the normalized Damerau-Levenshtein similarity and DLD its distance. Scoring rules are
    fair, and truth quantiles are randomized PITs of the observed value among the draws.
    """

    signature_kernel_score: float = _family('joint')

    dls_sample_mean: float = _family('activity')
    activity_dld_spread: float = _family('activity')
    # Zero for a calibrated forecast, negative when overconfident.
    activity_spread_error_gap: float = _family('activity')
    unique_suffix_ratio: float = _family('activity')
    energy_score_dls: float = _family('activity')
    energy_score_exact: float = _family('activity')
    # On the multiset Jaccard distance between padded bigrams.
    energy_score_bigram: float = _family('activity')

    suffix_length_crps: float = _family('suffix_length')
    suffix_length_crps_error: float = _family('suffix_length')
    suffix_length_crps_spread: float = _family('suffix_length')
    suffix_length_signed_error: float = _family('suffix_length')
    suffix_length_truth_quantile: float = _family('suffix_length')
    truncated_draw_rate: float = _family('suffix_length')

    remaining_time_crps_days: float = _family('remaining_time')
    remaining_time_crps_error_days: float = _family('remaining_time')
    remaining_time_crps_spread_days: float = _family('remaining_time')
    # On `log1p` days.
    remaining_time_log_error: float = _family('remaining_time')
    remaining_time_truth_quantile: float = _family('remaining_time')

    full_conformance_sample_rate: float = _family('conformance')
    full_conformance_observed: float = _family('conformance')
    # Mean fair Brier score of the completed trace witnessing each soft Declare rule.
    declare_support_brier: float = _family('support')
    # Mean fair Brier score of the completed trace holding each vocabulary activity.
    declare_activation_brier: float = _family('support')

    @classmethod
    def of(cls, inputs: MetricInputs) -> Self:
        """Score one prefix."""
        return cls(
            **joint(inputs),
            **activity(inputs),
            **suffix_length(inputs),
            **remaining_time(inputs),
            **conformance(inputs),
            **support(inputs),
        )


METRIC_KEYS = tuple(metric.name for metric in fields(PrefixScores))

# Every metric key under its family, both in `METRIC_KEYS` order.
METRIC_FAMILIES = {
    family: tuple(
        metric.name for metric in fields(PrefixScores) if metric.metadata['family'] == family
    )
    for family in dict.fromkeys(metric.metadata['family'] for metric in fields(PrefixScores))
}

# The proper scoring rules, lower is better. Models are ranked on them.
PROPER_SCORES = (
    'signature_kernel_score',
    'energy_score_dls',
    'energy_score_exact',
    'energy_score_bigram',
    'suffix_length_crps',
    'remaining_time_crps_days',
    'declare_support_brier',
    'declare_activation_brier',
)
