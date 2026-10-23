from dataclasses import dataclass, fields
from typing import Self

from pseval.metrics.definitions.activity import activity
from pseval.metrics.definitions.conformance import conformance
from pseval.metrics.definitions.joint import joint
from pseval.metrics.definitions.remaining_time import remaining_time
from pseval.metrics.definitions.suffix_length import suffix_length
from pseval.metrics.prepared import PreparedPrefix


@dataclass(frozen=True, slots=True)
class PrefixScores:
    """Every metric of one prefix. Field order is the scores column order.

    DLS is the normalized Damerau-Levenshtein similarity and DLD its distance, both with a shared
    terminal token. Draw-weighted means over no draws are zero.
    """

    # Fair kernel score of the joint activity and time draws, under the cosine-normalized
    # signature kernel.
    signature_kernel_score: float

    # Draw-weighted DLS of the activity suffixes to the observed one.
    dls_sample_mean: float
    # Mean DLD over ordered pairs of different draws; zero with fewer than two draws.
    activity_dld_spread: float
    # The DLD spread minus the mean DLD to the observed suffix. For a calibrated forecast the
    # observed suffix is one more draw, so it is zero; negative is overconfident. Zero with fewer
    # than two draws.
    activity_spread_error_gap: float
    # Distinct activity suffixes per draw, 1.0 when every draw differs.
    unique_suffix_ratio: float
    # Expected highest DLS to the observed suffix among 5 and 10 random draws.
    best_of_5_dls: float
    best_of_10_dls: float
    # Fair activity energy scores on DLD, exact-match distance, and multiset Jaccard distance
    # over padded bigrams.
    energy_score_dls: float
    energy_score_exact: float
    energy_score_bigram: float

    # Fair suffix length CRPS in events, then its error and spread terms: the mean absolute error
    # of the draws, and half the mean distance between them.
    suffix_length_crps: float
    suffix_length_crps_error: float
    suffix_length_crps_spread: float
    # Mean sampled suffix length minus the observed one.
    suffix_length_signed_error: float
    # Randomized PIT of the observed suffix length among the draws.
    suffix_length_truth_quantile: float
    # Share of draws closed at the length cap without ending.
    truncated_draw_rate: float

    # Fair remaining-time CRPS in days, then its error and spread terms.
    remaining_time_crps_days: float
    remaining_time_crps_error_days: float
    remaining_time_crps_spread_days: float
    # Mean `log1p` sampled remaining days minus the observed one.
    remaining_time_log_error: float
    # Randomized PIT of the observed remaining time among the draws.
    remaining_time_truth_quantile: float

    # Draw-weighted share of the Declare constraints the full trace satisfies, its difference to
    # the observed trace's, and the observed trace's.
    conformance_sample_mean: float
    conformance_share_bias: float
    conformance_observed: float
    # Draw-weighted rate of traces satisfying every constraint, and 1.0 when the observed one
    # does.
    full_conformance_sample_rate: float
    full_conformance_observed: float

    @classmethod
    def of(cls, prefix: PreparedPrefix) -> Self:
        """Score one prefix."""
        return cls(
            **joint(prefix),
            **activity(prefix),
            **suffix_length(prefix),
            **remaining_time(prefix),
            **conformance(prefix),
        )


METRIC_KEYS = tuple(field.name for field in fields(PrefixScores))

# The fewest draws per prefix every metric is defined on: best-of-10 needs ten.
MIN_DRAWS = 10

# The proper scoring rules, lower is better. Models are ranked on them.
PROPER_SCORES = (
    'signature_kernel_score',
    'energy_score_dls',
    'energy_score_exact',
    'energy_score_bigram',
    'suffix_length_crps',
    'remaining_time_crps_days',
)
