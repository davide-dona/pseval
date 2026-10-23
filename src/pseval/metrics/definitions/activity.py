import numpy as np

from pseval.metrics.functions.scoring_rules import best_of_k, energy_score, mean_between_draws
from pseval.metrics.functions.sequences import (
    bigram_distances,
    dld_distances,
    exact_distances,
    sequence_similarity,
)
from pseval.metrics.prepared import PreparedPrefix
from pseval.metrics.registry import METRICS, Direction, MetricGroup


def _draw_similarities(context: PreparedPrefix) -> np.ndarray:
    """`[U]` DLS similarity of each distinct sampled suffix to the observed one."""
    samples, truth = context.generation.samples, context.generation.truth
    return np.array(
        [sequence_similarity(suffix, truth.activities) for suffix in samples.suffixes],
        dtype=np.float64,
    )


@METRICS.register(
    'dls_sample_mean',
    label='DLS sample mean',
    group=MetricGroup.ACTIVITY,
    bounds=(0.0, 1.0),
    diagnostic=True,
    direction=Direction.HIGHER,
)
def dls_sample_mean(context: PreparedPrefix) -> float:
    """Draw-weighted DLS similarity of the activity suffixes to the observed one; zero without
    draws."""
    return context.generation.samples.mean(_draw_similarities(context))


@METRICS.register(
    'activity_dld_spread',
    label='Activity DLD spread',
    group=MetricGroup.ACTIVITY,
    bounds=(0.0, 1.0),
    diagnostic=True,
)
def activity_dld_spread(context: PreparedPrefix) -> float:
    """Mean DLD over ordered pairs of different draws; zero with fewer than two draws."""
    samples = context.generation.samples
    if len(samples) < 2:
        return 0.0
    return mean_between_draws(samples.counts, dld_distances(samples.suffixes, samples.suffixes))


@METRICS.register(
    'activity_spread_error_gap',
    label='Activity spread error gap',
    group=MetricGroup.ACTIVITY,
    bounds=(-1.0, 1.0),
    diagnostic=True,
    direction=Direction.ZERO,
)
def activity_spread_error_gap(context: PreparedPrefix) -> float:
    """Mean DLD between draws minus their mean DLD to the observed suffix.

    For a calibrated forecast the observed suffix is one more draw, so both means agree. A
    negative gap means draws agree with each other more than with the truth, which is
    overconfidence. Fewer than two draws score zero.
    """
    samples, truth = context.generation.samples, context.generation.truth
    if len(samples) < 2:
        return 0.0
    distances = dld_distances(samples.suffixes, (*samples.suffixes, truth.activities))
    return mean_between_draws(samples.counts, distances[:, :-1]) - samples.mean(distances[:, -1])


@METRICS.register(
    'unique_suffix_ratio',
    label='Unique suffix ratio',
    group=MetricGroup.ACTIVITY,
    bounds=(0.0, 1.0),
    diagnostic=True,
)
def unique_suffix_ratio(context: PreparedPrefix) -> float:
    """Distinct activity suffixes per draw, 1.0 when every draw differs; zero without draws."""
    samples = context.generation.samples
    return len(samples.suffixes) / len(samples) if len(samples) else 0.0


@METRICS.register(
    'best_of_5_dls',
    label='Best-of-5 DLS',
    group=MetricGroup.ACTIVITY,
    bounds=(0.0, 1.0),
    diagnostic=True,
    direction=Direction.HIGHER,
)
def best_of_5_dls(context: PreparedPrefix) -> float:
    """Expected highest DLS to the observed suffix among 5 random draws."""
    return best_of_k(_draw_similarities(context)[context.generation.samples.taken], 5)


@METRICS.register(
    'best_of_10_dls',
    label='Best-of-10 DLS',
    group=MetricGroup.ACTIVITY,
    bounds=(0.0, 1.0),
    diagnostic=True,
    direction=Direction.HIGHER,
)
def best_of_10_dls(context: PreparedPrefix) -> float:
    """Expected highest DLS to the observed suffix among 10 random draws."""
    return best_of_k(_draw_similarities(context)[context.generation.samples.taken], 10)


@METRICS.register(
    'energy_score_dls',
    label='DLS energy score',
    group=MetricGroup.ACTIVITY,
    direction=Direction.LOWER,
)
def energy_score_dls(context: PreparedPrefix) -> float:
    """Activity energy score on normalized Damerau-Levenshtein distance."""
    samples, truth = context.generation.samples, context.generation.truth
    return energy_score(
        samples.suffixes, truth.activities, distance=dld_distances, draw_counts=samples.counts
    )


@METRICS.register(
    'energy_score_exact',
    label='Exact energy score',
    group=MetricGroup.ACTIVITY,
    direction=Direction.LOWER,
)
def energy_score_exact(context: PreparedPrefix) -> float:
    """Activity energy score on exact-match distance."""
    samples, truth = context.generation.samples, context.generation.truth
    return energy_score(
        samples.suffixes, truth.activities, distance=exact_distances, draw_counts=samples.counts
    )


@METRICS.register(
    'energy_score_bigram',
    label='Bigram energy score',
    group=MetricGroup.ACTIVITY,
    direction=Direction.LOWER,
)
def energy_score_bigram(context: PreparedPrefix) -> float:
    """Activity energy score on multiset Jaccard distance over padded bigrams."""
    samples, truth = context.generation.samples, context.generation.truth
    return energy_score(
        samples.suffixes, truth.activities, distance=bigram_distances, draw_counts=samples.counts
    )
