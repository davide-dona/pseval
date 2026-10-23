import math

import numpy as np


def energy_score(draw_counts: np.ndarray, distances: np.ndarray) -> float:
    """Return the fair energy score from distances between distinct draw values.

    Repeated draws share one distinct value. Their counts preserve draw multiplicity while the
    distance matrix keeps one row per distinct value.

    Args:
        draw_counts: `[U]` draws of each distinct value, summing to at least two.
        distances: `[U, U + 1]` distances from each distinct value to every distinct value, then
            to the observation.

    Returns:
        Draw-weighted mean distance to the observation minus the fair diversity correction over
        pairs of different draws.
    """
    draw_count = float(draw_counts.sum())
    accuracy = float(draw_counts @ distances[:, -1]) / draw_count
    pair_sum = float(draw_counts @ distances[:, :-1] @ draw_counts)
    return accuracy - pair_sum / (2.0 * draw_count * (draw_count - 1.0))


def mean_between_draws(draw_counts: np.ndarray, distances: np.ndarray) -> float:
    """Return the mean distance over ordered pairs of different draws.

    Two draws of the same distinct value sit on the zero diagonal, so they add nothing to the
    sum while the pair count still includes them.

    Args:
        draw_counts: `[U]` draws of each distinct value, summing to at least two.
        distances: `[U, U]` distances between the distinct values.
    """
    draw_count = float(draw_counts.sum())
    return float(draw_counts @ distances @ draw_counts) / (draw_count * (draw_count - 1.0))


def best_of_k(similarities: np.ndarray, k: int) -> float:
    """Return the expected best similarity over k draws taken without replacement.

    With the K draw similarities sorted ascending, the i-th smallest (1-indexed) is the best of
    a uniformly random k-subset with probability C(i - 1, k - 1) / C(K, k). The estimate is
    unbiased for every k <= K, so it does not depend on how many draws were generated.

    Args:
        similarities: `[K]` one similarity per draw, repeated draws retained.
        k: Subset size, at most K.
    """
    draw_count = len(similarities)
    ordered = np.sort(similarities)  # [K]
    subsets = math.comb(draw_count, k)
    weights = [math.comb(rank, k - 1) / subsets for rank in range(draw_count)]  # [K]
    return float(ordered @ weights)


def crps(draws: np.ndarray, truth: float) -> float:
    """Return the fair CRPS of scalar draws, integrated over gaps between sorted samples.

    The fair score equals a sum of gap lengths with nonnegative weights, so it never
    subtracts two large accuracy and diversity totals. That keeps it exact and nonnegative
    when a few draws are orders of magnitude larger than the rest.

    Args:
        draws: `[K]` numeric samples, at least two, with repeated draws retained.
        truth: Observed quantity, in the same units as the samples.

    Returns:
        Fair absolute-distance energy score in the input units.
    """
    draw_count = len(draws)
    ordered = np.sort(draws)  # [K]
    lower, upper = ordered[:-1], ordered[1:]  # [K - 1]
    below_truth = np.clip(np.minimum(upper, truth) - lower, 0.0, None)  # [K - 1]
    above_truth = np.clip(upper - np.maximum(lower, truth), 0.0, None)  # [K - 1]
    # Number of draws at or below each gap between consecutive sorted draws.
    at_or_below = np.arange(1, draw_count, dtype=np.float64)  # [K - 1]
    at_or_above = draw_count - at_or_below
    between = (
        at_or_below * (at_or_below - 1.0) * below_truth
        + at_or_above * (at_or_above - 1.0) * above_truth
    ).sum() / (draw_count * (draw_count - 1.0))
    outside = max(ordered[0] - truth, 0.0) + max(truth - ordered[-1], 0.0)
    return float(between + outside)


def mae(draws: np.ndarray, truth: float) -> float:
    """Return the mean absolute error of scalar draws, the accuracy term of the CRPS."""
    return float(np.abs(draws - truth).mean())


def crps_spread(draws: np.ndarray) -> float:
    """Return half the mean absolute difference over pairs of different draws, at least two.

    It is the spread term of the fair CRPS, which equals the mean absolute error minus it. The
    sum over pairs is taken over gaps between sorted samples with nonnegative weights.
    """
    draw_count = len(draws)
    gaps = np.diff(np.sort(draws))  # [K - 1]
    # Each gap separates the draws at or below it from those above it.
    at_or_below = np.arange(1, draw_count, dtype=np.float64)  # [K - 1]
    pair_sum = float((at_or_below * (draw_count - at_or_below) * gaps).sum())
    return pair_sum / (draw_count * (draw_count - 1.0))


def randomized_pit(draws: np.ndarray, truth: float, *, noise: float) -> float:
    """Return the randomized probability integral transform of the truth among the draws.

    Ties between draws and the truth are spread uniformly by the noise, so the value is uniform
    on [0, 1] for a calibrated forecast even when draws and truth are discrete.

    Args:
        draws: `[K]` numeric samples, with repeated draws retained.
        truth: Observed quantity, in the same units as the samples.
        noise: Uniform value on [0, 1] that places the truth among its ties.

    Returns:
        (draws below truth + noise * (draws equal to truth + 1)) / (K + 1).
    """
    below = float((draws < truth).sum())
    ties = float((draws == truth).sum())
    return (below + noise * (ties + 1.0)) / (len(draws) + 1.0)


def kernel_score(gram: np.ndarray) -> float:
    """Return the fair kernel score from a normalized Gram matrix with the observation last.

    With a unit self-kernel the score reads `1/2 E k(X, X') - E k(X, y) + 1/2`, the expectation
    over pairs of different draws.

    Args:
        gram: `[K + 1, K + 1]` kernel between every pair of at least two draws, the observation
            last.
    """
    draws = len(gram) - 1
    to_truth = float(gram[:draws, draws].mean())
    between = gram[:draws, :draws]
    pairs = (float(between.sum()) - float(between.trace())) / (draws * (draws - 1))
    return 0.5 * pairs - to_truth + 0.5
