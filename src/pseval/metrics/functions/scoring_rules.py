import numpy as np


def energy_score(distances_between: np.ndarray, distances_to_truth: np.ndarray) -> float:
    """Return the fair energy score of the draws, which is equivalent to:
    `mean_i d(x_i, y) - 1/2 mean_{i!=j} d(x_i, x_j)`.

    Args:
        distances_between: `[K, K]` distance between each pair of at least two draws, zero on
            the diagonal.
        distances_to_truth: `[K]` distance from each draw to the observation.
    """
    return float(distances_to_truth.mean()) - 0.5 * mean_between_draws(distances_between)


def mean_between_draws(distances_between: np.ndarray) -> float:
    """Return the mean distance over ordered pairs of different draws, `mean_{i!=j} d(x_i, x_j)`.

    Args:
        distances_between: `[K, K]` distance between each pair of at least two draws, zero on
            the diagonal.
    """
    draws = len(distances_between)
    return float(distances_between.sum()) / (draws * (draws - 1.0))


def crps(draws: np.ndarray, truth: float) -> float:
    """Return the fair CRPS of scalar draws, `mean_i |x_i - y| - 1/2 mean_{i!=j} |x_i - x_j|`.

    It is summed over the gaps between sorted draws with nonnegative weights, never as that
    difference, which cancels catastrophically when a few draws are orders of magnitude larger
    than the rest.

    Args:
        draws: `[K]` at least two draws.
        truth: The observation, in the units of the draws.
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
    """Return the error term of the CRPS, `mean_i |x_i - y|`.

    Args:
        draws: `[K]` draws.
        truth: The observation, in the units of the draws.
    """
    return float(np.abs(draws - truth).mean())


def crps_spread(draws: np.ndarray) -> float:
    """Return the spread term of the CRPS, `1/2 mean_{i!=j} |x_i - x_j|`.

    Like `crps`, it is summed over the gaps between sorted draws with nonnegative weights.

    Args:
        draws: `[K]` at least two draws.
    """
    draw_count = len(draws)
    gaps = np.diff(np.sort(draws))  # [K - 1]
    # Each gap separates the draws at or below it from those above it.
    at_or_below = np.arange(1, draw_count, dtype=np.float64)  # [K - 1]
    pair_sum = float((at_or_below * (draw_count - at_or_below) * gaps).sum())
    return pair_sum / (draw_count * (draw_count - 1.0))


def randomized_pit(draws: np.ndarray, truth: float, *, noise: float) -> float:
    """Return the randomized PIT of the observation, `(#{x_i < y} + V (#{x_i = y} + 1)) / (K + 1)`.

    The noise `V` spreads the observation uniformly among its ties, so the value is uniform on
    [0, 1] for a calibrated forecast even when draws and observation are discrete.

    Args:
        draws: `[K]` draws.
        truth: The observation, in the units of the draws.
        noise: `V`, uniform on [0, 1].
    """
    below = float((draws < truth).sum())
    ties = float((draws == truth).sum())
    return (below + noise * (ties + 1.0)) / (len(draws) + 1.0)


def kernel_score(kernels_between: np.ndarray, kernels_to_truth: np.ndarray) -> float:
    """Return the fair kernel score, `1/2 mean_{i!=j} k(x_i, x_j) - mean_i k(x_i, y) + 1/2`.

    The constant makes the score of draws all equal to the observation zero under a normalized
    kernel.

    Args:
        kernels_between: `[K, K]` kernel between each pair of at least two draws, one on the
            diagonal.
        kernels_to_truth: `[K]` kernel between each draw and the observation.
    """
    draws = len(kernels_between)
    pair_sum = float(kernels_between.sum()) - float(kernels_between.trace())
    pairs = pair_sum / (draws * (draws - 1.0))
    return 0.5 * pairs - float(kernels_to_truth.mean()) + 0.5
