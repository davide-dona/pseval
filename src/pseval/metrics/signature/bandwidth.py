import numpy as np
import torch

# Pairwise distances of the bandwidth median are visited in square tiles of at most this side.
_TILE = 8192


# Bit widths of the radix passes selecting the median among float64 bit patterns, high to low.
_RADIX_BITS = (22, 21, 21)


def weighted_median_squared_distance(points: torch.Tensor, weights: torch.Tensor) -> float:
    """Return the lower weighted median of squared distances over pairs of distinct points.

    The pair of points `i < j` weighs `weights[i] * weights[j]`. Nonnegative float64 values
    order as their bit patterns, so a radix selection over `_RADIX_BITS` narrows the bit pattern
    of the median one group of bits per pass without storing any distance.

    Args:
        points: `[N, D]` distinct float64 path points: a time, then integer activity counts.
        weights: `[N]` float64 multiplicity of each point.
    """
    device = points.device
    tile = min(_TILE, len(points))
    # Pad to whole tiles with weightless points, so every tile has one compiled shape.
    padding = -len(points) % tile
    times = torch.nn.functional.pad(points[:, 0], (0, padding))  # [N']
    counts = torch.nn.functional.pad(points[:, 1:].float(), (0, 0, 0, padding))  # [N', A]
    weights = torch.nn.functional.pad(weights, (0, padding))  # [N']
    total = (weights.sum() ** 2 - weights.square().sum()).item() / 2.0
    target = total / 2.0
    prefix, known = 0, 0
    for bits in _RADIX_BITS:
        histogram = torch.zeros(1 << bits, dtype=torch.float64, device=device)
        # A shift of 63 leaves only the sign bit, zero for every distance, while nothing is known.
        settings = torch.tensor(
            [prefix, min(64 - known, 63), 64 - known - bits, (1 << bits) - 1], device=device
        )
        for row in range(0, len(times), tile):
            for column in range(row, len(times), tile):
                rows, columns = slice(row, row + tile), slice(column, column + tile)
                _tile_histogram(
                    histogram,
                    times[rows],
                    times[columns],
                    counts[rows],
                    counts[columns],
                    weights[rows],
                    weights[columns],
                    torch.tensor(column - row, device=device),
                    settings,
                )
        cumulative = torch.cumsum(histogram, dim=0)
        target_tensor = torch.tensor(target, dtype=torch.float64, device=device)
        bucket = min(int(torch.searchsorted(cumulative, target_tensor).item()), (1 << bits) - 1)
        if bucket:
            target -= cumulative[bucket - 1].item()
        prefix = (prefix << bits) | bucket
        known += bits
    return float(np.int64(prefix).view(np.float64))


@torch.compile(dynamic=False)
def _tile_histogram(
    histogram: torch.Tensor,
    row_times: torch.Tensor,
    column_times: torch.Tensor,
    row_counts: torch.Tensor,
    column_counts: torch.Tensor,
    row_weights: torch.Tensor,
    column_weights: torch.Tensor,
    offset: torch.Tensor,
    settings: torch.Tensor,
) -> None:
    """Add the weights of one tile's pairs `i < j` whose distance matches the known prefix.

    `settings` holds the known bit prefix, the shift exposing it, the shift and the mask
    selecting this pass's bits. The count part of each distance is an integer far below 2**24,
    so it is exact in float32; the time part is formed in float64.
    """
    prefix, prefix_shift, shift, mask = settings
    row_norms = row_counts.square().sum(dim=1)  # [T]
    column_norms = column_counts.square().sum(dim=1)  # [T]
    count_part = row_norms[:, None] + column_norms[None, :] - 2.0 * row_counts @ column_counts.T
    squared = count_part.double() + (row_times[:, None] - column_times[None, :]).square()
    keys = squared.view(torch.int64)  # [T, T]
    index = torch.arange(len(row_times), device=keys.device)
    upper = index[:, None] < index[None, :] + offset
    matches = upper & ((keys >> prefix_shift) == prefix)
    pair_weights = torch.where(matches, row_weights[:, None] * column_weights[None, :], 0.0)
    histogram.index_add_(0, ((keys >> shift) & mask).flatten(), pair_weights.flatten())
