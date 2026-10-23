import torch

# Each cell of the PDE grid is split into 2**DYADIC_ORDER steps along both paths.
DYADIC_ORDER = 1


def _solve_diagonals(increments: torch.Tensor) -> torch.Tensor:
    """Solve the Goursat PDE one anti-diagonal of all pairs at a time.

    Args:
        increments: `[B, H, W]` static kernel second differences of each pair's coarse cells,
            with `1 <= H <= W`.

    Returns:
        `[B]` float32 kernel values.
    """
    device = increments.device
    steps = 2**DYADIC_ORDER
    batch, height, width = increments.shape
    rows, columns = steps * height, steps * width
    diagonals = rows + columns - 1
    # The refined cells of a coarse cell share its increment, so the update coefficients are
    # computed once per coarse cell. The extra last cell holds the coefficients of a zero
    # increment, both exactly one, for refined cells outside the grid.
    scaled = increments.float().div_(steps**2).reshape(batch, height * width)
    coefficients = torch.ones(batch, 2, height * width + 1, dtype=torch.float32, device=device)
    coefficients[:, 1, :-1] = scaled.square().mul_(-1.0 / 12.0).add_(1.0)
    coefficients[:, 0, :-1] = scaled.mul_(0.5).add_(2.0).sub_(coefficients[:, 1, :-1])
    del scaled
    # Skew the refined grid so cell (i, j) sits at [d, i] with d = i + j, making every
    # anti-diagonal one row of `cells`, which holds the coarse cell of each refined cell.
    row = torch.arange(rows, device=device)
    column = torch.arange(diagonals, device=device)[:, None] - row  # [diagonals, rows]
    outside = (column < 0) | (column >= columns)
    cells = column.clamp_(min=0, max=columns - 1).floor_divide_(steps)
    cells += row.floor_divide(steps).mul_(width)
    cells.masked_fill_(outside, height * width)
    del outside
    # Points are skewed the same way, point (i, j) at [i + j, i], keeping only the three
    # latest anti-diagonals, in slot d % 3. Every point starts at the boundary value one; a
    # point left of the grid stays one, since the cell producing it has a zero increment and
    # only ones as inputs, and a point right of the grid is never read by a point inside it.
    solution = torch.ones(batch, 3, rows + 1, dtype=torch.float32, device=device)
    for diagonal in range(diagonals):
        first, second = coefficients.index_select(dim=2, index=cells[diagonal]).unbind(1)
        previous, earlier = solution[:, (diagonal + 1) % 3], solution[:, diagonal % 3]
        solution[:, (diagonal + 2) % 3, 1:] = (
            previous[:, 1:] + previous[:, :-1]
        ) * first - earlier[:, :-1] * second
    return solution[:, (diagonals + 1) % 3, -1]


def solve(increments: torch.Tensor) -> torch.Tensor:
    """Solve the Goursat PDE of each pair, on the device of `increments`.

    Args:
        increments: `[B, H, W]` static kernel second differences of each pair's coarse cells,
            with `1 <= H <= W`.

    Returns:
        `[B]` float32 kernel values.
    """
    if increments.is_cuda:
        from pseval.metrics.signature.goursat import sweep

        return sweep(increments, DYADIC_ORDER)
    return _solve_diagonals(increments)
