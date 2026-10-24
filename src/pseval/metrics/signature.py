from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Self

import numpy as np
import pysiglib
import torch
from scipy.spatial.distance import pdist

from pseval.data import Suffix

# Each cell of the PDE grid is split into 2**DYADIC_ORDER steps along both paths.
DYADIC_ORDER = 1

# Upper bound on the coarse grid cells of one solver call, which bounds its device memory.
CHUNK_CELLS = 1 << 26

# Path points, drawn with this seed from those of every training suffix, whose distances set the
# bandwidth.
BANDWIDTH_POINTS = 4096
BANDWIDTH_SEED = 0

# The current CUDA device, where kernels are solved.
_DEVICE = torch.device('cuda')


@dataclass(frozen=True)
class SignatureKernel:
    """Cosine-normalized signature kernel between suffixes, with an RBF static kernel.

    A suffix is embedded as a path through `[tau, counts]`: `tau` accumulates
    `time_scale * log1p(minutes)` over its inter-event times, and `counts` holds the cumulative
    occurrences of each activity, one axis per vocabulary entry. Each event first advances `tau`,
    then steps its activity count. Both constants are fitted on the train split. Kernels are
    solved by pySigLib on CUDA, in float32, since float64 runs at a small fraction of its speed
    on workstation GPUs.
    """

    time_scale: float
    bandwidth: float
    alphabet: int

    @classmethod
    def fit(cls, traces: Sequence[tuple[np.ndarray, np.ndarray]], *, alphabet: int) -> Self:
        """Fit the time scale and bandwidth on training suffixes.

        The time scale is the inverse mean `log1p` of every inter-event minute the suffixes
        hold. The bandwidth is the median Euclidean distance between different points among
        `BANDWIDTH_POINTS` drawn from the points of all their paths.

        Args:
            traces: One `(activity axes [L], inter-event minutes [L])` pair per suffix.
            alphabet: Number of activity axes; every axis lies in `[0, alphabet)`.
        """
        minutes = np.concatenate([np.zeros(0), *(times for _, times in traces)])
        time_scale = 1.0 / float(np.log1p(minutes).mean())
        points = np.concatenate(
            [embed(axes, times, time_scale=time_scale, alphabet=alphabet) for axes, times in traces]
        )
        chosen = np.random.default_rng(BANDWIDTH_SEED).choice(
            len(points), size=min(BANDWIDTH_POINTS, len(points)), replace=False
        )
        distances = pdist(points[chosen])
        return cls(
            time_scale=time_scale,
            bandwidth=float(np.median(distances[distances > 0])),
            alphabet=alphabet,
        )

    @classmethod
    def from_dict(cls, value: dict[str, float]) -> Self:
        """Read a stored calibration."""
        return cls(
            time_scale=float(value['time_scale']),
            bandwidth=float(value['bandwidth']),
            alphabet=int(value['alphabet']),
        )

    def as_dict(self) -> dict[str, float]:
        """The calibration as stored in a benchmark."""
        return {
            'time_scale': self.time_scale,
            'bandwidth': self.bandwidth,
            'alphabet': self.alphabet,
        }

    def embed(self, suffix: Suffix, *, axes: Mapping[str, int]) -> np.ndarray:
        """Embed one suffix as path points, each activity on its axis in `axes`.

        Returns:
            Points shaped `[2L + 1, 1 + alphabet]`, float64.
        """
        return embed(
            np.fromiter(map(axes.__getitem__, suffix.activities), np.int64, len(suffix)),
            suffix.inter_event_time_minutes,
            time_scale=self.time_scale,
            alphabet=self.alphabet,
        )

    def grams(self, groups: Sequence[Sequence[np.ndarray]]) -> list[np.ndarray]:
        """Return the normalized kernel matrix of each group of paths, solved in one batch.

        Identical paths within a group are solved once.

        Args:
            groups: Paths as `[N, 1 + alphabet]` points, all with the same width.

        Returns:
            One symmetric `[P, P]` float64 matrix per group with a unit diagonal.
        """
        paths: list[np.ndarray] = []
        layouts: list[tuple[np.ndarray, int]] = []
        left: list[np.ndarray] = []
        right: list[np.ndarray] = []
        for group in groups:
            distinct: dict[bytes, int] = {}
            index = np.fromiter(
                (distinct.setdefault(points.tobytes(), i) for i, points in enumerate(group)),
                dtype=np.int64,
                count=len(group),
            )
            first = np.unique(index)
            remap = np.zeros(len(group), dtype=np.int64)
            remap[first] = np.arange(len(first))
            start = len(paths)
            paths.extend(group[i] for i in first)
            rows, columns = np.triu_indices(len(first))
            left.append(start + rows)
            right.append(start + columns)
            layouts.append((remap[index], len(first)))
        values = self._pairs(
            paths,
            np.concatenate([np.zeros(0, np.int64), *left]),
            np.concatenate([np.zeros(0, np.int64), *right]),
        )
        grams = []
        cursor = 0
        for index, size in layouts:
            rows, columns = np.triu_indices(size)
            full = np.zeros((size, size))
            full[rows, columns] = values[cursor : cursor + len(rows)]
            full[columns, rows] = full[rows, columns]
            cursor += len(rows)
            norms = np.sqrt(np.diag(full))
            normalized = full / np.outer(norms, norms)
            np.fill_diagonal(normalized, 1.0)
            grams.append(normalized[np.ix_(index, index)])
        return grams

    def _pairs(
        self, paths: Sequence[np.ndarray], left: np.ndarray, right: np.ndarray
    ) -> np.ndarray:
        """Return the unnormalized signature kernel of each pair of paths.

        Pairs are solved in chunks of similar grid size, each padded by repeating its paths'
        last points, which adds zero increments and leaves the kernel unchanged.

        Args:
            paths: Path points shaped `[N, 1 + alphabet]`, all with the same width.
            left: `[P]` indices into `paths` of each pair's first path.
            right: `[P]` indices into `paths` of each pair's second path.

        Returns:
            `[P]` float64 kernel values in pair order.
        """
        sizes = np.fromiter(map(len, paths), np.int64, len(paths))
        starts = np.cumsum(sizes) - sizes
        points = torch.from_numpy(np.concatenate(paths)).to(_DEVICE, torch.float32)
        static_kernel = pysiglib.RBFKernel(2.0 * self.bandwidth**2)

        def padded(chosen: np.ndarray) -> torch.Tensor:
            width = np.arange(sizes[chosen].max())
            offsets = np.minimum(width[None, :], sizes[chosen, None] - 1)  # [B, width]
            return points[torch.from_numpy(starts[chosen, None] + offsets).to(_DEVICE)]

        swap = sizes[left] < sizes[right]
        long, short = np.where(swap, right, left), np.where(swap, left, right)
        order = np.lexsort((sizes[short], sizes[long]))
        long, short = long[order], short[order]
        # A chunk [start, stop) fits when (stop - start) * rows**2 cells of its longest path fit
        # the budget. `latest[i]` is the earliest start a chunk ending at pair i may have, and it
        # strictly increases since rows only grow along the sorted pairs.
        capacity = CHUNK_CELLS // sizes[long] ** 2
        latest = np.arange(1, len(order) + 1) - np.maximum(capacity, 1)
        values = np.empty(len(order))
        start = 0
        while start < len(order):
            stop = int(np.searchsorted(latest, start, side='right'))
            values[start:stop] = pysiglib.sig_kernel(
                padded(long[start:stop]),
                padded(short[start:stop]),
                dyadic_order=DYADIC_ORDER,
                static_kernel=static_kernel,
            ).cpu()
            start = stop
        unsorted = np.empty(len(order))
        unsorted[order] = values
        return unsorted


def embed(axes: np.ndarray, minutes: np.ndarray, *, time_scale: float, alphabet: int) -> np.ndarray:
    """Embed one suffix as the points of its time and activity counting path.

    Args:
        axes: `[L]` activity axis of each event, in `[0, alphabet)`.
        minutes: `[L]` nonnegative inter-event minutes before each event.
        time_scale: Factor applied to `log1p` of each inter-event minute.
        alphabet: Number of activity axes.

    Returns:
        `[2L + 1, 1 + alphabet]` float64 points: the origin, then for each event its point after
        the time step and its point after the activity step.
    """
    length = len(axes)
    tau = time_scale * np.cumsum(np.log1p(np.asarray(minutes, dtype=np.float64)))  # [L]
    counts = np.zeros((length + 1, alphabet))
    counts[np.arange(1, length + 1), axes] = 1.0
    counts = np.cumsum(counts, axis=0)  # [L + 1, A]
    points = np.zeros((2 * length + 1, 1 + alphabet))
    points[1::2, 0] = tau
    points[2::2, 0] = tau
    points[1::2, 1:] = counts[:-1]
    points[2::2, 1:] = counts[1:]
    return points
