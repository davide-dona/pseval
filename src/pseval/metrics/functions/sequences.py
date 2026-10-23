from collections import Counter
from collections.abc import Hashable, Sequence
from itertools import pairwise

import numpy as np
from rapidfuzz import process
from rapidfuzz.distance import DamerauLevenshtein
from scipy.spatial.distance import cdist

# Boundaries sequences are padded with, distinct from every activity name.
_START = object()
_END = object()


def damerau_levenshtein_distances(
    queries: Sequence[tuple[str, ...]], choices: Sequence[tuple[str, ...]]
) -> np.ndarray:
    """Return the normalized Damerau-Levenshtein distance between each query and each choice.

    Both sequences end with a shared terminal token, so the end of the case counts as one more
    activity.

    Args:
        queries: `[Q]` activity name sequences.
        choices: `[C]` activity name sequences.

    Returns:
        `[Q, C]` float64 distances in [0, 1].
    """
    return process.cdist(
        queries=[(*sequence, _END) for sequence in queries],
        choices=[(*sequence, _END) for sequence in choices],
        scorer=DamerauLevenshtein.normalized_distance,
        dtype=np.float64,
    )


def exact_distances(
    queries: Sequence[tuple[str, ...]], choices: Sequence[tuple[str, ...]]
) -> np.ndarray:
    """Return zero where a query equals a choice and one elsewhere.

    Args:
        queries: `[Q]` activity name sequences.
        choices: `[C]` activity name sequences.

    Returns:
        `[Q, C]` float64 distances.
    """
    distinct: dict[tuple[str, ...], int] = {}
    left = np.fromiter((distinct.setdefault(s, len(distinct)) for s in queries), np.int64)
    right = np.fromiter((distinct.setdefault(s, len(distinct)) for s in choices), np.int64)
    return np.not_equal.outer(left, right).astype(np.float64)


def bigram_distances(
    queries: Sequence[tuple[str, ...]], choices: Sequence[tuple[str, ...]]
) -> np.ndarray:
    """Return the multiset Jaccard distance between the bigrams of each query and each choice.

    Each sequence is padded with a start and an end token, so its first and last activities form
    bigrams too and every sequence has at least one.

    Args:
        queries: `[Q]` activity name sequences.
        choices: `[C]` activity name sequences.

    Returns:
        `[Q, C]` float64 distances in [0, 1].
    """

    def bigrams(sequence: tuple[str, ...]) -> Counter[tuple[Hashable, Hashable]]:
        padded = (_START, *sequence, _END)
        return Counter(pairwise(padded))

    counted = [bigrams(sequence) for sequence in (*queries, *choices)]
    vocabulary = {
        pair: column for column, pair in enumerate({pair for counts in counted for pair in counts})
    }
    matrix = np.zeros((len(counted), len(vocabulary)), dtype=np.float64)
    for row, counts in enumerate(counted):
        for pair, count in counts.items():
            matrix[row, vocabulary[pair]] = count

    left, right = matrix[: len(queries)], matrix[len(queries) :]
    absolute_differences = cdist(left, right, metric='cityblock')
    sizes = left.sum(axis=1)[:, None] + right.sum(axis=1)[None, :]
    # With L1 the summed count differences, the union holds (sizes + L1) / 2 bigrams and the
    # intersection (sizes - L1) / 2, so one minus their ratio is 2 L1 / (sizes + L1).
    return 2.0 * absolute_differences / (sizes + absolute_differences)
