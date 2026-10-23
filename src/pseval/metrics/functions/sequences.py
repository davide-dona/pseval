from collections import Counter
from collections.abc import Hashable, Sequence

import numpy as np
from rapidfuzz import process
from rapidfuzz.distance import DamerauLevenshtein
from scipy.spatial.distance import cdist

from pseval.metrics.vocabulary import END_CODE, START_CODE


def damerau_levenshtein_distances(queries: Sequence[str], choices: Sequence[str]) -> np.ndarray:
    """Return pairwise normalized Damerau-Levenshtein distances with a shared terminal token.

    Sequences are strings spelling one activity per character.
    """
    return process.cdist(
        queries=[sequence + END_CODE for sequence in queries],
        choices=[sequence + END_CODE for sequence in choices],
        scorer=DamerauLevenshtein.normalized_distance,
        dtype=np.float64,
    )


def exact_distances(
    queries: Sequence[Sequence[Hashable]], choices: Sequence[Sequence[Hashable]]
) -> np.ndarray:
    """Return pairwise sequence inequality as zero or one."""
    # Tuples make strings and lists comparable as sequences and avoid NumPy array walks.
    left = np.empty(len(queries), dtype=object)
    left[:] = [tuple(sequence) for sequence in queries]
    right = np.empty(len(choices), dtype=object)
    right[:] = [tuple(sequence) for sequence in choices]
    return np.not_equal.outer(left, right).astype(np.float64)


def bigram_distances(
    queries: Sequence[Sequence[Hashable]], choices: Sequence[Sequence[Hashable]]
) -> np.ndarray:
    """Return pairwise multiset Jaccard distances over bigrams padded with start and end."""

    def bigrams(sequence: Sequence[Hashable]) -> Counter[tuple[Hashable, Hashable]]:
        padded = (START_CODE, *sequence, END_CODE)
        return Counter(zip(padded[:-1], padded[1:], strict=True))

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
    # Padding gives every sequence at least one pair, so the union is never empty.
    return 2.0 * absolute_differences / (sizes + absolute_differences)
