from collections import Counter
from collections.abc import Sequence

import numpy as np
from rapidfuzz import process
from rapidfuzz.distance import DamerauLevenshtein
from scipy.spatial.distance import cdist

# Boundaries sequences are padded with, distinct and outside the private use area activity codes
# are drawn from.
START_CODE = '\x02'
END_CODE = '\x03'


def damerau_levenshtein_distances(queries: Sequence[str], choices: Sequence[str]) -> np.ndarray:
    """Return the normalized Damerau-Levenshtein distance between each query and each choice.

    Both sequences end with a shared terminal token, so the end of the case counts as one more
    activity.

    Args:
        queries: `[Q]` sequences spelled one activity per character.
        choices: `[C]` sequences spelled one activity per character.

    Returns:
        `[Q, C]` float64 distances in [0, 1].
    """
    return process.cdist(
        queries=[sequence + END_CODE for sequence in queries],
        choices=[sequence + END_CODE for sequence in choices],
        scorer=DamerauLevenshtein.normalized_distance,
        dtype=np.float64,
    )


def exact_distances(queries: Sequence[str], choices: Sequence[str]) -> np.ndarray:
    """Return zero where a query equals a choice and one elsewhere.

    Args:
        queries: `[Q]` sequences spelled one activity per character.
        choices: `[C]` sequences spelled one activity per character.

    Returns:
        `[Q, C]` float64 distances.
    """
    left = np.array(queries, dtype=object)
    right = np.array(choices, dtype=object)
    return np.not_equal.outer(left, right).astype(np.float64)


def bigram_distances(queries: Sequence[str], choices: Sequence[str]) -> np.ndarray:
    """Return the multiset Jaccard distance between the bigrams of each query and each choice.

    Each sequence is padded with a start and an end token, so its first and last activities form
    bigrams too and every sequence has at least one.

    Args:
        queries: `[Q]` sequences spelled one activity per character.
        choices: `[C]` sequences spelled one activity per character.

    Returns:
        `[Q, C]` float64 distances in [0, 1].
    """

    def bigrams(sequence: str) -> Counter[str]:
        padded = START_CODE + sequence + END_CODE
        return Counter(padded[i : i + 2] for i in range(len(padded) - 1))

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
