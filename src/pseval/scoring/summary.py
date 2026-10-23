import math
from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True, eq=False)
class Summary:
    """Equal-weight means of every metric, overall and by length.

    `overall` holds one mean per metric. `by_prefix_length` and `by_suffix_length` hold one row
    per length present, ascending, with a `prefixes` column counting its prefixes.
    """

    prefixes: int
    overall: pd.Series
    by_prefix_length: pd.DataFrame
    by_suffix_length: pd.DataFrame


def means(values: pd.DataFrame) -> pd.Series:
    """Column means, each sum exactly rounded, so the row order does not change them."""
    return pd.Series({column: math.fsum(values[column]) / len(values) for column in values})


def _bucket_means(frame: pd.DataFrame, column: str, keys: list[str]) -> pd.DataFrame:
    rows = {
        length: pd.concat([pd.Series({'prefixes': len(group)}), means(group[keys])])
        for length, group in frame.groupby(column, sort=True)
    }
    buckets = pd.DataFrame.from_dict(rows, orient='index')
    buckets.index.name = column
    return buckets.astype({'prefixes': np.int64})


def summarize(frame: pd.DataFrame, keys: list[str]) -> Summary:
    """Equal-weight means of the metric columns `keys`, overall, by prefix length, and by true
    suffix length.

    Args:
        frame: One row per prefix.
        keys: The metric columns to average.
    """
    return Summary(
        prefixes=len(frame),
        overall=means(frame[keys]),
        by_prefix_length=_bucket_means(frame, 'prefix_len', keys),
        by_suffix_length=_bucket_means(frame, 'suffix_len', keys),
    )
