import numpy as np
import pandas as pd

from pseval.data import CASE_KEY, TIMESTAMP_KEY


def case_durations(log: pd.DataFrame) -> pd.Series:
    """The duration of every case in days, first event to last, indexed by case."""
    edges = log.groupby(CASE_KEY)[TIMESTAMP_KEY].agg(['min', 'max'])
    return (edges['max'] - edges['min']).dt.total_seconds() / 86_400.0


def case_length_cutoff(log: pd.DataFrame, *, percentile: float) -> int:
    """The case length in events at a percentile, rounded up."""
    lengths = log.groupby(CASE_KEY)[CASE_KEY].size()
    return int(np.ceil(np.percentile(lengths, percentile)))


def drop_cases_by_length(log: pd.DataFrame, *, max_length: int) -> pd.DataFrame:
    """Keep the cases of at most `max_length` events."""
    lengths = log.groupby(CASE_KEY)[CASE_KEY].transform('size')
    return log[lengths <= max_length]


def drop_cases_by_duration(log: pd.DataFrame, *, max_duration_days: float) -> pd.DataFrame:
    """Keep the cases running at most `max_duration_days`, first event to last.

    A case spanning decades is a broken timestamp rather than a slow process.
    """
    return log[log[CASE_KEY].map(case_durations(log)) <= max_duration_days]


def drop_cases_starting_late(log: pd.DataFrame, *, min_observed_days: float) -> pd.DataFrame:
    """Keep the cases starting at least `min_observed_days` before the last event of the log.

    A log is a snapshot: a case still running when it was extracted ends at its last recorded
    event, which would then be scored as the true end of the trace. Selecting on start time
    rather than on how a case ends keeps the selection independent of the scored suffixes.
    """
    cutoff = log[TIMESTAMP_KEY].max() - pd.Timedelta(days=min_observed_days)
    starts = log.groupby(CASE_KEY)[TIMESTAMP_KEY].min()
    return log[log[CASE_KEY].map(starts) <= cutoff]
