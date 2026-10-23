import pandas as pd

from pseval.data import CASE_KEY, TIMESTAMP_KEY


def case_durations(log: pd.DataFrame) -> pd.Series:
    """The duration of every case in days, first event to last, indexed by case."""
    edges = log.groupby(CASE_KEY)[TIMESTAMP_KEY].agg(['min', 'max'])
    return (edges['max'] - edges['min']).dt.total_seconds() / 86_400.0


def case_lengths(log: pd.DataFrame) -> pd.Series:
    """The length of every case in events, indexed by case."""
    return log.groupby(CASE_KEY).size()


def drop_outlier_cases(
    log: pd.DataFrame, *, max_duration_days: float, max_length: int
) -> pd.DataFrame:
    """Keep the cases running at most `max_duration_days`, first event to last, and holding at
    most `max_length` events.

    A case spanning decades is a broken timestamp rather than a slow process.
    """
    kept = (case_durations(log) <= max_duration_days) & (case_lengths(log) <= max_length)
    return log[log[CASE_KEY].isin(kept.index[kept])]


def drop_cases_starting_late(log: pd.DataFrame, *, min_observed_days: float) -> pd.DataFrame:
    """Keep the cases starting at least `min_observed_days` before the last event of the log.

    A log is a snapshot: a case still running when it was extracted ends at its last recorded
    event, which would then be scored as the true end of the trace. Selecting on start time
    rather than on how a case ends keeps the selection independent of the scored suffixes.
    """
    cutoff = log[TIMESTAMP_KEY].max() - pd.Timedelta(days=min_observed_days)
    starts = log.groupby(CASE_KEY)[TIMESTAMP_KEY].min()
    return log[log[CASE_KEY].map(starts) <= cutoff]
