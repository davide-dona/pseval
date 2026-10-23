import numpy as np
import pandas as pd

from pseval.data import CASE_KEY, TIMESTAMP_KEY


def sort_log(log: pd.DataFrame) -> pd.DataFrame:
    """Order a log by case start time, and the events of a case by their own timestamp.

    Returns:
        A reindexed copy. The sort is stable, so events sharing a timestamp keep the order the
        log gave them, which is the order cuts count events in.
    """
    case_start = log.groupby(CASE_KEY)[TIMESTAMP_KEY].min()
    rank = pd.Series(
        data=range(len(case_start)), index=case_start.sort_values(kind='mergesort').index
    )
    log = log.assign(_case_rank=log[CASE_KEY].map(rank))
    log = log.sort_values(by=['_case_rank', TIMESTAMP_KEY], kind='mergesort')
    return log.drop(columns='_case_rank').reset_index(drop=True)


def out_of_time_split(
    log: pd.DataFrame, *, validation_fraction: float, test_fraction: float, seed: int
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series]:
    """Split a log by time into train and test, and draw validation at random from train.

    The separation time is the start of the case that begins once `1 - test_fraction` of them
    already have. Train keeps every case finished before it. Test keeps every case still running
    at it, so a case running across it is a test case whose earlier prefixes a training run could
    already have seen; those prefixes are withheld from scoring through its first cut rather than
    the case being dropped.

    Validation is a random sample of whole training cases. A time slice ending at the separation
    would lose every long case starting inside it to test, so its prefixes would face far shorter
    remaining times than test's.

    Args:
        log: The filtered log, sorted by case start and then by timestamp.
        validation_fraction: Fraction of all cases drawn from the train cases for validation.
        test_fraction: Fraction of all cases the separation time is placed at.
        seed: Seed of the validation draw.

    Returns:
        `(train, validation, test, first_cuts)`: row subsets of `log`, and the first cut of every
        test case, indexed by case.
    """
    edges = log.groupby(CASE_KEY)[TIMESTAMP_KEY].agg(['min', 'max'])
    starts_ts, stops_ts = edges['min'], edges['max']

    first_test_case = int(len(edges) * (1 - test_fraction))
    separation_ts = np.sort(starts_ts.values)[first_test_case]

    train_cases = stops_ts.index[stops_ts.values < separation_ts]
    train = log[log[CASE_KEY].isin(train_cases)]

    test_cases = stops_ts.index[stops_ts.values >= separation_ts]
    crossing = test_cases.difference(starts_ts.sort_values().index[first_test_case:])
    test = log[log[CASE_KEY].isin(test_cases)]
    first_cuts = _first_cuts(test, separation_ts=separation_ts, crossing=crossing)

    # The candidates are taken in log order, so the draw depends on the seed alone.
    validation_share = validation_fraction / (1 - test_fraction)
    train_cases = train[CASE_KEY].unique()
    rng = np.random.default_rng(seed)
    validation_cases = rng.choice(
        train_cases, size=int(len(train_cases) * validation_share), replace=False
    )

    validation = train[train[CASE_KEY].isin(validation_cases)]
    train = train[~train[CASE_KEY].isin(validation_cases)]
    return train, validation, test, first_cuts


def _first_cuts(
    split: pd.DataFrame, *, separation_ts: np.datetime64, crossing: pd.Index
) -> pd.Series:
    """Each case's first cut, indexed by case: on the crossing cases the first one whose prefix
    reaches past the separation, and 1 on every other case."""
    rows = split[split[CASE_KEY].isin(crossing)]
    # A cut of k takes the first k events, so a bound is the 1-based position of the event the
    # prefix has to reach.
    positions = rows.groupby(CASE_KEY).cumcount() + 1

    # Strictly after the separation: an event landing exactly on it is one a training case could
    # still have been running through. Compared as numpy values, since a tz-aware column and a
    # numpy timestamp are not comparable as pandas objects.
    selected = rows[TIMESTAMP_KEY].values > separation_ts
    bounds = positions[selected].groupby(rows[CASE_KEY][selected]).min()
    cases = pd.Index(split[CASE_KEY].unique())
    return pd.Series(cases.map(bounds), index=cases).fillna(1).astype(int)
