from collections.abc import Collection
from pathlib import Path

import numpy as np
import pandas as pd

from pseval.data import ACTIVITY_KEY, CASE_KEY, RESOURCE_KEY, TIMESTAMP_KEY, Log

# Separator of every split file.
CSV_SEPARATOR = ';'


# The first cut of each case, a column of the test split file only. It is 1 except on the test
# cases running across the split separation, whose earlier prefixes a training run could have
# seen.
PREFIX_START_KEY = '_prefix_start'


def write_split(events: pd.DataFrame, path: Path, *, first_cuts: pd.Series | None = None) -> None:
    """Write one split file of a bundle.

    Args:
        events: The split's events, each case's rows consecutive and in time order.
        path: Destination file.
        first_cuts: The first cut of every case, indexed by case, stored for the test split.
    """
    if first_cuts is not None:
        events = events.assign(**{PREFIX_START_KEY: events[CASE_KEY].map(first_cuts)})
    events.to_csv(path, sep=CSV_SEPARATOR, index=False)


def read_split(path: Path, *, text_columns: Collection[str]) -> Log:
    """Read one split file written by `write_split`.

    Args:
        path: The split, as `write_split` writes it.
        text_columns: The columns held as text beside case, activity, and resource.

    Returns:
        The split, with the first cuts its `PREFIX_START_KEY` column states when it has one.
    """
    columns = {CASE_KEY, ACTIVITY_KEY, RESOURCE_KEY, *text_columns}
    events = pd.read_csv(path, sep=CSV_SEPARATOR, dtype=dict.fromkeys(columns, str))
    events[TIMESTAMP_KEY] = pd.to_datetime(events[TIMESTAMP_KEY], format='mixed')
    if PREFIX_START_KEY not in events.columns:
        return Log(events)
    first_rows = events[CASE_KEY].ne(events[CASE_KEY].shift()).to_numpy()
    prefix_starts = events.loc[first_rows, PREFIX_START_KEY].to_numpy(np.int64)
    return Log(events.drop(columns=PREFIX_START_KEY), prefix_starts=prefix_starts)
