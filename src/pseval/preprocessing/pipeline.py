from pathlib import Path

import numpy as np
import pandas as pd

from pseval.data import ACTIVITY_KEY, CASE_KEY, RESOURCE_KEY, TIMESTAMP_KEY
from pseval.preprocessing.filters import (
    case_durations,
    case_lengths,
    drop_cases_starting_late,
    drop_outlier_cases,
)
from pseval.preprocessing.split import add_inter_event_minutes, out_of_time_split, sort_log
from pseval.specs import DatasetSpec


def read_original(spec: DatasetSpec, path: Path) -> pd.DataFrame:
    """Read an original log, renaming its structural columns to the canonical ones.

    Returns:
        The log as read, one row per event. The case and text columns are text, the other
        columns as pandas infers them, and the timestamp parsed.
    """
    log = pd.read_csv(
        path,
        sep=spec.separator,
        dtype=dict.fromkeys([spec.case_column, *spec.text_columns], str),
    )
    log = log.rename(
        columns={
            spec.case_column: CASE_KEY,
            spec.activity_column: ACTIVITY_KEY,
            spec.resource_column: RESOURCE_KEY,
            spec.timestamp_column: TIMESTAMP_KEY,
        }
    )
    log[TIMESTAMP_KEY] = pd.to_datetime(log[TIMESTAMP_KEY], format='mixed')
    return log


def split_log(
    spec: DatasetSpec, original: Path
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, int]:
    """Filter, sort, and split an original log.

    Drops the cases running longer than the duration percentile or longer than the length
    percentile, both read off the log as read, then the cases starting too late to be seen
    finishing, read off what the outlier filter left. Sorts the rest, adds each event's
    inter-event minutes, and splits it out of time.

    Returns:
        `(train, validation, test, first_cuts, max_length)`: the splits and first cuts as
        `out_of_time_split` returns them, and the length cutoff, the length cap of every draw.
    """
    log = read_original(spec, original)
    max_duration_days = float(np.percentile(case_durations(log), spec.max_duration_percentile))
    max_length = int(np.ceil(np.percentile(case_lengths(log), spec.max_length_percentile)))
    log = drop_outlier_cases(log, max_duration_days=max_duration_days, max_length=max_length)
    min_observed_days = float(
        np.percentile(case_durations(log), spec.min_observed_duration_percentile)
    )
    log = drop_cases_starting_late(log, min_observed_days=min_observed_days)
    train, validation, test, first_cuts = out_of_time_split(
        add_inter_event_minutes(sort_log(log)),
        validation_fraction=spec.validation_fraction,
        test_fraction=spec.test_fraction,
        seed=spec.split_seed,
    )
    return train, validation, test, first_cuts, max_length
