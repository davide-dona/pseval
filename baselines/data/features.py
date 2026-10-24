import numpy as np
import pandas as pd

from pseval.data import TIMESTAMP_KEY

# Minutes since the first event of the case, offered to the encoders as an event feature.
CASE_ELAPSED_KEY = 'ts_start'
# The calendar position of the event, offered the same way. Cyclical: sin/cos of the day of the
# week and of the second of the day, so the encoders read the wrap-around (Sunday to Monday,
# midnight to midnight) rather than a raw count that treats it as a jump.
DAY_SIN_KEY = 'day_sin'
DAY_COS_KEY = 'day_cos'
SECONDS_SIN_KEY = 'seconds_sin'
SECONDS_COS_KEY = 'seconds_cos'


def derive_features(events: pd.DataFrame, offsets: np.ndarray) -> pd.DataFrame:
    """Add each event's case elapsed minutes and calendar position.

    Both are read off the event's own timestamp, so a prefix derives them exactly as its whole case
    does: the calendar from the event alone, and the elapsed minutes from the first event of the
    run, which is the first event of its case.

    Args:
        events: One row per event, each run's rows consecutive and in time order.
        offsets: `[runs + 1]` row where each run starts, then the past-the-end row. A run is a
            case, or a prefix starting at its case's first event.

    Returns:
        A copy of `events` with `CASE_ELAPSED_KEY` and the four calendar columns added.
    """
    timestamps = events[TIMESTAMP_KEY].reset_index(drop=True)
    run_starts = timestamps.iloc[np.repeat(offsets[:-1], np.diff(offsets))].reset_index(drop=True)
    calendar = timestamps.dt
    day = calendar.weekday  # 0 (Monday) .. 6 (Sunday)
    seconds = calendar.hour * 3600 + calendar.minute * 60 + calendar.second  # 0 .. 86399
    day_angle = (2 * np.pi * day / 7).to_numpy()
    seconds_angle = (2 * np.pi * seconds / 86400).to_numpy()
    return events.assign(
        **{
            CASE_ELAPSED_KEY: ((timestamps - run_starts).dt.total_seconds() / 60.0).to_numpy(),
            DAY_SIN_KEY: np.sin(day_angle),
            DAY_COS_KEY: np.cos(day_angle),
            SECONDS_SIN_KEY: np.sin(seconds_angle),
            SECONDS_COS_KEY: np.cos(seconds_angle),
        }
    )
