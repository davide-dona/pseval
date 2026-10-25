import numpy as np
import pandas as pd

from pseval.data import INTER_EVENT_MINUTES_KEY, TIMESTAMP_KEY

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

    Args:
        events: One row per event, each run's rows consecutive and in time order.
        offsets: `[runs + 1]` row where each run starts, then the past-the-end row. A run is a
            case, or a prefix starting at its case's first event.

    Returns:
        A copy of `events` with `CASE_ELAPSED_KEY` and the four calendar columns added.
    """
    total = np.cumsum(events[INTER_EVENT_MINUTES_KEY].to_numpy(np.float64))
    calendar = events[TIMESTAMP_KEY].dt
    day = calendar.weekday  # 0 (Monday) .. 6 (Sunday)
    seconds = calendar.hour * 3600 + calendar.minute * 60 + calendar.second  # 0 .. 86399
    day_angle = (2 * np.pi * day / 7).to_numpy()
    seconds_angle = (2 * np.pi * seconds / 86400).to_numpy()
    return events.assign(
        **{
            CASE_ELAPSED_KEY: total - np.repeat(total[offsets[:-1]], np.diff(offsets)),
            DAY_SIN_KEY: np.sin(day_angle),
            DAY_COS_KEY: np.cos(day_angle),
            SECONDS_SIN_KEY: np.sin(seconds_angle),
            SECONDS_COS_KEY: np.cos(seconds_angle),
        }
    )
