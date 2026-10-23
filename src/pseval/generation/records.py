import math
from collections.abc import Sequence
from dataclasses import dataclass
from functools import cached_property

import numpy as np


@dataclass(frozen=True, eq=False)
class DecodedEvents:
    """A run of events in log units, its activities spelled by `ActivityCodes`."""

    activities: str
    # [L] float64 minutes before each activity, from the previous event.
    inter_event_time_minutes: np.ndarray

    def __len__(self) -> int:
        return len(self.activities)

    @property
    def remaining_time_minutes(self) -> float:
        """Minutes from the last prefix event to the end of the case, the times summed."""
        return math.fsum(self.inter_event_time_minutes)


@dataclass(frozen=True, eq=False)
class Draws:
    """One prefix's drawn suffixes.

    Draw `i` has `suffixes[i]` as its activities and
    `inter_event_time_minutes[offsets[i]:offsets[i + 1]]` as their times.
    """

    # [K] The activities of each draw, in draw order.
    suffixes: tuple[str, ...]
    # [E] float64 inter-event minutes of every draw, draw after draw.
    inter_event_time_minutes: np.ndarray
    # [K] Whether each draw was closed at the length cap without ending.
    truncated: np.ndarray

    @classmethod
    def of(cls, drawn: Sequence[DecodedEvents], *, truncated: Sequence[bool]) -> 'Draws':
        """Gather one prefix's draws, in the order they were drawn."""
        return cls(
            suffixes=tuple(events.activities for events in drawn),
            inter_event_time_minutes=np.concatenate(
                [np.zeros(0), *(events.inter_event_time_minutes for events in drawn)]
            ),
            truncated=np.asarray(truncated, dtype=bool),
        )

    def __len__(self) -> int:
        """How many draws were taken."""
        return len(self.suffixes)

    @cached_property
    def lengths(self) -> np.ndarray:
        """[K] How many events each draw holds."""
        return np.fromiter(map(len, self.suffixes), np.int64, len(self))

    @cached_property
    def offsets(self) -> np.ndarray:
        """[K + 1] Where each draw's times start in `inter_event_time_minutes`, then their end."""
        return np.concatenate([np.zeros(1, np.int64), np.cumsum(self.lengths)])

    @property
    def remaining_time_minutes(self) -> np.ndarray:
        """[K] Each draw's inter-event minutes summed in event order."""
        draw_of_event = np.repeat(np.arange(len(self)), self.lengths)  # [E]
        return np.bincount(
            draw_of_event, weights=self.inter_event_time_minutes, minlength=len(self)
        )


@dataclass(frozen=True, eq=False)
class Generation:
    """One prefix's drawn suffixes and the observed suffix they are scored against."""

    case_id: str
    prefix_activities: str
    samples: Draws
    truth: DecodedEvents

    @property
    def prefix_len(self) -> int:
        """How many events the prefix holds."""
        return len(self.prefix_activities)
