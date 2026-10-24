import math
from collections.abc import Iterator
from dataclasses import dataclass
from functools import cached_property

import numpy as np

from pseval.data import SuffixDraws


@dataclass(frozen=True, eq=False)
class DecodedEvents:
    """A run of events in log units."""

    # [L] The activity name of each event.
    activities: tuple[str, ...]
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

    Draw `i` has `activities[i]` as its activities and
    `inter_event_time_minutes[offsets[i]:offsets[i + 1]]` as their times.
    """

    # [K] The activity names of each draw.
    activities: tuple[tuple[str, ...], ...]
    # [E] float64 inter-event minutes of every draw, draw after draw.
    inter_event_time_minutes: np.ndarray
    # [K] Whether each draw was closed at the length cap without ending.
    truncated: np.ndarray

    @classmethod
    def of(cls, draws: SuffixDraws) -> 'Draws':
        """Gather one prefix's draws from a model's."""
        return cls(
            activities=tuple(tuple(names) for names in draws.activities),
            inter_event_time_minutes=np.concatenate(
                [np.zeros(0), *(np.asarray(minutes, dtype=np.float64) for minutes in draws.minutes)]
            ),
            truncated=np.asarray(draws.truncated, dtype=bool),
        )

    def __len__(self) -> int:
        """How many draws were taken."""
        return len(self.activities)

    def __iter__(self) -> Iterator[DecodedEvents]:
        """Each draw's activities and times."""
        times, offsets = self.inter_event_time_minutes, self.offsets
        for draw, activities in enumerate(self.activities):
            yield DecodedEvents(activities, times[offsets[draw] : offsets[draw + 1]])

    @cached_property
    def lengths(self) -> np.ndarray:
        """[K] How many events each draw holds."""
        return np.fromiter(map(len, self.activities), np.int64, len(self))

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
