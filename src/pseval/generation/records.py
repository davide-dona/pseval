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

    def __post_init__(self) -> None:
        if self.inter_event_time_minutes.shape != (len(self.activities),):
            raise ValueError('Inter-event times do not match their activities')

    def __len__(self) -> int:
        return len(self.activities)

    @property
    def remaining_time_minutes(self) -> float:
        """Minutes from the last prefix event to the end of the case, the times summed."""
        return math.fsum(self.inter_event_time_minutes)


@dataclass(frozen=True, eq=False)
class Draws:
    """One prefix's drawn suffixes, held as the distinct ones and which draw took each.

    Draw `i` has `suffixes[taken[i]]` as its activities and
    `inter_event_time_minutes[offsets[i]:offsets[i + 1]]` as their times. Folding the activity
    suffixes lets conformance and the sequence distances be solved over distinct suffixes only.
    """

    # The distinct suffixes, in the order they were first drawn.
    suffixes: tuple[str, ...]
    # [S] Which of them each draw took, in draw order.
    taken: np.ndarray
    # [E] float64 inter-event minutes of every draw, draw after draw.
    inter_event_time_minutes: np.ndarray
    # [S] Whether each draw was closed at the length cap without ending.
    truncated: np.ndarray

    def __post_init__(self) -> None:
        draws = len(self.taken)
        if self.taken.ndim != 1 or self.truncated.shape != (draws,):
            raise ValueError('Generation draws disagree on how many draws were taken')
        if len(set(self.suffixes)) != len(self.suffixes):
            raise ValueError('Generation suffixes must be distinct')
        if draws and (self.taken.min() < 0 or self.taken.max() >= len(self.suffixes)):
            raise ValueError('Generation draw references an unknown suffix')
        if len(np.unique(self.taken)) != len(self.suffixes):
            raise ValueError('Every generation suffix must be taken by a draw')
        if self.inter_event_time_minutes.shape != (int(self.offsets[-1]),):
            raise ValueError('Generation draw times do not match their suffix lengths')

    @classmethod
    def of(cls, drawn: Sequence[DecodedEvents], *, truncated: Sequence[bool]) -> 'Draws':
        """Fold one prefix's draws, in the order they were drawn."""
        rows: dict[str, int] = {}
        taken = np.fromiter(
            (rows.setdefault(events.activities, len(rows)) for events in drawn),
            dtype=np.int64,
            count=len(drawn),
        )
        return cls(
            suffixes=tuple(rows),
            taken=taken,
            inter_event_time_minutes=np.concatenate(
                [np.zeros(0), *(events.inter_event_time_minutes for events in drawn)]
            ),
            truncated=np.asarray(truncated, dtype=bool),
        )

    def __len__(self) -> int:
        """How many draws were taken, which every mean over them divides by."""
        return len(self.taken)

    @cached_property
    def lengths(self) -> np.ndarray:
        """[S] How many events each draw holds."""
        suffix_lengths = np.fromiter(map(len, self.suffixes), np.int64, len(self.suffixes))
        return suffix_lengths[self.taken]

    @cached_property
    def offsets(self) -> np.ndarray:
        """[S + 1] Where each draw's times start in `inter_event_time_minutes`, then their end."""
        return np.concatenate([np.zeros(1, np.int64), np.cumsum(self.lengths)])

    @property
    def remaining_time_minutes(self) -> np.ndarray:
        """[S] Each draw's inter-event minutes summed in event order."""
        draw_of_event = np.repeat(np.arange(len(self)), self.lengths)  # [E]
        return np.bincount(
            draw_of_event, weights=self.inter_event_time_minutes, minlength=len(self)
        )

    @cached_property
    def counts(self) -> np.ndarray:
        """[U] How many draws took each of `suffixes`, summing to `len(self)`."""
        return np.bincount(self.taken, minlength=len(self.suffixes)).astype(np.float64)

    def mean(self, values: Sequence[float] | np.ndarray) -> float:
        """Return the draw-weighted mean of one value per distinct suffix, zero without draws."""
        draw_count = len(self)
        return float(self.counts @ values) / draw_count if len(values) and draw_count else 0.0


@dataclass(frozen=True, eq=False)
class Generation:
    """One prefix's drawn suffixes and the observed suffix they are scored against."""

    case_id: str
    prefix_activities: str
    # [P] float64 minutes before each prefix event, zero before the case's first event.
    prefix_inter_event_time_minutes: np.ndarray
    samples: Draws
    truth: DecodedEvents

    def __post_init__(self) -> None:
        if self.prefix_inter_event_time_minutes.shape != (len(self.prefix_activities),):
            raise ValueError('Generation prefix times do not match its activities')

    @property
    def prefix_len(self) -> int:
        """How many events the prefix holds."""
        return len(self.prefix_activities)
