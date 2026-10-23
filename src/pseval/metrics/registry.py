from collections.abc import Callable, Mapping
from dataclasses import dataclass
from enum import StrEnum
from types import MappingProxyType

from pseval.metrics.prepared import PreparedPrefix

type MetricCompute = Callable[[PreparedPrefix], float]


class Direction(StrEnum):
    """Which values are preferred when comparing models."""

    HIGHER = 'higher'
    LOWER = 'lower'
    ZERO = 'zero'
    NONE = 'none'


class Owner(StrEnum):
    """Whether a value belongs to a model or the observed log."""

    MODEL = 'model'
    LOG = 'log'


class MetricGroup(StrEnum):
    """The evaluation question a metric answers."""

    JOINT = 'joint'
    ACTIVITY = 'activity'
    SUFFIX_LENGTH = 'suffix_length'
    REMAINING_TIME = 'remaining_time'
    CONFORMANCE = 'conformance'


@dataclass(frozen=True, slots=True)
class Metric:
    """One per-prefix value and the function computing it.

    The key names the scores column. A ranked metric compares models: it is not a diagnostic and
    has a higher or lower preferred value. A diagnostic describes how a model forecasts and never
    ranks.
    """

    key: str
    label: str
    group: MetricGroup
    direction: Direction
    compute: MetricCompute
    unit: str | None = None
    bounds: tuple[float | None, float | None] = (None, None)
    owner: Owner = Owner.MODEL
    diagnostic: bool = False

    @property
    def ranked(self) -> bool:
        return not self.diagnostic and self.direction in (Direction.HIGHER, Direction.LOWER)

    def describe(self) -> dict[str, object]:
        """The metric's record in a scores file's metric table."""
        return {
            'key': self.key,
            'label': self.label,
            'group': str(self.group),
            'ranked': self.ranked,
            'diagnostic': self.diagnostic,
            'direction': str(self.direction),
            'owner': str(self.owner),
            'unit': self.unit,
            'bounds': list(self.bounds),
        }


class MetricRegistry:
    """Ordered declarations of every metric. Declaration order is the scores column order."""

    def __init__(self) -> None:
        self._entries: dict[str, Metric] = {}

    def register(
        self,
        key: str,
        *,
        label: str,
        group: MetricGroup,
        unit: str | None = None,
        bounds: tuple[float | None, float | None] = (None, None),
        direction: Direction = Direction.NONE,
        owner: Owner = Owner.MODEL,
        diagnostic: bool = False,
    ) -> Callable[[MetricCompute], MetricCompute]:
        """Register a prefix-scoring function under a stable metric key.

        Returns:
            Decorator that records the metric in declaration order and returns the function.

        Raises:
            ValueError: When the decorator encounters an already registered key.
        """

        def decorator(compute: MetricCompute) -> MetricCompute:
            if key in self._entries:
                raise ValueError(f'metric {key!r} is registered more than once.')
            self._entries[key] = Metric(
                key=key,
                label=label,
                group=group,
                unit=unit,
                bounds=bounds,
                direction=direction,
                owner=owner,
                diagnostic=diagnostic,
                compute=compute,
            )
            return compute

        return decorator

    @property
    def entries(self) -> Mapping[str, Metric]:
        """Every metric, in declaration order."""
        return MappingProxyType(self._entries)

    @property
    def ranked(self) -> Mapping[str, Metric]:
        """The metrics that compare models, in declaration order."""
        return MappingProxyType({key: m for key, m in self._entries.items() if m.ranked})

    def table(self) -> list[dict[str, object]]:
        """Every metric's record, in declaration order."""
        return [metric.describe() for metric in self._entries.values()]

    def __getitem__(self, key: str) -> Metric:
        """Look up a registered metric.

        Raises:
            ValueError: If no metric has this key.
        """
        if key not in self._entries:
            raise ValueError(f'no metric is registered as {key!r}.')
        return self._entries[key]


METRICS = MetricRegistry()
