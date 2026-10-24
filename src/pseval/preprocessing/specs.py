import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Self


@dataclass(frozen=True)
class DatasetSpec:
    """How one benchmark is built from its original log."""

    name: str
    case_column: str
    activity_column: str
    resource_column: str
    timestamp_column: str
    separator: str
    original_sha256: str
    text_columns: tuple[str, ...] = ()
    validation_fraction: float = 0.15
    test_fraction: float = 0.2
    split_seed: int = 42
    # Percentile of case length, in events, above which a case is dropped. Measured on the whole
    # log as read, and the length cap of every draw.
    max_length_percentile: float = 98.5
    # Percentile of case duration above which a case is dropped, on the whole log as read.
    max_duration_percentile: float = 99.9
    # Percentile of case duration, after the outlier filters, that a kept case must start at
    # least that long before the end of the log.
    min_observed_duration_percentile: float = 95.0
    declare_hard_min_support: float = 0.05
    declare_hard_min_confidence: float = 1.0
    declare_max_cardinality: int = 2

    def save(self, path: Path) -> None:
        """Write this spec as JSON, every field included."""
        path.write_text(json.dumps(asdict(self), indent=2, sort_keys=True) + '\n')

    @classmethod
    def load(cls, path: Path) -> Self:
        """Read a spec written as JSON, its omitted fields taking their defaults.

        Raises:
            TypeError: If the JSON misses a required field or names an unknown one.
        """
        fields = json.loads(path.read_text())
        return cls(**fields | {'text_columns': tuple(fields.get('text_columns', ()))})
