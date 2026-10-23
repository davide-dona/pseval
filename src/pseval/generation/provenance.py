import json
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from typing import Self

import pyarrow as pa
import pyarrow.parquet as pq

from pseval.metrics import METRIC_KEYS

_PARQUET_KEY = b'pseval'


@dataclass(frozen=True)
class Selection:
    """How the generating model was selected, stored with its generations and scores.

    `metric` is the validation metric the selection minimized or maximized, a scores column,
    or None for a model without a selection step. `procedure` says in words how the
    selection ran, for instance which validation prefixes, draws, and checkpoints it compared.
    """

    metric: str | None
    procedure: str

    def __post_init__(self) -> None:
        if self.metric is not None and self.metric not in METRIC_KEYS:
            raise ValueError(f'Selection metric {self.metric!r} is not a scores column')

    @classmethod
    def none(cls) -> Self:
        """The selection of a model estimated once, with nothing compared on validation."""
        return cls(metric=None, procedure='none')


@dataclass(frozen=True)
class Provenance:
    """What a generations or scores file was produced from.

    `dataset` and `fingerprint` identify the benchmark. `num_samples` is the draw count of
    every prefix and `seed` the one the batch seeds are derived from. `label` names the model and
    `metadata` holds whatever the model author records with it, such as a checkpoint hash or a
    configuration, as a JSON object. `generations_sha256` is set on a scores file only, naming the
    generations file it scored.
    """

    dataset: str
    fingerprint: str
    label: str
    selection: Selection
    num_samples: int
    seed: int
    metadata: Mapping[str, object] = field(default_factory=dict)
    generations_sha256: str | None = None

    def attach_to_schema(self, schema: pa.Schema) -> pa.Schema:
        """Attach this record to a Parquet schema."""
        return schema.with_metadata(
            (schema.metadata or {}) | {_PARQUET_KEY: json.dumps(asdict(self)).encode()}
        )

    @classmethod
    def from_parquet(cls, parquet: pq.ParquetFile) -> Self:
        """Read this record from a Parquet file written by pseval.

        Raises:
            ValueError: If the file has none.
        """
        raw = (parquet.schema_arrow.metadata or {}).get(_PARQUET_KEY)
        if raw is None:
            raise ValueError('Missing pseval provenance: this file was not written by pseval.')
        value = json.loads(raw)
        return cls(**value | {'selection': Selection(**value['selection'])})

    def require_benchmark(self, dataset: str, fingerprint: str) -> None:
        """Require this record to name the given benchmark.

        Raises:
            ValueError: If the dataset or the fingerprint differs.
        """
        if self.dataset != dataset or self.fingerprint != fingerprint:
            raise ValueError(
                f'Produced on benchmark {self.dataset} ({self.fingerprint[:12]}), not on '
                f'{dataset} ({fingerprint[:12]})'
            )
