import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Self

import pyarrow as pa
import pyarrow.parquet as pq

from pseval.hashing import canonical_json, validate_sha256
from pseval.metrics import METRIC_KEYS

# The major version of pseval. Every file pseval writes carries it, and a file written under
# another one is refused: a major version changes the reported numbers.
PROTOCOL_VERSION = 1

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
        if not isinstance(self.procedure, str) or not self.procedure.strip():
            raise ValueError('A selection procedure must be described')

    @classmethod
    def none(cls) -> Self:
        """The selection of a model estimated once, with nothing compared on validation."""
        return cls(metric=None, procedure='none')

    def as_dict(self) -> dict[str, str | None]:
        return {'metric': self.metric, 'procedure': self.procedure}

    @classmethod
    def from_dict(cls, value: object) -> Self:
        if not isinstance(value, dict) or set(value) != {'metric', 'procedure'}:
            raise ValueError('Invalid selection record')
        return cls(metric=value['metric'], procedure=value['procedure'])


@dataclass(frozen=True)
class Provenance:
    """What a generations or scores file was produced from.

    `dataset` and `fingerprint` identify the benchmark bundle. `label` names the model and
    `metadata` holds whatever the model author records with it, such as a checkpoint hash or a
    configuration, as a JSON object. `generations_sha256` is set on a scores file only, naming the
    generations file it scored.
    """

    dataset: str
    fingerprint: str
    label: str
    selection: Selection
    metadata: Mapping[str, object] = field(default_factory=dict)
    generations_sha256: str | None = None
    protocol_version: int = PROTOCOL_VERSION

    def __post_init__(self) -> None:
        validate_sha256(self.fingerprint, 'benchmark fingerprint')
        if not isinstance(self.label, str) or not self.label.strip():
            raise ValueError('A model label must be a nonempty string')
        if not isinstance(self.metadata, Mapping):
            raise ValueError('Model metadata must be a JSON object')
        try:
            canonical_json(dict(self.metadata))
        except (TypeError, ValueError) as error:
            raise ValueError(f'Model metadata is not JSON serializable: {error}') from error
        if self.generations_sha256 is not None:
            validate_sha256(self.generations_sha256, 'generations SHA-256')
        if self.protocol_version != PROTOCOL_VERSION:
            raise ValueError(
                f'Written under pseval protocol {self.protocol_version}, but this is protocol '
                f'{PROTOCOL_VERSION}. Generate and score it again with this version.'
            )

    def as_dict(self) -> dict[str, object]:
        return {
            'protocol_version': self.protocol_version,
            'dataset': self.dataset,
            'fingerprint': self.fingerprint,
            'label': self.label,
            'selection': self.selection.as_dict(),
            'metadata': dict(self.metadata),
            'generations_sha256': self.generations_sha256,
        }

    @classmethod
    def from_dict(cls, value: object) -> Self:
        """Read a stored record.

        Raises:
            ValueError: If the record is malformed or written under another protocol version.
        """
        fields = {
            'protocol_version',
            'dataset',
            'fingerprint',
            'label',
            'selection',
            'metadata',
            'generations_sha256',
        }
        if not isinstance(value, dict) or set(value) != fields:
            raise ValueError('Invalid pseval provenance')
        return cls(
            protocol_version=value['protocol_version'],
            dataset=value['dataset'],
            fingerprint=value['fingerprint'],
            label=value['label'],
            selection=Selection.from_dict(value['selection']),
            metadata=value['metadata'],
            generations_sha256=value['generations_sha256'],
        )

    def scored(self, generations_sha256: str) -> Self:
        """This record as stored on the scores of the generations file with that hash."""
        return type(self)(
            dataset=self.dataset,
            fingerprint=self.fingerprint,
            label=self.label,
            selection=self.selection,
            metadata=self.metadata,
            generations_sha256=generations_sha256,
        )

    def attach_to_schema(self, schema: pa.Schema) -> pa.Schema:
        """Attach this record to a Parquet schema."""
        return schema.with_metadata(
            (schema.metadata or {}) | {_PARQUET_KEY: json.dumps(self.as_dict()).encode()}
        )

    @classmethod
    def from_parquet(cls, parquet: pq.ParquetFile) -> Self:
        """Read this record from a Parquet file written by pseval.

        Raises:
            ValueError: If the file has none, or it is written under another protocol version.
        """
        raw = (parquet.schema_arrow.metadata or {}).get(_PARQUET_KEY)
        if raw is None:
            raise ValueError('Missing pseval provenance: this file was not written by pseval.')
        return cls.from_dict(json.loads(raw))

    def require_benchmark(self, dataset: str, fingerprint: str) -> None:
        """Require this record to name the given benchmark bundle.

        Raises:
            ValueError: If the dataset or the fingerprint differs.
        """
        if self.dataset != dataset or self.fingerprint != fingerprint:
            raise ValueError(
                f'Produced on benchmark {self.dataset} ({self.fingerprint[:12]}), not on '
                f'{dataset} ({fingerprint[:12]})'
            )
