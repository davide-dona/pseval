import hashlib
import json
import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Self

import pyarrow as pa
import pyarrow.parquet as pq

from pseval.metrics import METRIC_KEYS, MIN_DRAWS

# The major version of pseval. Every file pseval writes carries it, and a file written under
# another one is refused: a major version changes the reported numbers.
PROTOCOL_VERSION = 1

_PARQUET_KEY = b'pseval'
_SHA256 = re.compile(r'[0-9a-f]{64}')


def sha256(path: Path) -> str:
    """Return the SHA-256 digest of a file's bytes."""
    with path.open('rb') as file:
        return hashlib.file_digest(file, 'sha256').hexdigest()


def canonical_json(value: object) -> bytes:
    """Encode a JSON value with sorted keys and no whitespace, the form fingerprints hash."""
    return json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False).encode()


def validate_sha256(value: object, field_name: str) -> str:
    """Require a lowercase hexadecimal SHA-256 digest.

    Raises:
        ValueError: If `value` is not one.
    """
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ValueError(f'Invalid {field_name}: {value!r}')
    return value


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

    `dataset` and `fingerprint` identify the benchmark bundle. `num_samples` is the draw count of
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
    protocol_version: int = PROTOCOL_VERSION

    def __post_init__(self) -> None:
        validate_sha256(self.fingerprint, 'benchmark fingerprint')
        if not isinstance(self.label, str) or not self.label.strip():
            raise ValueError('A model label must be a nonempty string')
        if not _is_int(self.num_samples) or self.num_samples < MIN_DRAWS:
            raise ValueError(f'The draw count must be an integer of at least {MIN_DRAWS}')
        if not _is_int(self.seed):
            raise ValueError('The seed must be an integer')
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
            'num_samples': self.num_samples,
            'seed': self.seed,
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
            'num_samples',
            'seed',
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
            num_samples=value['num_samples'],
            seed=value['seed'],
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
            num_samples=self.num_samples,
            seed=self.seed,
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


def _is_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)
