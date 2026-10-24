import hashlib
import json
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Self

import pyarrow as pa

_PARQUET_KEY = b'pseval'


def digest(path: Path) -> str:
    """Return the SHA-256 of a file's bytes."""
    with path.open('rb') as file:
        return hashlib.file_digest(file, 'sha256').hexdigest()


@dataclass(frozen=True)
class Record:
    """What an artifact holds and what it was produced from, stored in its Parquet metadata.

    `dataset` names the benchmark the artifact belongs to. `label` names the model whose draws a
    generations or scores file holds, and is None on a benchmark. `inputs` maps the role of each
    artifact this one was produced from directly to its digest: `original` for a benchmark,
    `benchmark` for generations, and `benchmark` and `generations` for scores.
    """

    dataset: str
    label: str | None
    inputs: Mapping[str, str]

    def attach_to(self, schema: pa.Schema) -> pa.Schema:
        """Return `schema` with this record in its metadata."""
        return schema.with_metadata(
            (schema.metadata or {}) | {_PARQUET_KEY: json.dumps(asdict(self)).encode()}
        )

    @classmethod
    def from_schema(cls, schema: pa.Schema) -> Self:
        """Read the record from the schema of a Parquet file written by pseval.

        Raises:
            ValueError: If the schema holds none.
        """
        raw = (schema.metadata or {}).get(_PARQUET_KEY)
        if raw is None:
            raise ValueError('Missing pseval record: this file was not written by pseval.')
        return cls(**json.loads(raw))

    def require_input(self, role: str, expected: str) -> None:
        """Require the input of `role` to be the artifact whose digest is `expected`.

        Raises:
            ValueError: If it is another one.
        """
        if self.inputs.get(role) != expected:
            raise ValueError(
                f'Produced from {role} {str(self.inputs.get(role))[:12]}, not from {expected[:12]}'
            )
