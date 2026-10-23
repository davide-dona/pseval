import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from types import TracebackType
from typing import Self

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from pseval.data import PrefixKey
from pseval.metrics import METRICS
from pseval.provenance import Provenance
from pseval.scoring.summary import Summary, summarize

# The scores file: one row per test prefix, every registered metric a column.
SCHEMA = pa.schema(
    [
        ('case_id', pa.large_string()),
        ('prefix_len', pa.int64()),
        ('suffix_len', pa.int64()),
        *((key, pa.float64()) for key in METRICS.entries),
    ]
)
_METRIC_TABLE = b'metrics'
# Rows buffered before a row group of the scores file is written.
_ROW_GROUP = 16_384


class ScoresWriter:
    """A scores file, open for writing.

    Used as a context manager: the buffered rows and the footer are written when it closes, and
    a failure inside the block removes the file, so an interrupted scoring leaves nothing
    readable.
    """

    def __init__(self, path: Path, provenance: Provenance) -> None:
        """
        Args:
            path: Destination file, replaced if it exists.
            provenance: The provenance of the scored generations, naming their file.
        """
        if provenance.generations_sha256 is None:
            raise ValueError('A scores file must name the generations it scored')
        schema = provenance.attach_to_schema(SCHEMA)
        schema = schema.with_metadata(
            schema.metadata | {_METRIC_TABLE: json.dumps(METRICS.table()).encode()}
        )
        self._path = path
        self._writer = pq.ParquetWriter(where=path, schema=schema)
        self._columns = _empty_columns()

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exception_type: type[BaseException] | None,
        exception: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        try:
            if exception_type is None and self._columns['case_id']:
                self._flush()
            self._writer.close()
        except BaseException:
            self._path.unlink(missing_ok=True)
            raise
        if exception_type is not None:
            self._path.unlink(missing_ok=True)

    def write(self, rows: Sequence[tuple[PrefixKey, int, dict[str, float]]]) -> None:
        """Append scored prefixes, each its key, true suffix length, and metric values.

        A row group is written once at least `_ROW_GROUP` rows are buffered.
        """
        columns = self._columns
        for (case_id, prefix_len), suffix_len, values in rows:
            columns['case_id'].append(case_id)
            columns['prefix_len'].append(prefix_len)
            columns['suffix_len'].append(suffix_len)
            for key, value in values.items():
                columns[key].append(value)
        if len(columns['case_id']) >= _ROW_GROUP:
            self._flush()

    def _flush(self) -> None:
        self._writer.write_table(pa.Table.from_pydict(self._columns, schema=SCHEMA))
        self._columns = _empty_columns()


def _empty_columns() -> dict[str, list]:
    return {name: [] for name in SCHEMA.names}


@dataclass(frozen=True, eq=False)
class Scores:
    """A scores file: one row per test prefix, its provenance, and its metric table."""

    frame: pd.DataFrame
    provenance: Provenance
    metrics: tuple[dict[str, object], ...]

    def summary(self) -> Summary:
        """Equal-weight means overall, by prefix length, and by true suffix length."""
        return summarize(self.frame, [metric['key'] for metric in self.metrics])


def read_scores(path: Path) -> Scores:
    """Read a scores file.

    Raises:
        ValueError: If it is not a pseval scores file of this protocol version.
    """
    with pq.ParquetFile(path) as parquet:
        if not parquet.schema_arrow.remove_metadata().equals(SCHEMA):
            raise ValueError(f'{path} does not have the scores schema of this pseval version')
        provenance = Provenance.from_parquet(parquet)
        if provenance.generations_sha256 is None:
            raise ValueError(f'{path} is a generations file, not a scores file')
        metrics = tuple(json.loads(parquet.schema_arrow.metadata[_METRIC_TABLE]))
        frame = parquet.read().to_pandas()
    return Scores(frame=frame, provenance=provenance, metrics=metrics)
