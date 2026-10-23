from dataclasses import dataclass
from pathlib import Path
from types import TracebackType
from typing import Self

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from pseval.generation.provenance import Provenance
from pseval.metrics import METRIC_KEYS
from pseval.scoring.summary import Summary, summarize

# The scores file: one row per test prefix, every metric a column.
SCHEMA = pa.schema(
    [
        ('case_id', pa.large_string()),
        ('prefix_len', pa.int64()),
        ('suffix_len', pa.int64()),
        *((key, pa.float64()) for key in METRIC_KEYS),
    ]
)
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
        self._path = path
        self._writer = pq.ParquetWriter(where=path, schema=provenance.attach_to_schema(SCHEMA))
        self._buffered: list[pd.DataFrame] = []

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exception_type: type[BaseException] | None,
        exception: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        try:
            if exception_type is None and self._buffered:
                self._flush()
            self._writer.close()
        except BaseException:
            self._path.unlink(missing_ok=True)
            raise
        if exception_type is not None:
            self._path.unlink(missing_ok=True)

    def write(self, rows: pd.DataFrame) -> None:
        """Append scored prefixes, one row each holding every `SCHEMA` column.

        A row group is written once at least `_ROW_GROUP` rows are buffered.
        """
        self._buffered.append(rows)
        if sum(map(len, self._buffered)) >= _ROW_GROUP:
            self._flush()

    def _flush(self) -> None:
        rows = pd.concat(self._buffered, ignore_index=True)
        self._writer.write_table(pa.Table.from_pandas(rows, schema=SCHEMA, preserve_index=False))
        self._buffered = []


@dataclass(frozen=True, eq=False)
class Scores:
    """A scores file: one row per test prefix, and its provenance."""

    frame: pd.DataFrame
    provenance: Provenance

    def summary(self) -> Summary:
        """Equal-weight means overall, by prefix length, and by true suffix length."""
        return summarize(self.frame, list(METRIC_KEYS))


def read_scores(path: Path) -> Scores:
    """Read a scores file.

    Raises:
        ValueError: If it is not a pseval scores file.
    """
    with pq.ParquetFile(path) as parquet:
        if not parquet.schema_arrow.remove_metadata().equals(SCHEMA):
            raise ValueError(f'{path} does not have the scores schema of this pseval version')
        provenance = Provenance.from_parquet(parquet)
        frame = parquet.read().to_pandas()
    return Scores(frame=frame, provenance=provenance)
