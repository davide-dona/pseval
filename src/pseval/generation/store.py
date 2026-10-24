from collections.abc import Sequence
from dataclasses import dataclass
from itertools import pairwise
from pathlib import Path
from types import TracebackType
from typing import Self

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from pseval.data import Cut, Draw, Forecast
from pseval.record import Record

# The generations file: one row per test prefix, the draws nested inside it.
# Prefixes and observed suffixes are not stored, but rather read from the benchmark.
SCHEMA = pa.schema(
    [
        ('case_id', pa.large_string()),
        ('prefix_len', pa.int64()),
        (
            # List of draws
            'forecast',
            pa.list_(
                # Each draw is a struct of the suffix's activities, inter-event minutes, and
                # truncation flag.
                pa.field(
                    name='element',
                    type=pa.struct(
                        [
                            # Activity names, one per event, dictionary encoded on disk.
                            (
                                'activities',
                                pa.list_(pa.field(name='element', type=pa.string())),
                            ),
                            # Inter-event minutes, one per activity.
                            (
                                'inter_event_time_minutes',
                                pa.list_(pa.field(name='element', type=pa.float32())),
                            ),
                            ('truncated', pa.bool_()),
                        ]
                    ),
                )
            ),
        ),
    ]
)


def _row(cut: Cut, forecast: Forecast) -> dict:
    """Convert one prefix's forecast to a generations file row."""
    return {
        'case_id': cut.case_id,
        'prefix_len': cut.prefix_len,
        'forecast': [
            {
                'activities': list(draw.activities),
                'inter_event_time_minutes': draw.inter_event_time_minutes.tolist(),
                'truncated': draw.truncated,
            }
            for draw in forecast
        ],
    }


def _forecasts_from_table(table: pa.Table) -> list[tuple[Cut, Forecast]]:
    """Decode a block column by column, each draw's times being a view of the block's.

    Raises:
        ValueError: If a draw's times do not match its activities.
    """
    structs = table.column('forecast').combine_chunks()
    names = pc.struct_field(structs.values, 'activities')
    # Decoded through the dictionary, so each distinct name is one shared string object.
    encoded = pc.dictionary_encode(names.values)
    events = np.array(encoded.dictionary.to_pylist(), dtype=object)[encoded.indices.to_numpy()]
    lists = pc.struct_field(structs.values, 'inter_event_time_minutes')
    times = lists.values.to_numpy().astype(np.float64)
    truncated = pc.struct_field(structs.values, 'truncated').to_pylist()
    draws = [
        Draw(activities=events[a:b], inter_event_time_minutes=times[c:d], truncated=flag)
        for (a, b), (c, d), flag in zip(
            pairwise(names.offsets.to_numpy()),
            pairwise(lists.offsets.to_numpy()),
            truncated,
            strict=True,
        )
    ]
    return [
        (cut, Forecast(draws=tuple(draws[first:last])))
        for cut, (first, last) in zip(
            _cuts_from_table(table), pairwise(structs.offsets.to_numpy()), strict=True
        )
    ]


def _cuts_from_table(table: pa.Table) -> list[Cut]:
    """The cut of every row, from the two key columns."""
    return list(
        map(
            Cut,
            table.column('case_id').to_pylist(),
            table.column('prefix_len').to_pylist(),
        )
    )


class GenerationsWriter:
    """A generations file, open for writing, one row group per batch.

    Used as a context manager: the footer is written when it closes, and a failure inside the
    block removes the file, so an interrupted generation leaves nothing readable.
    """

    def __init__(self, path: Path, record: Record) -> None:
        """
        Args:
            path: Destination file, replaced if it exists.
            record: The model and the benchmark that produced the generations.
        """
        self._path = path
        self._writer = pq.ParquetWriter(
            where=path,
            schema=record.attach_to(SCHEMA),
            compression='zstd',
            compression_level=9,
            # Nested Parquet columns are named by their path in `SCHEMA`, `list.element` per list.
            use_dictionary=['forecast.list.element.activities.list.element'],
            # Byte stream split compresses floats better than the plain encoding.
            use_byte_stream_split=['forecast.list.element.inter_event_time_minutes.list.element'],
        )

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exception_type: type[BaseException] | None,
        exception: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        try:
            self._writer.close()
        except BaseException:
            self._path.unlink(missing_ok=True)
            raise
        if exception_type is not None:
            self._path.unlink(missing_ok=True)

    def write(self, forecasts: Sequence[tuple[Cut, Forecast]]) -> None:
        """Write one batch of prefixes and their forecasts as one row group."""
        rows = [_row(cut, forecast) for cut, forecast in forecasts]
        self._writer.write_table(table=pa.Table.from_pylist(mapping=rows, schema=SCHEMA))


@dataclass(frozen=True, eq=False)
class Generations:
    """The contents of a generations file: each prefix's cut and forecast, and its record."""

    forecasts: list[tuple[Cut, Forecast]]
    record: Record


class GenerationsReader:
    """A generations file, open for reading.

    Used as a context manager, and opened once per process that reads it: each scoring worker
    opens its own and decodes the row groups it scores.
    """

    def __init__(self, path: Path) -> None:
        """
        Raises:
            ValueError: If the schema differs or the record is missing.
        """
        self._parquet = pq.ParquetFile(path)
        try:
            if not self._parquet.schema_arrow.remove_metadata().equals(SCHEMA):
                raise ValueError(f'{path} does not have the generations schema of pseval')
            self._record = Record.from_schema(self._parquet.schema_arrow)
        except Exception:
            self._parquet.close()
            raise

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exception_type: type[BaseException] | None,
        exception: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self._parquet.close()

    @property
    def record(self) -> Record:
        return self._record

    @property
    def blocks(self) -> int:
        """How many row groups the file holds, the units scoring splits it into."""
        return self._parquet.num_row_groups

    @property
    def prefixes(self) -> int:
        """How many prefixes the file answers, one per row."""
        return self._parquet.metadata.num_rows

    def cuts(self) -> list[Cut]:
        """The cut of every row, in file order, read from the two key columns only."""
        return _cuts_from_table(self._parquet.read(columns=['case_id', 'prefix_len']))

    def num_samples(self) -> int:
        """The draw count every prefix shares, read from the truncation flags only.

        Raises:
            ValueError: If the prefixes hold different draw counts.
        """
        table = self._parquet.read(columns=['forecast.list.element.truncated'])
        counts = pc.unique(pc.list_value_length(table.column('forecast'))).to_pylist()
        if len(counts) != 1:
            raise ValueError(f'The prefixes hold different draw counts: {sorted(counts)}')
        return counts[0]

    def block(self, block: int) -> list[tuple[Cut, Forecast]]:
        """Decode one row group: each prefix's cut and forecast, in file order."""
        return _forecasts_from_table(self._parquet.read_row_group(block))

    def read(self) -> Generations:
        """Decode every row of the file at once, holding all its draws in memory."""
        return Generations(
            forecasts=_forecasts_from_table(self._parquet.read()), record=self._record
        )
