from collections.abc import Sequence
from itertools import pairwise
from pathlib import Path
from types import TracebackType
from typing import Self

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from pseval.data import PrefixKey
from pseval.generation.records import Draws, Generation
from pseval.record import Record

# Activity names, one per event, dictionary encoded on disk.
_ACTIVITIES = pa.list_(pa.field(name='element', type=pa.string()))
# Inter-event minutes, one per activity. Stored in float32, the width models emit.
_MINUTES = pa.list_(pa.field(name='element', type=pa.float32()))

# The generations file: one row per test prefix, the draws nested inside it. Prefixes and observed
# suffixes are not stored: scoring reads them from the benchmark.
SCHEMA = pa.schema(
    [
        ('case_id', pa.large_string()),
        ('prefix_len', pa.int64()),
        (
            'generated_draws',
            pa.list_(
                pa.field(
                    name='element',
                    type=pa.struct(
                        [
                            ('activities', _ACTIVITIES),
                            ('inter_event_time_minutes', _MINUTES),
                            ('truncated', pa.bool_()),
                        ]
                    ),
                )
            ),
        ),
    ]
)

_KEY_COLUMNS = ['case_id', 'prefix_len']
_NAME_LEAVES = ['generated_draws.list.element.activities.list.element']
_FLOAT_LEAVES = ['generated_draws.list.element.inter_event_time_minutes.list.element']
_TRUNCATED_LEAF = 'generated_draws.list.element.truncated'


def _row(generation: Generation) -> dict:
    draws = generation.draws
    return {
        'case_id': generation.case_id,
        'prefix_len': generation.prefix_len,
        'generated_draws': [
            {
                'activities': list(events.activities),
                'inter_event_time_minutes': events.inter_event_time_minutes.tolist(),
                'truncated': bool(truncated),
            }
            for events, truncated in zip(draws, draws.truncated, strict=True)
        ],
    }


def _draws_from_table(table: pa.Table) -> list[tuple[PrefixKey, Draws]]:
    """Decode a block column by column, each prefix's arrays being views of the block's.

    Raises:
        ValueError: If a draw's times do not match its activities.
    """
    draws = table.column('generated_draws').combine_chunks()
    draw_offsets = draws.offsets.to_numpy()
    names = pc.struct_field(draws.values, 'activities')
    # Decoded through the dictionary, so each distinct name is one shared string object.
    encoded = pc.dictionary_encode(names.values)
    events = np.array(encoded.dictionary.to_pylist(), dtype=object)[encoded.indices.to_numpy()]
    activities = [tuple(events[a:b]) for a, b in pairwise(names.offsets.to_numpy())]
    truncated = pc.struct_field(draws.values, 'truncated').to_numpy(zero_copy_only=False)
    lists = pc.struct_field(draws.values, 'inter_event_time_minutes')
    times, time_offsets = lists.values.to_numpy().astype(np.float64), lists.offsets.to_numpy()

    result = []
    for row, key in enumerate(
        zip(
            table.column('case_id').to_pylist(),
            table.column('prefix_len').to_pylist(),
            strict=True,
        )
    ):
        first, last = draw_offsets[row], draw_offsets[row + 1]
        row_time_offsets = time_offsets[first : last + 1]
        draws = Draws(
            activities=tuple(activities[first:last]),
            inter_event_time_minutes=times[row_time_offsets[0] : row_time_offsets[-1]],
            truncated=truncated[first:last],
        )
        if not np.array_equal(row_time_offsets - row_time_offsets[0], draws.offsets):
            raise ValueError(f'{key} has draw times that do not match their activities')
        result.append((key, draws))
    return result


class GenerationWriter:
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
            use_dictionary=_NAME_LEAVES,
            use_byte_stream_split=_FLOAT_LEAVES,
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

    def write(self, generations: Sequence[Generation]) -> None:
        """Write one batch of prefixes as one row group."""
        rows = [_row(generation) for generation in generations]
        self._writer.write_table(table=pa.Table.from_pylist(mapping=rows, schema=SCHEMA))


class Generations:
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

    def prefix_keys(self) -> list[PrefixKey]:
        """The key of every row, in file order, read from the two key columns only."""
        table = self._parquet.read(columns=_KEY_COLUMNS)
        return list(
            zip(
                table.column('case_id').to_pylist(),
                table.column('prefix_len').to_pylist(),
                strict=True,
            )
        )

    def num_samples(self) -> int:
        """The draw count every prefix shares, read from the truncation flags only.

        Raises:
            ValueError: If the prefixes hold different draw counts.
        """
        table = self._parquet.read(columns=[_TRUNCATED_LEAF])
        counts = pc.unique(pc.list_value_length(table.column('generated_draws'))).to_pylist()
        if len(counts) != 1:
            raise ValueError(f'The prefixes hold different draw counts: {sorted(counts)}')
        return counts[0]

    def block(self, block: int) -> list[tuple[PrefixKey, Draws]]:
        """Decode one row group: each prefix's key and draws, in file order."""
        return _draws_from_table(self._parquet.read_row_group(block))
