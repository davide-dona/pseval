import json
from collections.abc import Sequence
from pathlib import Path
from types import TracebackType
from typing import Self

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from pseval.data import PrefixKey
from pseval.provenance import Provenance
from pseval.records import DecodedEvents, Draws, Generation
from pseval.vocabulary import ActivityCodes

_VOCABULARY = b'vocabulary'

# One activity sequence, one character per activity. Names live once in the vocabulary metadata.
_ACTIVITIES = pa.string()
# Inter-event minutes, one per activity. Stored in float32, the width models emit.
_MINUTES = pa.list_(pa.field(name='element', type=pa.float32()))

# The generations file: one row per test prefix, the draws nested inside it.
SCHEMA = pa.schema(
    [
        ('case_id', pa.large_string()),
        ('prefix_len', pa.int64()),
        ('prefix_activities', _ACTIVITIES),
        ('prefix_inter_event_time_minutes', _MINUTES),
        ('generated_suffixes', pa.list_(pa.field(name='element', type=_ACTIVITIES))),
        (
            'generated_draws',
            pa.list_(
                pa.field(
                    name='element',
                    type=pa.struct(
                        [
                            ('suffix_index', pa.int32()),
                            ('inter_event_time_minutes', _MINUTES),
                            ('truncated', pa.bool_()),
                        ]
                    ),
                )
            ),
        ),
        (
            'truth',
            pa.struct([('activities', _ACTIVITIES), ('inter_event_time_minutes', _MINUTES)]),
        ),
    ]
)

_KEY_COLUMNS = ['case_id', 'prefix_len']
_FLOAT_LEAVES = [
    'prefix_inter_event_time_minutes.list.element',
    'generated_draws.list.element.inter_event_time_minutes.list.element',
    'truth.inter_event_time_minutes.list.element',
]


def _row(generation: Generation) -> dict:
    samples = generation.samples
    times = samples.inter_event_time_minutes
    return {
        'case_id': generation.case_id,
        'prefix_len': generation.prefix_len,
        'prefix_activities': generation.prefix_activities,
        'prefix_inter_event_time_minutes': generation.prefix_inter_event_time_minutes.tolist(),
        'generated_suffixes': list(samples.suffixes),
        'generated_draws': [
            {
                'suffix_index': int(index),
                'inter_event_time_minutes': times[start:stop].tolist(),
                'truncated': bool(truncated),
            }
            for index, start, stop, truncated in zip(
                samples.taken,
                samples.offsets[:-1],
                samples.offsets[1:],
                samples.truncated,
                strict=True,
            )
        ],
        'truth': {
            'activities': generation.truth.activities,
            'inter_event_time_minutes': generation.truth.inter_event_time_minutes.tolist(),
        },
    }


def _require_values(*arrays: pa.Array | pa.ChunkedArray) -> None:
    """Raise a `ValueError` if any of the arrays holds a null."""
    if any(array.null_count for array in arrays):
        raise ValueError('Generations contain a null value')


def _flat_times(lists: pa.ListArray) -> tuple[np.ndarray, np.ndarray]:
    """Return a list column's values widened to float64 and its offsets into them."""
    _require_values(lists, lists.values)
    return lists.values.to_numpy().astype(np.float64), lists.offsets.to_numpy()


def _generations_from_table(table: pa.Table) -> list[Generation]:
    """Decode a block column by column, each prefix's arrays being views of the block's.

    Raises:
        ValueError: If a value is null or the rows contradict themselves.
    """
    _require_values(*table.columns)
    suffixes = table.column('generated_suffixes').combine_chunks()
    _require_values(suffixes.values)
    suffix_offsets = suffixes.offsets.to_numpy()
    suffix_values = suffixes.values.to_pylist()
    draws = table.column('generated_draws').combine_chunks()
    draw_offsets = draws.offsets.to_numpy()
    taken = pc.struct_field(draws.values, 'suffix_index')
    truncated = pc.struct_field(draws.values, 'truncated')
    _require_values(taken, truncated)
    taken = taken.to_numpy()
    truncated = truncated.to_numpy(zero_copy_only=False)
    times, time_offsets = _flat_times(pc.struct_field(draws.values, 'inter_event_time_minutes'))
    truth = table.column('truth').combine_chunks()
    truth_times, truth_offsets = _flat_times(pc.struct_field(truth, 'inter_event_time_minutes'))
    truth_activities = pc.struct_field(truth, 'activities')
    _require_values(truth_activities)
    prefix_times, prefix_offsets = _flat_times(
        table.column('prefix_inter_event_time_minutes').combine_chunks()
    )

    generations = []
    for row, (case_id, prefix_len, prefix, true_activities) in enumerate(
        zip(
            table.column('case_id').to_pylist(),
            table.column('prefix_len').to_pylist(),
            table.column('prefix_activities').to_pylist(),
            truth_activities.to_pylist(),
            strict=True,
        )
    ):
        if prefix_len != len(prefix):
            raise ValueError('Generation prefix length does not match its activities')
        first, last = draw_offsets[row], draw_offsets[row + 1]
        row_time_offsets = time_offsets[first : last + 1]
        samples = Draws(
            suffixes=tuple(suffix_values[suffix_offsets[row] : suffix_offsets[row + 1]]),
            taken=taken[first:last],
            inter_event_time_minutes=times[row_time_offsets[0] : row_time_offsets[-1]],
            truncated=truncated[first:last],
        )
        if not np.array_equal(row_time_offsets - row_time_offsets[0], samples.offsets):
            raise ValueError('Generation draw times do not match their suffix lengths')
        generations.append(
            Generation(
                case_id=case_id,
                prefix_activities=prefix,
                prefix_inter_event_time_minutes=prefix_times[
                    prefix_offsets[row] : prefix_offsets[row + 1]
                ],
                samples=samples,
                truth=DecodedEvents(
                    activities=true_activities,
                    inter_event_time_minutes=truth_times[
                        truth_offsets[row] : truth_offsets[row + 1]
                    ],
                ),
            )
        )
    return generations


class GenerationWriter:
    """A generations file, open for writing, one row group per batch.

    Used as a context manager: the footer is written when it closes, and a failure inside the
    block removes the file, so an interrupted generation leaves nothing readable.
    """

    def __init__(self, path: Path, provenance: Provenance, *, vocabulary: Sequence[str]) -> None:
        """
        Args:
            path: Destination file, replaced if it exists.
            provenance: The benchmark, model, and selection that produced the generations.
            vocabulary: The activity names the suffixes are spelled on, in code order.
        """
        if provenance.generations_sha256 is not None:
            raise ValueError('A generations file cannot name a generations source')
        schema = provenance.attach_to_schema(SCHEMA)
        schema = schema.with_metadata(
            schema.metadata | {_VOCABULARY: json.dumps(list(vocabulary)).encode()}
        )
        self._path = path
        self._writer = pq.ParquetWriter(
            where=path,
            schema=schema,
            compression='zstd',
            compression_level=9,
            use_dictionary=False,
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
            ValueError: If the schema differs, or the provenance or vocabulary is missing,
                malformed, or written under another protocol version.
        """
        self._parquet = pq.ParquetFile(path)
        try:
            if not self._parquet.schema_arrow.remove_metadata().equals(SCHEMA):
                raise ValueError(f'{path} does not have the generations schema of pseval')
            self._provenance = Provenance.from_parquet(self._parquet)
            if self._provenance.generations_sha256 is not None:
                raise ValueError(f'{path} is a scores file, not a generations file')
            raw = (self._parquet.schema_arrow.metadata or {}).get(_VOCABULARY)
            if raw is None:
                raise ValueError(f'{path} has no activity vocabulary')
            self._codes = ActivityCodes(json.loads(raw))
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
    def provenance(self) -> Provenance:
        return self._provenance

    @property
    def codes(self) -> ActivityCodes:
        """The codes the file spells activities with, from its vocabulary."""
        return self._codes

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

    def block(self, block: int) -> list[Generation]:
        """Decode one row group, its prefixes in file order."""
        return _generations_from_table(self._parquet.read_row_group(block))
