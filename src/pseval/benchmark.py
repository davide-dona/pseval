import hashlib
import json
from functools import cached_property
from importlib.metadata import version
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from pseval.data import CASE_KEY, Log
from pseval.metrics.scorer import Scorer
from pseval.record import Record, digest

# The Hugging Face dataset repository holding the published benchmarks, one directory per dataset
# and one tag per pseval release.
HUB_REPOSITORY = '446f6e6e79/pseval'

# The files of a benchmark directory, by name. The benchmark digest covers exactly these.
FILES = {
    'train': 'train.parquet',
    'validation': 'validation.parquet',
    'test': 'test.parquet',
    'scorer': 'scorer.json',
    'declare_hard': 'declare_hard.json',
    'declare_soft': 'declare_soft.json',
}

# The splits, each holding the benchmark's record in its Parquet metadata.
SPLITS = ('train', 'validation', 'test')

# The scores of the unconditional reference. Outside the digest, since they are scored on the
# benchmark and name its digest in their record.
REFERENCE_SCORES = 'reference_scores.parquet'

# The first cut of each case, a column of the test split only. It is 1 except on the test cases
# running across the split separation, whose prefixes ending before it would be predicted with
# training data from their future.
PREFIX_START_KEY = '_prefix_start'


class Benchmark:
    """One dataset's benchmark: a directory holding `FILES` and the reference scores.

    `train` and `validation` are what a model may learn and select from. The test split is
    reachable only through `pseval.generate`, which cuts its prefixes and hands a model nothing
    past a prefix's last event.
    """

    def __init__(self, directory: Path) -> None:
        """
        Args:
            directory: The benchmark directory.

        Raises:
            ValueError: If the splits hold different records.
        """
        records = [Record.from_schema(pq.read_schema(directory / FILES[s])) for s in SPLITS]
        if any(record != records[0] for record in records):
            raise ValueError(f'The splits under {directory} hold different records')
        self._directory = directory
        self._dataset = records[0].dataset
        self._digest = _directory_digest(directory)
        self._scorer = Scorer.load(
            path=directory / FILES['scorer'],
            hard=directory / FILES['declare_hard'],
            soft=directory / FILES['declare_soft'],
        )

    @property
    def directory(self) -> Path:
        return self._directory

    @property
    def dataset(self) -> str:
        return self._dataset

    @property
    def digest(self) -> str:
        """The identity every generations and scores file of this benchmark names."""
        return self._digest

    @property
    def vocabulary(self) -> tuple[str, ...]:
        """Every activity name a draw may hold: the train activities, sorted, then `UNK`.

        An observed activity unseen in train is scored as `UNK`.
        """
        return self._scorer.vocabulary

    @property
    def max_suffix_length(self) -> int:
        """The most events a draw may hold, the length of the longest case kept."""
        return self._scorer.max_suffix_length

    @property
    def scorer(self) -> Scorer:
        """The scorer fitted on train."""
        return self._scorer

    @cached_property
    def train(self) -> Log:
        return read_split(self._directory / FILES['train'])

    @cached_property
    def validation(self) -> Log:
        return read_split(self._directory / FILES['validation'])

    def _test(self) -> Log:
        """The test split, cut only by `pseval.generate` and read by scoring."""
        return read_split(self._directory / FILES['test'])

    def _reference_scores(self) -> Path:
        """The scores of the unconditional reference."""
        return self._directory / REFERENCE_SCORES


def _directory_digest(directory: Path) -> str:
    """Return the digest of a benchmark directory: the SHA-256 of the digests of its `FILES`."""
    digests = {name: digest(directory / file) for name, file in FILES.items()}
    return hashlib.sha256(json.dumps(digests, sort_keys=True).encode()).hexdigest()


def load_benchmark(dataset: str, *, root: Path | None = None) -> Benchmark:
    """Load a benchmark.

    Args:
        dataset: One of `sepsis`, `bpic12`, `bpic17`, `bpic19`.
        root: A local root holding `<dataset>/`, instead of downloading the benchmark of this
            pseval release from the Hugging Face Hub into its cache.

    Raises:
        ValueError: If the root holds no benchmark of that dataset under its name.
    """
    if root is None:
        from huggingface_hub import snapshot_download

        root = Path(
            snapshot_download(
                repo_id=HUB_REPOSITORY,
                repo_type='dataset',
                revision=f'v{version("pseval")}',
                allow_patterns=f'{dataset}/**',
            )
        )
    if not (root / dataset).is_dir():
        raise ValueError(f'No {dataset!r} benchmark under {root}')
    benchmark = Benchmark(root / dataset)
    if benchmark.dataset != dataset:
        raise ValueError(f'{root / dataset} holds the {benchmark.dataset!r} benchmark')
    return benchmark


def write_split(
    events: pd.DataFrame, path: Path, record: Record, *, first_cuts: pd.Series | None = None
) -> None:
    """Write one split of a benchmark.

    Args:
        events: The split's events, each case's rows consecutive and in time order.
        path: Destination file.
        record: The benchmark's record, stored in the split's metadata.
        first_cuts: The first cut of every case, indexed by case, stored for the test split.
    """
    if first_cuts is not None:
        events = events.assign(**{PREFIX_START_KEY: events[CASE_KEY].map(first_cuts)})
    table = pa.Table.from_pandas(events, preserve_index=False)
    pq.write_table(table.cast(record.attach_to(table.schema)), path)


def read_split(path: Path) -> Log:
    """Read one split written by `write_split`, with the first cuts it stores when it has them."""
    events = pd.read_parquet(path)
    if PREFIX_START_KEY not in events.columns:
        return Log(events)
    first_rows = events[CASE_KEY].ne(events[CASE_KEY].shift()).to_numpy()
    prefix_starts = events.loc[first_rows, PREFIX_START_KEY].to_numpy(np.int64)
    return Log(events.drop(columns=PREFIX_START_KEY), prefix_starts=prefix_starts)
