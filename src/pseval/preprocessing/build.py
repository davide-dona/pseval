import tempfile
from pathlib import Path

from pseval.benchmark import FILES, REFERENCE_SCORES, Benchmark, write_split
from pseval.data import Log
from pseval.generation.generate import generate
from pseval.metrics.scorer import Scorer
from pseval.preprocessing.pipeline import split_log
from pseval.preprocessing.specs import SPECS, DatasetSpec
from pseval.record import Record, digest
from pseval.scoring.reference import Unconditional
from pseval.scoring.score import score


def build(
    spec: DatasetSpec, original: Path, root: Path, *, workers: int | None = None
) -> Benchmark:
    """Build a benchmark under a local root, its reference scores included.

    Args:
        spec: The dataset's spec.
        original: The original log.
        root: The local root; the benchmark is written to `root/<dataset>/`, the layout
            `load_benchmark(dataset, root=root)` reads.
        workers: Scoring processes for the reference, as `score` defaults them when None.

    Raises:
        FileExistsError: If `root/<dataset>/` exists.
        ValueError: If the original log is not the one `spec` names.
    """
    directory = root / spec.name
    _write_files(spec, original, directory)
    benchmark = Benchmark(directory)
    with tempfile.TemporaryDirectory() as temporary:
        generations = generate(
            Unconditional(benchmark.train),
            benchmark,
            Path(temporary) / 'generations.parquet',
            label='unconditional',
        )
        score(generations, benchmark, path=directory / REFERENCE_SCORES, workers=workers)
    return benchmark


def rebuild(benchmark: Benchmark, original: Path) -> None:
    """Rebuild a benchmark's `FILES` from the original log and require the same digest.

    Raises:
        ValueError: If the rebuilt digest differs, naming the files that differ.
    """
    with tempfile.TemporaryDirectory() as temporary:
        directory = Path(temporary) / benchmark.dataset
        _write_files(SPECS[benchmark.dataset], original, directory)
        differing = [
            name
            for name, file in FILES.items()
            if digest(directory / file) != digest(benchmark.directory / file)
        ]
    if differing:
        raise ValueError(
            f'Rebuilding {benchmark.dataset} gives another digest; differing: '
            f'{", ".join(differing)}'
        )


def _write_files(spec: DatasetSpec, original: Path, directory: Path) -> None:
    """Write the splits and the scorer fitted on train, every file of `FILES`."""
    if digest(original) != spec.original_sha256:
        raise ValueError(f'{original} is not the original {spec.name} log')
    directory.mkdir(parents=True)
    record = Record(dataset=spec.name, label=None, inputs={'original': spec.original_sha256})
    train, validation, test, first_cuts, max_length = split_log(spec, original)
    write_split(train, directory / FILES['train'], record)
    write_split(validation, directory / FILES['validation'], record)
    write_split(test, directory / FILES['test'], record, first_cuts=first_cuts)
    scorer = Scorer.fit(
        Log(train),
        max_suffix_length=max_length,
        declare_hard_min_support=spec.declare_hard_min_support,
        declare_hard_min_confidence=spec.declare_hard_min_confidence,
        declare_max_cardinality=spec.declare_max_cardinality,
    )
    scorer.save(
        path=directory / FILES['scorer'],
        hard=directory / FILES['declare_hard'],
        soft=directory / FILES['declare_soft'],
    )
