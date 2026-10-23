import tempfile
from dataclasses import replace
from pathlib import Path

from pseval.benchmark.bundle import Benchmark
from pseval.benchmark.manifest import (
    MANIFEST,
    PROTOCOL_FILES,
    REFERENCE_FILES,
    Manifest,
    sha256,
)
from pseval.benchmark.specs import DatasetSpec
from pseval.benchmark.splits import write_split
from pseval.data import Log
from pseval.generation.generate import generate
from pseval.generation.provenance import Selection
from pseval.metrics.scorer import Scorer
from pseval.preprocessing.pipeline import split_log
from pseval.preprocessing.reference import Unconditional
from pseval.scoring.score import score


def build_protocol(spec: DatasetSpec, original: Path, directory: Path) -> Manifest:
    """Write the protocol files of a bundle and its manifest, without the reference.

    Fits the vocabulary, mines the Declare model, and calibrates the signature kernel on the
    train split alone.

    Args:
        spec: The dataset's spec.
        original: The original log.
        directory: A directory to create for the bundle.

    Returns:
        The written manifest.

    Raises:
        FileExistsError: If `directory` exists.
    """
    directory.mkdir(parents=True)
    train, validation, test, first_cuts, max_length = split_log(spec, original)
    write_split(train, directory / PROTOCOL_FILES['train'])
    write_split(validation, directory / PROTOCOL_FILES['validation'])
    write_split(test, directory / PROTOCOL_FILES['test'], first_cuts=first_cuts)

    scorer = Scorer.fit(
        Log(train),
        max_suffix_length=max_length,
        declare_min_support=spec.declare_min_support,
        declare_min_confidence=spec.declare_min_confidence,
        declare_max_cardinality=spec.declare_max_cardinality,
    )
    scorer.save(
        protocol=directory / PROTOCOL_FILES['protocol'],
        declare=directory / PROTOCOL_FILES['declare'],
    )

    manifest = Manifest.create(directory, spec, original_sha256=sha256(original))
    manifest.write(directory / MANIFEST)
    return manifest


def add_reference(benchmark: Benchmark, *, workers: int | None = None) -> Benchmark:
    """Generate and score the unconditional reference into a bundle and list it in the manifest.

    Returns:
        The bundle reloaded with its reference.
    """
    paths = {name: benchmark.directory / file for name, file in REFERENCE_FILES.items()}
    paths['generations'].parent.mkdir(exist_ok=True)
    generate(
        Unconditional(benchmark.train),
        benchmark,
        paths['generations'],
        selection=Selection.none(),
        label='unconditional',
    )
    score(paths['generations'], benchmark, path=paths['scores'], workers=workers)
    manifest = replace(
        benchmark.manifest, reference={name: sha256(path) for name, path in paths.items()}
    )
    manifest.write(benchmark.directory / MANIFEST)
    return Benchmark(benchmark.directory)


def build(
    spec: DatasetSpec,
    original: Path,
    root: Path,
    *,
    workers: int | None = None,
) -> Benchmark:
    """Build a complete bundle under a local bundle root.

    Args:
        spec: The dataset's spec.
        original: The original log.
        root: The bundle root; the bundle is written to `root/<dataset>/`, the layout
            `load_benchmark(dataset, root=root)` reads.
        workers: Scoring processes for the reference, one per CPU when None.

    Returns:
        The verified bundle.
    """
    directory = root / spec.name
    build_protocol(spec, original, directory)
    return add_reference(Benchmark(directory), workers=workers)


def rebuild(benchmark: Benchmark, original: Path) -> None:
    """Rebuild a bundle's protocol files from the original log and require its fingerprint.

    Raises:
        ValueError: If the rebuilt fingerprint differs, naming the files that differ.
    """
    with tempfile.TemporaryDirectory() as temporary:
        manifest = build_protocol(benchmark.spec, original, Path(temporary) / 'bundle')
    if manifest.fingerprint != benchmark.fingerprint:
        expected = benchmark.manifest.files | {'original': benchmark.manifest.original_sha256}
        actual = manifest.files | {'original': manifest.original_sha256}
        differing = sorted(name for name in expected if expected[name] != actual[name])
        raise ValueError(
            f'Rebuilding {benchmark.dataset} gives another fingerprint; differing: '
            f'{", ".join(differing) or "spec"}'
        )
