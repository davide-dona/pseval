from functools import cached_property
from pathlib import Path

from pseval.benchmark.manifest import MANIFEST, PROTOCOL_FILES, REFERENCE_FILES, Manifest
from pseval.benchmark.specs import SPECS, DatasetSpec
from pseval.benchmark.splits import read_split
from pseval.data import Log
from pseval.metrics.scorer import Scorer

# The Hugging Face dataset repository holding the published bundles, one directory per dataset.
HUB_REPOSITORY = '446f6e6e79/pseval'


class Benchmark:
    """One verified benchmark bundle.

    `train` and `validation` are what a model may learn and select from. The test split is
    reachable only through `pseval.generate`, which cuts its prefixes and hands a model nothing
    past a prefix's last event.
    """

    def __init__(self, directory: Path) -> None:
        """
        Args:
            directory: The bundle directory of one dataset.

        Raises:
            ValueError: If the manifest or a file does not verify.
        """
        self._directory = directory
        self._manifest = Manifest.read(directory / MANIFEST)
        self._manifest.verify(directory)
        self._scorer = Scorer.load(
            protocol=directory / PROTOCOL_FILES['protocol'],
            declare=directory / PROTOCOL_FILES['declare'],
        )

    @property
    def directory(self) -> Path:
        return self._directory

    @property
    def spec(self) -> DatasetSpec:
        return SPECS[self._manifest.dataset]

    @property
    def dataset(self) -> str:
        return self._manifest.dataset

    @property
    def fingerprint(self) -> str:
        return self._manifest.fingerprint

    @property
    def vocabulary(self) -> tuple[str, ...]:
        """Every activity name a draw may hold: the train activities, sorted, then `UNK`.

        An observed activity unseen in train is scored as `UNK`.
        """
        return self._scorer.codes.vocabulary

    @property
    def max_suffix_length(self) -> int:
        """The most events a draw may hold, the length of the longest case kept."""
        return self._scorer.max_suffix_length

    @cached_property
    def train(self) -> Log:
        return self._read('train')

    @cached_property
    def validation(self) -> Log:
        return self._read('validation')

    @property
    def manifest(self) -> Manifest:
        return self._manifest

    def _read(self, split: str) -> Log:
        return read_split(
            self._directory / PROTOCOL_FILES[split], text_columns=self.spec.text_columns
        )

    def _test(self) -> Log:
        """The test split, cut only by `pseval.generate` and read by scoring."""
        return self._read('test')

    @property
    def scorer(self) -> Scorer:
        """The protocol fitted on train."""
        return self._scorer

    def _reference_scores(self) -> Path:
        """The scores of the unconditional reference."""
        return self._directory / REFERENCE_FILES['scores']


def load_benchmark(dataset: str, *, root: Path | None = None) -> Benchmark:
    """Load and verify a benchmark bundle.

    Args:
        dataset: One of `sepsis`, `bpic12`, `bpic17`, `bpic19`.
        root: A local bundle root, holding `<dataset>/`, instead of downloading the bundle from
            the Hugging Face Hub into its cache.

    Returns:
        The verified benchmark.

    Raises:
        ValueError: If the dataset is unknown or the bundle does not verify.
    """
    if dataset not in SPECS:
        raise ValueError(f'Unknown dataset {dataset!r}; choose one of {", ".join(SPECS)}')
    if root is None:
        from huggingface_hub import snapshot_download

        root = Path(
            snapshot_download(
                repo_id=HUB_REPOSITORY,
                repo_type='dataset',
                allow_patterns=f'{dataset}/**',
            )
        )
    return Benchmark(root / dataset)
