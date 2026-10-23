import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Self

from pseval.benchmark.specs import SPECS, DatasetSpec

MANIFEST = 'manifest.json'


# The files a bundle's fingerprint covers, by name.
PROTOCOL_FILES = {
    'train': 'train.csv',
    'validation': 'validation.csv',
    'test': 'test.csv',
    'protocol': 'protocol.json',
    'declare': 'declare.json',
}


# The unconditional reference, derived from the files above and verified but not fingerprinted.
REFERENCE_FILES = {
    'generations': 'reference/generations.parquet',
    'scores': 'reference/scores.parquet',
}


@dataclass(frozen=True)
class Manifest:
    """Identity and contents of one benchmark bundle.

    The fingerprint hashes the dataset spec, the original log's hash, and the hash of every
    protocol file. The reference files are listed beside it, since they are
    produced from a bundle that already has its fingerprint.
    """

    dataset: str
    spec: dict[str, object]
    original_sha256: str
    files: dict[str, str]
    fingerprint: str
    reference: dict[str, str] | None = None

    def __post_init__(self) -> None:
        if self.fingerprint != _fingerprint(
            self.dataset, self.spec, self.original_sha256, self.files
        ):
            raise ValueError(f'The {self.dataset} manifest fingerprint does not match its contents')

    @classmethod
    def create(
        cls,
        directory: Path,
        spec: DatasetSpec,
        *,
        original_sha256: str,
        reference: dict[str, str] | None = None,
    ) -> Self:
        """Describe the protocol files written in a bundle directory."""
        files = {name: sha256(directory / file) for name, file in PROTOCOL_FILES.items()}
        return cls(
            dataset=spec.name,
            spec=spec.as_dict(),
            original_sha256=original_sha256,
            files=files,
            fingerprint=_fingerprint(spec.name, spec.as_dict(), original_sha256, files),
            reference=reference,
        )

    @classmethod
    def read(cls, path: Path) -> Self:
        """Read a manifest file.

        Raises:
            ValueError: If its fingerprint does not match its contents.
        """
        return cls(**json.loads(path.read_bytes()))

    def write(self, path: Path) -> None:
        payload = {
            'dataset': self.dataset,
            'spec': self.spec,
            'original_sha256': self.original_sha256,
            'files': self.files,
            'fingerprint': self.fingerprint,
            'reference': self.reference,
        }
        path.write_text(json.dumps(payload, indent=4) + '\n')

    def verify(self, directory: Path) -> None:
        """Require the bundle directory to hold exactly the files this manifest describes.

        Raises:
            ValueError: If the spec differs from the one shipped with this version of pseval, or
                a file is missing or changed.
        """
        if self.dataset not in SPECS or self.spec != SPECS[self.dataset].as_dict():
            raise ValueError(f'The {self.dataset} bundle was built from another dataset spec')
        recorded = self.files | {
            f'reference {name}': value for name, value in (self.reference or {}).items()
        }
        paths = {name: directory / file for name, file in PROTOCOL_FILES.items()} | {
            f'reference {name}': directory / file for name, file in REFERENCE_FILES.items()
        }
        changed = [
            name
            for name, value in recorded.items()
            if not paths[name].is_file() or sha256(paths[name]) != value
        ]
        if changed:
            raise ValueError(
                f'The {self.dataset} bundle at {directory} is missing or has changed: '
                f'{", ".join(changed)}'
            )


def sha256(path: Path) -> str:
    """Return the SHA-256 digest of a file's bytes."""
    with path.open('rb') as file:
        return hashlib.file_digest(file, 'sha256').hexdigest()


def _fingerprint(
    dataset: str, spec: dict[str, object], original_sha256: str, files: dict[str, str]
) -> str:
    contents = {
        'dataset': dataset,
        'spec': spec,
        'original_sha256': original_sha256,
        'files': files,
    }
    # Sorted keys and no whitespace, so the same contents always hash the same.
    canonical = json.dumps(contents, sort_keys=True, separators=(',', ':'), ensure_ascii=False)
    return hashlib.sha256(canonical.encode()).hexdigest()
