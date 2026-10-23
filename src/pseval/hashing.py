import hashlib
import json
import re
from pathlib import Path

_SHA256 = re.compile(r'[0-9a-f]{64}')


def sha256(path: Path) -> str:
    """Return the SHA-256 digest of a file's bytes."""
    with path.open('rb') as file:
        return hashlib.file_digest(file, 'sha256').hexdigest()


def canonical_json(value: object) -> bytes:
    """Encode a JSON value with sorted keys and no whitespace, the form fingerprints hash."""
    return json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False).encode()


def validate_sha256(value: object, field_name: str) -> str:
    """Require a lowercase hexadecimal SHA-256 digest.

    Raises:
        ValueError: If `value` is not one.
    """
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ValueError(f'Invalid {field_name}: {value!r}')
    return value
