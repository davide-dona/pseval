# pseval

pseval is the evaluation protocol for probabilistic suffix prediction: a model draws suffixes for
observed process prefixes, and pseval scores them, owning everything that decides what a reported
number means, from splits to metrics. Training and selection belong to the model author.

## Safety and Correctness

- Do not build bundles, generate, or score full benchmarks locally. Verify with small, disposable
  checks under `tmp/` and remove them afterward. Do not add tests unless Davide asks for them.
- Never expose a model to anything past a prefix's last event, and fit protocol state on the train
  split only.
- Any change to a reported number is a major version: bump `PROTOCOL_VERSION` and the package
  major version together. Do not add compatibility for files of an earlier version.
- Use Python 3.13 typing, four-space indentation, single quotes, 100-character lines, and no
  comment on the first line of a file. Run `uv run ruff check .` and `uv run ruff format --check .`.

## Task Routing

| Task | Area |
| --- | --- |
| Model boundary: `Log`, `PrefixBatch`, `SuffixDraws`, `SuffixModel` | `src/pseval/data.py`, `src/pseval/protocol.py` |
| File identity, provenance, and fingerprint hashing | `src/pseval/provenance.py` |
| Dataset specs, bundle layout, manifests, and loading | `src/pseval/benchmark/` |
| Log filtering, the out-of-time split, bundle builds, and the reference | `src/pseval/preprocessing/` |
| Test-split sampling and the generations file | `src/pseval/generation/` |
| Metric definitions, vocabulary, Declare checks, and the signature kernel | `src/pseval/metrics/` |
| The scores file, summaries, skill, and validation | `src/pseval/scoring/` |
| Trained baselines using the public API | [`baselines/AGENTS.md`](baselines/AGENTS.md) |

See [`README.md`](README.md) for the user-facing workflow.
