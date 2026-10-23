# pseval

pseval is the evaluation protocol for probabilistic suffix prediction: a model draws suffixes for
observed process prefixes, and pseval scores them, owning everything that decides what a reported
number means, from splits to metrics. Training and selection belong to the model author.

## Safety and Correctness

- Do not add tests unless Davide asks for them.
- Never expose a model to anything past a prefix's last event, and fit protocol state on the train
  split only.
- Keep each module specialized and small, and avoid cross-cutting concerns.
- Use Python 3.13 typing, four-space indentation, single quotes, 100-character lines, and no
  comment on the first line of a file. Run `uv run ruff check .` and `uv run ruff format --check .`.

## Task Routing

| Task | Area |
| --- | --- |
| Model boundary: `Log`, `PrefixBatch`, `SuffixDraws`, `SuffixModel` | `src/pseval/data.py`, `src/pseval/protocol.py` |
| Dataset specs, bundle layout, manifests, fingerprints, and loading | `src/pseval/benchmark/` |
| Log filtering, the out-of-time split, bundle builds, and the reference | `src/pseval/preprocessing/` |
| Test-split sampling, the generations file, and its provenance | `src/pseval/generation/` |
| Metric definitions, vocabulary, Declare checks, and the signature kernel | `src/pseval/metrics/` |
| The scores file, summaries, skill, and validation | `src/pseval/scoring/` |
| Trained baselines using the public API | [`baselines/AGENTS.md`](baselines/AGENTS.md) |

See [`README.md`](README.md) for the user-facing workflow.
