# pseval

## Objective

pseval is the evaluation protocol for probabilistic suffix prediction. Given an observed process
prefix, a model draws several suffixes, activities with inter-event times, and pseval scores them
against the observed continuation. Models are ranked only by fair, proper scoring rules
estimated from the draws; every other metric is a diagnostic that explains a ranking.

pseval owns everything that decides what a reported number means: splits, prefixes, vocabulary,
length cap, metrics, aggregation, and the unconditional reference. The model author owns
training, features, selection, logging, figures, tables, and significance tests, and composes
the public operations `load_benchmark`, `validate`, `generate`, `score`, and `skill` into their
own pipeline.

## Research Knowledge

Research notes live in the Obsidian vault at `~/Obsidian`. Read the relevant notes before
changing a metric, a split, or the reference.

- [Projects](~/Obsidian/Projects/): `Evaluation Framework for Probabilistic Suffix Prediction`
  explains the research objective and why ranking uses proper scores only.
- [Implementations](~/Obsidian/Implementations/): `Probabilistic Suffix Evaluation` describes the
  metric design.
- [Concepts](~/Obsidian/Concepts/): defines proper scoring rules, the signature kernel, and
  Declare.

The code remains the authority for current contracts.

## Safety and Correctness

- Do not build bundles, generate, or score full benchmarks locally. Verify with small,
  disposable checks under `tmp/` and remove them afterward.
- Hand a model observed prefixes only, through `sample`. Test cuts are built inside `generate`
  and observed suffixes are joined after sampling; never expose anything past a prefix's last
  event.
- Draws use the benchmark vocabulary, `UNK` allowed, finite nonnegative inter-event minutes, and
  at most `max_suffix_length` events. Observed activities unseen in train are scored as `UNK`.
- Test cuts respect each case's first cut, narrowed on cases crossing the split separation.
- Fit the vocabulary, the Declare model, and the signature kernel on the train split only.
- Every reported number is sample based. The draw count, at least ten, and the seed are chosen at
  `generate`. Means take equal prefix weight, summed in the scores file's row order, with buckets
  by exact prefix and true suffix length.
- Every generations and scores file carries the protocol version, the benchmark fingerprint, the
  draw count and seed, the model label and metadata, and the `Selection`. Refuse files of another
  version or fingerprint, and require every test prefix exactly once with the benchmark's own
  prefix and observed suffix.
- Any change to a reported number is a major version: bump `PROTOCOL_VERSION` and the package
  major version together, build new bundles, and rerun every baseline. Do not add compatibility
  for files of an earlier version.
- Bundles are fingerprinted from their file bytes. Build and rebuild them with the locked
  dependencies (`uv sync --locked`).
- Keep each module to one task. The public API is exported from `pseval/__init__.py`;
  subpackage initializers hold imports and exports only.
- Use Python 3.13 typing, four-space indentation, single quotes, 100-character lines,
  `snake_case` names, and `PascalCase` classes. Never put a comment on the first line of a file.
  Public operations need a summary and `Args`, `Returns`, and `Raises` where they apply. Use tensor
  comments in the form `[B, T, D]`. Keep immutable records frozen and random draws behind explicit
  seeds.
- Do not add tests unless Davide asks for them. Use `uv run ruff check .` and
  `uv run ruff format --check .` for lint and format validation.

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
