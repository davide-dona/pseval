# pseval

pseval is the evaluation protocol for probabilistic suffix prediction: given an observed process
prefix, a model draws suffixes, activities with inter-event times, and pseval scores them. It owns
everything that decides what a reported number means: splits, prefixes, vocabulary, length cap,
metrics, aggregation, and the unconditional reference. The model author owns training, features,
logging, selection, figures, tables, and significance tests.

## Layout

| Path | Responsibility |
| --- | --- |
| `src/pseval/data.py` | `Log`, `PrefixBatch`, `SuffixDraws`: the records crossing the model boundary, in log units. |
| `src/pseval/protocol.py` | The `SuffixModel` protocol, the draw count, and the protocol seed. |
| `src/pseval/vocabulary.py` | `UNK` and `ActivityCodes`, spelling each activity as one character. |
| `src/pseval/records.py` | The coded records generated and scored: `Generation`, `Draws`, `DecodedEvents`. |
| `src/pseval/hashing.py` | File digests and the canonical JSON that fingerprints hash. |
| `src/pseval/provenance.py` | Protocol version, `Selection`, and the `Provenance` stored in every file. |
| `src/pseval/benchmark/specs.py` | The `DatasetSpec` of each benchmark. |
| `src/pseval/benchmark/manifest.py` | The bundle layout, its `Manifest`, and the fingerprint. |
| `src/pseval/benchmark/splits.py` | The split file format: `write_split` and `read_split`. |
| `src/pseval/benchmark/bundle.py` | `Benchmark` and `load_benchmark`. |
| `src/pseval/preprocessing/filters.py` | Case filters and the cutoffs they use. |
| `src/pseval/preprocessing/split.py` | Log ordering and the out-of-time split with its first cuts. |
| `src/pseval/preprocessing/pipeline.py` | An original log read, filtered, and split. |
| `src/pseval/preprocessing/build.py` | The bundle build, rebuild, and reference. |
| `src/pseval/generation/generate.py` | `generate` over the test split. |
| `src/pseval/generation/join.py` | `join_draws`: a model's draws joined with their prefixes and observed suffixes. |
| `src/pseval/generation/store.py` | The generations file. |
| `src/pseval/metrics/columns.py` | `PrefixScores`, every scores column in order, `METRIC_KEYS`, and `PROPER_SCORES`. |
| `src/pseval/metrics/definitions/` | One function per metric group, computing its columns of one prefix. |
| `src/pseval/metrics/functions/` | Pure functions the metrics are computed with: sequence distances and scoring rules. |
| `src/pseval/metrics/signature/` | The signature kernel: embedding and Gram matrices, bandwidth fit, CPU and CUDA solvers. |
| `src/pseval/metrics/prepared.py` | `PreparedPrefix`, the per-prefix inputs every metric shares. |
| `src/pseval/metrics/scorer.py` | `Scorer`: the protocol fitted on train, its files, the draw contract, and batch scoring. |
| `src/pseval/scoring/score.py` | `score`: input checks and the scoring processes. |
| `src/pseval/scoring/store.py` | The scores file: `ScoresWriter`, `Scores`, `read_scores`. |
| `src/pseval/scoring/summary.py` | `Summary`: equal-weight means overall and by length. |
| `src/pseval/scoring/skill.py` | `skill` against the unconditional reference. |
| `src/pseval/validation.py` | `validate`: every metric on the validation split. |
| `src/pseval/reference.py` | `Unconditional`, the reference model. |
| `src/pseval/evaluation.py` | `evaluate`: fit, generate, score. |
| `baselines/` | Trained baselines using pseval like any external user. Not part of the wheel; see [`baselines/AGENTS.md`](baselines/AGENTS.md). |

Each module holds one task; split a module before it takes on a second. The public API is
exported from `pseval/__init__.py`. Subpackage initializers hold imports and exports only.

## Protocol Contracts

- Every number pseval reports is sample based: 100 draws per test prefix, equal prefix weight,
  means summed in the scores file's row order, buckets by exact prefix and true suffix length.
  These constants are not configurable.
- A model only ever sees train and validation through `fit` and observed prefixes through
  `sample`. Test cuts are built inside `generate`; observed suffixes are joined after sampling.
  Never hand a model anything past a prefix's last event.
- Draws use the benchmark vocabulary, `UNK` allowed, finite nonnegative inter-event minutes, at
  most `max_suffix_length` events. Observed activities unseen in train are scored as `UNK`.
- Test cuts respect each case's first cut, narrowed on cases crossing the split separation.
- Vocabulary, Declare model, and signature kernel are fitted on the train split only.
- Every generations and scores file carries the protocol version, the benchmark fingerprint,
  the model label and metadata, and the `Selection`. Files of another version or fingerprint are
  refused. `score` also requires every test prefix exactly once, with the benchmark's own prefix
  and observed suffix.
- Any change to a reported number is a major version: bump `PROTOCOL_VERSION` and the package
  major version together, build new bundles, and rerun every baseline. Do not add compatibility
  for files of an earlier version.
- Bundles are fingerprinted from their file bytes. Build and rebuild them with the locked
  dependencies (`uv sync --locked`).

## Coding and Validation

Use Python 3.13 typing, four-space indentation, single quotes, `snake_case` names, `PascalCase`
classes, and lines no longer than 100 characters. Never put a comment on the first line of a file.
Public operations need a summary and `Args`, `Returns`, and `Raises` where they apply. Comments
explain current invariants, never the history of a change. Keep immutable records frozen and
random draws behind explicit seeds.

Use tensor comments in the form `[B, T, D]`. Run `uv run ruff check .` and
`uv run ruff format --check .`. Do not build bundles, generate, or score full benchmarks locally;
verify with small disposable checks under `tmp/` and remove them afterward. Do not add tests unless
Davide asks for them.
