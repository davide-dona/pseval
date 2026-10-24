# Preprocessing Guide

Preprocessing turns an original log into a benchmark directory. It is the maintainer's tool, not
part of the model author's workflow, and the only code that writes benchmark files.

## Layout

| Module | Owns |
| --- | --- |
| `specs.py` | `DatasetSpec`, its JSON form: original log columns and every protocol constant. |
| `filters.py` | Case duration and length measures and the case filters. |
| `split.py` | Log sorting, inter-event minutes, and the out-of-time split with its first cuts. |
| `pipeline.py` | Reading an original log by its spec and the full filter, sort, and split order. |
| `build.py` | `build`, the only way to make a benchmark. |

The benchmark directory format, `FILES`, its digest, and loading live in
`pseval/benchmark.py`, which knows nothing of the specs.

A spec lives only in its benchmark, as `spec.json` with every field written, so the Hub is its
one source. To change the protocol, edit the spec of the current release's benchmark and build
from it.

## Benchmark Directory

- A benchmark is one directory per dataset, named after it: the `FILES` (three Parquet splits,
  `scorer.json`, `declare_hard.json`, `declare_soft.json`, and `spec.json`) and
  `reference_scores.parquet`.
- Every split stores the benchmark's `Record` in its metadata: the dataset and the original
  log's digest. The splits must hold the same record, and `load_benchmark` requires its dataset
  to be the directory's name.
- The digest is computed from the `FILES` at load and is the input every generations and
  scores record names. The reference scores sit outside it, since they are scored on the
  benchmark; their record names the digest, and `skill` requires it. Their generations are
  not kept, so the generations digest they record cannot be checked.
- A benchmark is never patched. Any change, a spec field, a metric, or a filter, means
  rebuilding every benchmark with `build`, which also invalidates every scores file.
- `train` and `validation` are public. The test split stays private: `Benchmark._test` is read
  only by `generation` and `scoring`, and `_reference_scores` only by `skill`. Never expose
  either through a public attribute.
- Splits use the pm4py column names from `pseval.data`; every other column keeps its original log
  name. Only the test split stores `_prefix_start`, which `read_split` turns into the `Log`'s
  first cuts.
- The Hub holds one tag per pseval release, and `load_benchmark` downloads the tag of the
  installed version.

## Contracts

- `build` refuses an original log whose hash differs from its spec's `original_sha256`.
- `split_log` fixes the order: outlier percentiles of duration and length on the log as read,
  then the late-start filter on what they leave, then sorting, inter-event minutes, and the
  split. Reordering these steps changes which cases survive.
- The split is out of time. Train keeps cases finished before the separation; test keeps cases
  still running at it. A test case crossing the separation keeps its earlier events, and its
  first cut is the first prefix reaching past the separation. An earlier prefix ends before
  train does, so the model would predict it with training data from its future; every scored
  prefix ends after all of train.
- Validation is a seeded random sample of whole train cases, not a time slice: a slice ending at
  the separation would lose every long case starting inside it to test, so its prefixes would
  face far shorter remaining times than test's.
- Everything fitted, the vocabulary, both Declare models, and the signature kernel calibration,
  is fitted on train alone through `Scorer.fit`.
- A build must be deterministic: stable sorts, seeds from the spec, and canonical outputs. To
  check a benchmark, build from its `spec.json` and original log into a scratch root and compare
  the two digests. The Parquet bytes depend on the locked pandas and pyarrow versions, so build
  in the locked environment.
- The reference model `Unconditional` lives in `pseval.scoring.reference` and is generated and
  scored through `generate` and `score`, exactly like any other model.
