# Preprocessing Guide

Preprocessing turns an original log into a benchmark bundle. It is the maintainer's tool, not
part of the model author's workflow, and the only code that writes bundle files.

## Layout

| Module | Owns |
| --- | --- |
| `filters.py` | Case duration and length measures and the case filters. |
| `split.py` | Log sorting, inter-event minutes, and the out-of-time split with its first cuts. |
| `pipeline.py` | Reading an original log by its spec and the full filter, sort, and split order. |
| `reference.py` | `Unconditional`, the reference model whose scores every bundle stores. |
| `build.py` | `build_protocol`, `add_reference`, `build`, and the reproducibility check `rebuild`. |

## Contracts

- `split_log` fixes the order: outlier percentiles of duration and length on the log as read,
  then the late-start filter on what they leave, then sorting, inter-event minutes, and the
  split. Reordering these steps changes which cases survive.
- The split is out of time. Train keeps cases finished before the separation; test keeps cases
  still running at it. A test case crossing the separation keeps its earlier events, and its
  first cut is the first prefix reaching past the separation. Validation is a seeded random
  sample of whole train cases.
- Everything fitted, the vocabulary, the Declare model, and the signature kernel calibration,
  is fitted on train alone through `Scorer.fit`.
- A build must be deterministic: stable sorts, seeds from the spec, and canonical outputs.
  `rebuild` checks that rebuilding from the original log reproduces the fingerprint, so any
  nondeterminism surfaces there.
- `Unconditional` uses the public `SuffixModel` interface and is generated and scored through
  `generate` and `score`, exactly like any other model.
