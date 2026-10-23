# Scoring Guide

Scoring turns a generations file into a scores file, one row per test prefix, and reduces scores
to the numbers a paper reports. It also scores validation prefixes in memory for model selection.

## Layout

| Module | Owns |
| --- | --- |
| `score.py` | `score`: input checks and the spawned worker pool over row groups. |
| `store.py` | The scores Parquet schema, `ScoresWriter`, `Scores`, and `read_scores`. |
| `summary.py` | `Summary`: equal-weight means overall and by prefix and true suffix length. |
| `skill.py` | `skill`: each proper score against the bundle's unconditional reference. |
| `validation.py` | `validate`: in-memory means over every validation prefix, writing no file. |

## Contracts

- `score` rejects a generations file from another bundle, spelled on another vocabulary, or not
  answering every test cut exactly once. Each prefix's draws are checked again with
  `Scorer.require_draws`, since the file may come from outside `generate`.
- Workers are spawned, not forked, so each opens its own CUDA context. Each worker opens its own
  generations file, test split, and scorer, gets `1 / processes` of the GPU memory, and scores one
  row group per task. Results are written in file order.
- The scores file stores `case_id`, `prefix_len`, `suffix_len`, then `METRIC_KEYS` as float64,
  and the generations `Provenance` with the generations file's hash. `read_scores` rejects any
  other schema.
- Every mean weighs prefixes equally and sums with `math.fsum`, so row order never changes a
  reported number.
- `skill` is `1 - mean / reference mean` over `PROPER_SCORES` and requires the same prefixes as
  the reference. Ranking by skill equals ranking by the mean.
- `validate` reads only the validation split. Selection never touches test prefixes.
