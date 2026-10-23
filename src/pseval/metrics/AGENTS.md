# Metrics Guide

Metrics score each forecast of a given prefix, represented by its draws, against the single
observed suffix. `Scorer` scores batches of `Generation` records in memory; file handling
belongs to `generation` and `scoring`. In shape comments, `K` counts a prefix's draws.

## Layout

| Module | Owns |
| --- | --- |
| `scorer.py` | `Scorer`: fit, save, load, `require_draws`, and batch scoring. |
| `columns.py` | `PrefixScores`, `METRIC_KEYS`, `PROPER_SCORES`, and `MIN_DRAWS`. |
| `prepared.py` | `PreparedPrefix`: values every metric shares, built one batch at a time. |
| `definitions/` | One function per metric group, mapping a `PreparedPrefix` to its columns. |
| `functions/` | Pure numeric helpers: scoring rules, randomized PIT, and sequence distances. |
| `signature.py` | `SignatureKernel`: path embedding, train calibration, and batched Gram matrices. |
| `vocabulary.py` | `ActivityCodes`, spelling each activity as one private-use character, and `UNK`. |

## Contracts

- `PrefixScores` field order is the scores file column order. Adding a metric means a commented
  field there and an entry in one `definitions/` function; it changes the scores schema, so older
  scores files stop loading.
- Every scored prefix has at least `MIN_DRAWS` draws, so neither definitions nor helpers guard
  smaller draw counts.
- A `definitions/` function unpacks its inputs and converts units at the top, names only values
  that several columns use, and returns one expression per column in `PrefixScores` order.
- `PROPER_SCORES` are the fair scoring rules models are ranked on, lower is better. Every other
  column is a diagnostic and never ranks.
- `PreparedPrefix.of_batch` makes one Declare checker call and one signature kernel call per
  batch, since both have a fixed cost per call. Score batches, not single prefixes.
- Compute CRPS and its spread from gaps between sorted draws with nonnegative weights. Never as
  accuracy minus spread, which cancels catastrophically under extreme draws.
- Inter-event times have no metric of their own, since aligning draws by position to the observed
  suffix is not proper. The signature kernel score covers activities and times jointly.
- The PIT noise is seeded from the prefix key alone, so a score never depends on the batch or the
  worker that computed it.
- pySigLib solves the signature kernel on the current CUDA device, in float32. Pairs of paths
  with different lengths share a solver call by repeating each path's last point.
