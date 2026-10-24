# Generation Guide

Generation draws a model's suffixes for every test prefix and writes them to a generations file,
the only artifact a model contributes to its scores.

## Layout

| Module | Owns |
| --- | --- |
| `generate.py` | `generate`: test cuts, batching, batch seeds, and the write loop. |
| `records.py` | `DecodedEvents` and the `Draws` of one prefix, gathered from a model's `SuffixDraws`. |
| `store.py` | The generations Parquet schema, `GenerationWriter`, and the `Generations` reader. |

## Contracts

- A model sees only `Log.prefixes` output, through `pseval.model.draw`. Generation never reads an
  observed suffix: only `Scorer.score` joins draws with them.
- Test prefixes are sorted by prefix length and batched; batch `b` gets `batch_seed(seed, b)`.
- Draws are checked before they are written: names must be in the vocabulary, and
  `Scorer.require_draws` checks the count, at least two, the length cap, and finite nonnegative
  minutes.
  Observed activities unseen in train are named `UNK`; a drawn one is an error.
- The file holds one row per prefix: the key and, per draw, its activity names, dictionary
  encoded, float32 minutes, and truncation flag. Its `Record`, in the schema metadata, names
  the dataset, the model label, and the benchmark digest. The draw count is not stored: it is
  read from the draws. Prefixes and observed suffixes are not stored; scoring reads them from
  the benchmark.
- `GenerationWriter` writes one row group per batch, and scoring parallelizes over row groups. A
  failure inside the writer removes the file.
