# Benchmark Guide

A benchmark is one verified bundle directory per dataset. `Benchmark` is the only way code reads
a bundle, and it verifies every file before exposing any of them.

## Layout

| Module | Owns |
| --- | --- |
| `specs.py` | `DatasetSpec` and the shipped `SPECS`: original log columns and every protocol constant. |
| `manifest.py` | Bundle file names, `Manifest`, file hashes, and the fingerprint. |
| `splits.py` | The split CSV format, its canonical columns, and the test split's `_prefix_start`. |
| `bundle.py` | `Benchmark` and `load_benchmark`, local or from the Hugging Face Hub. |

## Contracts

- The fingerprint hashes the dataset name, its spec, the original log's hash, and every
  `PROTOCOL_FILES` hash. The `REFERENCE_FILES` are hashed in the manifest but sit outside the
  fingerprint, since they are produced from an already fingerprinted bundle.
- A bundle loads only when its recorded spec equals the shipped one and every file matches its
  hash. Changing a `DatasetSpec` field therefore invalidates every published bundle of that
  dataset; do it only together with a rebuild.
- `train` and `validation` are public. The test split stays private: `Benchmark._test` is read
  only by `generation` and `scoring`, and `_reference_scores` only by `skill`. Never expose either
  through a public attribute.
- Splits use the pm4py column names from `pseval.data`; every other column keeps its original log
  name. Only the test split stores `_prefix_start`, which `read_split` turns into the `Log`'s
  first cuts.
- Never edit a bundle by hand. Bundles are written only by `pseval.preprocessing`.
