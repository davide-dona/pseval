# Baselines

The trained baselines reported with pseval: Retrieval, U-ED-Transformer, and U-ED-LSTM. The
unconditional reference lives in `pseval.reference`.

Baselines use pseval exactly as an external user does: through its public API, fitting on
`Benchmark.train` and `Benchmark.validation`, selecting with `pseval.validate`, and generating and
scoring with `pseval.generate` and `pseval.score`. Nothing under `src/pseval/` may import from
here, and this directory is not part of the wheel.

Their extra dependencies (Hydra, W&B) are the `baselines` dependency group:

```sh
uv sync --group baselines
```

Model-side preprocessing belongs here: codecs, scaling, and derived features such as calendar
position and case elapsed time. Follow the root guide's coding contracts, and never start training
locally.
