# Baselines

The trained baselines reported with pseval: Retrieval, U-ED-Transformer, U-ED-LSTM, and the plain
autoregressive ablation. The unconditional reference lives in `pseval.scoring.reference`.

Baselines use pseval exactly as an external user does: through its public API, fitting on
`Benchmark.train` and `Benchmark.validation`, selecting with `pseval.validate`, and generating and
scoring with `pseval.generate` and `pseval.score`. Nothing under `src/pseval/` may import from
here, and this directory is not part of the wheel.

Their extra dependencies (Hydra, W&B) are the `baselines` dependency group:

```sh
uv sync --group baselines
```

## Layout

| Path | Owns |
| --- | --- |
| `data/features.py` | Derived event features: case elapsed minutes and calendar position. |
| `data/codec.py` | `Codec`, fitted on the train `Log`: activity, resource, feature, and time encodings. |
| `data/dataset.py` | `Events`, `TraceCut`, `TraceDataset` over a `Log`, and `encode_prefixes` over a `PrefixBatch`. |
| `models/embeddings.py` | Event content embedding and sinusoidal position tables. |
| `models/dropout.py` | `SequenceDropout`, the variational dropout every U-ED site uses. |
| `models/transformer.py` | The SuTraN encoder-decoder backbone with its key/value cache. |
| `models/lstm.py` | The stacked LSTM backbone and its compiled variational cell. |
| `models/u_ed.py` | The shared U-ED recipe: heads, loss, and generation over either backbone. |
| `models/retrieval.py` | Retrieval, a `pseval.SuffixModel` on activity names, without torch. |
| `adapter.py` | `TorchSuffixModel`, a U-ED model as a `pseval.SuffixModel`, and seeded draws. |
| `checkpoint.py` | Checkpoint contents, saving, and loading. |
| `training/` | The loss record, trace normalization, and the training loop. |
| `pipelines/` | Hydra entry points, one per stage. |
| `config/` | Hydra configuration: stages, `dataset/`, `model/`, `regime/`, `training/`. |

## Commands

Production commands, for suitable compute only. Never start training, full generation, or
scoring locally.

```sh
uv run python -m baselines.pipelines.train dataset=sepsis model=u_ed_transformer
uv run python -m baselines.pipelines.generate checkpoint=outputs/train/sepsis/u_ed_transformer/<run>/best.pt
uv run python -m baselines.pipelines.retrieval dataset=sepsis
uv run python -m baselines.pipelines.score generations=/path/to/generations.parquet
```

`model` is `u_ed_transformer`, `u_ed_lstm`, or `plain_autoregressive`. Training writes `best.pt`
under `outputs/train/<dataset>/<label>/<run>/`, generation writes `generations.parquet` under the
checkpoint's `generate/<invocation>/`, and scoring writes `scores.parquet` under the generations'
`score/<invocation>/`, printing the means and the skill. Inspect a resolved configuration with
`--cfg job --resolve`.

## Contracts

- Model-side preprocessing belongs here: codecs, scaling, and derived features. Everything is
  fitted on `Benchmark.train`; generation refits the codec from the checkpoint's configuration,
  and refuses a checkpoint whose stored benchmark digest differs from the loaded benchmark's.
- A model's activity index is the name's position in `Benchmark.vocabulary`, `UNK` included, then
  PAD, EOT, and SOS. Only PAD and SOS cannot be drawn; every index below PAD decodes to a name.
- `model.label` names a configuration's runs, output paths, and the label of its generations and
  scores. Changing a model's recipe means a new label.
- Training selects the checkpoint by the minimum validation `signature_kernel_score` from
  `pseval.validate` on one fixed seeded subset of `training.validation_prefixes` validation cuts,
  and reports the teacher-forced loss on `training.validation_pairs` pairs.
- `TorchSuffixModel.sample` draws inside `seeded`, so draws depend on the prefixes and the seed
  alone and leave the training random stream untouched. Generation reads nothing past a prefix.
- Retrieval has nothing to select, so it reads neither validation data nor a checkpoint.

## U-ED Recipe

Both U-ED models share one recipe and differ only in the backbone: `d_model` 64, 4 encoder and 4
decoder layers, SOS first decoder input, MLP heads for activity-logit and time means and
log-variances clamped to [-10, 10], the log-mean-exp activity loss over 20 Gaussian logit draws,
the Gaussian time NLL, per-trace normalization, AdamW weight decay, activity input dropout 0.3,
full-suffix teacher forcing, negative times read as zero, and the per-dataset `regime/`.

- Dropout is variational: `SequenceDropout` draws one Bernoulli mask per sequence and site, shared
  across positions, with p = 0.1, in training and in generation. A generation draws one mask per
  suffix for encoder and decoder alike, so cached and new positions share it. `resample_masks`
  draws the masks at the start of every pass, independently of the module's training mode.
- The transformer masks embeddings, sublayer outputs, feedforward hidden features, and the head
  trunk with `[B, 1, D]` masks. It has no attention-weight dropout.
- The LSTM backbone runs standard stacked layers of width `d_model`, the encoder's final
  per-layer `(h, c)` starting the decoder, with no attention and no position table. Its cell
  masks each gate's input and hidden state, which cuDNN's `nn.LSTM` cannot, and its step is
  compiled with `torch.compile`.
- `uncertainty: null` is the plain autoregressive ablation: no logit variance head, cross-entropy
  activity loss, and no dropout at generation. It is trained and selected like U-ED-Transformer.

Follow the root guide's coding contracts. Verify changes with disposable checks under `tmp/` on
small inputs; pseval's signature kernel runs on CUDA only.
