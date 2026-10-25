# Baselines

The trained baselines reported with pseval: Retrieval, U-ED-Transformer, U-ED-LSTM, and the
plain Transformer ablation. The unconditional reference lives in `pseval.scoring.reference`.

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
| `data/codec.py` | `Codec`, fitted on a benchmark's train split: activity, resource, feature, and time encodings, without torch. |
| `data/dataset.py` | `Events`, `TraceCut`, `TraceDataset` and its `batches` loader over a `Log`, and `encode_prefixes` over a `PrefixBatch`. |
| `models/` | What draws suffixes. |
| `models/retrieval.py` | Retrieval, a `pseval.SuffixModel` on activity names, without torch. |
| `models/autoregressive.py` | `AutoregressiveSuffixModel`, the shared base: backbone, heads, time loss, and the generation loop, and `build_model`. |
| `models/u_ed.py` | `UncertaintyAwareEncoderDecoder`: Gaussian activity logits and dropout in generation. |
| `models/ed.py` | `EncoderDecoder`: softmax activities from one deterministic network in generation. |
| `backbones/transformer.py` | The SuTraN encoder-decoder backbone with its key/value cache. |
| `backbones/lstm.py` | The stacked LSTM backbone and its compiled gate-dropout cell. |
| `modules/embeddings.py` | Event content embedding, each stack's `EmbeddingInput`, and sinusoidal position tables. |
| `modules/dropout.py` | `SequenceDropout`, the dropout schemes, and the mask resampling every site shares. |
| `runtime/adapter.py` | `TorchSuffixModel`, an autoregressive model as a `pseval.SuffixModel`, and seeded draws. |
| `runtime/checkpoint.py` | Checkpoint contents, saving, loading, and `restore_model`. |
| `training/` | The loss record, trace normalization, and the training loop with its W&B logging. |
| `pipelines/` | Hydra entry points, one per stage. |
| `config/` | Hydra configuration: stages, `dataset/` with each dataset's regime, and `model/` with `_sizes.yaml`, `backbone/`, and one file per model. |

## Commands

Production commands, for suitable compute only. Never start training, full generation, or
scoring locally.

```sh
uv run python -m baselines.pipelines.train dataset=sepsis model=u_ed_transformer
uv run python -m baselines.pipelines.generate train_dir=outputs/train/sepsis/u_ed_transformer/<run>
uv run python -m baselines.pipelines.retrieval data.name=sepsis
uv run python -m baselines.pipelines.score generate_dir=outputs/generate/sepsis/retrieval/<run>
```

`model` is `u_ed_transformer`, `u_ed_lstm`, or `transformer`. Training writes `best.pt`
under `outputs/train/<dataset>/<label>/<run>/`, generation reads a training run's `best.pt` and
writes `generations.parquet` under its `generate/<invocation>/`, and scoring reads a generation
run's `generations.parquet` and writes `scores.parquet` under its `score/<invocation>/`, printing
the means and the skill. Inspect a resolved configuration with
`--cfg job --resolve`.

## Contracts

- Model-side preprocessing belongs here: codecs, scaling, and derived features. Everything is
  fitted on `Benchmark.train`. `restore_model` refits the codec from the checkpoint's
  configuration, and refuses a benchmark whose digest differs from the checkpoint's.
- Row 0 of every embedding table is PAD. A model's activity rows are PAD, EOT, and SOS, then each
  name of `Benchmark.vocabulary` in order, `UNK` included. Only PAD and SOS cannot be drawn; every
  row from `ActivityColumn.first_name` on decodes to a name.
- `TraceDataset` reads whole batches of cuts by vectorized gathers over the encoded split, each
  prefix and suffix padded to the longest of its batch, and `batches` loads them into pinned
  memory.
- `model.label` names a configuration's runs, output paths, and the label of its generations and
  scores. Changing a model's recipe means a new label.
- Training selects the checkpoint by the minimum validation `signature_kernel_score` from
  `pseval.validate` on its seeded subset of `training.validation_prefixes` validation prefixes,
  and reports the teacher-forced loss on `training.validation_pairs` pairs.
- `TorchSuffixModel.sample` draws inside `seeded`, so draws depend on the prefixes and the seed
  alone and leave the training random stream untouched. Generation reads nothing past a prefix.
- Retrieval has nothing to select, so it reads neither validation data nor a checkpoint.

## Autoregressive Recipe

The three autoregressive baselines share `AutoregressiveSuffixModel`: `d_model` 64, 4 encoder and
4 decoder layers, SOS first decoder input, an MLP trunk shared by the heads, time means and
log-variances clamped to [-10, 10], the Gaussian time NLL, per-trace normalization, AdamW weight
decay, activity input dropout 0.3, full-suffix teacher forcing, negative times read as zero, and
the per-dataset regime of `dataset/` over the defaults of `train.yaml`. `model/_sizes.yaml` holds
the widths they share, and each `model/backbone/` file a backbone's `_target_` and its own depth.
Each model's file picks its backbone in its defaults, names its class by `_target_`, which
`build_model` instantiates, and states its own recipe. The base builds the backbone its
`_target_` names, every backbone taking the same arguments.

- `UncertaintyAwareEncoderDecoder`, U-ED, adds activity-logit log-variances and the log-mean-exp
  activity loss over `logit_samples` (20) Gaussian logit draws on real suffix positions, and keeps
  dropout active in generation. U-ED-Transformer and U-ED-LSTM differ only in their backbone.
- `EncoderDecoder` is a plain autoregressive encoder-decoder: softmax logits trained by
  cross-entropy, standard dropout, and one deterministic network that encodes each prefix once
  for all its draws. The Transformer, `EncoderDecoder` over the transformer backbone, is trained
  and selected like U-ED-Transformer.

- Each model class passes its `DropoutScheme` to the base, which hands it to every site, so no
  configuration field chooses it. U-ED's is `variational_dropout`: `SequenceDropout` draws one
  Bernoulli mask per sequence and site, shared across positions, with p = 0.1, in training and in
  generation, and there is no attention-weight dropout. `EncoderDecoder`'s is `standard_dropout`:
  `nn.Dropout` at the same sites and on the attention weights, with p = 0.1, in training mode
  only.
- A U-ED generation draws one mask per suffix for encoder and decoder alike, so cached and new
  positions share it. `resample_masks` draws the masks at the start of every pass, independently
  of the module's training mode, and generation clears them when it ends.
- Generation stops decoding finished rows once they are half the rows, selecting the live rows'
  backbone state and dropout masks, and checks the rows still writing only every `CHECK_EVERY`
  steps, since the check waits for the device.
- The transformer drops embeddings, sublayer outputs, feedforward hidden features, and the head
  trunk, and its attention weights at the scheme's `attention` rate.
- The LSTM backbone runs standard stacked layers of width `d_model`, the encoder's final
  per-layer `(h, c)` starting the decoder, with no attention and no position table. It runs a
  layer at a time, each layer's input projection one product over every position. Its cell drops
  each gate's input and hidden state, which cuDNN's `nn.LSTM` cannot, and its recurrent step is
  compiled with `torch.compile`.

Follow the root guide's coding contracts. Verify changes with disposable checks under `tmp/` on
small inputs; pseval's signature kernel runs on CUDA only.
