from pathlib import Path

import torch
from omegaconf import OmegaConf
from torch import nn

import pseval
from baselines.data.codec import Codec
from baselines.models.autoregressive import AutoregressiveSuffixModel, build_model

# What every checkpoint holds: the resolved run configuration, the digest of the benchmark it was
# trained on, the selected weights, and the step and validation score they were selected at.
CHECKPOINT_KEYS = ('config', 'benchmark', 'model_state_dict', 'step', 'selection_score')


def save_checkpoint(
    model: nn.Module,
    *,
    config: dict,
    benchmark: str,
    step: int,
    selection_score: float,
    path: Path,
) -> Path:
    """Save the model weights and run metadata to `path`, replacing it atomically."""
    temp = path.with_suffix('.pt.tmp')
    torch.save(
        obj={
            'config': config,
            'benchmark': benchmark,
            'model_state_dict': model.state_dict(),
            'step': step,
            'selection_score': selection_score,
        },
        f=temp,
    )
    temp.replace(path)
    return path


def load_checkpoint(path: Path) -> dict:
    """Load a checkpoint on the CPU.

    Raises:
        ValueError: If it lacks a `CHECKPOINT_KEYS` entry.
    """
    checkpoint = torch.load(f=path, map_location='cpu', weights_only=True)
    missing = [key for key in CHECKPOINT_KEYS if key not in checkpoint]
    if missing:
        raise ValueError(f'{path} is missing {", ".join(missing)}; train the model again')
    return checkpoint


def restore_model(
    checkpoint: dict, benchmark: pseval.Benchmark
) -> tuple[AutoregressiveSuffixModel, Codec]:
    """Rebuild a checkpoint's model, refitting its codec on the benchmark it was trained on.

    Raises:
        ValueError: If `benchmark` is not the one the checkpoint was trained on, whose codec the
            weights expect.
    """
    if benchmark.digest != checkpoint['benchmark']:
        raise ValueError(
            f'The checkpoint was trained on another benchmark than {benchmark.dataset}'
        )
    config = OmegaConf.create(checkpoint['config'])
    codec = Codec.fit(benchmark, config.data)
    model = build_model(config.model, codec)
    model.load_state_dict(checkpoint['model_state_dict'])
    return model, codec
