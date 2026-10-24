from pathlib import Path

import numpy as np
import torch
from torch import nn

# What every checkpoint holds: the resolved run configuration, the digest of the benchmark it was
# trained on, the selected weights, and the step and validation score they were selected at.
CHECKPOINT_KEYS = ('config', 'benchmark', 'model_state_dict', 'step', 'selection_score')

# Checkpoints may store NumPy float64 metadata, which weights-only unpickling rejects by default.
_SAFE_GLOBALS = [np._core.multiarray.scalar, np.dtype, np.dtypes.Float64DType]


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
    with torch.serialization.safe_globals(_SAFE_GLOBALS):
        checkpoint = torch.load(f=path, map_location='cpu', weights_only=True)
    missing = [key for key in CHECKPOINT_KEYS if key not in checkpoint]
    if missing:
        raise ValueError(f'{path} is missing {", ".join(missing)}; train the model again')
    return checkpoint
