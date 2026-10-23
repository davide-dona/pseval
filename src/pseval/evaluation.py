from collections.abc import Mapping
from pathlib import Path

import torch

from pseval.benchmark.bundle import Benchmark
from pseval.generation.generate import generate
from pseval.protocol import SuffixModel
from pseval.provenance import Selection
from pseval.scoring.score import score


def evaluate(
    model: SuffixModel,
    benchmark: Benchmark,
    directory: Path,
    *,
    selection: Selection,
    label: str,
    metadata: Mapping[str, object] | None = None,
    batch_size: int = 256,
    device: torch.device | str = 'cpu',
    workers: int | None = None,
) -> Path:
    """Fit a model on train and validation, generate on test, and score.

    Writes `generations.parquet` and `scores.parquet` into `directory`. See `generate` and
    `score` for the arguments.

    Returns:
        The scores file.
    """
    model.fit(benchmark.train, benchmark.validation)
    directory.mkdir(parents=True, exist_ok=True)
    generations = generate(
        model,
        benchmark,
        directory / 'generations.parquet',
        selection=selection,
        label=label,
        metadata=metadata,
        batch_size=batch_size,
    )
    return score(generations, benchmark, device=device, workers=workers)
