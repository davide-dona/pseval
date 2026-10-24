from pathlib import Path

from tqdm import tqdm

from pseval.benchmark import Benchmark
from pseval.generation.store import GenerationsWriter
from pseval.model import SuffixModel, batch_seed, forecast
from pseval.record import Record


def generate(
    model: SuffixModel,
    benchmark: Benchmark,
    path: Path,
    *,
    label: str,
    num_samples: int = 100,
    seed: int = 42,
    batch_size: int = 256,
) -> Path:
    """Draw suffixes for every test prefix and write the generations file.

    Args:
        model: The trained model.
        benchmark: The benchmark to generate on.
        path: Destination file, replaced if it exists.
        label: The model's name, stored in the record of the generations and their scores.
        num_samples: Draws per prefix, at least two.
        seed: Seed every batch seed is derived from.
        batch_size: Prefixes per `sample` call.

    Returns:
        `path`.

    Raises:
        ValueError: If the draws break the protocol contract, which also asks for at least two
            draws per prefix. The file is removed.
    """
    # Retrieve the test set and sort by prefix length
    test = benchmark._test()
    cuts = sorted(test.cuts(), key=lambda cut: cut.prefix_len)
    # Build the record of the generation file
    record = Record(dataset=benchmark.dataset, label=label, inputs={'benchmark': benchmark.digest})
    scorer = benchmark.scorer

    # Batched writing of the generations
    with GenerationsWriter(path, record) as writer:
        starts = range(0, len(cuts), batch_size)
        for batch, start in enumerate(tqdm(starts, desc='Generating', unit='batch')):
            batch_cuts = cuts[start : start + batch_size]
            forecasts = forecast(
                model, test.prefixes(batch_cuts), num_samples, seed=batch_seed(seed, batch)
            )
            for cut, prefix_forecast in zip(batch_cuts, forecasts, strict=True):
                scorer.require_forecast(cut, prefix_forecast, num_samples=num_samples)
            writer.write(list(zip(batch_cuts, forecasts, strict=True)))
    return path
