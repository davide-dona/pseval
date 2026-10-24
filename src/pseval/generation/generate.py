from pathlib import Path

from tqdm import tqdm

from pseval.benchmark import Benchmark
from pseval.generation.join import join_draws
from pseval.generation.store import GenerationWriter
from pseval.model import SuffixModel, batch_seed
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

    The test cuts are built here, and the model receives only their observed events through
    `sample`; the observed suffixes are joined after sampling. Prefixes are batched by ascending
    prefix length, each batch with its own seed derived from `seed`, so the batch size changes
    nothing but which seed a prefix's draws come from.

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
    test = benchmark._test()
    keys = sorted(test.cuts(), key=lambda key: key[1])
    record = Record(dataset=benchmark.dataset, label=label, inputs={'benchmark': benchmark.digest})
    scorer = benchmark.scorer
    with GenerationWriter(path, record) as writer:
        starts = range(0, len(keys), batch_size)
        for batch, start in enumerate(tqdm(starts, desc='Generating', unit='batch')):
            batch_keys = keys[start : start + batch_size]
            draws = model.sample(
                test.prefixes(batch_keys), num_samples, seed=batch_seed(seed, batch)
            )
            writer.write(
                join_draws(
                    test,
                    batch_keys,
                    draws,
                    scorer=scorer,
                    num_samples=num_samples,
                )
            )
    return path
