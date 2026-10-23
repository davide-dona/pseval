from collections.abc import Mapping
from pathlib import Path

from tqdm import tqdm

from pseval.benchmark import Benchmark
from pseval.generation.join import join_draws
from pseval.generation.provenance import Provenance, Selection
from pseval.generation.store import GenerationWriter
from pseval.model import SuffixModel, batch_seed


def generate(
    model: SuffixModel,
    benchmark: Benchmark,
    path: Path,
    *,
    selection: Selection,
    label: str,
    metadata: Mapping[str, object] | None = None,
    num_samples: int = 100,
    seed: int = 42,
    batch_size: int = 256,
) -> Path:
    """Draw suffixes for every test prefix and write the generations file.

    The test cuts are built here, and the model receives only their observed events through
    `sample`; the observed suffixes are joined after sampling. Prefixes are batched by ascending
    prefix length, each batch with its own seed derived from `seed`, so the batch size changes
    nothing but which seed a prefix's draws come from. The draw count and the seed are stored with
    the generations and their scores.

    Args:
        model: The trained model.
        benchmark: The benchmark to generate on.
        path: Destination file, replaced if it exists.
        selection: How the model was selected on validation.
        label: The model's name, stored with the generations and their scores.
        metadata: Anything the author records with the model, as a JSON object, such as a
            checkpoint hash or a configuration.
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
    provenance = Provenance(
        dataset=benchmark.dataset,
        fingerprint=benchmark.fingerprint,
        label=label,
        selection=selection,
        num_samples=num_samples,
        seed=seed,
        metadata=dict(metadata or {}),
    )
    scorer = benchmark.scorer
    with GenerationWriter(path, provenance) as writer:
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
