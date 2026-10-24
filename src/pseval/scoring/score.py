import multiprocessing
import os
from collections.abc import Iterator
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import torch
from tqdm import tqdm

from pseval.benchmark import Benchmark
from pseval.data import Log
from pseval.generation.join import observe
from pseval.generation.store import Generations
from pseval.metrics.scorer import Scorer
from pseval.record import Record, digest
from pseval.scoring.store import ScoresWriter


@dataclass(frozen=True)
class _Worker:
    """What one pool process holds for the whole of its life."""

    generations: Generations
    num_samples: int
    test: Log
    scorer: Scorer


# Set by `_init_worker` in each pool process and read by `_score_block` there.
_worker: _Worker


def _init_worker(generations: Path, num_samples: int, directory: Path, processes: int) -> None:
    global _worker
    # Each process caches device memory its peers cannot reclaim, so each gets an equal share.
    torch.cuda.set_per_process_memory_fraction(1 / processes)
    benchmark = Benchmark(directory)
    _worker = _Worker(
        generations=Generations(generations),
        num_samples=num_samples,
        test=benchmark._test(),
        scorer=benchmark.scorer,
    )


def _score_block(block: int) -> pd.DataFrame:
    """Validate and score one row group, one scores row per prefix."""
    scorer = _worker.scorer
    generations = [
        observe(_worker.test, key, draws, vocabulary=scorer.axes.keys())
        for key, draws in _worker.generations.block(block)
    ]
    for generation in generations:
        scorer.require_draws(generation, num_samples=_worker.num_samples)
    keys = pd.DataFrame(
        {
            'case_id': [generation.case_id for generation in generations],
            'prefix_len': [generation.prefix_len for generation in generations],
            'suffix_len': [len(generation.observed) for generation in generations],
        }
    )
    return pd.concat([keys, scorer.score(generations)], axis=1)


def _scored_blocks(
    generations: Path, benchmark: Benchmark, *, num_samples: int, blocks: int, processes: int
) -> Iterator[pd.DataFrame]:
    """Score every row group across spawned processes, yielding them in file order.

    Spawned rather than forked, so each process can open its own CUDA context.
    """
    with ProcessPoolExecutor(
        max_workers=processes,
        mp_context=multiprocessing.get_context('spawn'),
        initializer=_init_worker,
        initargs=(generations, num_samples, benchmark.directory, processes),
    ) as executor:
        yield from executor.map(_score_block, range(blocks))


def score(
    generations: Path,
    benchmark: Benchmark,
    *,
    path: Path | None = None,
    workers: int | None = None,
) -> Path:
    """Score a generations file and write the scores file.

    The file must come from this benchmark and answer every test prefix exactly once, each with
    the same draw count. Prefixes and observed suffixes are read from the benchmark. Each row group
    is scored in a worker process; the scores file holds one row per prefix in file order, with
    every metric, and a record naming the model, the benchmark, and the generations file.

    Args:
        generations: The generations file, from `pseval.generate` or written to its format.
        benchmark: The benchmark it was generated on.
        path: Destination, `scores.parquet` beside the generations by default, replaced if it
            exists.
        workers: Scoring processes, one per CPU when None.

    Returns:
        The scores file.

    Raises:
        ValueError: If the generations break the protocol contract. A partial scores file is
            removed.
    """
    path = generations.with_name('scores.parquet') if path is None else path
    with Generations(generations) as file:
        file.record.require_input('benchmark', benchmark.digest)
        label, blocks, prefixes = file.record.label, file.blocks, file.prefixes
        keys = file.prefix_keys()
        num_samples = file.num_samples()
    if len(set(keys)) != len(keys) or set(keys) != set(benchmark._test().cuts()):
        raise ValueError('The generations must answer every test prefix exactly once')
    record = Record(
        dataset=benchmark.dataset,
        label=label,
        inputs={'benchmark': benchmark.digest, 'generations': digest(generations)},
    )
    processes = workers if workers is not None else os.cpu_count() or 1
    with (
        ScoresWriter(path, record) as writer,
        tqdm(total=prefixes, desc='Scoring', unit='prefix') as progress,
    ):
        for block in _scored_blocks(
            generations, benchmark, num_samples=num_samples, blocks=blocks, processes=processes
        ):
            writer.write(block)
            progress.update(len(block))
    return path
