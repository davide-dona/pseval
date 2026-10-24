import multiprocessing
import os
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import torch
from tqdm import tqdm

from pseval.benchmark import Benchmark
from pseval.data import Log
from pseval.generation.store import GenerationsReader
from pseval.metrics.scorer import Scorer
from pseval.record import Record, digest
from pseval.scoring.store import ScoresWriter

# The most scoring processes started when none are asked for. Each opens its own CUDA context on
# the shared device and gets an equal share of its memory.
MAX_DEFAULT_WORKERS = 8


@dataclass(frozen=True)
class _Worker:
    """What one pool process holds for the whole of its life."""

    generations: GenerationsReader
    num_samples: int
    test: Log
    scorer: Scorer


# Set by `_init_worker` in each pool process and read by `_score_block` there.
_worker: _Worker


def _init_worker(
    generations: Path, num_samples: int, test: Log, scorer: Scorer, processes: int
) -> None:
    global _worker
    # Set a limit on the GPU memory fraction per process, to avoid OOM errors
    torch.cuda.set_per_process_memory_fraction(1 / processes)
    _worker = _Worker(
        generations=GenerationsReader(generations),
        num_samples=num_samples,
        test=test,
        scorer=scorer,
    )


def _score_block(block: int) -> pd.DataFrame:
    """Validate and score one row group, one scores row per prefix."""
    test, forecasts = _worker.test, _worker.generations.block(block)
    keys = pd.DataFrame(
        {
            'case_id': [cut.case_id for cut, _ in forecasts],
            'prefix_len': [cut.prefix_len for cut, _ in forecasts],
            'suffix_len': [len(test.suffix_trace(cut)[0]) for cut, _ in forecasts],
        }
    )
    scores = _worker.scorer.score(test, forecasts, num_samples=_worker.num_samples)
    return pd.concat([keys, scores], axis=1)


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
        workers: Scoring processes, one per CPU up to `MAX_DEFAULT_WORKERS` when None.

    Returns:
        The scores file.

    Raises:
        ValueError: If the generations break the protocol contract. A partial scores file is
            removed.
    """
    path = generations.with_name('scores.parquet') if path is None else path
    with GenerationsReader(generations) as file:
        file.record.require_input('benchmark', benchmark.digest)
        label, blocks, prefixes = file.record.label, file.blocks, file.prefixes
        cuts = file.cuts()
        num_samples = file.num_samples()
    test = benchmark._test()
    if len(set(cuts)) != len(cuts) or set(cuts) != set(test.cuts()):
        raise ValueError('The generations must answer every test prefix exactly once')
    record = Record(
        dataset=benchmark.dataset,
        label=label,
        inputs={'benchmark': benchmark.digest, 'generations': digest(generations)},
    )
    processes = workers if workers is not None else min(os.cpu_count() or 1, MAX_DEFAULT_WORKERS)
    with (
        ScoresWriter(path, record) as writer,
        tqdm(total=prefixes, desc='Scoring', unit='prefix') as progress,
        # Spawned rather than forked, so each process can open its own CUDA context
        ProcessPoolExecutor(
            max_workers=processes,
            mp_context=multiprocessing.get_context('spawn'),
            initializer=_init_worker,
            initargs=(generations, num_samples, test, benchmark.scorer, processes),
        ) as executor,
    ):
        for block in executor.map(_score_block, range(blocks)):
            writer.write(block)
            progress.update(len(block))
    return path
