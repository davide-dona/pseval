import multiprocessing
import os
from collections.abc import Iterator
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from pseval.benchmark.bundle import Benchmark
from pseval.data import Log
from pseval.generation.store import Generations
from pseval.hashing import sha256
from pseval.metrics.scorer import Scorer
from pseval.protocol import NUM_SAMPLES
from pseval.records import Generation
from pseval.scoring.store import ScoresWriter
from pseval.vocabulary import ActivityCodes


def _require_observed(generation: Generation, log: Log, codes: ActivityCodes) -> None:
    """Require a stored prefix and observed suffix to be the split's, as stored in float32.

    Raises:
        ValueError: If they differ.
    """
    key = (generation.case_id, generation.prefix_len)
    for stored_activities, stored_minutes, (names, minutes) in (
        (
            generation.prefix_activities,
            generation.prefix_inter_event_time_minutes,
            log.prefix_trace(key),
        ),
        (
            generation.truth.activities,
            generation.truth.inter_event_time_minutes,
            log.suffix_trace(key),
        ),
    ):
        if stored_activities != codes.spell(names, unseen_as_unk=True) or not np.array_equal(
            stored_minutes, minutes.astype(np.float32).astype(np.float64)
        ):
            raise ValueError(f'The observed events stored for {key} are not the test split')


@dataclass(frozen=True)
class _Worker:
    """What one pool process holds for the whole of its life."""

    generations: Generations
    test: Log
    scorer: Scorer


# Set by `_init_worker` in each pool process and read by `_score_block` there.
_worker: _Worker


def _init_worker(generations: Path, directory: Path, device: torch.device, processes: int) -> None:
    global _worker
    # Each process caches device memory its peers cannot reclaim, so each gets an equal share.
    if device.type == 'cuda':
        torch.cuda.set_per_process_memory_fraction(1 / processes, device)
    benchmark = Benchmark(directory)
    _worker = _Worker(
        generations=Generations(generations),
        test=benchmark._test(),
        scorer=benchmark.scorer(device),
    )


def _score_block(block: int) -> pd.DataFrame:
    """Validate and score one row group, one scores row per prefix."""
    generations = _worker.generations.block(block)
    scorer = _worker.scorer
    for generation in generations:
        _require_observed(generation, _worker.test, scorer.codes)
        scorer.require_draws(generation, num_samples=NUM_SAMPLES)
    keys = pd.DataFrame(
        {
            'case_id': [generation.case_id for generation in generations],
            'prefix_len': [generation.prefix_len for generation in generations],
            'suffix_len': [len(generation.truth) for generation in generations],
        }
    )
    return pd.concat([keys, scorer.score(generations)], axis=1)


def _scored_blocks(
    generations: Path, benchmark: Benchmark, *, device: torch.device, blocks: int, processes: int
) -> Iterator[pd.DataFrame]:
    """Score every row group across spawned processes, yielding them in file order.

    Spawned rather than forked, so each process can open its own CUDA context on `device`.
    """
    with ProcessPoolExecutor(
        max_workers=processes,
        mp_context=multiprocessing.get_context('spawn'),
        initializer=_init_worker,
        initargs=(generations, benchmark.directory, device, processes),
    ) as executor:
        yield from executor.map(_score_block, range(blocks))


def score(
    generations: Path,
    benchmark: Benchmark,
    *,
    path: Path | None = None,
    device: torch.device | str = 'cpu',
    workers: int | None = None,
) -> Path:
    """Score a generations file and write the scores file.

    The file must answer every test prefix exactly once with the protocol's draw count, hold the
    benchmark's own prefixes and observed suffixes, and come from this benchmark and protocol
    version. Each row group is scored in a worker process; the scores file holds one row per
    prefix in file order, with every metric.

    Args:
        generations: The generations file, from `pseval.generate` or written to its format.
        benchmark: The benchmark it was generated on.
        path: Destination, `scores.parquet` beside the generations by default, replaced if it
            exists.
        device: Where the signature kernel is solved, shared by every worker.
        workers: Scoring processes, one per CPU when None.

    Returns:
        The scores file.

    Raises:
        ValueError: If the generations break the protocol contract. A partial scores file is
            removed.
    """
    device = torch.device(device)
    path = generations.with_name('scores.parquet') if path is None else path
    with Generations(generations) as file:
        provenance = file.provenance
        vocabulary = file.codes.vocabulary
        blocks, prefixes = file.blocks, file.prefixes
        keys = file.prefix_keys()
    provenance.require_benchmark(benchmark.dataset, benchmark.fingerprint)
    if vocabulary != benchmark.vocabulary:
        raise ValueError('The generations are spelled on another activity vocabulary')
    if len(set(keys)) != len(keys) or set(keys) != set(benchmark._test().cuts()):
        raise ValueError('The generations must answer every test prefix exactly once')
    processes = workers if workers is not None else os.cpu_count() or 1
    with (
        ScoresWriter(path, provenance.scored(sha256(generations))) as writer,
        tqdm(total=prefixes, desc='Scoring', unit='prefix') as progress,
    ):
        for block in _scored_blocks(
            generations, benchmark, device=device, blocks=blocks, processes=processes
        ):
            writer.write(block)
            progress.update(len(block))
    return path
