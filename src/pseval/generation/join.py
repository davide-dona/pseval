from collections.abc import Container, Iterable, Sequence

import numpy as np

from pseval.data import Log, PrefixKey, SuffixDraws
from pseval.generation.records import DecodedEvents, Draws, Generation
from pseval.metrics.scorer import UNK, Scorer


def observe(log: Log, key: PrefixKey, draws: Draws, *, vocabulary: Container[str]) -> Generation:
    """Join one prefix's draws with its prefix and observed suffix from the split.

    Observed activities outside `vocabulary`, unseen in train, are named `UNK`.
    """
    observed_activities, observed_minutes = log.suffix_trace(key)
    return Generation(
        case_id=key[0],
        prefix_activities=_known(log.prefix_activities(key), vocabulary),
        draws=draws,
        observed=DecodedEvents(
            activities=_known(observed_activities, vocabulary),
            inter_event_time_minutes=observed_minutes,
        ),
    )


def _known(names: Iterable[str], vocabulary: Container[str]) -> tuple[str, ...]:
    return tuple(name if name in vocabulary else UNK for name in names)


def join_draws(
    log: Log,
    keys: Sequence[PrefixKey],
    draws: Sequence[SuffixDraws],
    *,
    scorer: Scorer,
    num_samples: int,
) -> list[Generation]:
    """Join a batch's draws with their prefixes and observed suffixes.

    Args:
        log: The split the prefixes were cut from.
        keys: The prefixes, in the order they were sampled.
        draws: The model's draws, one entry per key.
        scorer: The benchmark's fitted scorer, whose vocabulary the draws are named from.
        num_samples: How many draws each prefix was asked for.

    Returns:
        One generation per key.

    Raises:
        ValueError: If the draws break the protocol contract.
    """
    if not isinstance(draws, Sequence) or len(draws) != len(keys):
        raise ValueError(f'Expected draws for {len(keys)} prefixes')
    result = []
    for key, suffix_draws in zip(keys, draws, strict=True):
        if not isinstance(suffix_draws, SuffixDraws):
            raise ValueError('A model must return SuffixDraws for every prefix')
        drawn = Draws.of(
            [
                DecodedEvents(
                    activities=tuple(names),
                    inter_event_time_minutes=np.asarray(minutes, dtype=np.float64),
                )
                for names, minutes in zip(
                    suffix_draws.activities, suffix_draws.minutes, strict=True
                )
            ],
            truncated=suffix_draws.truncated,
        )
        generation = observe(log, key, drawn, vocabulary=scorer.axes.keys())
        scorer.require_draws(generation, num_samples=num_samples)
        result.append(generation)
    return result
