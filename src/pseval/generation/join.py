from collections.abc import Sequence

import numpy as np

from pseval.data import Log, PrefixKey, SuffixDraws
from pseval.metrics.scorer import Scorer
from pseval.records import DecodedEvents, Draws, Generation


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
        scorer: The benchmark's fitted protocol, whose codes spell the generations.
        num_samples: How many draws each prefix was asked for.

    Returns:
        One generation per key. Observed activities unseen in train are spelled as `UNK`.

    Raises:
        ValueError: If the draws break the protocol contract.
    """
    if not isinstance(draws, Sequence) or len(draws) != len(keys):
        raise ValueError(f'Expected draws for {len(keys)} prefixes')
    codes = scorer.codes
    result = []
    for key, sample in zip(keys, draws, strict=True):
        if not isinstance(sample, SuffixDraws):
            raise ValueError('A model must return SuffixDraws for every prefix')
        try:
            spelled = [codes.spell(names) for names in sample.activities]
        except KeyError as error:
            raise ValueError(
                f'{key} has a draw outside the vocabulary: {error.args[0]!r}'
            ) from error
        prefix_activities, prefix_minutes = log.prefix_trace(key)
        truth_activities, truth_minutes = log.suffix_trace(key)
        generation = Generation(
            case_id=key[0],
            prefix_activities=codes.spell(prefix_activities, unseen_as_unk=True),
            prefix_inter_event_time_minutes=prefix_minutes,
            samples=Draws.of(
                [
                    DecodedEvents(
                        activities=activities,
                        inter_event_time_minutes=np.asarray(minutes, dtype=np.float64),
                    )
                    for activities, minutes in zip(spelled, sample.minutes, strict=True)
                ],
                truncated=sample.truncated,
            ),
            truth=DecodedEvents(
                activities=codes.spell(truth_activities, unseen_as_unk=True),
                inter_event_time_minutes=truth_minutes,
            ),
        )
        scorer.require_draws(generation, num_samples=num_samples)
        result.append(generation)
    return result
