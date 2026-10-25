import numpy as np
import pandas as pd

import pseval
from pseval.metrics import PROPER_SCORES

ALPHA = 0.05
RESAMPLES = 10_000
SEED = 42
# Bootstrap draws resampled at once.
BATCH_SIZE = 250


def _by_case(models: dict[str, pseval.Scores]) -> tuple[np.ndarray, np.ndarray]:
    """Sum each case's proper scores, `[cases, models, PROPER_SCORES]`, and count its prefixes,
    `[cases]`. Every model scores the same prefixes, as `pseval.skill` checks."""
    keys = ['case_id', 'prefix_len']
    frames = [scores.frame.sort_values(keys) for scores in models.values()]
    values = np.stack([frame[list(PROPER_SCORES)].to_numpy() for frame in frames], axis=1)
    cases = frames[0]['case_id'].to_numpy()
    starts = np.flatnonzero(np.r_[True, cases[1:] != cases[:-1]])
    return np.add.reduceat(values, starts, axis=0), np.diff(np.r_[starts, len(cases)])


def _bootstrap_means(totals: np.ndarray, counts: np.ndarray) -> np.ndarray:
    """Draw prefix means under a case bootstrap, `[RESAMPLES, ...]`.

    Args:
        totals: Case sums, `[cases, ...]`.
        counts: Prefixes per case, `[cases]`.
    """
    generator = np.random.default_rng(SEED)
    cases = len(counts)
    flat = totals.reshape(cases, -1)
    draws = []
    for start in range(0, RESAMPLES, BATCH_SIZE):
        size = min(BATCH_SIZE, RESAMPLES - start)
        picks = (
            generator.integers(0, cases, size=(size, cases)) + (np.arange(size) * cases)[:, None]
        )
        multiplicities = np.bincount(picks.ravel(), minlength=size * cases).reshape(size, cases)
        draws.append((multiplicities @ flat) / (multiplicities @ counts)[:, None])
    return np.concatenate(draws).reshape(RESAMPLES, *totals.shape[1:])


def emphasis(models: dict[str, pseval.Scores]) -> pd.DataFrame:
    """Mark, on each proper score, the lowest mean and every model not detectably worse.

    A model is detectably worse when the `ALPHA / (models - 1)` quantile of its paired
    case-bootstrap mean difference to the lowest mean is above zero: a one-sided test,
    Bonferroni-corrected over the models compared with the lowest mean. Not detectably worse is
    no evidence of equivalence. A single model is never marked.

    Args:
        models: One dataset's scores by label, all on the same prefixes.

    Returns:
        A boolean frame, one row per label and one column per proper score.
    """
    if len(models) < 2:
        return pd.DataFrame(False, index=list(models), columns=list(PROPER_SCORES))
    totals, counts = _by_case(models)
    best = (totals.sum(axis=0) / counts.sum()).argmin(axis=0)
    differences = totals - totals[:, best, np.arange(len(PROPER_SCORES))][:, None]
    bound = np.quantile(_bootstrap_means(differences, counts), ALPHA / (len(models) - 1), axis=0)
    return pd.DataFrame(bound <= 0.0, index=list(models), columns=list(PROPER_SCORES))
