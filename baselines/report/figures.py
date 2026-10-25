import math

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.ticker import MaxNLocator

import pseval
from baselines.report import Summaries
from baselines.report.style import (
    COLUMN_WIDTH,
    DATASETS,
    FIGURE_OVERHEAD,
    LOG_STYLE,
    MAX_MARKERS,
    MIN_BUCKET_PREFIXES,
    MODELS,
    PAGE_WIDTH,
    REFERENCE_LINE,
    ROW_HEIGHT,
    Y_HEADROOM,
    ModelStyle,
)

PIT_BINS = 10
# Legend entries per row in a column-wide figure.
LEGEND_COLUMNS = 3


def _grid(datasets: list[str], columns: int, *, width: float, **shared: object) -> np.ndarray:
    """Lay out one row of panels per dataset, each labeled on its left.

    Returns:
        The axes, `[datasets, columns]`.
    """
    figure, grid = plt.subplots(
        nrows=len(datasets),
        ncols=columns,
        figsize=(width, len(datasets) * ROW_HEIGHT + FIGURE_OVERHEAD),
        squeeze=False,
        constrained_layout=True,
        **shared,
    )
    for dataset, axes in zip(datasets, grid[:, 0], strict=True):
        axes.set_ylabel(DATASETS[dataset], rotation=0, ha='right', va='center', labelpad=12)
    figure.align_ylabels(grid[:, 0])
    return grid


def _draw_lines(axes: Axes, lines: list[tuple[pd.Series, ModelStyle]]) -> None:
    """Draw values by length, from 1 to the longest length drawn.

    Args:
        axes: The panel.
        lines: Each line's values, indexed by length, with its style.
    """
    longest = 2
    for values, style in lines:
        axes.plot(
            values.index,
            values,
            label=style.label,
            color=style.color,
            marker=style.marker,
            linestyle=style.linestyle,
            markevery=max(1, math.ceil(len(values) / MAX_MARKERS)),
        )
        longest = max(longest, int(values.index.max()))
    axes.set_xlim(1, longest)
    axes.xaxis.set_major_locator(MaxNLocator(nbins=5, integer=True))


def _readable(buckets: pd.DataFrame) -> pd.DataFrame:
    """Leave out the length buckets with too few prefixes for a readable mean."""
    return buckets[buckets['prefixes'] >= MIN_BUCKET_PREFIXES]


def _legend(figure: Figure) -> None:
    """Show each label once above the panels."""
    entries = {}
    for axes in figure.axes:
        for handle, label in zip(*axes.get_legend_handles_labels(), strict=True):
            entries.setdefault(label, handle)
    figure.legend(
        list(entries.values()),
        list(entries),
        loc='outside upper center',
        ncols=min(len(entries), LEGEND_COLUMNS),
    )


def skill_by_length(skills: Summaries) -> Figure:
    """Draw each model's signature kernel score skill by prefix and by true suffix length.

    Args:
        skills: Each dataset's `pseval.skill` by model label.

    Returns:
        A column-wide figure, one row per dataset.
    """
    grid = _grid(list(skills), 2, width=COLUMN_WIDTH, sharey='col')
    for models, row in zip(skills.values(), grid, strict=True):
        for axes, length in zip(row, ('by_prefix_length', 'by_suffix_length'), strict=True):
            lines = []
            for label, summary in models.items():
                buckets = _readable(getattr(summary, length))
                lines.append((buckets['signature_kernel_score'], MODELS[label]))
            _draw_lines(axes, lines)
            axes.axhline(0.0, **REFERENCE_LINE)
    for column, length in enumerate(('Prefix length', 'Suffix length')):
        grid[0, column].set_title('Signature kernel score\nskill\N{NO-BREAK SPACE}↑')
        grid[-1, column].set_xlabel(length)
    figure = grid[0, 0].figure
    _legend(figure)
    return figure


def conformance_by_suffix_length(means: Summaries) -> Figure:
    """Draw the rate of fully conforming draws by true suffix length, with the observed log's.

    Args:
        means: Each dataset's `Scores.summary` by model label.

    Returns:
        A column-wide figure, one row per dataset.
    """
    grid = _grid(list(means), 1, width=COLUMN_WIDTH)
    for models, axes in zip(means.values(), grid[:, 0], strict=True):
        buckets = {label: _readable(summary.by_suffix_length) for label, summary in models.items()}
        lines = [
            (frame['full_conformance_sample_rate'], MODELS[label])
            for label, frame in buckets.items()
        ]
        observed = next(iter(buckets.values()))['full_conformance_observed']
        _draw_lines(axes, [*lines, (observed, LOG_STYLE)])
        axes.set_ylim(0.0, 1.0 + Y_HEADROOM)
        axes.set_yticks([0.0, 0.5, 1.0])
    grid[0, 0].set_title('Full conformance rate\N{NO-BREAK SPACE}↑')
    figure = grid[0, 0].figure
    figure.supxlabel('Suffix length')
    _legend(figure)
    return figure


def pit_histograms(scores: dict[str, dict[str, pseval.Scores]], key: str) -> Figure:
    """Draw one quantity's randomized PIT histogram per dataset and model.

    Args:
        scores: Each dataset's scores by model label.
        key: The truth quantile column of the quantity.

    Returns:
        A page-wide figure, one row per dataset and one column per model, where a calibrated
        model follows the uniform density.
    """
    models = [label for label in MODELS if any(label in runs for runs in scores.values())]
    grid = _grid(list(scores), len(models), width=PAGE_WIDTH, sharex=True, sharey=True)
    edges = np.linspace(0.0, 1.0, PIT_BINS + 1)
    for runs, row in zip(scores.values(), grid, strict=True):
        for label, axes in zip(models, row, strict=True):
            axes.axhline(1.0, **REFERENCE_LINE)
            if label not in runs:
                axes.text(0.5, 0.5, 'no scores', ha='center', va='center', transform=axes.transAxes)
                continue
            counts, _ = np.histogram(runs[label].frame[key], bins=edges)
            density = counts * PIT_BINS / counts.sum()
            axes.stairs(density, edges, fill=True, color=MODELS[label].color, alpha=0.8)
    for label, axes in zip(models, grid[0], strict=True):
        axes.set_title(MODELS[label].label)
    grid[0, 0].set_xlim(0.0, 1.0)
    grid[0, 0].set_xticks([0.0, 0.5, 1.0])
    grid[0, 0].set_ylim(bottom=0.0)
    figure = grid[0, 0].figure
    figure.supxlabel('PIT')
    figure.suptitle(key.removesuffix('_truth_quantile').replace('_', ' ').capitalize())
    return figure
