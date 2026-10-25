import logging
from dataclasses import dataclass

import matplotlib as mpl


@dataclass(frozen=True)
class ModelStyle:
    """A model's display label and line style."""

    label: str
    color: str
    marker: str
    linestyle: str


# Every reported model by its scores label, in display order.
MODELS = {
    'diffusion_transformer': ModelStyle('Diffusion Transformer', '#2A7F9E', '^', '-'),
    'head_sampling_transformer': ModelStyle('SuTraN-PH', '#A05A4B', 'D', ':'),
    'u_ed_transformer': ModelStyle('U-ED-Transformer', '#57834B', 'P', '--'),
    'u_ed_lstm': ModelStyle('U-ED-LSTM', '#C2843A', 'X', '--'),
    'transformer': ModelStyle('Transformer', '#5B6FA8', 'o', ':'),
    'retrieval': ModelStyle('Retrieval', '#8A6FA8', 's', '-'),
}

# The observed log, drawn beside the models where a metric has an observed value.
LOG_STYLE = ModelStyle('Log', '#737373', 'o', '-.')

# Every dataset's display label, in display order.
DATASETS = {'sepsis': 'Sepsis', 'bpic19': 'BPIC19', 'bpic17': 'BPIC17', 'bpic12': 'BPIC12'}

# IEEE conference text widths in inches.
COLUMN_WIDTH = 3.487
PAGE_WIDTH = 7.140
# Height per dataset row, and room for headings, legend, and axis labels, in inches.
ROW_HEIGHT = 1.0
FIGURE_OVERHEAD = 0.8
# Fewest prefixes a length bucket needs to be drawn.
MIN_BUCKET_PREFIXES = 30
# Most visible markers per line.
MAX_MARKERS = 8
# Upper headroom of a bounded axis, as a share of its range.
Y_HEADROOM = 0.05
# The line marking a null or uniform value.
REFERENCE_LINE = {'color': '#A3A3A3', 'linestyle': '--', 'linewidth': 0.6, 'zorder': 0}

_PAPER_RC = {
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'Nimbus Roman', 'Times', 'STIX Two Text', 'DejaVu Serif'],
    'font.size': 8,
    'axes.labelsize': 8,
    'axes.titlesize': 8,
    'xtick.labelsize': 7,
    'ytick.labelsize': 7,
    'legend.fontsize': 7,
    'figure.titlesize': 9,
    'figure.labelsize': 8,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.linewidth': 0.6,
    'axes.labelpad': 3.0,
    'xtick.major.width': 0.6,
    'ytick.major.width': 0.6,
    'xtick.major.size': 2.5,
    'ytick.major.size': 2.5,
    'axes.grid': False,
    'lines.linewidth': 1.0,
    'lines.markersize': 2.4,
    'lines.markeredgewidth': 0.3,
    'legend.frameon': False,
    'legend.borderpad': 0.0,
    'legend.handlelength': 1.8,
    'legend.columnspacing': 0.9,
    'savefig.dpi': 400,
    'savefig.pad_inches': 0.01,
    # Keeps PDF text selectable.
    'pdf.fonttype': 42,
}


def ordered[T](present: dict[str, T], declared: dict[str, object], *, kind: str) -> dict[str, T]:
    """Return `present` in the order of `declared`.

    Raises:
        ValueError: If `present` holds a name `declared` lacks.
    """
    unknown = present.keys() - declared.keys()
    if unknown:
        raise ValueError(f'No display label for {kind} {", ".join(sorted(unknown))}')
    return {name: present[name] for name in declared if name in present}


def apply_style() -> None:
    """Draw without a display, in the paper style."""
    logging.getLogger('matplotlib').setLevel(logging.WARNING)
    logging.getLogger('fontTools').setLevel(logging.WARNING)
    mpl.use('Agg')
    mpl.rcParams.update(_PAPER_RC)
