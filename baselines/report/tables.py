import pandas as pd

from baselines.report import Summaries
from baselines.report.style import DATASETS, LOG_STYLE, MODELS

# The header of each proper score, in `PROPER_SCORES` order, under its group's.
SKILL_HEADERS = {
    'signature_kernel_score': r'$KS_{\mathrm{sig}}$',
    'energy_score_dls': r'$ES_{\mathrm{DL}}$',
    'energy_score_exact': r'$ES_{\mathrm{exact}}$',
    'energy_score_bigram': r'$ES_{\mathrm{2-\text{gram}}}$',
    'suffix_length_crps': 'CRPS',
    'remaining_time_crps_days': 'CRPS',
    'declare_support_brier': 'Support',
    'declare_activation_brier': 'Activation',
}
SKILL_GROUPS = (('Joint', 1), ('Activity', 3), ('Length', 1), ('Time', 1), ('Declare Brier', 2))


def _body(blocks: dict[str, list[list[str]]]) -> list[str]:
    """Render each dataset's rows of cells, the dataset named once beside them."""
    lines = []
    for dataset, rows in blocks.items():
        if lines:
            lines.append(r'\midrule')
        lines.append(f'  \\multirow{{{len(rows)}}}{{*}}{{{DATASETS[dataset]}}}')
        lines.extend('   & ' + ' & '.join(cells) + r' \\' for cells in rows)
    return lines


def skill_table(skills: Summaries, marked: dict[str, pd.DataFrame]) -> str:
    """Render the skill of every proper score, one row per dataset and model.

    Args:
        skills: Each dataset's `pseval.skill` by model label.
        marked: Each dataset's emphasis, from `emphasis`.

    Returns:
        A page-wide booktabs `tabular*`, bolding the marked cells.
    """
    blocks = {
        dataset: [
            [
                MODELS[label].label,
                *(
                    rf'\textbf{{{value:.3f}}}'
                    if marked[dataset].loc[label, key]
                    else f'{value:.3f}'
                    for key, value in summary.overall.items()
                ),
            ]
            for label, summary in models.items()
        ]
        for dataset, models in skills.items()
    }
    groups = ' & '.join(f'\\multicolumn{{{span}}}{{c}}{{{label}}}' for label, span in SKILL_GROUPS)
    spans, start = [], 3
    for _, span in SKILL_GROUPS:
        spans.append(f'\\cmidrule(lr){{{start}-{start + span - 1}}}')
        start += span
    return '\n'.join(
        [
            rf'\begin{{tabular*}}{{\linewidth}}{{@{{\extracolsep{{\fill}}}}ll|*{{{len(SKILL_HEADERS)}}}{{r}}}}',
            r'\toprule',
            r'  \multirow{2}{*}{Dataset} & \multirow{2}{*}{Model} & ' + groups + r' \\',
            *spans,
            '  & & ' + ' & '.join(SKILL_HEADERS.values()) + r' \\',
            r'\midrule',
            *_body(blocks),
            r'\bottomrule',
            r'\end{tabular*}',
            '',
        ]
    )


def conformance_table(means: Summaries) -> str:
    """Render the rate of fully conforming draws, one row per dataset and model, with the
    observed log's rate.

    Args:
        means: Each dataset's `Scores.summary` by model label.

    Returns:
        A booktabs `tabularx`.
    """
    blocks = {
        dataset: [
            *(
                [MODELS[label].label, f'{summary.overall["full_conformance_sample_rate"]:.3f}']
                for label, summary in models.items()
            ),
            [
                LOG_STYLE.label,
                f'{next(iter(models.values())).overall["full_conformance_observed"]:.3f}',
            ],
        ]
        for dataset, models in means.items()
    }
    return '\n'.join(
        [
            r'\begin{tabularx}{\linewidth}{ll|>{\centering\arraybackslash}X}',
            r'\toprule',
            r'  Dataset & Model & Full conformance ($\uparrow$) \\',
            r'\midrule',
            *_body(blocks),
            r'\bottomrule',
            r'\end{tabularx}',
            '',
        ]
    )
