from pathlib import Path

import hydra
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from omegaconf import DictConfig

import pseval
from baselines.pipelines.helpers import output_path
from baselines.report.figures import conformance_by_suffix_length, pit_histograms, skill_by_length
from baselines.report.significance import emphasis
from baselines.report.style import DATASETS, MODELS, apply_style, ordered
from baselines.report.tables import conformance_table, skill_table

# The truth quantile columns drawn as PIT histograms.
PIT_KEYS = ('suffix_length_truth_quantile', 'remaining_time_truth_quantile')


def _read(paths: list[Path]) -> dict[str, dict[str, pseval.Scores]]:
    """Read every scores file, by dataset and model label in display order.

    Raises:
        ValueError: If a dataset or label has no display label, or a dataset repeats a label.
    """
    runs: dict[str, dict[str, pseval.Scores]] = {}
    for path in paths:
        with pseval.ScoresReader(path) as reader:
            scores = reader.read()
        models = runs.setdefault(scores.record.dataset, {})
        if scores.record.label in models:
            raise ValueError(
                f'{scores.record.dataset} has two {scores.record.label} scores: {path}'
            )
        models[scores.record.label] = scores
    return ordered(
        {dataset: ordered(models, MODELS, kind='model') for dataset, models in runs.items()},
        DATASETS,
        kind='dataset',
    )


def _save(figure: Figure, name: str) -> None:
    path = output_path(f'figures/{name}.pdf')
    path.parent.mkdir(exist_ok=True)
    figure.savefig(path)
    plt.close(figure)


def _write(table: str, name: str) -> None:
    path = output_path(f'tables/{name}.tex')
    path.parent.mkdir(exist_ok=True)
    path.write_text(table)


def run(config: DictConfig) -> None:
    """Draw the figures and write the tables comparing every scores file found."""
    paths = [
        found
        for given in map(Path, config.scores)
        for found in (sorted(given.rglob('scores.parquet')) if given.is_dir() else [given])
    ]
    if not paths:
        raise ValueError(f'No scores files under {list(config.scores)}')
    scores = _read(paths)
    skills = {}
    for dataset, models in scores.items():
        benchmark = pseval.load_benchmark(dataset, root=config.benchmark_root)
        skills[dataset] = {
            label: pseval.skill(scored, benchmark) for label, scored in models.items()
        }
    means = {
        dataset: {label: scored.summary() for label, scored in models.items()}
        for dataset, models in scores.items()
    }
    apply_style()
    _save(skill_by_length(skills), 'skill-by-length')
    _save(conformance_by_suffix_length(means), 'conformance-by-suffix-length')
    for key in PIT_KEYS:
        _save(pit_histograms(scores, key), f'pit-{key.removesuffix("_truth_quantile")}')
    marked = {dataset: emphasis(models) for dataset, models in scores.items()}
    _write(skill_table(skills, marked), 'skill')
    _write(conformance_table(means), 'conformance')
    print(f'Wrote {len(paths)} runs to {output_path("figures")} and {output_path("tables")}')


@hydra.main(version_base='1.3', config_path='../config', config_name='report')
def main(config: DictConfig) -> None:
    run(config)


if __name__ == '__main__':
    main()
