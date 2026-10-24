from pathlib import Path

import hydra
from omegaconf import DictConfig

import pseval
from baselines.pipelines.helpers import load_benchmark, output_path


def run(config: DictConfig) -> Path:
    """Score a generations file through `pseval.score` and print its means and skill."""
    generations = Path(config.generations)
    with pseval.GenerationsReader(generations) as reader:
        dataset = reader.record.dataset
    benchmark = load_benchmark(dataset, config.benchmark_root)
    path = pseval.score(
        generations, benchmark, path=output_path('scores.parquet'), workers=config.workers
    )
    with pseval.ScoresReader(path) as reader:
        scores = reader.read()
    print(f'Wrote {path}\n\nMeans\n{scores.summary().overall.to_string()}\n\nSkill')
    for key, value in pseval.skill(scores, benchmark).items():
        print(f'{key}  {value:.4f}')
    return path


@hydra.main(version_base='1.3', config_path='../config', config_name='score')
def main(config: DictConfig) -> None:
    run(config)


if __name__ == '__main__':
    main()
