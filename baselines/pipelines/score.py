from pathlib import Path

import hydra
from omegaconf import DictConfig

import pseval
from baselines.pipelines.helpers import output_path


def run(config: DictConfig) -> Path:
    """Score a generation run through `pseval.score` and print its means and skill."""
    generations = Path(config.generate_dir) / 'generations.parquet'
    with pseval.GenerationsReader(generations) as reader:
        dataset = reader.record.dataset
    benchmark = pseval.load_benchmark(dataset, root=config.benchmark_root)
    path = pseval.score(
        generations, benchmark, path=output_path('scores.parquet'), workers=config.workers
    )
    with pseval.ScoresReader(path) as reader:
        scores = reader.read()
    print(f'Wrote {path}\n\nMeans\n{scores.summary().overall.to_string()}\n\nSkill')
    for key, value in pseval.skill(scores, benchmark).overall.items():
        print(f'{key}  {value:.4f}')
    return path


@hydra.main(version_base='1.3', config_path='../config', config_name='score')
def main(config: DictConfig) -> None:
    run(config)


if __name__ == '__main__':
    main()
