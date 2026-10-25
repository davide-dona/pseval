from pathlib import Path

import hydra
from omegaconf import DictConfig

import pseval
from baselines.models.retrieval import Retrieval
from baselines.pipelines.helpers import output_path


def run(config: DictConfig) -> Path:
    """Fit Retrieval on the train split and draw its suffixes for every test prefix.

    Retrieval has nothing to select, so it reads no validation data.
    """
    benchmark = pseval.load_benchmark(config.data.name, root=config.benchmark_root)
    model = Retrieval(benchmark.train, vocabulary=benchmark.vocabulary)
    path = pseval.generate(
        model,
        benchmark,
        output_path('generations.parquet'),
        label='retrieval',
        num_samples=config.num_samples,
        seed=config.seed,
        batch_size=config.batch_size,
    )
    print(f'Wrote {path}')
    return path


@hydra.main(version_base='1.3', config_path='../config', config_name='retrieval')
def main(config: DictConfig) -> None:
    run(config)


if __name__ == '__main__':
    main()
