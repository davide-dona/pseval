from pathlib import Path

import hydra
import torch
from omegaconf import DictConfig

import pseval
from baselines.adapter import TorchSuffixModel
from baselines.checkpoint import load_checkpoint, restore_model
from baselines.pipelines.helpers import output_path


def run(config: DictConfig) -> Path:
    """Draw a trained model's suffixes for every test prefix through `pseval.generate`."""
    checkpoint = load_checkpoint(Path(config.train_dir) / 'best.pt')
    trained = checkpoint['config']
    benchmark = pseval.load_benchmark(trained['data']['name'], root=config.benchmark_root)
    model, codec = restore_model(checkpoint, benchmark)
    device = torch.device(config.device)
    label = trained['model']['label']
    print(
        f'Generating {config.num_samples} suffixes per test prefix with {label} '
        f'(step {checkpoint["step"]}, {checkpoint["selection_score"]:.4f})',
        flush=True,
    )
    path = pseval.generate(
        TorchSuffixModel(model.to(device), codec, device=device, max_rows=config.max_rows),
        benchmark,
        output_path('generations.parquet'),
        label=label,
        num_samples=config.num_samples,
        seed=config.seed,
        batch_size=config.batch_size,
    )
    print(f'Wrote {path}')
    return path


@hydra.main(version_base='1.3', config_path='../config', config_name='generate')
def main(config: DictConfig) -> None:
    run(config)


if __name__ == '__main__':
    main()
