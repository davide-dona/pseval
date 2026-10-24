from pathlib import Path

import hydra
import torch
from omegaconf import DictConfig, OmegaConf

import pseval
from baselines.adapter import TorchSuffixModel
from baselines.checkpoint import load_checkpoint
from baselines.data.codec import Codec
from baselines.models.u_ed import UncertaintyAwareEncoderDecoder
from baselines.pipelines.helpers import load_benchmark, output_path


def run(config: DictConfig) -> Path:
    """Draw a trained model's suffixes for every test prefix through `pseval.generate`.

    Raises:
        ValueError: If the checkpoint was trained on another benchmark.
    """
    checkpoint = load_checkpoint(Path(config.checkpoint))
    trained = OmegaConf.create(checkpoint['config'])
    benchmark = load_benchmark(trained.data.name, config.benchmark_root)
    if benchmark.digest != checkpoint['benchmark']:
        raise ValueError(f'{config.checkpoint} was trained on another benchmark')
    device = torch.device(config.device)
    codec = Codec.fit(
        benchmark.train,
        vocabulary=benchmark.vocabulary,
        max_trace_length=benchmark.max_suffix_length,
        data_config=trained.data,
    )
    model = UncertaintyAwareEncoderDecoder(trained.model, codec)
    model.load_state_dict(checkpoint['model_state_dict'])
    adapter = TorchSuffixModel(
        model.to(device),
        codec,
        max_suffix_length=benchmark.max_suffix_length,
        device=device,
        max_rows=config.max_rows,
    )
    print(
        f'Generating {config.num_samples} suffixes per test prefix with {trained.model.label} '
        f'(step {checkpoint["step"]}, {checkpoint["selection_score"]:.4f})',
        flush=True,
    )
    path = pseval.generate(
        adapter,
        benchmark,
        output_path('generations.parquet'),
        label=trained.model.label,
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
