import hydra
import torch
import wandb
from omegaconf import DictConfig, OmegaConf

import pseval
from baselines.data.codec import Codec
from baselines.data.dataset import TraceDataset, batches, fixed_subset
from baselines.models.autoregressive import build_model
from baselines.pipelines.helpers import output_path
from baselines.runtime.adapter import TorchSuffixModel
from baselines.runtime.checkpoint import save_checkpoint
from baselines.training.train import SELECTION_METRIC, TrainingResult, train


def run(config: DictConfig) -> TrainingResult:
    """Train a baseline on a benchmark and keep the checkpoint its validation score selects."""
    benchmark = pseval.load_benchmark(config.data.name, root=config.benchmark_root)
    # Seeded before anything is built, so initialization, shuffling, and subsets reproduce.
    torch.manual_seed(config.seed)
    generator = torch.Generator().manual_seed(config.seed)
    device = torch.device(config.device)

    codec = Codec.fit(benchmark, config.data)
    model = build_model(config.model, codec).to(device)
    parameters = sum(parameter.numel() for parameter in model.parameters())
    print(f'{config.model.label} on {config.data.name}: {parameters:,} parameters', flush=True)

    batch_size = config.dataloader.batch_size
    train_loader = batches(
        TraceDataset(benchmark.train, codec), batch_size=batch_size, generator=generator
    )
    validation_loader = batches(
        fixed_subset(
            TraceDataset(benchmark.validation, codec),
            size=config.training.validation_pairs,
            generator=generator,
        ),
        batch_size=batch_size,
    )
    adapter = TorchSuffixModel(model, codec, device=device, max_rows=config.inference.max_rows)
    print(
        f'Training on {len(train_loader.dataset):,} pairs, reporting the loss on '
        f'{len(validation_loader.dataset):,} validation pairs and scoring at most '
        f'{config.training.validation_prefixes:,} validation prefixes',
        flush=True,
    )

    experiment_config = OmegaConf.to_container(config, resolve=True)
    checkpoint = output_path('best.pt')

    def on_best(step: int, score: float) -> None:
        save_checkpoint(
            model,
            config=experiment_config,
            benchmark=benchmark.digest,
            step=step,
            selection_score=score,
            path=checkpoint,
        )
        print(f'New best model (step {step}, score {score:.4f}) saved at {checkpoint}')

    wandb.init(
        project=config.wandb.project,
        mode=config.wandb.mode,
        name=f'{config.data.name}/{config.model.label}/{config.run_id}',
        group=f'{config.data.name}/{config.model.label}',
        job_type='train',
        tags=[config.data.name, config.model.label],
        config=experiment_config,
    )
    wandb.define_metric(f'validation/{SELECTION_METRIC}', summary='min')
    try:
        result = train(
            model=model,
            train_loader=train_loader,
            validation_loader=validation_loader,
            score=lambda: pseval.validate(
                adapter,
                benchmark,
                max_prefixes=config.training.validation_prefixes,
                num_samples=config.inference.validation_samples,
                seed=config.seed,
            ),
            config=config,
            on_best=on_best,
        )
        wandb.summary['selection_score'] = result.selection_score
        wandb.summary['best_step'] = result.best_step
    finally:
        wandb.finish()
    print(
        f'Finished after {result.step} steps ({result.reason}); selected step '
        f'{result.best_step} with {SELECTION_METRIC} {result.selection_score:.4f}'
    )
    return result


@hydra.main(version_base='1.3', config_path='../config', config_name='train')
def main(config: DictConfig) -> None:
    run(config)


if __name__ == '__main__':
    main()
