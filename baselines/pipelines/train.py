from dataclasses import dataclass
from pathlib import Path

import hydra
import torch
import wandb
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader

import pseval
from baselines.adapter import TorchSuffixModel
from baselines.checkpoint import save_checkpoint
from baselines.data.codec import Codec
from baselines.data.dataset import TraceDataset, fixed_subset
from baselines.models.u_ed import UncertaintyAwareEncoderDecoder
from baselines.pipelines.helpers import load_benchmark, output_path
from baselines.training.loss import Loss
from baselines.training.train import (
    SELECTION_METRIC,
    OptimizerSettings,
    TrainingResult,
    TrainingSettings,
    ValidationReport,
    train,
)


@dataclass(frozen=True, slots=True)
class _Reporter:
    """Log training updates to W&B and save the selected checkpoint."""

    config: dict
    benchmark: str
    checkpoint: Path
    max_steps: int

    def on_batch(self, step: int, metrics: Loss, batch_size: int, learning_rate: float) -> None:
        wandb.log(
            {'train/lr': learning_rate}
            | {f'train/{key}': value for key, value in (metrics / batch_size).as_metrics().items()},
            step=step,
        )

    def on_validation(self, report: ValidationReport) -> None:
        wandb.log(
            {f'val/{key}': value for key, value in report.val_metrics.as_metrics().items()}
            | {f'validation/{key}': value for key, value in report.scores.items()},
            step=report.step,
        )
        print(
            f'Step {report.step:>{len(str(self.max_steps))}}/{self.max_steps}  '
            f'train {report.train_metrics.loss:.4f}  val {report.val_metrics.loss:.4f}  '
            f'{SELECTION_METRIC} {report.scores[SELECTION_METRIC]:.4f}  '
            f'scoring {report.scoring_seconds:.1f}s',
            flush=True,
        )

    def on_best(self, model: UncertaintyAwareEncoderDecoder, step: int, score: float) -> None:
        save_checkpoint(
            model,
            config=self.config,
            benchmark=self.benchmark,
            step=step,
            selection_score=score,
            path=self.checkpoint,
        )
        print(f'New best model (step {step}, score {score:.4f}) saved at {self.checkpoint}')


def run(config: DictConfig) -> TrainingResult:
    """Train a U-ED model on a benchmark and keep the checkpoint its validation score selects."""
    benchmark = load_benchmark(config.data.name, config.benchmark_root)
    # Seeded before anything is built, so initialization, shuffling, and subsets reproduce.
    torch.manual_seed(config.seed)
    generator = torch.Generator().manual_seed(config.seed)
    device = torch.device(config.device)

    codec = Codec.fit(
        benchmark.train,
        vocabulary=benchmark.vocabulary,
        max_trace_length=benchmark.max_suffix_length,
        data_config=config.data,
    )
    model = UncertaintyAwareEncoderDecoder(config.model, codec).to(device)
    parameters = sum(parameter.numel() for parameter in model.parameters())
    print(f'{config.model.label} on {config.data.name}: {parameters:,} parameters', flush=True)

    # Without persistent workers a loader forks its pool every time it is iterated, and the
    # validation loader is iterated once per check.
    workers = config.dataloader.num_workers
    train_loader = DataLoader(
        dataset=TraceDataset(benchmark.train, codec),
        batch_size=config.dataloader.batch_size,
        shuffle=True,
        generator=generator,
        num_workers=workers,
        persistent_workers=workers > 0,
    )
    validation_loader = DataLoader(
        dataset=fixed_subset(
            TraceDataset(benchmark.validation, codec),
            size=config.training.validation_pairs,
            generator=generator,
        ),
        batch_size=config.dataloader.batch_size,
        num_workers=workers,
        persistent_workers=workers > 0,
    )
    # One fixed subset of validation prefixes for every check, sorted by length like generation.
    cuts = benchmark.validation.cuts()
    drawn = torch.randperm(n=len(cuts), generator=generator)[: config.training.validation_prefixes]
    validation_cuts = sorted((cuts[index] for index in drawn.tolist()), key=lambda c: c.prefix_len)
    adapter = TorchSuffixModel(
        model,
        codec,
        max_suffix_length=benchmark.max_suffix_length,
        device=device,
        max_rows=config.inference.generation_rows_upper_bound,
    )
    print(
        f'Training on {len(train_loader.dataset):,} pairs, reporting the loss on '
        f'{len(validation_loader.dataset):,} validation pairs and scoring '
        f'{len(validation_cuts):,} validation prefixes',
        flush=True,
    )

    experiment_config = OmegaConf.to_container(config, resolve=True)
    run_name = f'{config.data.name}/{config.model.label}/{config.run_id}'
    wandb.init(
        project=config.wandb.project,
        mode=config.wandb.mode,
        name=run_name,
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
                cuts=validation_cuts,
                num_samples=config.inference.validation_samples,
                seed=config.seed,
            ),
            settings=TrainingSettings(
                optimizer=OptimizerSettings(
                    lr=config.optimizer.lr,
                    betas=(config.optimizer.beta1, config.optimizer.beta2),
                    weight_decay=config.optimizer.weight_decay,
                    warmup_steps=config.optimizer.warmup_steps,
                    min_lr_factor=config.optimizer.min_lr_factor,
                ),
                device=device,
                seed=config.seed,
                max_steps=config.training.max_steps,
                val_every_n_steps=config.training.val_every_n_steps,
                grad_clip_norm=config.training.grad_clip_norm,
                patience_validations=config.early_stopping.patience_validations,
                min_delta_perc=config.early_stopping.min_delta_perc,
            ),
            observer=_Reporter(
                config=experiment_config,
                benchmark=benchmark.digest,
                checkpoint=output_path('best.pt'),
                max_steps=config.training.max_steps,
            ),
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
