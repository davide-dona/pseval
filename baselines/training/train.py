import math
from collections.abc import Callable
from dataclasses import dataclass
from time import perf_counter

import torch
import wandb
from omegaconf import DictConfig
from torch import optim
from torch.utils.data import DataLoader

from baselines.models.autoregressive import AutoregressiveSuffixModel
from baselines.runtime.adapter import seeded
from baselines.training.loss import Loss

# The validation metric that selects the checkpoint, lower is better.
SELECTION_METRIC = 'signature_kernel_score'


@dataclass(frozen=True, slots=True)
class TrainingResult:
    """Selected checkpoint step, score, and training stop reason."""

    best_step: int
    selection_score: float
    step: int
    reason: str


def _lr_factor(step: int, *, warmup_steps: int, max_steps: int, min_lr_factor: float) -> float:
    """Return the linear warmup and cosine-decay learning-rate multiplier of a step."""
    if warmup_steps and step < warmup_steps:
        return step / warmup_steps
    decay_steps = max(max_steps - warmup_steps, 1)
    decay_progress = min(max((step - warmup_steps) / decay_steps, 0.0), 1.0)
    return min_lr_factor + (1 - min_lr_factor) * 0.5 * (1 + math.cos(math.pi * decay_progress))


@torch.no_grad()
def validate_loss(
    model: AutoregressiveSuffixModel, loader: DataLoader, *, device: torch.device, seed: int
) -> Loss:
    """Average the teacher-forced loss over validation pairs, with draws seeded by `seed`."""
    with seeded(seed=seed, device=device):
        model.eval()
        totals = Loss()
        for batch in loader:
            batch = batch.to(device)
            _, metrics = model.compute_loss(model(batch), batch)
            totals += metrics
        return totals / len(loader.dataset)


def train(
    *,
    model: AutoregressiveSuffixModel,
    train_loader: DataLoader,
    validation_loader: DataLoader,
    score: Callable[[], dict[str, float]],
    config: DictConfig,
    on_best: Callable[[int, float], None],
) -> TrainingResult:
    """Optimize the model, logging to the active W&B run, and select it by its validation
    `SELECTION_METRIC`.

    Args:
        model: The model, already on `config.device`.
        train_loader: Training batches, reshuffled every epoch.
        validation_loader: The fixed validation pairs the loss is reported on.
        score: Scores the current weights on the fixed validation prefixes, through
            `pseval.validate`.
        config: The run configuration: `optimizer`, `training`, `early_stopping`, `device`, and
            `seed`.
        on_best: Called with the step and score of each newly selected model, before the next
            optimizer step.

    Returns:
        The selected step, score, and stop summary.

    Raises:
        ValueError: If the selection metric is not finite.
    """
    device = torch.device(config.device)
    settings, stopping = config.training, config.early_stopping
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config.optimizer.lr,
        betas=tuple(config.optimizer.betas),
        weight_decay=config.optimizer.weight_decay,
    )
    step = 0
    best_step = 0
    best_score = math.inf
    checks_without_improvement = 0
    interval_totals, seen = Loss(), 0
    train_iterator = iter(train_loader)
    while step < settings.max_steps:
        try:
            batch = next(train_iterator)
        except StopIteration:
            train_iterator = iter(train_loader)
            batch = next(train_iterator)
        batch = batch.to(device)
        step += 1

        model.train()
        loss, metrics = model.compute_loss(model(batch), batch)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), settings.grad_clip_norm)
        learning_rate = config.optimizer.lr * _lr_factor(
            step,
            warmup_steps=config.optimizer.warmup_steps,
            max_steps=settings.max_steps,
            min_lr_factor=config.optimizer.min_lr_factor,
        )
        for group in optimizer.param_groups:
            group['lr'] = learning_rate
        optimizer.step()

        batch_size = batch.suffix.activities.size(dim=0)
        interval_totals += metrics
        seen += batch_size
        wandb.log(
            {'train/lr': learning_rate}
            | {f'train/{key}': value for key, value in (metrics / batch_size).as_metrics().items()},
            step=step,
        )
        if step % settings.val_every_n_steps != 0 and step != settings.max_steps:
            continue

        val_metrics = validate_loss(model, validation_loader, device=device, seed=config.seed)
        started = perf_counter()
        scores = score()
        scoring_seconds = perf_counter() - started
        wandb.log(
            {f'val/{key}': value for key, value in val_metrics.as_metrics().items()}
            | {f'validation/{key}': value for key, value in scores.items()},
            step=step,
        )
        selection = scores[SELECTION_METRIC]
        print(
            f'Step {step:>{len(str(settings.max_steps))}}/{settings.max_steps}  '
            f'train {(interval_totals / seen).loss:.4f}  val {val_metrics.loss:.4f}  '
            f'dls {scores["dls_sample_mean"]:.4f}  energy {scores["energy_score_dls"]:.4f}  '
            f'kernel {selection:.4f}  scoring {scoring_seconds:.1f}s',
            flush=True,
        )
        interval_totals, seen = Loss(), 0
        if not math.isfinite(selection):
            raise ValueError(f'Nonfinite validation {SELECTION_METRIC}: {selection}')
        if best_score == math.inf or best_score - selection > abs(best_score) * (
            stopping.min_delta_perc
        ):
            checks_without_improvement = 0
        else:
            checks_without_improvement += 1
        if selection < best_score:
            best_score, best_step = selection, step
            on_best(step, selection)
        if checks_without_improvement >= stopping.patience_validations:
            return TrainingResult(
                best_step=best_step,
                selection_score=best_score,
                step=step,
                reason=f'no validation improvement for {stopping.patience_validations} checks',
            )
    return TrainingResult(
        best_step=best_step, selection_score=best_score, step=step, reason='reached max_steps'
    )
