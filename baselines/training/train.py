import math
from collections.abc import Callable
from dataclasses import dataclass
from time import perf_counter
from typing import Protocol

import torch
from torch import optim
from torch.utils.data import DataLoader

from baselines.adapter import seeded
from baselines.models.u_ed import UncertaintyAwareEncoderDecoder
from baselines.training.loss import Loss

# The validation metric that selects the checkpoint, lower is better.
SELECTION_METRIC = 'signature_kernel_score'


@dataclass(frozen=True, slots=True)
class OptimizerSettings:
    """AdamW parameters and step-relative learning rate schedule."""

    lr: float
    betas: tuple[float, float]
    weight_decay: float
    warmup_steps: int
    min_lr_factor: float


@dataclass(frozen=True, slots=True)
class TrainingSettings:
    """Optimization, validation, and stopping settings for one training run."""

    optimizer: OptimizerSettings
    device: torch.device
    seed: int
    max_steps: int
    val_every_n_steps: int
    grad_clip_norm: float | None
    patience_validations: int
    min_delta_perc: float


@dataclass(frozen=True, slots=True)
class ValidationReport:
    """Metrics from one scheduled validation check."""

    step: int
    train_metrics: Loss
    val_metrics: Loss
    # Every pseval metric's mean over the validation prefixes.
    scores: dict[str, float]
    scoring_seconds: float


class TrainingObserver(Protocol):
    """Receive training updates while the current model weights are still available."""

    def on_batch(self, step: int, metrics: Loss, batch_size: int, learning_rate: float) -> None:
        """Handle metrics for one optimized batch."""

    def on_validation(self, report: ValidationReport) -> None:
        """Handle metrics from one validation check."""

    def on_best(self, model: UncertaintyAwareEncoderDecoder, step: int, score: float) -> None:
        """Handle a newly selected model before the next optimizer step."""


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
    model: UncertaintyAwareEncoderDecoder, loader: DataLoader, *, device: torch.device, seed: int
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
    model: UncertaintyAwareEncoderDecoder,
    train_loader: DataLoader,
    validation_loader: DataLoader,
    score: Callable[[], dict[str, float]],
    settings: TrainingSettings,
    observer: TrainingObserver,
) -> TrainingResult:
    """Optimize the model and select it by its validation `SELECTION_METRIC`.

    Args:
        model: The model, already on the configured device.
        train_loader: Training batches, reshuffled every epoch.
        validation_loader: The fixed validation pairs the loss is reported on.
        score: Scores the current weights on the fixed validation prefixes, through
            `pseval.validate`.
        settings: Optimization, validation, and stopping settings.
        observer: Synchronous callbacks for batch, validation, and best-model updates.

    Returns:
        The selected step, score, and stop summary.

    Raises:
        ValueError: If the selection metric is not finite.
    """
    optimizer = optim.AdamW(
        model.parameters(),
        lr=settings.optimizer.lr,
        betas=settings.optimizer.betas,
        weight_decay=settings.optimizer.weight_decay,
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
        batch = batch.to(settings.device)
        step += 1

        model.train()
        loss, metrics = model.compute_loss(model(batch), batch)
        optimizer.zero_grad()
        loss.backward()
        if settings.grad_clip_norm is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), settings.grad_clip_norm)
        learning_rate = settings.optimizer.lr * _lr_factor(
            step,
            warmup_steps=settings.optimizer.warmup_steps,
            max_steps=settings.max_steps,
            min_lr_factor=settings.optimizer.min_lr_factor,
        )
        for group in optimizer.param_groups:
            group['lr'] = learning_rate
        optimizer.step()

        batch_size = batch.suffix.activities.size(dim=0)
        interval_totals += metrics
        seen += batch_size
        observer.on_batch(step, metrics, batch_size, learning_rate)
        if step % settings.val_every_n_steps != 0 and step != settings.max_steps:
            continue

        val_metrics = validate_loss(
            model, validation_loader, device=settings.device, seed=settings.seed
        )
        started = perf_counter()
        scores = score()
        observer.on_validation(
            ValidationReport(
                step=step,
                train_metrics=interval_totals / seen,
                val_metrics=val_metrics,
                scores=scores,
                scoring_seconds=perf_counter() - started,
            )
        )
        interval_totals, seen = Loss(), 0
        selection = scores[SELECTION_METRIC]
        if not math.isfinite(selection):
            raise ValueError(f'Nonfinite validation {SELECTION_METRIC}: {selection}')
        if best_score == math.inf or best_score - selection > abs(best_score) * (
            settings.min_delta_perc
        ):
            checks_without_improvement = 0
        else:
            checks_without_improvement += 1
        if selection < best_score:
            best_score, best_step = selection, step
            observer.on_best(model, step, selection)
        if checks_without_improvement >= settings.patience_validations:
            return TrainingResult(
                best_step=best_step,
                selection_score=best_score,
                step=step,
                reason=f'no validation improvement for {settings.patience_validations} checks',
            )
    return TrainingResult(
        best_step=best_step, selection_score=best_score, step=step, reason='reached max_steps'
    )
