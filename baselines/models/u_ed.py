import math
from dataclasses import dataclass
from typing import Any, Protocol

import torch
import torch.nn.functional as F
from omegaconf import DictConfig
from torch import nn

from baselines.data.codec import Codec
from baselines.data.dataset import Events, TraceCut
from baselines.models.dropout import SequenceDropout, resample_masks
from baselines.models.embeddings import EventContentEmbedding
from baselines.models.lstm import LSTMBackbone
from baselines.models.transformer import TransformerBackbone
from baselines.training.loss import Loss, reconstruction_loss


@dataclass(frozen=True)
class DecoderOutput:
    """The predicted distributions at every suffix position."""

    activity_logit_means: torch.Tensor  # [B, T, V]
    # [B, T, V], None without logit uncertainty, the logits then being the means themselves.
    activity_log_variances: torch.Tensor | None
    inter_event_time_means: torch.Tensor  # [B, T], standardized
    inter_event_time_log_variances: torch.Tensor  # [B, T]


@dataclass(frozen=True)
class GeneratedSuffix:
    """Generated events, grouped by prefix and sample."""

    activities: torch.Tensor  # [B, S, T], PAD past each length
    lengths: torch.Tensor  # [B, S], events before EOT, or T if no EOT was emitted
    inter_event_times: torch.Tensor  # [B, S, T], standardized, 0.0 past each length
    truncated: torch.Tensor  # [B, S], whether the draw ran to the cap without EOT


class Backbone(Protocol):
    """An encoder-decoder over embedded events, as the U-ED recipe drives it."""

    def encode(self, prefix: Events) -> Any: ...

    def decode(self, inputs: Events, prefix: Any) -> torch.Tensor: ...

    def start(self, prefix: Any, *, max_steps: int) -> Any: ...

    def step(self, inputs: Events, state: Any, *, position: int) -> tuple[torch.Tensor, Any]: ...


class UncertaintyAwareEncoderDecoder(nn.Module):
    """The shared U-ED recipe over a transformer or LSTM backbone.

    MLP heads read Gaussian activity-logit and inter-event time distributions at every suffix
    position. Variational dropout is active in training and in generation, with one mask per
    suffix, so a draw comes from one network sampled the way training scores it. Without the
    `uncertainty` section the model is the plain autoregressive ablation: no logit variance and
    no dropout at generation, activities drawn from the softmax and times from the Gaussian head.
    """

    def __init__(self, config: DictConfig, codec: Codec) -> None:
        """
        Args:
            config: The `model` section.
            codec: The codec fitted on the train split.
        """
        super().__init__()
        d_model = config.d_model
        content = EventContentEmbedding(config=config.embeddings, codec=codec, d_model=d_model)
        backbones = {
            'transformer': lambda: TransformerBackbone(
                config.backbone,
                content,
                d_model=d_model,
                dropout=config.dropout,
                max_trace_length=codec.max_trace_length,
            ),
            'lstm': lambda: LSTMBackbone(
                config.backbone, content, d_model=d_model, dropout=config.dropout
            ),
        }
        if config.backbone.kind not in backbones:
            raise ValueError(f'model.backbone.kind must be one of {sorted(backbones)}')
        self.backbone: Backbone = backbones[config.backbone.kind]()
        self.uncertainty = config.uncertainty
        self.activity_dropout = config.activity_dropout
        self.log_variance_min = config.log_variance.min
        self.log_variance_max = config.log_variance.max
        self.num_categorical = len(codec.categorical_features)
        self.num_numeric = len(codec.numeric_features)
        self.sos_activity_index = codec.activity.sos_index
        self.pad_activity_index = codec.activity.pad_index
        self.eot_activity_index = codec.activity.eot_index
        self.pad_resource_index = codec.resource.pad_index
        self.input_inter_event_time = codec.input_inter_event_time
        self.inter_event_time = codec.inter_event_time

        num_activities = codec.activity.num_rows
        hidden = config.head_hidden_dim
        # A trunk shared by every head, so the heads can be smaller.
        self.shared_layer = nn.Sequential(
            nn.Linear(in_features=d_model, out_features=hidden),
            nn.ReLU(),
            SequenceDropout(p=config.dropout, shape=(hidden,)),
        )
        self.activity_head = nn.Linear(in_features=hidden, out_features=num_activities)
        self.activity_variance_head = (
            nn.Linear(in_features=hidden, out_features=num_activities)
            if self.uncertainty is not None
            else None
        )
        self.inter_event_time_head = nn.Linear(in_features=hidden, out_features=1)
        self.inter_event_time_variance_head = nn.Linear(in_features=hidden, out_features=1)
        # EOT and UNK remain valid output classes; only structural PAD and SOS are masked.
        self.register_buffer(
            name='unemittable_activities',
            tensor=torch.tensor(
                data=[codec.activity.pad_index, codec.activity.sos_index], dtype=torch.long
            ),
            persistent=False,
        )

    def forward(self, batch: TraceCut) -> DecoderOutput:
        """Predict every suffix position in one teacher-forced pass."""
        resample_masks(
            self,
            rows=batch.prefix.length.size(dim=0),
            device=batch.prefix.length.device,
            active=self.training,
        )
        prefix = self.backbone.encode(batch.prefix)
        activities = _teacher_forced(batch.suffix.activities, start=self.sos_activity_index)
        times = _teacher_forced(batch.suffix.inter_event_times, start=0.0)
        if self.training and self.activity_dropout > 0.0:
            dropped = (
                torch.rand_like(input=activities, dtype=torch.float32) < self.activity_dropout
            ) & (activities != self.sos_activity_index)
            activities = activities.masked_fill(mask=dropped, value=self.pad_activity_index)
        hidden = self.backbone.decode(self._decoder_events(activities, times), prefix)
        return self._distributions(self.shared_layer(hidden))

    def compute_loss(self, output: DecoderOutput, batch: TraceCut) -> tuple[torch.Tensor, Loss]:
        """Score a teacher-forced pass by its equal-weight trace reconstruction loss.

        With logit uncertainty, the activity term is the negative log of the target's softmax
        probability averaged over `categorical_samples` Gaussian logit draws; without it, the
        cross-entropy of the logits. The time term is the Gaussian negative log-likelihood.
        """
        targets = batch.suffix.activities  # [B, T]
        if output.activity_log_variances is None:
            activity_nll = F.cross_entropy(
                input=output.activity_logit_means.transpose(dim0=1, dim1=2),  # -> [B, V, T]
                target=targets,
                reduction='none',
            )  # [B, T]
        else:
            samples = self.uncertainty.categorical_samples
            logits = _sample_gaussian(
                mean=output.activity_logit_means.unsqueeze(dim=0).expand(samples, -1, -1, -1),
                log_variance=output.activity_log_variances.unsqueeze(dim=0),
            )  # [M, B, T, V]
            target_log_probabilities = (
                logits.log_softmax(dim=-1)
                .gather(dim=-1, index=targets[None, :, :, None].expand(samples, -1, -1, -1))
                .squeeze(dim=-1)
            )  # [M, B, T, V] -> [M, B, T]
            activity_nll = math.log(samples) - target_log_probabilities.logsumexp(dim=0)
        activity_loss = activity_nll.masked_fill(
            mask=targets == self.pad_activity_index, value=0.0
        ).sum(dim=1)

        log_variances = output.inter_event_time_log_variances
        time_nll = 0.5 * (
            (-log_variances).exp()
            * (batch.inter_event_times - output.inter_event_time_means).square()
            + log_variances
        )
        time_loss = time_nll.masked_fill(mask=~batch.timed_positions(), value=0.0).sum(dim=1)
        return reconstruction_loss(
            activity_loss=activity_loss, time_loss=time_loss, lengths=batch.suffix.length
        )

    @torch.no_grad()
    def generate(self, prefix: Events, *, num_samples: int, max_steps: int) -> GeneratedSuffix:
        """Draw `num_samples` suffixes per prefix from the prefix alone.

        Args:
            prefix: The observed prefixes, `[B, P]` per field.
            num_samples: Draws per prefix.
            max_steps: The most events a draw may hold.

        Returns:
            The draws, `[B, S, ...]`.
        """
        batch_size = prefix.length.size(dim=0)
        prefix = prefix.repeat_rows(num_samples)  # [B, P] -> [B * S, P]
        rows = batch_size * num_samples
        device = prefix.length.device
        resample_masks(self, rows=rows, device=device, active=self.uncertainty is not None)
        state = self.backbone.start(self.backbone.encode(prefix), max_steps=max_steps)
        # SOS at time 0.0 first, exactly how teacher forcing opens, then the previous draw.
        next_activity = torch.full(
            size=(rows, 1), fill_value=self.sos_activity_index, dtype=torch.long, device=device
        )
        next_time = torch.zeros(size=(rows, 1), device=device)
        activities = torch.zeros(size=(rows, max_steps), dtype=torch.long, device=device)
        times = torch.zeros(size=(rows, max_steps), device=device)
        # A row that never emits EOT ran to the cap, so that is the length it keeps.
        lengths = torch.full(size=(rows,), fill_value=max_steps, dtype=torch.long, device=device)
        finished = torch.zeros(size=(rows,), dtype=torch.bool, device=device)
        steps_taken = max_steps
        for position in range(max_steps):
            hidden, state = self.backbone.step(
                self._decoder_events(next_activity, next_time), state, position=position
            )
            activity, time = self._sample(self.shared_layer(hidden))
            activities[:, position] = activity
            times[:, position] = time
            next_activity = activity.unsqueeze(dim=1)
            next_time = self._input_time(time).unsqueeze(dim=1)
            # A suffix ends at its first EOT, so a later one cannot move the length back.
            just_finished = ~finished & (activity == self.eot_activity_index)
            lengths = lengths.masked_fill(mask=just_finished, value=position)
            finished |= just_finished
            if bool(finished.all()):
                steps_taken = position + 1
                break

        activities, times = activities[:, :steps_taken], times[:, :steps_taken]
        positions = torch.arange(end=steps_taken, device=device)
        kept = positions.unsqueeze(dim=0) < lengths.unsqueeze(dim=1)  # [B * S, T]
        return GeneratedSuffix(
            activities=activities.masked_fill(~kept, self.pad_activity_index).view(
                batch_size, num_samples, steps_taken
            ),  # [B * S, T] -> [B, S, T]
            lengths=lengths.view(batch_size, num_samples),
            inter_event_times=times.masked_fill(~kept, 0.0).view(
                batch_size, num_samples, steps_taken
            ),
            truncated=lengths.eq(max_steps).view(batch_size, num_samples),
        )

    def _distributions(self, features: torch.Tensor) -> DecoderOutput:
        """Read means and bounded log-variances from shared features `[..., H]`."""
        return DecoderOutput(
            activity_logit_means=self.activity_head(features),
            activity_log_variances=(
                self._bounded(self.activity_variance_head(features))
                if self.activity_variance_head is not None
                else None
            ),
            inter_event_time_means=self.inter_event_time_head(features).squeeze(dim=-1),
            inter_event_time_log_variances=self._bounded(
                self.inter_event_time_variance_head(features).squeeze(dim=-1)
            ),
        )

    def _bounded(self, log_variances: torch.Tensor) -> torch.Tensor:
        return log_variances.clamp(min=self.log_variance_min, max=self.log_variance_max)

    def _sample(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Draw one activity and one standardized time per row of features `[R, H]`."""
        output = self._distributions(features)
        logits = output.activity_logit_means
        if output.activity_log_variances is not None:
            logits = _sample_gaussian(mean=logits, log_variance=output.activity_log_variances)
        logits = logits.index_fill(dim=-1, index=self.unemittable_activities, value=-torch.inf)
        activities = torch.multinomial(input=logits.softmax(dim=-1), num_samples=1).squeeze(dim=-1)
        times = _sample_gaussian(
            mean=output.inter_event_time_means, log_variance=output.inter_event_time_log_variances
        )
        return activities, times

    def _decoder_events(self, activities: torch.Tensor, times: torch.Tensor) -> Events:
        """Wrap decoder input activities and input-encoded times `[B, T]` as events.

        Activity and time are the channels the decoder feeds itself, so they are the only ones
        carrying content: teacher forcing would otherwise read ground-truth values generation has
        no prediction of. Every other channel is blanked to a PAD row or 0.0.
        """
        batch_size, seq_len = activities.shape
        device = activities.device
        numeric = torch.zeros(size=(batch_size, seq_len, self.num_numeric), device=device)
        return Events(
            activities=activities,
            resources=torch.full(
                size=(batch_size, seq_len),
                fill_value=self.pad_resource_index,
                dtype=torch.long,
                device=device,
            ),
            inter_event_times=times,
            categorical_attributes=torch.zeros(
                size=(batch_size, seq_len, self.num_categorical), dtype=torch.long, device=device
            ),
            numeric_attributes=numeric,
            numeric_attributes_present=numeric,
            length=torch.full(
                size=(batch_size,), fill_value=seq_len, dtype=torch.long, device=device
            ),
        )

    def _input_time(self, times: torch.Tensor) -> torch.Tensor:
        """Re-encode predicted standardized times `[...]` as the input time channel.

        A negative duration is read as zero minutes, since the log1p input encoding is undefined
        below that. A log-scaled prediction is clamped in log space instead of passing through
        expm1, which overflows on large draws.
        """
        target = self.inter_event_time
        scaled = (times * target.std + target.mean).clamp_min(0.0)
        log_minutes = scaled if target.log else scaled.log1p()
        return (log_minutes - self.input_inter_event_time.mean) / self.input_inter_event_time.std


def _teacher_forced(channel: torch.Tensor, *, start: float) -> torch.Tensor:
    """Shift a suffix channel `[B, T]` right behind `start` without changing its width."""
    return F.pad(input=channel[:, :-1], pad=(1, 0), value=start)  # [B, T - 1] -> [B, T]


def _sample_gaussian(*, mean: torch.Tensor, log_variance: torch.Tensor) -> torch.Tensor:
    """Reparameterize a Gaussian with an already bounded log-variance."""
    return mean + (0.5 * log_variance).exp() * torch.randn_like(mean)
