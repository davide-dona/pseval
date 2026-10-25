from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Protocol

import torch
import torch.nn.functional as F
from hydra.utils import get_class
from omegaconf import DictConfig
from torch import nn

from baselines.data.codec import Codec
from baselines.data.dataset import Events, TraceCut
from baselines.modules.dropout import (
    DropoutScheme,
    resample_masks,
    select_masks,
)
from baselines.modules.embeddings import EventContentEmbedding
from baselines.training.loss import Loss, reconstruction_loss

# Generation checks how many rows are still writing once every this many steps, since the check
# waits for the device.
CHECK_EVERY = 8


@dataclass(frozen=True)
class DecoderOutput:
    """The predicted distributions at every suffix position."""

    activity_logits: torch.Tensor  # [B, T, V], the Gaussian means under logit uncertainty
    inter_event_time_means: torch.Tensor  # [B, T], standardized
    inter_event_time_log_variances: torch.Tensor  # [B, T]


@dataclass(frozen=True)
class GeneratedSuffix:
    """Generated events, grouped by prefix and sample."""

    activities: torch.Tensor  # [B, S, T], PAD past each length
    lengths: torch.Tensor  # [B, S], events before EOT, or max_steps if no EOT was emitted
    inter_event_times: torch.Tensor  # [B, S, T], standardized, 0.0 past each length


class Backbone(Protocol):
    """An encoder-decoder over embedded events, as an autoregressive model drives it."""

    def encode(self, prefix: Events) -> Any: ...

    def decode(self, inputs: Events, prefix: Any) -> torch.Tensor: ...

    def start(self, prefix: Events, *, max_steps: int) -> Any: ...

    def step(self, inputs: Events, state: Any, *, position: int) -> tuple[torch.Tensor, Any]: ...

    def select(self, state: Any, index: torch.Tensor) -> Any: ...


def build_model(config: DictConfig, codec: Codec) -> 'AutoregressiveSuffixModel':
    """Build the model class `config._target_` names.

    Args:
        config: The `model` section.
        codec: The codec fitted on the train split.
    """
    return get_class(config._target_)(config, codec)


class AutoregressiveSuffixModel[OutputT: DecoderOutput](nn.Module, ABC):
    """A transformer or LSTM encoder-decoder that writes a suffix one event at a time.

    An MLP trunk shared by every head reads the activity logits and the Gaussian inter-event time
    at every suffix position. Subclasses own the activity distribution, its loss, the dropout,
    and the networks a generation draws from.
    """

    def __init__(self, config: DictConfig, codec: Codec, *, dropout: DropoutScheme) -> None:
        """
        Args:
            config: The `model` section.
            codec: The codec fitted on the train split.
            dropout: The dropout of every site, the head trunk's included.
        """
        super().__init__()
        d_model = config.d_model
        content = EventContentEmbedding(config=config.embeddings, codec=codec, d_model=d_model)
        self.backbone: Backbone = get_class(config.backbone._target_)(
            config.backbone, content, d_model=d_model, dropout=dropout
        )
        self.activity_dropout = config.activity_dropout
        self.log_variance_min = config.log_variance.min
        self.log_variance_max = config.log_variance.max
        self.num_categorical = len(codec.categorical_features)
        self.num_numeric = len(codec.numeric_features)
        self.sos_activity_index = codec.activity.sos_index
        self.pad_activity_index = codec.activity.pad_index
        self.eot_activity_index = codec.activity.eot_index
        self.input_inter_event_time = codec.input_inter_event_time
        self.inter_event_time = codec.inter_event_time

        hidden = config.head_hidden_dim
        # A trunk shared by every head, so the heads can be smaller.
        self.shared_layer = nn.Sequential(
            nn.Linear(in_features=d_model, out_features=hidden),
            nn.ReLU(),
            dropout.site(shape=(hidden,)),
        )
        self.activity_head = nn.Linear(in_features=hidden, out_features=codec.activity.num_rows)
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

    @abstractmethod
    def _distributions(self, features: torch.Tensor) -> OutputT:
        """Read the distributions from shared features `[..., H]`."""

    @abstractmethod
    def _activity_nll(
        self, output: OutputT, *, real: torch.Tensor, targets: torch.Tensor
    ) -> torch.Tensor:
        """The negative log-likelihood `[K]` of the `targets` `[K]` at the `real` `[B, T]`
        positions."""

    @abstractmethod
    def _sampling_logits(self, output: OutputT) -> torch.Tensor:
        """The logits `[R, V]` one activity is drawn from, at each row of `output`."""

    @abstractmethod
    def _start(self, prefix: Events, *, num_samples: int, max_steps: int) -> Any:
        """Set the dropout masks of a generation and return the backbone state of its
        `B * S` rows, the samples of each prefix consecutive."""

    def forward(self, batch: TraceCut) -> OutputT:
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

    def compute_loss(self, output: OutputT, batch: TraceCut) -> tuple[torch.Tensor, Loss]:
        """Score a teacher-forced pass by its equal-weight trace reconstruction loss.

        The activity term is the subclass's negative log-likelihood and the time term the
        Gaussian one. Only real suffix positions, EOT included, are scored.
        """
        targets = batch.suffix.activities  # [B, T]
        real = targets != self.pad_activity_index  # [B, T]
        real_nll = self._activity_nll(output, real=real, targets=targets[real])  # [K]
        activity_nll = real_nll.new_zeros(size=targets.shape).masked_scatter(
            mask=real, source=real_nll
        )
        activity_loss = activity_nll.sum(dim=1)

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
        rows = batch_size * num_samples
        device = prefix.length.device
        state = self._start(prefix, num_samples=num_samples, max_steps=max_steps)
        # SOS at time 0.0 first, exactly how teacher forcing opens, then the previous draw.
        next_activity = torch.full(
            size=(rows, 1), fill_value=self.sos_activity_index, dtype=torch.long, device=device
        )
        next_time = torch.zeros(size=(rows, 1), device=device)
        activities = torch.zeros(size=(rows, max_steps), dtype=torch.long, device=device)
        times = torch.zeros(size=(rows, max_steps), device=device)
        # A row that never emits EOT ran to the cap, so that is the length it keeps.
        lengths = torch.full(size=(rows,), fill_value=max_steps, dtype=torch.long, device=device)
        # The output row each decoded row writes, and whether it has emitted its EOT.
        live = torch.arange(end=rows, device=device)
        finished = torch.zeros(size=(rows,), dtype=torch.bool, device=device)
        steps_taken = max_steps
        for position in range(max_steps):
            hidden, state = self.backbone.step(
                self._decoder_events(next_activity, next_time), state, position=position
            )
            activity, time = self._sample(self.shared_layer(hidden))
            activities[live, position] = activity
            times[live, position] = time
            next_activity = activity.unsqueeze(dim=1)
            next_time = self._input_time(time).unsqueeze(dim=1)
            # A suffix ends at its first EOT, so a later one cannot move the length back.
            just_finished = ~finished & (activity == self.eot_activity_index)
            lengths[live] = lengths[live].masked_fill(mask=just_finished, value=position)
            finished |= just_finished
            if (position + 1) % CHECK_EVERY != 0:
                continue
            writing = (~finished).nonzero().squeeze(dim=1)
            if len(writing) == 0:
                steps_taken = position + 1
                break
            # Finished rows only write past their length, so they stop being decoded once they
            # are half the rows.
            if 2 * len(writing) <= len(finished):
                live, finished = live[writing], finished[writing]
                next_activity, next_time = next_activity[writing], next_time[writing]
                state = self.backbone.select(state, writing)
                select_masks(self, writing)
        resample_masks(self, rows=0, device=device, active=False)

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
        )

    def _time_distribution(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Read the inter-event time means and bounded log-variances `[...]` from shared
        features `[..., H]`."""
        means = self.inter_event_time_head(features).squeeze(dim=-1)
        log_variances = self.inter_event_time_variance_head(features).squeeze(dim=-1)
        return means, self._bounded(log_variances)

    def _bounded(self, log_variances: torch.Tensor) -> torch.Tensor:
        return log_variances.clamp(min=self.log_variance_min, max=self.log_variance_max)

    def _sample(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Draw one activity and one standardized time per row of features `[R, H]`."""
        output = self._distributions(features)
        logits = self._sampling_logits(output).index_fill(
            dim=-1, index=self.unemittable_activities, value=-torch.inf
        )
        activities = torch.multinomial(input=logits.softmax(dim=-1), num_samples=1).squeeze(dim=-1)
        times = sample_gaussian(
            mean=output.inter_event_time_means, log_variance=output.inter_event_time_log_variances
        )
        return activities, times

    def _decoder_events(self, activities: torch.Tensor, times: torch.Tensor) -> Events:
        """Wrap decoder input activities and input-encoded times `[B, T]` as events.

        Activity and time are the channels the decoder feeds itself, so they are the only ones
        carrying content: teacher forcing would otherwise read ground-truth values generation has
        no prediction of. Every other channel is PAD, an expanded zero that allocates nothing.
        """
        batch_size, seq_len = activities.shape
        zero = activities.new_zeros(size=())
        numeric = times.new_zeros(size=()).expand(batch_size, seq_len, self.num_numeric)
        return Events(
            activities=activities,
            resources=zero.expand(batch_size, seq_len),
            inter_event_times=times,
            categorical_attributes=zero.expand(batch_size, seq_len, self.num_categorical),
            numeric_attributes=numeric,
            numeric_attributes_present=numeric,
            length=zero.add(seq_len).expand(batch_size),
        )

    def _input_time(self, times: torch.Tensor) -> torch.Tensor:
        """Re-encode predicted standardized times `[...]` as the input time channel.

        A negative duration is read as zero minutes, since the log1p input encoding is undefined
        below that.
        """
        minutes = self.inter_event_time.unstandardize(times).clamp_min(0.0)
        return self.input_inter_event_time.standardize(minutes.log1p())


def sample_gaussian(*, mean: torch.Tensor, log_variance: torch.Tensor) -> torch.Tensor:
    """Reparameterize a Gaussian with an already bounded log-variance."""
    return mean + (0.5 * log_variance).exp() * torch.randn_like(mean)


def _teacher_forced(channel: torch.Tensor, *, start: float) -> torch.Tensor:
    """Shift a suffix channel `[B, T]` right behind `start` without changing its width."""
    return F.pad(input=channel[:, :-1], pad=(1, 0), value=start)  # [B, T - 1] -> [B, T]
