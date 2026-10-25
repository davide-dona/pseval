import math
from dataclasses import dataclass
from typing import Any

import torch
from omegaconf import DictConfig
from torch import nn

from baselines.data.codec import Codec
from baselines.data.dataset import Events
from baselines.models.autoregressive import (
    AutoregressiveSuffixModel,
    DecoderOutput,
    sample_gaussian,
)
from baselines.models.dropout import resample_masks, variational_dropout


@dataclass(frozen=True)
class UncertaintyAwareDecoderOutput(DecoderOutput):
    """The predicted distributions, with Gaussian activity logits around `activity_logits`."""

    activity_log_variances: torch.Tensor  # [B, T, V]


class UncertaintyAwareEncoderDecoder(AutoregressiveSuffixModel[UncertaintyAwareDecoderOutput]):
    """U-ED: Gaussian activity logits and variational dropout, kept active in generation.

    A draw comes from one network sampled the way training scores it, with one dropout mask per
    suffix, and its activities from Gaussian logits drawn at every step.
    """

    def __init__(self, config: DictConfig, codec: Codec) -> None:
        """
        Args:
            config: The `model` section.
            codec: The codec fitted on the train split.
        """
        super().__init__(config, codec, dropout=variational_dropout(config.dropout))
        self.logit_samples = config.logit_samples
        self.activity_variance_head = nn.Linear(
            in_features=config.head_hidden_dim, out_features=codec.activity.num_rows
        )

    def _distributions(self, features: torch.Tensor) -> UncertaintyAwareDecoderOutput:
        means, log_variances = self._time_distribution(features)
        return UncertaintyAwareDecoderOutput(
            activity_logits=self.activity_head(features),
            activity_log_variances=self._bounded(self.activity_variance_head(features)),
            inter_event_time_means=means,
            inter_event_time_log_variances=log_variances,
        )

    def _activity_nll(
        self, output: UncertaintyAwareDecoderOutput, *, real: torch.Tensor, targets: torch.Tensor
    ) -> torch.Tensor:
        """The negative log of the target's softmax probability averaged over `logit_samples`
        Gaussian logit draws."""
        samples = self.logit_samples
        logits = sample_gaussian(
            mean=output.activity_logits[real].unsqueeze(dim=0).expand(samples, -1, -1),
            log_variance=output.activity_log_variances[real].unsqueeze(dim=0),
        )  # [M, K, V]
        target_log_probabilities = (
            logits.log_softmax(dim=-1)
            .gather(dim=-1, index=targets[None, :, None].expand(samples, -1, -1))
            .squeeze(dim=-1)
        )  # [M, K, V] -> [M, K]
        return math.log(samples) - target_log_probabilities.logsumexp(dim=0)

    def _sampling_logits(self, output: UncertaintyAwareDecoderOutput) -> torch.Tensor:
        return sample_gaussian(
            mean=output.activity_logits, log_variance=output.activity_log_variances
        )

    def _start(self, prefix: Events, *, num_samples: int, max_steps: int) -> Any:
        """Draw one dropout mask per suffix, so each prefix is encoded once per draw."""
        rows = prefix.length.size(dim=0) * num_samples
        resample_masks(self, rows=rows, device=prefix.length.device, active=True)
        return self.backbone.start(prefix.repeat_rows(num_samples), max_steps=max_steps)
