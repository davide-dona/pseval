from typing import Any

import torch
import torch.nn.functional as F
from omegaconf import DictConfig

from baselines.data.codec import Codec
from baselines.data.dataset import Events
from baselines.models.autoregressive import AutoregressiveSuffixModel, DecoderOutput
from baselines.modules.dropout import standard_dropout


class EncoderDecoder(AutoregressiveSuffixModel[DecoderOutput]):
    """A plain autoregressive encoder-decoder, the ablation of what U-ED's uncertainty adds.

    It trains with standard dropout and cross-entropy. A generation reads one deterministic
    network, dropout being inactive in eval mode, and draws activities from the softmax of the
    logits and times from the Gaussian head.
    """

    def __init__(self, config: DictConfig, codec: Codec) -> None:
        """
        Args:
            config: The `model` section.
            codec: The codec fitted on the train split.
        """
        super().__init__(config, codec, dropout=standard_dropout(config.dropout))

    def _distributions(self, features: torch.Tensor) -> DecoderOutput:
        means, log_variances = self._time_distribution(features)
        return DecoderOutput(
            activity_logits=self.activity_head(features),
            inter_event_time_means=means,
            inter_event_time_log_variances=log_variances,
        )

    def _activity_nll(
        self, output: DecoderOutput, *, real: torch.Tensor, targets: torch.Tensor
    ) -> torch.Tensor:
        return F.cross_entropy(input=output.activity_logits[real], target=targets, reduction='none')

    def _sampling_logits(self, output: DecoderOutput) -> torch.Tensor:
        return output.activity_logits

    def _start(self, prefix: Events, *, num_samples: int, max_steps: int) -> Any:
        """Encode each prefix once for all its draws."""
        rows = torch.arange(end=prefix.length.size(dim=0), device=prefix.length.device)
        return self.backbone.select(
            self.backbone.start(prefix, max_steps=max_steps),
            rows.repeat_interleave(num_samples),
        )
