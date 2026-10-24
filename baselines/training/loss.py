from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType

import torch


@dataclass(frozen=True)
class Loss:
    """The loss of one pass and its terms, each a sum over the batch's traces."""

    loss: float = 0.0
    activity_loss: float = 0.0
    inter_event_time_loss: float = 0.0
    optional_terms: Mapping[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if set(self.optional_terms) & {'loss', 'activity_loss', 'inter_event_time_loss'}:
            raise ValueError('Optional loss terms must not replace shared terms')
        object.__setattr__(self, 'optional_terms', MappingProxyType(dict(self.optional_terms)))

    def __add__(self, other: 'Loss') -> 'Loss':
        """Add loss terms from another batch."""
        optional_terms = dict(self.optional_terms)
        for name, value in other.optional_terms.items():
            optional_terms[name] = optional_terms.get(name, 0.0) + value
        return Loss(
            loss=self.loss + other.loss,
            activity_loss=self.activity_loss + other.activity_loss,
            inter_event_time_loss=self.inter_event_time_loss + other.inter_event_time_loss,
            optional_terms=optional_terms,
        )

    def __truediv__(self, divisor: float) -> 'Loss':
        """Average every loss term by the same divisor."""
        return Loss(
            loss=self.loss / divisor,
            activity_loss=self.activity_loss / divisor,
            inter_event_time_loss=self.inter_event_time_loss / divisor,
            optional_terms={name: value / divisor for name, value in self.optional_terms.items()},
        )

    def as_metrics(self) -> dict[str, float]:
        """Flatten the shared and optional terms for logging."""
        return {
            'loss': self.loss,
            'activity_loss': self.activity_loss,
            'inter_event_time_loss': self.inter_event_time_loss,
            **self.optional_terms,
        }


def reconstruction_loss(
    *, activity_loss: torch.Tensor, time_loss: torch.Tensor, lengths: torch.Tensor
) -> tuple[torch.Tensor, Loss]:
    """Normalize per-trace channel sums `[B]` by each trace's number of supervised targets.

    A suffix of `L` positions, its EOT included, supervises `L` activities and `L - 1` times, so
    long suffixes do not take more of the batch's weight than short ones.

    Returns:
        The mean normalized trace loss to backpropagate, and its terms summed over traces.
    """
    target_counts = (2 * lengths - 1).to(dtype=activity_loss.dtype)
    normalized_activity = activity_loss / target_counts
    normalized_time = time_loss / target_counts
    normalized_total = normalized_activity + normalized_time
    metrics = Loss(
        loss=normalized_total.sum().item(),
        activity_loss=normalized_activity.sum().item(),
        inter_event_time_loss=normalized_time.sum().item(),
    )
    return normalized_total.mean(), metrics
