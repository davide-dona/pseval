from dataclasses import asdict, dataclass

import torch


@dataclass(frozen=True)
class Loss:
    """The loss of one pass and its terms, each a sum over the batch's traces."""

    loss: float = 0.0
    activity_loss: float = 0.0
    inter_event_time_loss: float = 0.0

    def __add__(self, other: 'Loss') -> 'Loss':
        """Add loss terms from another batch."""
        return Loss(
            *(
                a + b
                for a, b in zip(
                    self.as_metrics().values(), other.as_metrics().values(), strict=True
                )
            )
        )

    def __truediv__(self, divisor: float) -> 'Loss':
        """Average every loss term by the same divisor."""
        return Loss(*(value / divisor for value in self.as_metrics().values()))

    def as_metrics(self) -> dict[str, float]:
        """The terms by name, for logging."""
        return asdict(self)


def reconstruction_loss(
    *, activity_loss: torch.Tensor, time_loss: torch.Tensor, lengths: torch.Tensor
) -> tuple[torch.Tensor, Loss]:
    """Normalize per-trace channel sums `[B]` by each trace's number of supervised targets.

    A suffix of `L` positions, its EOT included, supervises `L` activities and `L - 1` times, so
    long suffixes do not take more of the batch's weight than short ones.

    Returns:
        The mean normalized trace loss to backpropagate, and its terms summed over traces, read
        off the device in one transfer.
    """
    target_counts = (2 * lengths - 1).to(dtype=activity_loss.dtype)
    normalized_activity = activity_loss / target_counts
    normalized_time = time_loss / target_counts
    normalized_total = normalized_activity + normalized_time
    terms = torch.stack(
        tensors=(normalized_total.sum(), normalized_activity.sum(), normalized_time.sum())
    )
    return normalized_total.mean(), Loss(*terms.tolist())
