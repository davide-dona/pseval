import torch
from torch import nn


class SequenceDropout(nn.Module):
    """Variational dropout: one Bernoulli mask per sequence, shared by all of its positions.

    `resample` draws the masks of a pass, one per row; every call until the next one applies the
    same mask to a row. A generation therefore reads one sampled network per suffix, and the
    positions a cache already holds share their mask with the new ones.
    """

    def __init__(self, *, p: float, shape: tuple[int, ...]) -> None:
        """
        Args:
            p: Probability of dropping a feature.
            shape: The shape of one row's mask, `(W,)` for features or `(4, W)` for LSTM gates.
        """
        super().__init__()
        self.p = p
        self.shape = shape
        # [R, *shape], scaled by 1 / (1 - p); None while inactive.
        self.mask: torch.Tensor | None = None

    def resample(self, *, rows: int, device: torch.device, active: bool) -> None:
        """Draw one mask per row, or clear the masks when inactive."""
        if not active or self.p == 0.0:
            self.mask = None
            return
        keep = torch.rand(size=(rows, *self.shape), device=device) >= self.p
        self.mask = keep.float() / (1.0 - self.p)

    def select(self, index: torch.Tensor) -> None:
        """Keep the masks of the rows at `index` `[R']`, in that order."""
        if self.mask is not None:
            self.mask = self.mask[index]

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        """Mask `[R, *shape]` values, or `[R, T, *shape]` values with the mask broadcast over
        positions."""
        if self.mask is None:
            return values
        mask = self.mask if values.dim() == self.mask.dim() else self.mask.unsqueeze(dim=1)
        return values * mask


def resample_masks(module: nn.Module, *, rows: int, device: torch.device, active: bool) -> None:
    """Draw the masks of every `SequenceDropout` inside `module` for a pass over `rows` rows."""
    for submodule in module.modules():
        if isinstance(submodule, SequenceDropout):
            submodule.resample(rows=rows, device=device, active=active)


def select_masks(module: nn.Module, index: torch.Tensor) -> None:
    """Keep the masks of the rows at `index` in every `SequenceDropout` inside `module`."""
    for submodule in module.modules():
        if isinstance(submodule, SequenceDropout):
            submodule.select(index)
