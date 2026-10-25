from collections.abc import Iterator
from contextlib import contextmanager

import numpy as np
import torch

from baselines.data.codec import Codec
from baselines.data.dataset import encode_prefixes
from baselines.models.u_ed import GeneratedSuffix, UncertaintyAwareEncoderDecoder
from pseval import Draw, Forecast, PrefixBatch, SuffixModel


@contextmanager
def seeded(*, seed: int, device: torch.device) -> Iterator[None]:
    """Draw from the CPU and `device` generators seeded with `seed`, restoring their state after.

    Draws inside the block depend on the seed alone, and the training stream outside it is left
    where it was.
    """
    devices = [] if device.type == 'cpu' else [device.index or 0]
    with torch.random.fork_rng(devices=devices, device_type=device.type):
        torch.manual_seed(seed)
        yield


class TorchSuffixModel(SuffixModel):
    """A torch U-ED model as pseval samples it: prefixes in log units in, named draws out."""

    def __init__(
        self,
        model: UncertaintyAwareEncoderDecoder,
        codec: Codec,
        *,
        device: torch.device,
        max_rows: int,
    ) -> None:
        """
        Args:
            model: The model, already on `device`.
            codec: The codec the model was trained through. Its length cap bounds every draw.
            device: Where the model runs.
            max_rows: The most rows, prefixes times draws, decoded at once, to bound memory.
        """
        self.model = model
        self.codec = codec
        self.device = device
        self.max_rows = max_rows

    def sample(self, prefixes: PrefixBatch, num_samples: int, *, seed: int) -> list[Forecast]:
        """Encode the prefixes, generate in chunks of at most `max_rows` rows, and decode."""
        events = encode_prefixes(self.codec, prefixes).to(self.device)
        chunk = max(1, self.max_rows // num_samples)
        forecasts: list[Forecast] = []
        with seeded(seed=seed, device=self.device):
            self.model.eval()
            for start in range(0, len(prefixes), chunk):
                generated = self.model.generate(
                    events.rows(slice(start, start + chunk)),
                    num_samples=num_samples,
                    max_steps=self.codec.max_trace_length,
                )
                forecasts.extend(self._decode(generated))
        return forecasts

    def _decode(self, generated: GeneratedSuffix) -> list[Forecast]:
        """Read generated suffixes back as names and nonnegative inter-event minutes."""
        lengths = generated.lengths.cpu().numpy()  # [B, S]
        activities = generated.activities.cpu().numpy()  # [B, S, T]
        # PAD past each length decodes to an arbitrary name, sliced off below.
        names = self.codec.activity.decode(
            np.maximum(activities, self.codec.activity.first_name)
        )  # [B, S, T]
        minutes = np.maximum(
            self.codec.inter_event_time.denormalize(generated.inter_event_times.cpu().numpy()), 0.0
        )  # [B, S, T]
        truncated = lengths == self.codec.max_trace_length  # [B, S]
        return [
            Forecast(
                draws=tuple(
                    Draw(
                        activities=names[row, sample, :length],
                        inter_event_time_minutes=minutes[row, sample, :length],
                        truncated=truncated[row, sample],
                    )
                    for sample, length in enumerate(lengths[row])
                )
            )
            for row in range(len(lengths))
        ]
