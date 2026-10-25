from typing import NamedTuple

import pandas as pd
import torch
from torch.utils.data import (
    BatchSampler,
    DataLoader,
    Dataset,
    RandomSampler,
    SequentialSampler,
    Subset,
)

from baselines.data.codec import Codec
from baselines.data.features import derive_features
from pseval import Log, PrefixBatch
from pseval.data import ACTIVITY_KEY


class Events(NamedTuple):
    """The events of a batch of runs, as the model reads them, PAD (row 0, or 0.0) past each run.

    Encoded events not yet gathered into runs are one flat run `[E]` per field.
    """

    activities: torch.Tensor  # int64, [B, T]
    resources: torch.Tensor  # int64, [B, T]
    inter_event_times: torch.Tensor  # float32, standardized input encoding, [B, T]
    categorical_attributes: torch.Tensor  # int64, [B, T, C]
    numeric_attributes: torch.Tensor  # float32, standardized, [B, T, N]
    # float32 0/1, 0.0 where the log had no value, [B, T, N]
    numeric_attributes_present: torch.Tensor
    length: torch.Tensor  # int64, the real, unpadded length of each run, [B]

    def pad_mask(self) -> torch.Tensor:
        """`[B, T]`, True where a position holds padding."""
        positions = torch.arange(end=self.activities.size(dim=-1), device=self.length.device)
        return positions.unsqueeze(dim=0) >= self.length.unsqueeze(dim=1)

    def rows(self, index: slice | torch.Tensor) -> 'Events':
        """Select runs, every field along its first dimension."""
        return Events(*(field[index] for field in self))

    def repeat_rows(self, repeats: int) -> 'Events':
        """Repeat each run `repeats` times, the copies of a run adjacent."""
        return Events(*(field.repeat_interleave(repeats, dim=0) for field in self))

    def to(self, device: torch.device) -> 'Events':
        """Move every field to `device`, asynchronously from pinned memory."""
        return Events(*(field.to(device, non_blocking=True) for field in self))

    def gather(self, *, starts: torch.Tensor, lengths: torch.Tensor, width: int) -> 'Events':
        """Gather runs from flat events.

        Args:
            starts: `[B]` the flat row each run starts at.
            lengths: `[B]` the events of each run.
            width: The padded width, at least the longest run.

        Returns:
            The runs, `[B, width]` per field.
        """
        rows, real = _runs(starts=starts, lengths=lengths, width=width)
        return Events(
            *(_take(channel, rows=rows, real=real) for channel in self[:-1]), length=lengths
        )


class TraceCut(NamedTuple):
    """A batch of cases cut into a prefix and a suffix, with the decoder's time target aligned to
    the suffix positions it is read at."""

    prefix: Events  # the condition: the events before the cut, no EOT
    suffix: Events  # what the decoder must produce: content, EOT, then padding
    # Standardized minutes until the event at each suffix position, in the target encoding.
    inter_event_times: torch.Tensor  # float32, [B, T], 0.0 from the EOT on

    def timed_positions(self) -> torch.Tensor:
        """`[B, T]`, True at real suffix events, excluding the terminal EOT and padding."""
        positions = torch.arange(
            end=self.suffix.activities.size(dim=1), device=self.suffix.length.device
        )
        return positions.unsqueeze(dim=0) < (self.suffix.length - 1).unsqueeze(dim=1)

    def to(self, device: torch.device) -> 'TraceCut':
        """Move every tensor field to `device`, asynchronously from pinned memory."""
        return TraceCut(
            prefix=self.prefix.to(device),
            suffix=self.suffix.to(device),
            inter_event_times=self.inter_event_times.to(device, non_blocking=True),
        )


def encode_events(codec: Codec, events: pd.DataFrame) -> Events:
    """Encode events, their derived features included, as one flat run `[E]` of every row."""
    numeric_attributes, numeric_attributes_present = codec.encode_numeric_features(events)
    return Events(
        # Pandas can expose a read-only view, so the integer channels are copied.
        activities=torch.tensor(data=codec.activity.encode(events[ACTIVITY_KEY])),
        resources=torch.tensor(data=codec.resource.encode(events)),
        inter_event_times=torch.from_numpy(codec.input_inter_event_time.encode(events)),
        categorical_attributes=torch.from_numpy(codec.encode_categorical_features(events)),
        numeric_attributes=torch.from_numpy(numeric_attributes),
        numeric_attributes_present=torch.from_numpy(numeric_attributes_present),
        length=torch.tensor(data=len(events), dtype=torch.long),
    )


def encode_prefixes(codec: Codec, prefixes: PrefixBatch) -> Events:
    """Encode a batch of observed prefixes, padded to the longest of them.

    Returns:
        The prefixes, `[N, P]` per field. Nothing past a prefix's last event is read.
    """
    events = encode_events(codec, derive_features(prefixes.events, prefixes.offsets))
    offsets = torch.from_numpy(prefixes.offsets)
    lengths = offsets.diff()
    return events.gather(starts=offsets[:-1], lengths=lengths, width=int(lengths.max()))


class TraceDataset(Dataset):
    """Every cut of a split as a (prefix, suffix) pair, read a batch of cuts at a time.

    A cut of `k` events yields the prefix `events[:k]` and the suffix `events[k:]` followed by
    EOT, each padded to the longest of its batch.
    """

    def __init__(self, log: Log, codec: Codec) -> None:
        """Encode the split whole and enumerate its cuts.

        Args:
            log: A benchmark split.
            codec: The codec fitted on the train split.
        """
        self.eot_index = codec.activity.eot_index
        offsets = log.case_offsets
        events = derive_features(log.events, offsets)
        self._events = encode_events(codec, events)
        self._target = torch.from_numpy(codec.inter_event_time.encode(events))
        cuts = log.cuts()
        case = log.case_indices(cuts)
        # [N] The first row of every cut's case, its prefix length, and its suffix length.
        self._starts = torch.from_numpy(offsets[case])
        self._prefix_lengths = torch.tensor([cut.prefix_len for cut in cuts], dtype=torch.long)
        self._suffix_lengths = torch.from_numpy(offsets[case + 1]) - (
            self._starts + self._prefix_lengths
        )

    def __len__(self) -> int:
        return len(self._starts)

    def __getitem__(self, indices: list[int]) -> TraceCut:
        """Return a batch of cuts, prefixes and suffixes each padded to the longest of them."""
        index = torch.tensor(indices)
        starts, k, n = self._starts[index], self._prefix_lengths[index], self._suffix_lengths[index]
        prefix = self._events.gather(starts=starts, lengths=k, width=int(k.max()))
        # One position more than the longest suffix, for its EOT.
        width = int(n.max()) + 1
        suffix = self._events.gather(starts=starts + k, lengths=n, width=width)
        rows, real = _runs(starts=starts + k, lengths=n, width=width)
        return TraceCut(
            prefix=prefix,
            suffix=suffix._replace(
                activities=suffix.activities.scatter(
                    dim=1, index=n.unsqueeze(dim=1), value=self.eot_index
                ),
                length=n + 1,
            ),
            inter_event_times=_take(self._target, rows=rows, real=real),
        )


def batches(
    dataset: Dataset, *, batch_size: int, generator: torch.Generator | None = None
) -> DataLoader:
    """Load whole batches of `TraceDataset` cuts into pinned memory, shuffled under `generator`
    and in order without it."""
    sampler = (
        RandomSampler(data_source=dataset, generator=generator)
        if generator is not None
        else SequentialSampler(data_source=dataset)
    )
    return DataLoader(
        dataset=dataset,
        batch_size=None,
        sampler=BatchSampler(sampler=sampler, batch_size=batch_size, drop_last=False),
        pin_memory=torch.cuda.is_available(),
    )


def fixed_subset(dataset: Dataset, *, size: int, generator: torch.Generator) -> Dataset:
    """A random subset of `dataset` drawn once, or the whole of it if it is no bigger.

    Every validation of a run then reads the same pairs, so two of its points differ because the
    model moved rather than because the sample did.
    """
    if len(dataset) <= size:
        return dataset
    indices = torch.randperm(n=len(dataset), generator=generator)[:size]
    return Subset(dataset=dataset, indices=indices.tolist())


def _runs(
    *, starts: torch.Tensor, lengths: torch.Tensor, width: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """`[B, width]` flat rows of each run's positions, and whether each is a real event."""
    positions = torch.arange(end=width)
    real = positions.unsqueeze(dim=0) < lengths.unsqueeze(dim=1)  # [B, width]
    rows = (starts.unsqueeze(dim=1) + positions).masked_fill(mask=~real, value=0)  # [B, width]
    return rows, real


def _take(channel: torch.Tensor, *, rows: torch.Tensor, real: torch.Tensor) -> torch.Tensor:
    """Read a flat channel `[E, ...]` at `rows` `[B, W]`, zero where not `real`: `[B, W, ...]`."""
    taken = channel[rows]
    return taken.masked_fill(mask=~real.view(*real.shape, *[1] * (taken.dim() - 2)), value=0)
