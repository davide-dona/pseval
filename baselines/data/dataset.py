from typing import NamedTuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, Subset

from baselines.data.codec import Codec
from baselines.data.features import derive_features
from pseval import Log, PrefixBatch
from pseval.data import ACTIVITY_KEY, CASE_KEY


class Events(NamedTuple):
    """The events of one run, or of a batch of runs once stacked, as the model reads them."""

    activities: torch.Tensor  # int64, [..., T]
    resources: torch.Tensor  # int64, [..., T]
    inter_event_times: torch.Tensor  # float32, standardized input encoding, [..., T]
    categorical_attributes: torch.Tensor  # int64, [..., T, C]
    numeric_attributes: torch.Tensor  # float32, standardized, [..., T, N]
    # float32 0/1, 0.0 where the log had no value, [..., T, N]
    numeric_attributes_present: torch.Tensor
    length: torch.Tensor  # int64, the real, unpadded length of each run, [...]

    def _channels(self) -> dict[str, torch.Tensor]:
        """Return the per-event fields, excluding the event count."""
        return {name: getattr(self, name) for name in self._fields if name != 'length'}

    def cut(self, index: slice) -> 'Events':
        """Slice every per-event field of one run, updating `length`."""
        channels = {name: channel[index] for name, channel in self._channels().items()}
        return self._replace(
            **channels, length=torch.tensor(data=len(channels['activities']), dtype=torch.long)
        )

    def padded(self, to: int, *, pad_activity: int) -> 'Events':
        """Pad every per-event field of one run to `to` positions, leaving `length` unchanged.

        Activities are padded with `pad_activity` and every other channel with zeros, the PAD row
        of the resource and categorical feature tables.
        """
        return self._replace(
            **{
                name: torch.cat(
                    tensors=(
                        channel,
                        torch.full(
                            size=(to - channel.size(dim=0), *channel.shape[1:]),
                            fill_value=pad_activity if name == 'activities' else 0,
                            dtype=channel.dtype,
                        ),
                    )
                )
                for name, channel in self._channels().items()
            }
        )

    def pad_mask(self) -> torch.Tensor:
        """`[B, T]`, True where a position of a batch holds padding."""
        positions = torch.arange(end=self.activities.size(dim=-1), device=self.length.device)
        return positions.unsqueeze(dim=0) >= self.length.unsqueeze(dim=1)

    def rows(self, index: slice) -> 'Events':
        """Select runs of a batch, every field along its first dimension."""
        return Events(*(field[index] for field in self))

    def repeat_rows(self, repeats: int) -> 'Events':
        """Repeat each run of a batch `repeats` times, the copies of a run adjacent."""
        return Events(*(field.repeat_interleave(repeats, dim=0) for field in self))

    def to(self, device: torch.device) -> 'Events':
        """Move every field to `device`."""
        return Events(*(field.to(device) for field in self))

    @classmethod
    def stack(cls, runs: list['Events']) -> 'Events':
        """Stack runs padded to one width into a batch."""
        return cls(*(torch.stack(fields) for fields in zip(*runs, strict=True)))


class TraceCut(NamedTuple):
    """One case cut into a prefix and a suffix, with the decoder's time target aligned to the
    suffix positions it is read at."""

    prefix: Events  # the condition: the events before the cut, no EOT
    suffix: Events  # what the decoder must produce: content, EOT, then padding
    # Standardized minutes until the event at each suffix position, in the target encoding.
    inter_event_times: torch.Tensor  # float32, [max_trace_length], batched [B, ...]

    def timed_positions(self) -> torch.Tensor:
        """`[B, T]`, True at real suffix events, excluding the terminal EOT and padding."""
        positions = torch.arange(
            end=self.suffix.activities.size(dim=1), device=self.suffix.length.device
        )
        return positions.unsqueeze(dim=0) < (self.suffix.length - 1).unsqueeze(dim=1)

    def to(self, device: torch.device) -> 'TraceCut':
        """Move every tensor field to `device`."""
        return TraceCut(
            prefix=self.prefix.to(device),
            suffix=self.suffix.to(device),
            inter_event_times=self.inter_event_times.to(device),
        )


def encode_events(codec: Codec, events: pd.DataFrame) -> Events:
    """Encode events, their derived features included, as one run of every row."""
    numeric_attributes, numeric_attributes_present = codec.encode_numeric_features(events)
    return Events(
        # Pandas can expose a read-only view, so the integer channels are copied.
        activities=torch.tensor(data=codec.activity.encode(events[ACTIVITY_KEY])),
        resources=torch.tensor(data=codec.resource.encode(events)),
        inter_event_times=torch.from_numpy(codec.input_inter_event_time.encode(events)),
        categorical_attributes=codec.encode_categorical_features(events),
        numeric_attributes=numeric_attributes,
        numeric_attributes_present=numeric_attributes_present,
        length=torch.tensor(data=len(events), dtype=torch.long),
    )


def encode_prefixes(codec: Codec, prefixes: PrefixBatch) -> Events:
    """Encode a batch of observed prefixes, padded to the longest of them.

    Returns:
        The prefixes, `[N, P]` per field. Nothing past a prefix's last event is read.
    """
    events = encode_events(codec, derive_features(prefixes.events, prefixes.offsets))
    width = int(np.diff(prefixes.offsets).max())
    return Events.stack(
        [
            events.cut(slice(int(start), int(stop))).padded(
                to=width, pad_activity=codec.activity.pad_index
            )
            for start, stop in zip(prefixes.offsets[:-1], prefixes.offsets[1:], strict=True)
        ]
    )


class TraceDataset(Dataset):
    """Every cut of a split as a (prefix, suffix) pair, both padded to `max_trace_length`.

    A cut of `k` events yields the prefix `events[:k]` and the suffix `events[k:]` followed by
    EOT.
    """

    def __init__(self, log: Log, codec: Codec) -> None:
        """Encode the split whole and enumerate its cuts.

        Args:
            log: A benchmark split.
            codec: The codec fitted on the train split.
        """
        self.codec = codec
        self.max_len = codec.max_trace_length
        offsets = log.case_offsets
        events = derive_features(log.events, offsets)
        encoded = encode_events(codec, events)
        target = torch.from_numpy(codec.inter_event_time.encode(events))
        bounds = list(zip(offsets[:-1].tolist(), offsets[1:].tolist(), strict=True))
        # Each case's events and time target, encoded once and shared by all of its cuts.
        self._cases = [
            (encoded.cut(slice(start, stop)), target[start:stop]) for start, stop in bounds
        ]
        case_ids = events[CASE_KEY].astype(str).to_numpy(dtype=object)[offsets[:-1]]
        case_index = {case_id: index for index, case_id in enumerate(case_ids)}
        self._cuts = [(case_index[cut.case_id], cut.prefix_len) for cut in log.cuts()]

    def __len__(self) -> int:
        return len(self._cuts)

    def __getitem__(self, i: int) -> TraceCut:
        """Return the i-th cut, both of its runs padded to `max_trace_length`."""
        case_index, k = self._cuts[i]
        events, target = self._cases[case_index]
        suffix_len = int(events.length) - k
        pad = self.codec.activity.pad_index
        prefix = events.cut(slice(0, k)).padded(to=self.max_len, pad_activity=pad)
        suffix = events.cut(slice(k, None)).padded(to=self.max_len, pad_activity=pad)
        suffix.activities[suffix_len] = self.codec.activity.eot_index
        return TraceCut(
            prefix=prefix,
            suffix=suffix._replace(length=suffix.length + 1),
            # Zero past the suffix's content, which the loss masks out.
            inter_event_times=F.pad(input=target[k:], pad=(0, self.max_len - suffix_len)),
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
