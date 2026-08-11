"""Safe HDF5 reader for v2 training shards.

The reader is intentionally small: training projects may wrap it, but cannot
accidentally consume non-Volt, unpinned, channel-order-incompatible shards.
Mixed production epochs are rejected by default; perception-only consumers get
the recorded clean mask rather than an implicit time crop.
"""
from __future__ import annotations

from pathlib import Path


def _channel_hash(channels: list[str]) -> str:
    import hashlib
    return hashlib.sha256(("\n".join(channels) + "\n").encode()).hexdigest()


class TrainingShardDataset:
    def __init__(self, shard_path: str | Path, channel_order: list[str], split_index_sha256: str,
                 allow_mixed_production: bool = False):
        import h5py
        self.path = Path(shard_path)
        self.h5 = h5py.File(self.path, "r")
        if self.h5.attrs.get("eeg_unit") != "V":
            raise ValueError("refusing non-Volt EEG shard")
        if self.h5.attrs.get("channel_order_hash") != _channel_hash(channel_order):
            raise ValueError("refusing channel-order-incompatible shard")
        if not split_index_sha256 or self.h5.attrs.get("split_index_sha256", "") != split_index_sha256:
            raise ValueError("refusing shard without the requested frozen split hash")
        self.allow_mixed_production = allow_mixed_production
        self.ids = [x.decode() if isinstance(x, bytes) else x for x in self.h5["provenance/trial_id"][:]]

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index: int):
        record = {"trial_id": self.ids[index], "eeg": self.h5["eeg"][index],
                  "eeg_valid_mask": self.h5["eeg_valid_mask"][index],
                  "clean_perception_mask": self.h5["clean_perception_mask"][index],
                  "audio_loss_mask": self.h5["audio_loss_mask"][index]}
        mixed = not bool(record["clean_perception_mask"].all())
        if mixed and not self.allow_mixed_production:
            # Return precisely the safe samples.  No default caller can train
            # on response EMG/self-feedback by just forgetting a flag.
            mask = record["clean_perception_mask"]
            record["eeg"] = record["eeg"][:, mask]
            record["eeg_valid_mask"] = record["eeg_valid_mask"][mask]
            record["audio_loss_mask"] = record["audio_loss_mask"][mask]
        return record

    def close(self):
        self.h5.close()
