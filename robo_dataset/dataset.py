"""TrajectoryDataset — torch Dataset wrapping a TrajectoryStore + SequenceSampler."""

import copy
from typing import Dict, List, Optional

import numpy as np
import torch
from torch.utils.data import Dataset

from .backends.base import TrajectoryStore
from .sampler import SequenceSampler, get_val_mask


class TrajectoryDataset(Dataset):
    """
    Torch Dataset over a TrajectoryStore.

    Returns dict of tensors with shape (horizon, *frame_shape) per key.
    Image keys are kept as uint8; callers should normalize as needed.

    For use with DataLoader, pass worker_init_fn=TrajectoryDataset.worker_init_fn.
    For distributed training, wrap with torch.utils.data.DistributedSampler.
    """

    def __init__(
        self,
        store: TrajectoryStore,
        horizon: int,
        pad_before: int = 0,
        pad_after: int = 0,
        keys: Optional[List[str]] = None,
        key_first_k: Optional[Dict[str, int]] = None,
        episode_mask: Optional[np.ndarray] = None,
        val_ratio: float = 0.0,
        seed: int = 42,
    ):
        self.store = store
        self.horizon = horizon
        self.pad_before = pad_before
        self.pad_after = pad_after

        if episode_mask is None and val_ratio > 0:
            val_mask = get_val_mask(store.n_episodes, val_ratio, seed)
            episode_mask = ~val_mask

        self._sampler = SequenceSampler(
            store=store,
            sequence_length=horizon,
            pad_before=pad_before,
            pad_after=pad_after,
            keys=keys,
            key_first_k=key_first_k,
            episode_mask=episode_mask,
        )
        self._episode_mask = episode_mask

    def get_validation_dataset(self) -> "TrajectoryDataset":
        val = copy.copy(self)
        val._episode_mask = ~self._episode_mask
        val._sampler = SequenceSampler(
            store=self.store,
            sequence_length=self.horizon,
            pad_before=self.pad_before,
            pad_after=self.pad_after,
            keys=self._sampler.keys,
            key_first_k=self._sampler.key_first_k,
            episode_mask=val._episode_mask,
        )
        return val

    def __len__(self) -> int:
        return len(self._sampler)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self._sampler.sample_sequence(idx)
        return {key: torch.from_numpy(arr) for key, arr in sample.items()}

    @staticmethod
    def worker_init_fn(worker_id: int) -> None:
        """Pass to DataLoader(worker_init_fn=...) for safe multi-process file access."""
        info = torch.utils.data.get_worker_info()
        info.dataset.store.on_worker_init(worker_id)

    def make_dataloader(
        self,
        batch_size: int = 32,
        num_workers: int = 8,
        shuffle: bool = True,
        pin_memory: bool = True,
        persistent_workers: bool = True,
        sampler=None,
        **kwargs,
    ) -> torch.utils.data.DataLoader:
        """Convenience method; returns a DataLoader with correct worker_init_fn."""
        return torch.utils.data.DataLoader(
            self,
            batch_size=batch_size,
            shuffle=(shuffle and sampler is None),
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=(persistent_workers and num_workers > 0),
            worker_init_fn=TrajectoryDataset.worker_init_fn if num_workers > 0 else None,
            sampler=sampler,
            **kwargs,
        )
