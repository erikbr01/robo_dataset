"""CPU batch transforms — applied in DataLoader collate_fn.

Input/output: Dict[str, Tensor] with shapes:
  image keys:    (B, T, H, W, C)  uint8
  lowdim/action: (B, T, D)        float32
"""

from typing import Dict, List, Optional

import torch

from .base import CPUBatchTransform


class ActionChunk(CPUBatchTransform):
    """
    Selects n_obs_steps observation frames and the full action horizon.

    Converts images from (B, T, H, W, C) uint8 → (B, T, C, H, W) float32 / 255.

    Args:
        n_obs_steps:  How many obs timesteps to keep (from the start of the window).
        image_keys:   Keys to normalize to [0, 1] and permute to (C, H, W).
        action_key:   Key for the action sequence (kept as-is).
    """

    def __init__(
        self,
        n_obs_steps: int,
        image_keys: List[str] = (),
        action_key: str = "action",
    ):
        self.n_obs_steps = n_obs_steps
        self.image_keys = set(image_keys)
        self.action_key = action_key

    def __call__(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        out = {}
        for key, val in batch.items():
            if key in self.image_keys:
                # (B, T, H, W, C) uint8 → (B, n_obs, C, H, W) float32
                obs = val[:, : self.n_obs_steps]  # (B, n_obs, H, W, C)
                obs = obs.permute(0, 1, 4, 2, 3).float().div(255.0)
                out[key] = obs
            elif key == self.action_key:
                out[key] = val.float()
            else:
                # Generic lowdim: keep first n_obs_steps
                out[key] = val[:, : self.n_obs_steps].float()
        return out


class NormalizeRange(CPUBatchTransform):
    """
    Normalize selected keys from [in_min, in_max] → [out_min, out_max].
    Useful for lowdim observations and actions.
    """

    def __init__(
        self,
        keys: List[str],
        in_min: float,
        in_max: float,
        out_min: float = -1.0,
        out_max: float = 1.0,
    ):
        self.keys = set(keys)
        self.in_min = in_min
        self.in_max = in_max
        self.out_min = out_min
        self.out_max = out_max

    def __call__(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        out = dict(batch)
        scale = (self.out_max - self.out_min) / (self.in_max - self.in_min)
        for key in self.keys:
            if key in out:
                out[key] = (out[key].float() - self.in_min) * scale + self.out_min
        return out


class ComposeCPU(CPUBatchTransform):
    """Apply a sequence of CPUBatchTransforms in order."""

    def __init__(self, transforms: List[CPUBatchTransform]):
        self.transforms = transforms

    def __call__(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        for t in self.transforms:
            batch = t(batch)
        return batch


def make_collate_fn(transform: Optional[CPUBatchTransform] = None):
    """
    Return a DataLoader collate_fn that stacks samples and optionally applies transform.
    """
    import torch.utils.data

    def collate_fn(samples):
        batch = {key: torch.stack([s[key] for s in samples]) for key in samples[0]}
        if transform is not None:
            batch = transform(batch)
        return batch

    return collate_fn
