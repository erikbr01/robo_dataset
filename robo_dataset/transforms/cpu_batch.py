"""CPU batch transforms — applied in DataLoader collate_fn.

Tensors arrive from the DataLoader with shape (B, T, *frame_shape):
  image keys:    (B, T, H, W, C)  uint8
  lowdim/action: (B, T, D)        float32

The typical stack for diffusion-policy-style training is:

    ComposeCPU([
        SplitObsAction(n_obs_steps=2, obs_keys=[...], action_key="action"),
        ImageToTensor(image_keys=[...]),
    ])

which produces:
  image keys:    (B, n_obs_steps, C, H, W)  float32 in [0, 1]
  lowdim keys:   (B, n_obs_steps, D)        float32
  action:        (B, T, action_dim)         float32   ← full horizon kept
"""

from typing import Dict, List, Optional

import torch

from .base import CPUBatchTransform


class SplitObsAction(CPUBatchTransform):
    """
    Slice the observation window and separate it from the action sequence.

    Observation keys (obs_keys) are trimmed to the first n_obs_steps timesteps.
    The action key is kept at full horizon length T.

    Input shape:  (B, T, ...)  for all keys
    Output shape: obs keys → (B, n_obs_steps, ...) | action → (B, T, ...)

    Example for horizon=16, n_obs_steps=2:
      obs   : (B, 16, H, W, C) → (B, 2, H, W, C)
      action: (B, 16, D)       stays (B, 16, D)
    """

    def __init__(
        self,
        n_obs_steps: int,
        obs_keys: List[str],
        action_key: str = "action",
    ):
        self.n_obs_steps = n_obs_steps
        self.obs_keys = set(obs_keys)
        self.action_key = action_key

    def __call__(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        out = {}
        for key, val in batch.items():
            if key in self.obs_keys:
                out[key] = val[:, : self.n_obs_steps]
            else:
                out[key] = val
        return out


class ImageToTensor(CPUBatchTransform):
    """
    Convert image tensors from (B, T, H, W, C) uint8  →  (B, T, C, H, W) float32 in [0, 1].

    This is a format conversion (HWC→CHW, uint8→float), not statistical normalisation.
    Statistical normalisation (mean/std) belongs in the model.
    """

    def __init__(self, image_keys: List[str]):
        self.image_keys = set(image_keys)

    def __call__(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        out = dict(batch)
        for key in self.image_keys:
            if key not in batch:
                continue
            # (B, T, H, W, C) uint8  →  (B, T, C, H, W) float32
            out[key] = batch[key].permute(0, 1, 4, 2, 3).float().div(255.0)
        return out


class NormalizeRange(CPUBatchTransform):
    """
    Affine rescale: [in_min, in_max] → [out_min, out_max] per selected key.

    in_min / in_max may be scalars or arrays broadcastable to the key's shape.
    """

    def __init__(
        self,
        keys: List[str],
        in_min,
        in_max,
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
    """Return a DataLoader collate_fn that stacks samples and applies transform."""

    def collate_fn(samples):
        batch = {key: torch.stack([s[key] for s in samples]) for key in samples[0]}
        if transform is not None:
            batch = transform(batch)
        return batch

    return collate_fn
