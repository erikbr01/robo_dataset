"""Transform ABCs for the three-tier data-loading system + inference pipeline.

Tier 1 – DiskCachedTransform:  offline, one-time, applied during conversion.
Tier 2 – CPUBatchTransform:    online, applied in DataLoader collate_fn.
Tier 3 – GPUBatchTransform:    online, applied after DataLoader returns, on GPU.

Inference pipeline (separate from the data-loading tiers):
  InferenceTransform / InferencePipeline — applied identically at train time
  and inference time, to model inputs (obs) or model outputs (actions).
  These are serialised into checkpoints so inference only needs the .ckpt.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List

import torch


# ─── Inference pipeline ─────────────────────────────────────────────────────

class InferenceTransform(ABC):
    """Single transform applied identically at train and inference time.

    Subclasses operate on either an obs dict (preprocess) or an action
    tensor / array (postprocess).
    """

    @abstractmethod
    def __call__(self, x: Any) -> Any: ...


class InferencePipeline:
    """Chains InferenceTransforms sequentially.

    Designed for Hydra instantiation via _target_:

        _target_: robo_dataset.transforms.InferencePipeline
        transforms:
          - _target_: robo_dataset.transforms.NormalizeImages
            keys: [agentview_image]
          - ...

    Both a preprocess pipeline (obs → obs) and a postprocess pipeline
    (action → action) are serialised into the checkpoint so inference
    requires only the .ckpt file, not the original dataset.
    """

    def __init__(self, transforms: List[InferenceTransform]):
        self.transforms = list(transforms)

    def __call__(self, x: Any) -> Any:
        for t in self.transforms:
            x = t(x)
        return x

    def __repr__(self) -> str:
        names = ", ".join(type(t).__name__ for t in self.transforms)
        return f"InferencePipeline([{names}])"


# ─── Data-loading tiers ─────────────────────────────────────────────────────


class DiskCachedTransform(ABC):
    """
    Offline transform applied per-episode during a conversion pass.
    Results are written to a WritableTrajectoryStore and re-used on subsequent runs
    (keyed by a hash of the config + source code of apply_episode).
    """

    @abstractmethod
    def apply_episode(self, ep_data: Dict[str, Any]) -> Dict[str, Any]:
        """Transform one episode's data. Receives/returns dict of ndarray(T, *shape)."""

    def cache_key(self) -> str:
        """Return a string that uniquely identifies this transform version."""
        import hashlib
        import inspect
        src = inspect.getsource(self.__class__)
        cfg = repr(sorted(self.__dict__.items()))
        return hashlib.sha256((src + cfg).encode()).hexdigest()[:16]


class CPUBatchTransform(ABC):
    """
    Online CPU transform applied to a collated batch dict in the DataLoader collate_fn.
    Input/output: Dict[str, Tensor] with shape (B, T, *frame_shape).
    """

    @abstractmethod
    def __call__(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        ...


class GPUBatchTransform(ABC):
    """
    Online GPU transform applied after DataLoader returns, before model forward.
    Input/output: Dict[str, Tensor] already on device, shape (B, T, *frame_shape) or (B, *shape).
    """

    @abstractmethod
    def __call__(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        ...
