"""Transform ABCs for the three-tier system.

Tier 1 – DiskCachedTransform:  offline, one-time, applied during conversion.
Tier 2 – CPUBatchTransform:    online, applied in DataLoader collate_fn.
Tier 3 – GPUBatchTransform:    online, applied after DataLoader returns, on GPU.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List

import torch


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
