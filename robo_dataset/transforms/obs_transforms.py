"""Observation-space InferenceTransforms.

Applied to the obs dict BEFORE policy.predict_action() — identically at
train time and inference time.

Only includes transforms with FIXED (non-data-dependent) parameters.
Data-dependent normalisation (e.g. LinearNormalizer fitted from training
statistics) lives inside the policy and is handled separately.
"""

from typing import Dict, List, Tuple, Union

import numpy as np
import torch

from robo_dataset.transforms.base import InferenceTransform


class NormalizeImages(InferenceTransform):
    """Normalise RGB images using per-channel mean / std.

    Converts uint8 [0, 255] or float [0, 1] CHW images to normalised
    float tensors.  Default statistics are ImageNet mean / std.
    Accepts any number of leading batch / time dimensions.

    Args:
        keys:        obs-dict keys to normalise.
        mean:        per-channel mean (length C).
        std:         per-channel std  (length C).
        from_uint8:  divide uint8 input by 255 before normalising.
    """

    def __init__(
        self,
        keys: List[str],
        mean: Tuple[float, ...] = (0.485, 0.456, 0.406),
        std: Tuple[float, ...] = (0.229, 0.224, 0.225),
        from_uint8: bool = True,
    ):
        self.keys = list(keys)
        self.from_uint8 = from_uint8
        self._mean = list(mean)
        self._std = list(std)

    def _stats(self, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        mean = torch.tensor(self._mean, dtype=torch.float32, device=device).view(-1, 1, 1)
        std  = torch.tensor(self._std,  dtype=torch.float32, device=device).view(-1, 1, 1)
        return mean, std

    def __call__(
        self,
        obs_dict: Dict[str, Union[np.ndarray, torch.Tensor]],
    ) -> Dict[str, Union[np.ndarray, torch.Tensor]]:
        result = dict(obs_dict)
        for key in self.keys:
            if key not in result:
                continue
            x = result[key]
            was_numpy = isinstance(x, np.ndarray)
            if was_numpy:
                x = torch.from_numpy(x)
            if self.from_uint8 and x.dtype == torch.uint8:
                x = x.float() / 255.0
            elif x.dtype != torch.float32:
                x = x.float()
            mean, std = self._stats(x.device)
            x = (x - mean) / std
            result[key] = x.numpy() if was_numpy else x
        return result


class ToFloat(InferenceTransform):
    """Convert uint8 image observations to float32 in [0, 1].

    Use when downstream code handles its own normalisation.
    """

    def __init__(self, keys: List[str]):
        self.keys = list(keys)

    def __call__(
        self,
        obs_dict: Dict[str, Union[np.ndarray, torch.Tensor]],
    ) -> Dict[str, Union[np.ndarray, torch.Tensor]]:
        result = dict(obs_dict)
        for key in self.keys:
            if key not in result:
                continue
            x = result[key]
            was_numpy = isinstance(x, np.ndarray)
            if was_numpy:
                x = torch.from_numpy(x)
            if x.dtype == torch.uint8:
                x = x.float() / 255.0
            result[key] = x.numpy() if was_numpy else x
        return result


class IdentityObs(InferenceTransform):
    """No-op obs preprocess — pass-through."""

    def __call__(self, obs_dict: dict) -> dict:
        return obs_dict
