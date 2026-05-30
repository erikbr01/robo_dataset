"""GPU batch transforms — applied after DataLoader returns, before model forward.

Operates on batches already on the target device.
Image tensors are expected as (B, T, C, H, W) float32 in [0, 1].
"""

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

from .base import GPUBatchTransform


class RandomCrop(GPUBatchTransform):
    """
    Random crop augmentation for image keys.

    Pads each image by `pad` pixels on each side, then takes a random `size × size` crop.
    The same crop offset is applied to all timesteps within one batch element.

    Args:
        image_keys:  Keys to apply cropping to.
        size:        Output spatial size (H, W) after crop.
        pad:         Padding on each side before crop.
    """

    def __init__(
        self,
        image_keys: List[str],
        size: Tuple[int, int],
        pad: int = 4,
    ):
        self.image_keys = set(image_keys)
        self.size = size
        self.pad = pad

    @torch.no_grad()
    def __call__(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        out = dict(batch)
        for key in self.image_keys:
            if key not in batch:
                continue
            x = batch[key]  # (B, T, C, H, W)
            B, T, C, H, W = x.shape
            oh, ow = self.size

            # Pad: (B, T, C, H+2p, W+2p)
            x_pad = F.pad(x.view(B * T, C, H, W), [self.pad] * 4, mode="reflect")
            PH, PW = H + 2 * self.pad, W + 2 * self.pad

            # Sample one crop offset per batch element (shared across T)
            top = torch.randint(0, PH - oh + 1, (B,), device=x.device)
            left = torch.randint(0, PW - ow + 1, (B,), device=x.device)

            crops = []
            for b in range(B):
                # Extract crop for all T timesteps of this batch element
                patch = x_pad[b * T : (b + 1) * T, :, top[b] : top[b] + oh, left[b] : left[b] + ow]
                crops.append(patch)
            out[key] = torch.stack(crops)  # (B, T, C, oh, ow)
        return out


class ColorJitter(GPUBatchTransform):
    """
    Random brightness/contrast/saturation jitter for image keys.
    Applied independently per batch element (same jitter across timesteps).

    Operates on (B, T, C, H, W) float32 [0, 1] images.
    """

    def __init__(
        self,
        image_keys: List[str],
        brightness: float = 0.1,
        contrast: float = 0.1,
        saturation: float = 0.1,
    ):
        self.image_keys = set(image_keys)
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation

    @torch.no_grad()
    def __call__(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        out = dict(batch)
        for key in self.image_keys:
            if key not in batch:
                continue
            x = batch[key]  # (B, T, C, H, W) in [0, 1]
            B = x.shape[0]

            # Sample per-batch deltas
            b_delta = torch.empty(B, 1, 1, 1, 1, device=x.device).uniform_(
                1 - self.brightness, 1 + self.brightness
            )
            c_delta = torch.empty(B, 1, 1, 1, 1, device=x.device).uniform_(
                1 - self.contrast, 1 + self.contrast
            )
            mean = x.mean(dim=(-1, -2, -3), keepdim=True)
            x = ((x - mean) * c_delta + mean) * b_delta
            x = x.clamp(0.0, 1.0)
            out[key] = x
        return out


class ComposeGPU(GPUBatchTransform):
    """Apply a sequence of GPUBatchTransforms in order."""

    def __init__(self, transforms: List[GPUBatchTransform]):
        self.transforms = transforms

    def __call__(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        for t in self.transforms:
            batch = t(batch)
        return batch
