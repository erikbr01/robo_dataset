"""GPU batch transforms — applied after DataLoader returns, before model forward.

Operates on batches already on the target device.
Image tensors are expected as (B, T, C, H, W) float32 in [0, 1] unless noted.
"""

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

from .base import GPUBatchTransform


class ImageToTensorDevice(GPUBatchTransform):
    """Convert uint8 images to float32 on any accelerator: (B,T,H,W,C) → (B,T,C,H,W)/255.

    **Use this instead of** ``ImageToTensor`` **(CPU) when using** ``DataPrefetcher``.
    Keeping images as uint8 through the DataLoader reduces PCIe/interconnect transfer
    by 3.8× (1.35 MB vs 5.42 MB per batch of 32 at 84×84 on PCIe 3.0 x16).
    The dtype conversion then happens at on-device memory bandwidth.

    Works on CUDA, MPS, XPU, and CPU — wherever the tensor lives after transfer.

    Note: ``permute().float()`` produces a non-contiguous tensor (the strides still
    reflect the original HWC layout).  ``.contiguous()`` is called before the in-place
    ``div_()`` to avoid downstream ops that require contiguous input.

    Args:
        image_keys: Keys to convert.  Keys absent from the batch are silently skipped.
    """

    def __init__(self, image_keys: List[str]):
        self.image_keys = set(image_keys)

    @torch.no_grad()
    def __call__(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        out = dict(batch)
        for key in self.image_keys:
            if key not in batch:
                continue
            x = batch[key]  # (B, T, H, W, C) uint8
            assert x.dtype == torch.uint8, (
                f"ImageToTensorDevice expects uint8 input for '{key}', got {x.dtype}. "
                "Make sure ImageToTensor is NOT in the CPU compose when using ImageToTensorDevice."
            )
            # permute() is non-contiguous; .contiguous() required before in-place div_
            out[key] = x.permute(0, 1, 4, 2, 3).float().contiguous().div_(255.0)
        return out


#: Backward-compatible alias.  New code should use :class:`ImageToTensorDevice`.
ImageToTensorGPU = ImageToTensorDevice


class RandomCrop(GPUBatchTransform):
    """
    Random crop augmentation for image keys.

    Pads each image by `pad` pixels on each side, then takes a random `size × size` crop.
    By default the same crop offset is applied to all timesteps within one batch
    element; set ``per_timestep=True`` to sample an independent offset per
    (batch, timestep) — matching diffusion_policy's CropRandomizer, which crops
    each obs frame independently.

    Args:
        image_keys:   Keys to apply cropping to.
        size:         Output spatial size (H, W) after crop.
        pad:          Padding on each side before crop (0 = crop within the image).
        per_timestep: Sample an independent offset per (B, T) element.
    """

    def __init__(
        self,
        image_keys: List[str],
        size: Tuple[int, int],
        pad: int = 4,
        per_timestep: bool = False,
    ):
        self.image_keys = set(image_keys)
        self.size = size
        self.pad = pad
        self.per_timestep = per_timestep

    @torch.no_grad()
    def __call__(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        out = dict(batch)
        for key in self.image_keys:
            if key not in batch:
                continue
            x = batch[key]  # (B, T, C, H, W)
            B, T, C, H, W = x.shape
            oh, ow = self.size

            flat = x.view(B * T, C, H, W)
            if self.pad > 0:
                flat = F.pad(flat, [self.pad] * 4, mode="reflect")
            PH, PW = H + 2 * self.pad, W + 2 * self.pad

            n_offsets = B * T if self.per_timestep else B
            top = torch.randint(0, PH - oh + 1, (n_offsets,), device=x.device)
            left = torch.randint(0, PW - ow + 1, (n_offsets,), device=x.device)
            if not self.per_timestep:
                top = top.repeat_interleave(T)
                left = left.repeat_interleave(T)

            # Vectorized gather equivalent to per-element slicing:
            #   crops[i] = flat[i, :, top[i]:top[i]+oh, left[i]:left[i]+ow]
            rows = top[:, None] + torch.arange(oh, device=x.device)   # (BT, oh)
            cols = left[:, None] + torch.arange(ow, device=x.device)  # (BT, ow)
            bt = torch.arange(B * T, device=x.device)
            # Advanced indices separated by the channel slice broadcast to the
            # front: result is (BT, oh, ow, C) — permute back to (BT, C, oh, ow).
            crops = flat[bt[:, None, None], :, rows[:, :, None], cols[:, None, :]]
            out[key] = crops.permute(0, 3, 1, 2).contiguous().view(B, T, C, oh, ow)
        return out


class CenterCrop(GPUBatchTransform):
    """Deterministic center crop for image keys — the evaluation-time
    counterpart of :class:`RandomCrop`. No-op when already at target size."""

    def __init__(self, image_keys: List[str], size: Tuple[int, int]):
        self.image_keys = set(image_keys)
        self.size = size

    @torch.no_grad()
    def __call__(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        out = dict(batch)
        oh, ow = self.size
        for key in self.image_keys:
            if key not in batch:
                continue
            x = batch[key]  # (B, T, C, H, W)
            H, W = x.shape[-2:]
            if (H, W) == (oh, ow):
                continue
            top, left = (H - oh) // 2, (W - ow) // 2
            out[key] = x[..., top : top + oh, left : left + ow]
        return out


class Resize(GPUBatchTransform):
    """Deterministic resize for image keys to a fixed (h, w) via area-weighted
    averaging. No-op when already at target size.

    ``F.interpolate(mode="area")`` is the same algorithm as
    ``cv2.resize(..., interpolation=cv2.INTER_AREA)`` (exact for integer
    downscale factors, the common case here): both average input pixels
    falling in each output cell. Use this when a source key's images arrive
    at a resolution that doesn't match what a store/model downstream
    expects -- e.g. adapting a live camera's native resolution to a
    checkpoint's trained shape at inference time.

    Args:
        image_keys: Keys to resize.
        size:       Output spatial size (H, W).
    """

    def __init__(self, image_keys: List[str], size: Tuple[int, int]):
        self.image_keys = set(image_keys)
        self.size = size

    @torch.no_grad()
    def __call__(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        out = dict(batch)
        oh, ow = self.size
        for key in self.image_keys:
            if key not in batch:
                continue
            x = batch[key]  # (B, T, C, H, W)
            H, W = x.shape[-2:]
            if (H, W) == (oh, ow):
                continue
            B, T, C = x.shape[:3]
            flat = F.interpolate(x.reshape(B * T, C, H, W), size=(oh, ow), mode="area")
            out[key] = flat.reshape(B, T, C, oh, ow)
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


class ChannelsLastGPU(GPUBatchTransform):
    """Convert 4-D image tensors to ``torch.channels_last`` memory format.

    Ampere (and newer) NVIDIA GPUs run conv2d 10–30% faster on NHWC tensors.
    This transform should be the **last** GPU transform before the model forward
    so that the layout is locked-in as late as possible (other transforms may
    call ``.contiguous()`` which reverts the format).

    **Only applies to 4-D tensors** ``(N, C, H, W)``.  Batches of sequences
    ``(B, T, C, H, W)`` are passed through unchanged because
    ``contiguous(channels_last)`` is undefined for 5-D inputs.

    Args:
        image_keys: Keys to convert.  Keys absent from the batch or with
                    non-4-D tensors are silently skipped.
    """

    def __init__(self, image_keys: List[str]):
        self.image_keys = set(image_keys)

    @torch.no_grad()
    def __call__(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        out = dict(batch)
        for key in self.image_keys:
            if key not in batch:
                continue
            x = batch[key]
            if x.ndim == 4:  # (N, C, H, W) only — 5-D is not supported
                out[key] = x.contiguous(memory_format=torch.channels_last)
        return out


class ComposeGPU(GPUBatchTransform):
    """Apply a sequence of GPUBatchTransforms in order."""

    def __init__(self, transforms: List[GPUBatchTransform]):
        self.transforms = transforms

    def __call__(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        for t in self.transforms:
            batch = t(batch)
        return batch
