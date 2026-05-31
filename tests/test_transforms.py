"""Tests for the three-tier transform system."""

import numpy as np
import pytest
import torch

from .constants import OBS_KEYS, LOWDIM_KEY, H, W, C, ACTION_DIM, N_EPISODES

# All obs keys (images + lowdim) — both should be sliced by SplitObsAction
ALL_OBS_KEYS = OBS_KEYS + [LOWDIM_KEY]


# ── Tier 1: DiskCachedTransform ───────────────────────────────────────────────

class TestDiskCachedTransform:
    def test_resize_shape(self, synthetic_hdf5, tmp_path):
        from robo_dataset.backends.hdf5 import RobomimicHDF5Store
        from robo_dataset.transforms.disk_cached import ResizeImages, apply_disk_cached

        store = RobomimicHDF5Store(synthetic_hdf5, obs_keys=OBS_KEYS + [LOWDIM_KEY])
        transform = ResizeImages(image_keys=OBS_KEYS, target_h=16, target_w=16)
        cached = apply_disk_cached(store, transform, cache_root=str(tmp_path), image_keys=OBS_KEYS)

        seq = cached.get_sequence(OBS_KEYS[0], 0, 3)
        assert seq.shape == (3, 16, 16, C)

    def test_cache_hit(self, synthetic_hdf5, tmp_path):
        from robo_dataset.backends.hdf5 import RobomimicHDF5Store
        from robo_dataset.transforms.disk_cached import ResizeImages, apply_disk_cached

        store = RobomimicHDF5Store(synthetic_hdf5, obs_keys=OBS_KEYS + [LOWDIM_KEY])
        transform = ResizeImages(image_keys=OBS_KEYS, target_h=16, target_w=16)
        apply_disk_cached(store, transform, cache_root=str(tmp_path), image_keys=OBS_KEYS)
        import os
        cache_path = os.path.join(str(tmp_path), transform.cache_key())
        mtime_before = os.path.getmtime(os.path.join(cache_path, "meta.json"))
        apply_disk_cached(store, transform, cache_root=str(tmp_path), image_keys=OBS_KEYS)
        mtime_after = os.path.getmtime(os.path.join(cache_path, "meta.json"))
        assert mtime_before == mtime_after

    def test_cache_key_changes_with_config(self):
        from robo_dataset.transforms.disk_cached import ResizeImages
        t1 = ResizeImages(image_keys=OBS_KEYS, target_h=84, target_w=84, version="1")
        t2 = ResizeImages(image_keys=OBS_KEYS, target_h=64, target_w=64, version="1")
        assert t1.cache_key() != t2.cache_key()

    def test_cache_key_changes_with_version(self):
        from robo_dataset.transforms.disk_cached import ResizeImages
        t1 = ResizeImages(image_keys=OBS_KEYS, target_h=84, target_w=84, version="1")
        t2 = ResizeImages(image_keys=OBS_KEYS, target_h=84, target_w=84, version="2")
        assert t1.cache_key() != t2.cache_key()


# ── Tier 2: CPUBatchTransform ─────────────────────────────────────────────────

def _make_batch(B=4, T=8):
    """Synthetic batch as DataLoader would produce it."""
    batch = {}
    for key in OBS_KEYS:
        batch[key] = torch.randint(0, 256, (B, T, H, W, C), dtype=torch.uint8)
    batch[LOWDIM_KEY] = torch.randn(B, T, 3)
    batch["action"] = torch.randn(B, T, ACTION_DIM)
    return batch


class TestSplitObsAction:
    def test_obs_sliced_action_kept(self):
        from robo_dataset.transforms.cpu_batch import SplitObsAction
        t = SplitObsAction(n_obs_steps=2, obs_keys=ALL_OBS_KEYS, action_key="action")
        batch = _make_batch()
        out = t(batch)
        # All obs keys trimmed to n_obs_steps, shape unchanged (still HWC uint8)
        assert out[OBS_KEYS[0]].shape == (4, 2, H, W, C)
        assert out[OBS_KEYS[0]].dtype == torch.uint8
        assert out[LOWDIM_KEY].shape == (4, 2, 3)
        # Action kept at full horizon
        assert out["action"].shape == (4, 8, ACTION_DIM)

    def test_non_obs_keys_pass_through(self):
        from robo_dataset.transforms.cpu_batch import SplitObsAction
        t = SplitObsAction(n_obs_steps=2, obs_keys=OBS_KEYS, action_key="action")
        batch = _make_batch()
        out = t(batch)
        # LOWDIM_KEY not in obs_keys → passes through at full length
        assert out[LOWDIM_KEY].shape == (4, 8, 3)


class TestImageToTensor:
    def test_hwc_to_chw_float(self):
        from robo_dataset.transforms.cpu_batch import ImageToTensor
        t = ImageToTensor(image_keys=OBS_KEYS)
        batch = _make_batch()
        out = t(batch)
        # (B, T, H, W, C) uint8 → (B, T, C, H, W) float32 in [0, 1]
        assert out[OBS_KEYS[0]].shape == (4, 8, C, H, W)
        assert out[OBS_KEYS[0]].dtype == torch.float32
        assert out[OBS_KEYS[0]].min() >= 0.0
        assert out[OBS_KEYS[0]].max() <= 1.0 + 1e-6

    def test_non_image_keys_unchanged(self):
        from robo_dataset.transforms.cpu_batch import ImageToTensor
        t = ImageToTensor(image_keys=OBS_KEYS)
        batch = _make_batch()
        out = t(batch)
        # lowdim and action not in image_keys — pass through untouched
        assert out[LOWDIM_KEY] is batch[LOWDIM_KEY]
        assert out["action"] is batch["action"]

    def test_missing_key_skipped(self):
        from robo_dataset.transforms.cpu_batch import ImageToTensor
        t = ImageToTensor(image_keys=["nonexistent_key"] + OBS_KEYS)
        batch = _make_batch()
        out = t(batch)  # should not raise
        assert out[OBS_KEYS[0]].dtype == torch.float32


class TestComposeCPU:
    def test_compose_split_then_to_tensor(self):
        from robo_dataset.transforms.cpu_batch import SplitObsAction, ImageToTensor, ComposeCPU
        composed = ComposeCPU([
            SplitObsAction(n_obs_steps=2, obs_keys=ALL_OBS_KEYS, action_key="action"),
            ImageToTensor(image_keys=OBS_KEYS),
        ])
        batch = _make_batch()
        out = composed(batch)
        # Images: (B, n_obs, C, H, W) float32
        assert out[OBS_KEYS[0]].shape == (4, 2, C, H, W)
        assert out[OBS_KEYS[0]].dtype == torch.float32
        # Lowdim: (B, n_obs, D) float32
        assert out[LOWDIM_KEY].shape == (4, 2, 3)
        # Action: (B, T, action_dim) — full horizon
        assert out["action"].shape == (4, 8, ACTION_DIM)

    def test_compose_with_normalize_range(self):
        from robo_dataset.transforms.cpu_batch import SplitObsAction, ImageToTensor, NormalizeRange, ComposeCPU
        composed = ComposeCPU([
            SplitObsAction(n_obs_steps=2, obs_keys=ALL_OBS_KEYS, action_key="action"),
            ImageToTensor(image_keys=OBS_KEYS),
            NormalizeRange(keys=["action"], in_min=-1, in_max=1, out_min=-1, out_max=1),
        ])
        batch = _make_batch()
        out = composed(batch)
        assert OBS_KEYS[0] in out and "action" in out


# ── Tier 3: GPUBatchTransform ─────────────────────────────────────────────────

def _cpu_processed_batch(B=4, T=8, n_obs=4):
    from robo_dataset.transforms.cpu_batch import SplitObsAction, ImageToTensor, ComposeCPU
    batch = _make_batch(B=B, T=T)
    cpu = ComposeCPU([
        SplitObsAction(n_obs_steps=n_obs, obs_keys=ALL_OBS_KEYS, action_key="action"),
        ImageToTensor(image_keys=OBS_KEYS),
    ])
    return cpu(batch)


class TestGPUBatchTransform:
    def test_random_crop_shape(self):
        from robo_dataset.transforms.gpu_batch import RandomCrop
        batch = _cpu_processed_batch(n_obs=4)
        crop = RandomCrop(image_keys=OBS_KEYS, size=(24, 24), pad=4)
        out = crop(batch)
        assert out[OBS_KEYS[0]].shape == (4, 4, C, 24, 24)

    def test_random_crop_different_per_sample(self):
        from robo_dataset.transforms.gpu_batch import RandomCrop
        batch = _cpu_processed_batch(B=8, n_obs=4)
        crop = RandomCrop(image_keys=OBS_KEYS, size=(24, 24), pad=4)
        out1 = crop(batch)
        out2 = crop(batch)
        assert not torch.allclose(out1[OBS_KEYS[0]], out2[OBS_KEYS[0]])

    def test_color_jitter_range(self):
        from robo_dataset.transforms.gpu_batch import ColorJitter
        batch = _cpu_processed_batch(n_obs=4)
        jitter = ColorJitter(image_keys=OBS_KEYS, brightness=0.2, contrast=0.2)
        out = jitter(batch)
        assert out[OBS_KEYS[0]].min() >= 0.0
        assert out[OBS_KEYS[0]].max() <= 1.0 + 1e-5


# ── ImageToTensorGPU ──────────────────────────────────────────────────────────

def _uint8_batch(B=4, T=2):
    """Batch with uint8 images in HWC layout, as would arrive from DataLoader."""
    batch = {}
    for key in OBS_KEYS:
        batch[key] = torch.randint(0, 256, (B, T, H, W, C), dtype=torch.uint8)
    batch[LOWDIM_KEY] = torch.randn(B, T, 3)
    batch["action"] = torch.randn(B, T, ACTION_DIM)
    return batch


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
class TestImageToTensorDevice:
    """Tests for ImageToTensorDevice (works on any device; run on CUDA here)."""

    def test_shape_and_dtype(self):
        from robo_dataset.transforms.gpu_batch import ImageToTensorDevice
        t = ImageToTensorDevice(image_keys=OBS_KEYS)
        batch = {k: v.cuda() for k, v in _uint8_batch().items()}
        out = t(batch)
        assert out[OBS_KEYS[0]].shape == (4, 2, C, H, W)
        assert out[OBS_KEYS[0]].dtype == torch.float32

    def test_range(self):
        from robo_dataset.transforms.gpu_batch import ImageToTensorDevice
        t = ImageToTensorDevice(image_keys=OBS_KEYS)
        batch = {k: v.cuda() for k, v in _uint8_batch().items()}
        out = t(batch)
        assert out[OBS_KEYS[0]].min() >= 0.0
        assert out[OBS_KEYS[0]].max() <= 1.0 + 1e-6

    def test_contiguous_output(self):
        from robo_dataset.transforms.gpu_batch import ImageToTensorDevice
        t = ImageToTensorDevice(image_keys=OBS_KEYS)
        batch = {k: v.cuda() for k, v in _uint8_batch().items()}
        out = t(batch)
        assert out[OBS_KEYS[0]].is_contiguous()

    def test_non_image_keys_unchanged(self):
        from robo_dataset.transforms.gpu_batch import ImageToTensorDevice
        t = ImageToTensorDevice(image_keys=OBS_KEYS)
        batch = {k: v.cuda() for k, v in _uint8_batch().items()}
        out = t(batch)
        assert out[LOWDIM_KEY] is batch[LOWDIM_KEY]
        assert out["action"] is batch["action"]

    def test_wrong_dtype_raises(self):
        from robo_dataset.transforms.gpu_batch import ImageToTensorDevice
        t = ImageToTensorDevice(image_keys=OBS_KEYS)
        batch = _uint8_batch()
        batch[OBS_KEYS[0]] = batch[OBS_KEYS[0]].float().cuda()
        with pytest.raises(AssertionError):
            t(batch)

    def test_missing_key_skipped(self):
        from robo_dataset.transforms.gpu_batch import ImageToTensorDevice
        t = ImageToTensorDevice(image_keys=["nonexistent"] + OBS_KEYS)
        batch = {k: v.cuda() for k, v in _uint8_batch().items()}
        out = t(batch)
        assert out[OBS_KEYS[0]].dtype == torch.float32

    def test_gpu_alias_is_same_class(self):
        from robo_dataset.transforms.gpu_batch import ImageToTensorDevice, ImageToTensorGPU
        assert ImageToTensorGPU is ImageToTensorDevice

    def test_cpu_device(self):
        """ImageToTensorDevice works on CPU too."""
        from robo_dataset.transforms.gpu_batch import ImageToTensorDevice
        t = ImageToTensorDevice(image_keys=OBS_KEYS)
        batch = _uint8_batch()  # already on CPU
        out = t(batch)
        assert out[OBS_KEYS[0]].shape == (4, 2, C, H, W)
        assert out[OBS_KEYS[0]].dtype == torch.float32
        assert out[OBS_KEYS[0]].device.type == "cpu"


# ── DataPrefetcher ────────────────────────────────────────────────────────────

def _make_loader(n=20, batch_size=4, pin_memory=False):
    """Minimal DataLoader returning dict batches."""
    dataset = torch.utils.data.TensorDataset(
        torch.randint(0, 256, (n, H, W, C), dtype=torch.uint8),
        torch.randn(n, ACTION_DIM),
    )

    def collate(batch):
        imgs, acts = zip(*batch)
        return {OBS_KEYS[0]: torch.stack(imgs), "action": torch.stack(acts)}

    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        collate_fn=collate,
        pin_memory=pin_memory,
        num_workers=0,
    )


class TestDataPrefetcherCPU:
    """Tests for the sync strategy (CPU device, no special hardware needed)."""

    def test_strategy_is_sync(self):
        from robo_dataset.prefetch import DataPrefetcher
        pf = DataPrefetcher(_make_loader(), device="cpu")
        assert pf.strategy == "sync"

    def test_yields_all_batches(self):
        from robo_dataset.prefetch import DataPrefetcher
        loader = _make_loader(n=20, batch_size=4)
        pf = DataPrefetcher(loader, device="cpu")
        assert len(list(pf)) == len(loader)

    def test_empty_loader(self):
        from robo_dataset.prefetch import DataPrefetcher
        loader = _make_loader(n=0, batch_size=4)
        assert list(DataPrefetcher(loader, device="cpu")) == []

    def test_len(self):
        from robo_dataset.prefetch import DataPrefetcher
        loader = _make_loader(n=20, batch_size=4)
        assert len(DataPrefetcher(loader, device="cpu")) == len(loader)

    def test_cuda_alias(self):
        from robo_dataset.prefetch import DataPrefetcher, CUDADataPrefetcher
        assert CUDADataPrefetcher is DataPrefetcher


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
class TestDataPrefetcherCUDA:
    """Tests for the stream strategy on CUDA."""

    def test_strategy_is_stream(self):
        from robo_dataset.prefetch import DataPrefetcher
        loader = _make_loader(pin_memory=True)
        pf = DataPrefetcher(loader, device="cuda:0")
        assert pf.strategy == "stream"

    def test_yields_all_batches(self):
        from robo_dataset.prefetch import DataPrefetcher
        loader = _make_loader(n=20, batch_size=4, pin_memory=True)
        assert len(list(DataPrefetcher(loader, device="cuda:0"))) == len(loader)

    def test_tensors_on_cuda(self):
        from robo_dataset.prefetch import DataPrefetcher
        loader = _make_loader(n=8, batch_size=4, pin_memory=True)
        for batch in DataPrefetcher(loader, device="cuda:0"):
            assert batch[OBS_KEYS[0]].device.type == "cuda"
            assert batch["action"].device.type == "cuda"

    def test_empty_loader(self):
        from robo_dataset.prefetch import DataPrefetcher
        loader = _make_loader(n=0, batch_size=4, pin_memory=True)
        assert list(DataPrefetcher(loader, device="cuda:0")) == []

    def test_len(self):
        from robo_dataset.prefetch import DataPrefetcher
        loader = _make_loader(n=20, batch_size=4, pin_memory=True)
        assert len(DataPrefetcher(loader, device="cuda:0")) == len(loader)


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS not available")
class TestDataPrefetcherMPS:
    """Tests for the thread strategy on Apple Silicon MPS."""

    def test_strategy_is_thread(self):
        from robo_dataset.prefetch import DataPrefetcher
        pf = DataPrefetcher(_make_loader(), device="mps")
        assert pf.strategy == "thread"

    def test_yields_all_batches(self):
        from robo_dataset.prefetch import DataPrefetcher
        loader = _make_loader(n=20, batch_size=4)
        assert len(list(DataPrefetcher(loader, device="mps"))) == len(loader)

    def test_tensors_on_mps(self):
        from robo_dataset.prefetch import DataPrefetcher
        loader = _make_loader(n=8, batch_size=4)
        for batch in DataPrefetcher(loader, device="mps"):
            assert batch[OBS_KEYS[0]].device.type == "mps"


@pytest.mark.skipif(
    not (hasattr(torch, "xpu") and torch.xpu.is_available()),
    reason="XPU not available",
)
class TestDataPrefetcherXPU:
    """Tests for the stream strategy on Intel XPU."""

    def test_strategy_is_stream(self):
        from robo_dataset.prefetch import DataPrefetcher
        pf = DataPrefetcher(_make_loader(), device="xpu:0")
        assert pf.strategy == "stream"

    def test_yields_all_batches(self):
        from robo_dataset.prefetch import DataPrefetcher
        loader = _make_loader(n=20, batch_size=4)
        assert len(list(DataPrefetcher(loader, device="xpu:0"))) == len(loader)

    def test_tensors_on_xpu(self):
        from robo_dataset.prefetch import DataPrefetcher
        loader = _make_loader(n=8, batch_size=4)
        for batch in DataPrefetcher(loader, device="xpu:0"):
            assert batch[OBS_KEYS[0]].device.type == "xpu"
