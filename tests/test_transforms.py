"""Tests for the three-tier transform system."""

import numpy as np
import pytest
import torch

from tests.conftest import OBS_KEYS, LOWDIM_KEY, H, W, C, ACTION_DIM, N_EPISODES


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
        # First call builds cache
        apply_disk_cached(store, transform, cache_root=str(tmp_path), image_keys=OBS_KEYS)
        import os
        # Second call should not touch the source store (just opens meta.json)
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


class TestCPUBatchTransform:
    def test_action_chunk_shapes(self):
        from robo_dataset.transforms.cpu_batch import ActionChunk
        t = ActionChunk(n_obs_steps=2, image_keys=OBS_KEYS, action_key="action")
        batch = _make_batch()
        out = t(batch)
        # Images: (B, n_obs, C, H, W) float32
        assert out[OBS_KEYS[0]].shape == (4, 2, C, H, W)
        assert out[OBS_KEYS[0]].dtype == torch.float32
        # Action: (B, T, action_dim) — untouched horizon
        assert out["action"].shape == (4, 8, ACTION_DIM)

    def test_image_normalization(self):
        from robo_dataset.transforms.cpu_batch import ActionChunk
        t = ActionChunk(n_obs_steps=2, image_keys=OBS_KEYS, action_key="action")
        batch = _make_batch()
        out = t(batch)
        assert out[OBS_KEYS[0]].min() >= 0.0
        assert out[OBS_KEYS[0]].max() <= 1.0 + 1e-6

    def test_compose(self):
        from robo_dataset.transforms.cpu_batch import ActionChunk, NormalizeRange, ComposeCPU
        composed = ComposeCPU([
            ActionChunk(n_obs_steps=2, image_keys=OBS_KEYS, action_key="action"),
            NormalizeRange(keys=["action"], in_min=-1, in_max=1, out_min=-1, out_max=1),
        ])
        batch = _make_batch()
        out = composed(batch)
        assert OBS_KEYS[0] in out and "action" in out


# ── Tier 3: GPUBatchTransform ─────────────────────────────────────────────────

class TestGPUBatchTransform:
    def test_random_crop_shape(self):
        from robo_dataset.transforms.cpu_batch import ActionChunk
        from robo_dataset.transforms.gpu_batch import RandomCrop
        batch = _make_batch()
        # Apply CPU transform first to get (B, T, C, H, W) float
        cpu = ActionChunk(n_obs_steps=4, image_keys=OBS_KEYS)
        batch = cpu(batch)

        crop = RandomCrop(image_keys=OBS_KEYS, size=(24, 24), pad=4)
        out = crop(batch)
        assert out[OBS_KEYS[0]].shape == (4, 4, C, 24, 24)

    def test_random_crop_different_per_sample(self):
        from robo_dataset.transforms.cpu_batch import ActionChunk
        from robo_dataset.transforms.gpu_batch import RandomCrop
        batch = _make_batch(B=8)
        cpu = ActionChunk(n_obs_steps=4, image_keys=OBS_KEYS)
        batch = cpu(batch)
        crop = RandomCrop(image_keys=OBS_KEYS, size=(24, 24), pad=4)
        out1 = crop(batch)
        out2 = crop(batch)
        # Two crops should (very likely) differ
        assert not torch.allclose(out1[OBS_KEYS[0]], out2[OBS_KEYS[0]])

    def test_color_jitter_range(self):
        from robo_dataset.transforms.cpu_batch import ActionChunk
        from robo_dataset.transforms.gpu_batch import ColorJitter
        batch = _make_batch()
        cpu = ActionChunk(n_obs_steps=4, image_keys=OBS_KEYS)
        batch = cpu(batch)
        jitter = ColorJitter(image_keys=OBS_KEYS, brightness=0.2, contrast=0.2)
        out = jitter(batch)
        assert out[OBS_KEYS[0]].min() >= 0.0
        assert out[OBS_KEYS[0]].max() <= 1.0 + 1e-5
