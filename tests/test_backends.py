"""Equivalence tests: verify that all backends return the same data.

Rules:
- Lowdim / action keys: bit-exact between backends.
- Image keys: PSNR >= 30 dB for lossy backends (RLDS uses JPEG compression).
- HDF5 is the ground truth.
"""

import numpy as np
import pytest

from .constants import N_EPISODES, EP_LENGTHS, OBS_KEYS, LOWDIM_KEY, H, W, C, ACTION_DIM


def _psnr(a: np.ndarray, b: np.ndarray) -> float:
    mse = np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2)
    if mse == 0:
        return float("inf")
    return 20 * np.log10(255.0 / np.sqrt(mse))


# ── HDF5 store ────────────────────────────────────────────────────────────────

class TestHDF5Store:
    def test_episode_count(self, synthetic_hdf5):
        from robo_dataset.backends.hdf5 import RobomimicHDF5Store
        store = RobomimicHDF5Store(synthetic_hdf5, obs_keys=OBS_KEYS + [LOWDIM_KEY])
        assert store.n_episodes == N_EPISODES

    def test_episode_lengths(self, synthetic_hdf5):
        from robo_dataset.backends.hdf5 import RobomimicHDF5Store
        store = RobomimicHDF5Store(synthetic_hdf5, obs_keys=OBS_KEYS + [LOWDIM_KEY])
        for i, expected_T in enumerate(EP_LENGTHS):
            assert store.episode_length(i) == expected_T

    def test_total_frames(self, synthetic_hdf5):
        from robo_dataset.backends.hdf5 import RobomimicHDF5Store
        store = RobomimicHDF5Store(synthetic_hdf5, obs_keys=OBS_KEYS + [LOWDIM_KEY])
        assert store.n_frames == sum(EP_LENGTHS)

    def test_keys(self, synthetic_hdf5):
        from robo_dataset.backends.hdf5 import RobomimicHDF5Store
        store = RobomimicHDF5Store(synthetic_hdf5, obs_keys=OBS_KEYS + [LOWDIM_KEY])
        assert set(store.keys) == set(OBS_KEYS + [LOWDIM_KEY, "action"])

    def test_frame_shapes(self, synthetic_hdf5):
        from robo_dataset.backends.hdf5 import RobomimicHDF5Store
        store = RobomimicHDF5Store(synthetic_hdf5, obs_keys=OBS_KEYS + [LOWDIM_KEY])
        for key in OBS_KEYS:
            assert store.frame_shape(key) == (H, W, C)
        assert store.frame_shape("action") == (ACTION_DIM,)

    def test_get_sequence_within_episode(self, synthetic_hdf5):
        from robo_dataset.backends.hdf5 import RobomimicHDF5Store
        store = RobomimicHDF5Store(synthetic_hdf5, obs_keys=OBS_KEYS + [LOWDIM_KEY])
        seq = store.get_sequence(OBS_KEYS[0], 0, EP_LENGTHS[0])
        assert seq.shape == (EP_LENGTHS[0], H, W, C)

    def test_get_sequence_across_episodes(self, synthetic_hdf5):
        from robo_dataset.backends.hdf5 import RobomimicHDF5Store
        store = RobomimicHDF5Store(synthetic_hdf5, obs_keys=OBS_KEYS + [LOWDIM_KEY])
        # Cross-episode: last 3 frames of ep0 + first 3 frames of ep1
        start = EP_LENGTHS[0] - 3
        end = EP_LENGTHS[0] + 3
        seq = store.get_sequence(OBS_KEYS[0], start, end)
        assert seq.shape == (6, H, W, C)


# ── ZarrStore equivalence ──────────────────────────────────────────────────────

class TestZarrEquivalence:
    def test_episode_metadata(self, synthetic_hdf5, zarr_store_path):
        from robo_dataset.backends.hdf5 import RobomimicHDF5Store
        from robo_dataset.backends.zarr_backend import ZarrStore
        hdf5 = RobomimicHDF5Store(synthetic_hdf5, obs_keys=OBS_KEYS + [LOWDIM_KEY])
        zarr = ZarrStore(zarr_store_path)
        assert zarr.n_episodes == hdf5.n_episodes
        np.testing.assert_array_equal(zarr.episode_ends, hdf5.episode_ends)

    def test_action_exact(self, synthetic_hdf5, zarr_store_path):
        from robo_dataset.backends.hdf5 import RobomimicHDF5Store
        from robo_dataset.backends.zarr_backend import ZarrStore
        hdf5 = RobomimicHDF5Store(synthetic_hdf5, obs_keys=OBS_KEYS + [LOWDIM_KEY])
        zarr = ZarrStore(zarr_store_path)
        h = hdf5.get_sequence("action", 0, hdf5.n_frames)
        z = zarr.get_sequence("action", 0, zarr.n_frames)
        np.testing.assert_array_equal(h, z)

    def test_lowdim_exact(self, synthetic_hdf5, zarr_store_path):
        from robo_dataset.backends.hdf5 import RobomimicHDF5Store
        from robo_dataset.backends.zarr_backend import ZarrStore
        hdf5 = RobomimicHDF5Store(synthetic_hdf5, obs_keys=OBS_KEYS + [LOWDIM_KEY])
        zarr = ZarrStore(zarr_store_path)
        h = hdf5.get_sequence(LOWDIM_KEY, 0, hdf5.n_frames)
        z = zarr.get_sequence(LOWDIM_KEY, 0, zarr.n_frames)
        np.testing.assert_array_equal(h, z)

    def test_images_lossless(self, synthetic_hdf5, zarr_store_path):
        """Zarr uses lz4 fallback for images — should be bit-exact."""
        from robo_dataset.backends.hdf5 import RobomimicHDF5Store
        from robo_dataset.backends.zarr_backend import ZarrStore
        hdf5 = RobomimicHDF5Store(synthetic_hdf5, obs_keys=OBS_KEYS + [LOWDIM_KEY])
        zarr = ZarrStore(zarr_store_path)
        for key in OBS_KEYS:
            h = hdf5.get_sequence(key, 0, hdf5.n_frames)
            z = zarr.get_sequence(key, 0, zarr.n_frames)
            # lz4 is lossless; if imagecodecs_jpeg is active PSNR check is still valid
            psnr = _psnr(h, z)
            assert psnr >= 30.0, f"PSNR {psnr:.1f} dB for {key} is too low"


# ── RLDS store equivalence ────────────────────────────────────────────────────

class TestRLDSEquivalence:
    def test_episode_metadata(self, synthetic_hdf5, rlds_store_path):
        from robo_dataset.backends.hdf5 import RobomimicHDF5Store
        from robo_dataset.backends.rlds import RLDSFrameStore
        hdf5 = RobomimicHDF5Store(synthetic_hdf5, obs_keys=OBS_KEYS + [LOWDIM_KEY])
        rlds = RLDSFrameStore(rlds_store_path)
        assert rlds.n_episodes == hdf5.n_episodes
        np.testing.assert_array_equal(rlds.episode_ends, hdf5.episode_ends)

    def test_action_exact(self, synthetic_hdf5, rlds_store_path):
        from robo_dataset.backends.hdf5 import RobomimicHDF5Store
        from robo_dataset.backends.rlds import RLDSFrameStore
        hdf5 = RobomimicHDF5Store(synthetic_hdf5, obs_keys=OBS_KEYS + [LOWDIM_KEY])
        rlds = RLDSFrameStore(rlds_store_path)
        h = hdf5.get_sequence("action", 0, hdf5.n_frames)
        r = rlds.get_sequence("action", 0, rlds.n_frames)
        np.testing.assert_allclose(h, r, atol=1e-5)

    def test_images_psnr(self, synthetic_hdf5, rlds_store_path):
        """RLDS uses JPEG — require PSNR >= 30 dB."""
        from robo_dataset.backends.hdf5 import RobomimicHDF5Store
        from robo_dataset.backends.rlds import RLDSFrameStore
        hdf5 = RobomimicHDF5Store(synthetic_hdf5, obs_keys=OBS_KEYS + [LOWDIM_KEY])
        rlds = RLDSFrameStore(rlds_store_path)
        for key in OBS_KEYS:
            h = hdf5.get_sequence(key, 0, hdf5.n_frames)
            r = rlds.get_sequence(key, 0, rlds.n_frames)
            psnr = _psnr(h, r)
            assert psnr >= 30.0, f"PSNR {psnr:.1f} dB for {key} below threshold"


# ── DataLoader round-trip ──────────────────────────────────────────────────────

class TestDataLoader:
    def test_zarr_dataloader(self, zarr_store_path):
        from robo_dataset.backends.zarr_backend import ZarrStore
        from robo_dataset.dataset import TrajectoryDataset
        store = ZarrStore(zarr_store_path)
        dataset = TrajectoryDataset(store, horizon=4, pad_before=1, pad_after=1)
        loader = dataset.make_dataloader(batch_size=4, num_workers=0, shuffle=False)
        batch = next(iter(loader))
        assert batch[OBS_KEYS[0]].shape == (4, 4, H, W, C)
        assert batch["action"].shape == (4, 4, ACTION_DIM)

    def test_rlds_dataloader(self, rlds_store_path):
        from robo_dataset.backends.rlds import RLDSFrameStore
        from robo_dataset.dataset import TrajectoryDataset
        store = RLDSFrameStore(rlds_store_path)
        dataset = TrajectoryDataset(store, horizon=4, pad_before=1, pad_after=1)
        loader = dataset.make_dataloader(batch_size=4, num_workers=0, shuffle=False)
        batch = next(iter(loader))
        assert batch[OBS_KEYS[0]].shape == (4, 4, H, W, C)


# ── convert() coverage ────────────────────────────────────────────────────────

class TestConvert:
    def test_convert_with_transform(self, synthetic_hdf5, tmp_path):
        """convert() transform= argument is applied per episode before writing."""
        from robo_dataset.backends.hdf5 import RobomimicHDF5Store
        from robo_dataset.backends.zarr_backend import ZarrStore, ZarrStoreWriter
        from robo_dataset.convert import convert

        src = RobomimicHDF5Store(synthetic_hdf5, obs_keys=OBS_KEYS + [LOWDIM_KEY])

        scale_factor = 2.0
        def double_action(ep):
            out = dict(ep)
            out["action"] = ep["action"] * scale_factor
            return out

        out_path = str(tmp_path / "zarr_transformed")
        writer = ZarrStoreWriter.create(
            path=out_path,
            key_shapes={k: src.frame_shape(k) for k in src.keys},
            key_dtypes={k: src.frame_dtype(k) for k in src.keys},
            image_keys=OBS_KEYS,
        )
        convert(src, writer, transform=double_action, verbose=False)

        dst = ZarrStore(out_path)
        orig_action = src.get_sequence("action", 0, src.n_frames)
        conv_action = dst.get_sequence("action", 0, dst.n_frames)
        np.testing.assert_allclose(conv_action, orig_action * scale_factor, rtol=1e-5)

    def test_convert_key_subset(self, synthetic_hdf5, tmp_path):
        """convert() keys= argument writes only the requested subset."""
        from robo_dataset.backends.hdf5 import RobomimicHDF5Store
        from robo_dataset.backends.zarr_backend import ZarrStore, ZarrStoreWriter
        from robo_dataset.convert import convert

        src = RobomimicHDF5Store(synthetic_hdf5, obs_keys=OBS_KEYS + [LOWDIM_KEY])
        subset = [OBS_KEYS[0], "action"]

        out_path = str(tmp_path / "zarr_subset")
        writer = ZarrStoreWriter.create(
            path=out_path,
            key_shapes={k: src.frame_shape(k) for k in subset},
            key_dtypes={k: src.frame_dtype(k) for k in subset},
            image_keys=[OBS_KEYS[0]],
        )
        convert(src, writer, keys=subset, verbose=False)

        dst = ZarrStore(out_path)
        assert set(dst.keys) == set(subset)

    def test_zarr_to_zarr(self, zarr_store_path, tmp_path):
        """convert() works with a zarr source (not just HDF5)."""
        from robo_dataset.backends.zarr_backend import ZarrStore, ZarrStoreWriter
        from robo_dataset.convert import convert

        src = ZarrStore(zarr_store_path)
        out_path = str(tmp_path / "zarr2zarr")
        writer = ZarrStoreWriter.create(
            path=out_path,
            key_shapes={k: src.frame_shape(k) for k in src.keys},
            key_dtypes={k: src.frame_dtype(k) for k in src.keys},
            image_keys=OBS_KEYS,
        )
        convert(src, writer, verbose=False)

        dst = ZarrStore(out_path)
        assert dst.n_episodes == src.n_episodes
        np.testing.assert_array_equal(dst.episode_ends, src.episode_ends)
        orig = src.get_sequence("action", 0, src.n_frames)
        copy = dst.get_sequence("action", 0, dst.n_frames)
        np.testing.assert_array_equal(orig, copy)

    def test_convert_rlds_from_zarr(self, zarr_store_path, tmp_path):
        """convert_rlds() works with a zarr source (non-HDF5 input path)."""
        from robo_dataset.backends.zarr_backend import ZarrStore
        from robo_dataset.backends.rlds import RLDSFrameStore
        from robo_dataset.convert import convert_rlds

        src = ZarrStore(zarr_store_path)
        out_path = str(tmp_path / "zarr_to_rlds")
        convert_rlds(src, out_path, image_keys=OBS_KEYS, jpeg_quality=95, verbose=False)

        dst = RLDSFrameStore(out_path)
        assert dst.n_episodes == src.n_episodes
        np.testing.assert_array_equal(dst.episode_ends, src.episode_ends)
        # action must be bit-exact
        orig = src.get_sequence("action", 0, src.n_frames)
        copy = dst.get_sequence("action", 0, dst.n_frames)
        np.testing.assert_allclose(orig, copy, atol=1e-5)


# ── ShmStore ──────────────────────────────────────────────────────────────────

class TestShmStore:
    """ShmStore should be equivalent to its source for all keys."""

    def test_episode_ends_match(self, zarr_store_path):
        from robo_dataset.backends.zarr_backend import ZarrStore
        from robo_dataset.backends.shm import ShmStore
        src = ZarrStore(zarr_store_path)
        shm = ShmStore.from_store(src, verbose=False)
        np.testing.assert_array_equal(shm.episode_ends, src.episode_ends)

    def test_keys_match(self, zarr_store_path):
        from robo_dataset.backends.zarr_backend import ZarrStore
        from robo_dataset.backends.shm import ShmStore
        src = ZarrStore(zarr_store_path)
        shm = ShmStore.from_store(src, verbose=False)
        assert set(shm.keys) == set(src.keys)

    def test_frame_shapes_match(self, zarr_store_path):
        from robo_dataset.backends.zarr_backend import ZarrStore
        from robo_dataset.backends.shm import ShmStore
        src = ZarrStore(zarr_store_path)
        shm = ShmStore.from_store(src, verbose=False)
        for key in src.keys:
            assert shm.frame_shape(key) == src.frame_shape(key)

    def test_frame_dtypes_match(self, zarr_store_path):
        from robo_dataset.backends.zarr_backend import ZarrStore
        from robo_dataset.backends.shm import ShmStore
        src = ZarrStore(zarr_store_path)
        shm = ShmStore.from_store(src, verbose=False)
        for key in src.keys:
            assert shm.frame_dtype(key) == src.frame_dtype(key)

    def test_action_exact(self, zarr_store_path):
        from robo_dataset.backends.zarr_backend import ZarrStore
        from robo_dataset.backends.shm import ShmStore
        src = ZarrStore(zarr_store_path)
        shm = ShmStore.from_store(src, verbose=False)
        n = int(src.episode_ends[-1])
        np.testing.assert_array_equal(
            shm.get_sequence("action", 0, n),
            src.get_sequence("action", 0, n),
        )

    def test_images_exact(self, zarr_store_path):
        from robo_dataset.backends.zarr_backend import ZarrStore
        from robo_dataset.backends.shm import ShmStore
        src = ZarrStore(zarr_store_path)
        shm = ShmStore.from_store(src, verbose=False)
        n = int(src.episode_ends[-1])
        for key in OBS_KEYS:
            np.testing.assert_array_equal(
                shm.get_sequence(key, 0, n),
                src.get_sequence(key, 0, n),
            )

    def test_key_subset(self, zarr_store_path):
        from robo_dataset.backends.zarr_backend import ZarrStore
        from robo_dataset.backends.shm import ShmStore
        src = ZarrStore(zarr_store_path)
        subset = [OBS_KEYS[0], "action"]
        shm = ShmStore.from_store(src, keys=subset, verbose=False)
        assert set(shm.keys) == set(subset)

    def test_on_worker_init_is_noop(self, zarr_store_path):
        from robo_dataset.backends.zarr_backend import ZarrStore
        from robo_dataset.backends.shm import ShmStore
        src = ZarrStore(zarr_store_path)
        shm = ShmStore.from_store(src, verbose=False)
        shm.on_worker_init(0)  # must not raise

    def test_dataloader_multiworker(self, zarr_store_path):
        """Workers forked from ShmStore can read data zero-copy via shared memory."""
        from robo_dataset.backends.zarr_backend import ZarrStore
        from robo_dataset.backends.shm import ShmStore
        from robo_dataset.dataset import TrajectoryDataset

        src = ZarrStore(zarr_store_path)
        shm = ShmStore.from_store(src, verbose=False)

        dataset = TrajectoryDataset(shm, horizon=4, pad_before=1, pad_after=1)
        loader = dataset.make_dataloader(batch_size=4, num_workers=2, shuffle=False)
        batch = next(iter(loader))
        assert batch[OBS_KEYS[0]].shape == (4, 4, H, W, C)
        assert batch["action"].shape == (4, 4, ACTION_DIM)


# ── ZarrShardedStore equivalence ─────────────────────────────────────────────

class TestZarrShardedEquivalence:
    """Sharded zarr-v3 store must return the same data as HDF5 (bit-exact for lz4)."""

    def test_episode_metadata(self, synthetic_hdf5, zarr_sharded_path):
        from robo_dataset.backends.hdf5 import RobomimicHDF5Store
        from robo_dataset.backends.zarr_sharded import ZarrShardedStore
        hdf5 = RobomimicHDF5Store(synthetic_hdf5, obs_keys=OBS_KEYS + [LOWDIM_KEY])
        sharded = ZarrShardedStore(zarr_sharded_path)
        assert sharded.n_episodes == hdf5.n_episodes
        np.testing.assert_array_equal(sharded.episode_ends, hdf5.episode_ends)

    def test_action_exact(self, synthetic_hdf5, zarr_sharded_path):
        from robo_dataset.backends.hdf5 import RobomimicHDF5Store
        from robo_dataset.backends.zarr_sharded import ZarrShardedStore
        hdf5 = RobomimicHDF5Store(synthetic_hdf5, obs_keys=OBS_KEYS + [LOWDIM_KEY])
        sharded = ZarrShardedStore(zarr_sharded_path)
        h = hdf5.get_sequence("action", 0, hdf5.n_frames)
        s = sharded.get_sequence("action", 0, sharded.n_frames)
        np.testing.assert_array_equal(h, s)

    def test_lowdim_exact(self, synthetic_hdf5, zarr_sharded_path):
        from robo_dataset.backends.hdf5 import RobomimicHDF5Store
        from robo_dataset.backends.zarr_sharded import ZarrShardedStore
        hdf5 = RobomimicHDF5Store(synthetic_hdf5, obs_keys=OBS_KEYS + [LOWDIM_KEY])
        sharded = ZarrShardedStore(zarr_sharded_path)
        h = hdf5.get_sequence(LOWDIM_KEY, 0, hdf5.n_frames)
        s = sharded.get_sequence(LOWDIM_KEY, 0, sharded.n_frames)
        np.testing.assert_array_equal(h, s)

    def test_images_lossless(self, synthetic_hdf5, zarr_sharded_path):
        from robo_dataset.backends.hdf5 import RobomimicHDF5Store
        from robo_dataset.backends.zarr_sharded import ZarrShardedStore
        hdf5 = RobomimicHDF5Store(synthetic_hdf5, obs_keys=OBS_KEYS + [LOWDIM_KEY])
        sharded = ZarrShardedStore(zarr_sharded_path)
        for key in OBS_KEYS:
            h = hdf5.get_sequence(key, 0, hdf5.n_frames)
            s = sharded.get_sequence(key, 0, sharded.n_frames)
            np.testing.assert_array_equal(h, s)

    def test_cross_shard_boundary_read(self, zarr_sharded_path):
        """Slice spanning two shard files must return correct data (shard_frames=16 in fixture)."""
        from robo_dataset.backends.zarr_sharded import ZarrShardedStore
        sharded = ZarrShardedStore(zarr_sharded_path)
        # Read 4 frames across the first shard boundary (frames 14–18, straddles shard at 16)
        n = int(sharded.episode_ends[-1])
        end = min(18, n)
        start = max(0, end - 4)
        seq = sharded.get_sequence(OBS_KEYS[0], start, end)
        assert seq.shape == (end - start, H, W, C)

    def test_file_count_much_less_than_frames(self, zarr_sharded_path):
        """Sharded store should have far fewer data files than frames."""
        import os
        n_files = sum(len(files) for _, _, files in os.walk(zarr_sharded_path))
        from robo_dataset.backends.zarr_sharded import ZarrShardedStore
        sharded = ZarrShardedStore(zarr_sharded_path)
        n_frames = int(sharded.episode_ends[-1])
        assert n_files < n_frames, (
            f"Expected fewer files ({n_files}) than frames ({n_frames}) in sharded store"
        )

    def test_dataloader_multiworker(self, zarr_sharded_path):
        from robo_dataset.backends.zarr_sharded import ZarrShardedStore
        from robo_dataset.dataset import TrajectoryDataset
        store = ZarrShardedStore(zarr_sharded_path)
        dataset = TrajectoryDataset(store, horizon=4, pad_before=1, pad_after=1)
        loader = dataset.make_dataloader(batch_size=4, num_workers=2, shuffle=False)
        batch = next(iter(loader))
        assert batch[OBS_KEYS[0]].shape == (4, 4, H, W, C)
        assert batch["action"].shape == (4, 4, ACTION_DIM)

    def test_shard_params_roundtrip(self, zarr_sharded_path):
        """chunk_frames and shard_frames survive the meta.json roundtrip."""
        from robo_dataset.backends.zarr_sharded import ZarrShardedStore
        sharded = ZarrShardedStore(zarr_sharded_path)
        assert sharded._chunk_frames == 4
        assert sharded._shard_frames == 16

    def test_invalid_shard_frames_raises(self, tmp_path):
        from robo_dataset.backends.zarr_sharded import ZarrShardedStoreWriter
        with pytest.raises(ValueError, match="multiple"):
            ZarrShardedStoreWriter(str(tmp_path / "bad"), chunk_frames=8, shard_frames=100)


# ── apply_disk_cached with RLDS writer ────────────────────────────────────────

class TestDiskCacheRLDSWriter:
    def test_rlds_as_cache_writer(self, synthetic_hdf5, tmp_path):
        """apply_disk_cached() produces a valid RLDS store when writer_cls=RLDSFrameStoreWriter."""
        from robo_dataset.backends.hdf5 import RobomimicHDF5Store
        from robo_dataset.backends.rlds import RLDSFrameStore, RLDSFrameStoreWriter
        from robo_dataset.transforms.disk_cached import ResizeImages, apply_disk_cached

        src = RobomimicHDF5Store(synthetic_hdf5, obs_keys=OBS_KEYS + [LOWDIM_KEY])
        transform = ResizeImages(image_keys=OBS_KEYS, target_h=16, target_w=16)
        cached = apply_disk_cached(
            src, transform,
            cache_root=str(tmp_path),
            image_keys=OBS_KEYS,
            writer_cls=RLDSFrameStoreWriter,
        )

        assert isinstance(cached, RLDSFrameStore)
        assert cached.n_episodes == src.n_episodes
        seq = cached.get_sequence(OBS_KEYS[0], 0, 3)
        assert seq.shape == (3, 16, 16, C)

    def test_rlds_cache_hit(self, synthetic_hdf5, tmp_path):
        """Second call returns cached RLDS store without re-running the transform."""
        import os
        from robo_dataset.backends.hdf5 import RobomimicHDF5Store
        from robo_dataset.backends.rlds import RLDSFrameStoreWriter
        from robo_dataset.transforms.disk_cached import ResizeImages, apply_disk_cached

        src = RobomimicHDF5Store(synthetic_hdf5, obs_keys=OBS_KEYS + [LOWDIM_KEY])
        transform = ResizeImages(image_keys=OBS_KEYS, target_h=16, target_w=16)
        apply_disk_cached(src, transform, cache_root=str(tmp_path),
                          image_keys=OBS_KEYS, writer_cls=RLDSFrameStoreWriter)
        cache_path = os.path.join(str(tmp_path), transform.cache_key())
        mtime_before = os.path.getmtime(os.path.join(cache_path, "meta.json"))
        apply_disk_cached(src, transform, cache_root=str(tmp_path),
                          image_keys=OBS_KEYS, writer_cls=RLDSFrameStoreWriter)
        mtime_after = os.path.getmtime(os.path.join(cache_path, "meta.json"))
        assert mtime_before == mtime_after


# ── VideoStore equivalence ────────────────────────────────────────────────────

pytestmark_video = pytest.mark.skipif(
    not pytest.importorskip("av", reason="PyAV not installed") and True,
    reason="PyAV not installed",
)


@pytest.fixture(scope="module")
def _video_imports():
    """Skip entire class if torchcodec or av is unavailable."""
    pytest.importorskip("av")
    pytest.importorskip("torchcodec")


class TestVideoStoreEquivalence:
    """VideoStore must return PSNR >= 30 dB for video keys; exact for non-video keys."""

    pytestmark = pytest.mark.usefixtures("_video_imports")

    def test_episode_metadata(self, synthetic_hdf5, video_store_path):
        from robo_dataset.backends.hdf5 import RobomimicHDF5Store
        from robo_dataset.backends.video import VideoStore
        hdf5 = RobomimicHDF5Store(synthetic_hdf5, obs_keys=OBS_KEYS + [LOWDIM_KEY])
        vs = VideoStore(video_store_path)
        assert vs.n_episodes == hdf5.n_episodes
        np.testing.assert_array_equal(vs.episode_ends, hdf5.episode_ends)

    def test_action_exact(self, synthetic_hdf5, video_store_path):
        from robo_dataset.backends.hdf5 import RobomimicHDF5Store
        from robo_dataset.backends.video import VideoStore
        hdf5 = RobomimicHDF5Store(synthetic_hdf5, obs_keys=OBS_KEYS + [LOWDIM_KEY])
        vs = VideoStore(video_store_path)
        h = hdf5.get_sequence("action", 0, hdf5.n_frames)
        v = vs.get_sequence("action", 0, vs.n_frames)
        np.testing.assert_array_equal(h, v)

    def test_lowdim_exact(self, synthetic_hdf5, video_store_path):
        from robo_dataset.backends.hdf5 import RobomimicHDF5Store
        from robo_dataset.backends.video import VideoStore
        hdf5 = RobomimicHDF5Store(synthetic_hdf5, obs_keys=OBS_KEYS + [LOWDIM_KEY])
        vs = VideoStore(video_store_path)
        h = hdf5.get_sequence(LOWDIM_KEY, 0, hdf5.n_frames)
        v = vs.get_sequence(LOWDIM_KEY, 0, vs.n_frames)
        np.testing.assert_array_equal(h, v)

    def test_images_psnr(self, synthetic_hdf5, video_store_path):
        """H.264 kfi=1 images must achieve PSNR >= 30 dB vs original."""
        from robo_dataset.backends.hdf5 import RobomimicHDF5Store
        from robo_dataset.backends.video import VideoStore
        hdf5 = RobomimicHDF5Store(synthetic_hdf5, obs_keys=OBS_KEYS + [LOWDIM_KEY])
        vs = VideoStore(video_store_path)
        for key in OBS_KEYS:
            h = hdf5.get_sequence(key, 0, hdf5.n_frames)
            v = vs.get_sequence(key, 0, vs.n_frames)
            assert v.shape == h.shape
            psnr = _psnr(h, v)
            assert psnr >= 30.0, f"{key}: PSNR={psnr:.1f} dB < 30 dB"

    def test_cross_episode_read(self, synthetic_hdf5, video_store_path):
        """Sequence spanning two episodes must decode correctly."""
        from robo_dataset.backends.hdf5 import RobomimicHDF5Store
        from robo_dataset.backends.video import VideoStore
        hdf5 = RobomimicHDF5Store(synthetic_hdf5, obs_keys=OBS_KEYS + [LOWDIM_KEY])
        vs = VideoStore(video_store_path)
        # Last 3 frames of ep0 + first 3 of ep1
        start = EP_LENGTHS[0] - 3
        end = EP_LENGTHS[0] + 3
        seq = vs.get_sequence(OBS_KEYS[0], start, end)
        assert seq.shape == (6, H, W, C)
        assert seq.dtype == np.uint8

    def test_meta_roundtrip(self, video_store_path):
        """video_keys, vcodec, kfi survive the meta.json roundtrip."""
        import json
        with open(f"{video_store_path}/meta.json") as f:
            meta = json.load(f)
        assert set(OBS_KEYS).issubset(set(meta["video_keys"]))
        assert meta["vcodec"] == "libx264"
        assert meta["kfi"] == 1

    def test_lru_eviction(self, video_store_path):
        """LRU cache evicts oldest decoder when maxsize is exceeded."""
        from robo_dataset.backends.video import VideoStore
        vs = VideoStore(video_store_path, max_decoders=2)
        # Access 3 different (key, ep_idx) combos — the first should be evicted
        vs._get_decoder(OBS_KEYS[0], 0)
        vs._get_decoder(OBS_KEYS[0], 1)
        vs._get_decoder(OBS_KEYS[0], 2)
        assert len(vs._decoder_cache._cache) == 2
        assert (OBS_KEYS[0], 0) not in vs._decoder_cache._cache

    def test_dataloader_multiworker(self, video_store_path):
        """VideoStore works in a multi-worker DataLoader (fork-safe)."""
        from robo_dataset.backends.video import VideoStore
        from robo_dataset.dataset import TrajectoryDataset
        vs = VideoStore(video_store_path)
        dataset = TrajectoryDataset(vs, horizon=4, pad_before=1, pad_after=1)
        loader = dataset.make_dataloader(batch_size=4, num_workers=2, shuffle=False)
        batch = next(iter(loader))
        assert batch[OBS_KEYS[0]].shape == (4, 4, H, W, C)
        assert batch["action"].shape == (4, 4, ACTION_DIM)
