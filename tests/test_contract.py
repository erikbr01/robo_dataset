"""Tests for the shared episode-stitch helper and the writer episode contract."""

import importlib.util

import numpy as np
import pytest

from .constants import OBS_KEYS, LOWDIM_KEY

requires_rlds = pytest.mark.skipif(
    importlib.util.find_spec("tfrecord") is None,
    reason="tfrecord not installed (RLDS backend)",
)


# ── iter_episode_slices (H3) ──────────────────────────────────────────────────

class TestIterEpisodeSlices:
    EPISODE_ENDS = np.array([10, 25, 40], dtype=np.int64)

    def _slices(self, start, end):
        from robo_dataset.backends.base import iter_episode_slices
        return list(iter_episode_slices(self.EPISODE_ENDS, start, end))

    def test_single_episode_read(self):
        assert self._slices(3, 7) == [(0, 3, 7)]
        # entirely inside the second episode (global 10..25 → local offsets)
        assert self._slices(12, 20) == [(1, 2, 10)]

    def test_cross_episode_read(self):
        assert self._slices(8, 27) == [(0, 8, 10), (1, 0, 15), (2, 0, 2)]

    def test_full_range_read(self):
        assert self._slices(0, 40) == [(0, 0, 10), (1, 0, 15), (2, 0, 15)]

    def test_episode_boundary_exclusive_end(self):
        # end == episode_ends[0] must not touch episode 1
        assert self._slices(5, 10) == [(0, 5, 10)]

    def test_out_of_range_raises(self):
        for start, end in [(-1, 5), (0, 41), (5, 5), (7, 3), (40, 41)]:
            with pytest.raises(IndexError):
                self._slices(start, end)

    def test_slices_reconstruct_range(self):
        """Concatenating the yielded local slices reconstructs [start, end)."""
        flat = np.arange(40)
        episodes = [flat[0:10], flat[10:25], flat[25:40]]
        for start, end in [(3, 7), (8, 27), (0, 40), (24, 26)]:
            got = np.concatenate(
                [episodes[ep][lo:hi] for ep, lo, hi in self._slices(start, end)]
            )
            np.testing.assert_array_equal(got, flat[start:end])


# ── Writer episode contract (H1 + H2) ─────────────────────────────────────────

def _episode(T, lowdim_dim=3, dtype=np.float32, extra=None):
    rng = np.random.default_rng(0)
    ep = {
        "action": rng.random((T, 4)).astype(dtype),
        LOWDIM_KEY: rng.random((T, lowdim_dim)).astype(dtype),
    }
    if extra:
        ep.update(extra)
    return ep


def _zarr_writer(tmp_path, declared=True):
    from robo_dataset.backends.zarr_backend import ZarrStoreWriter
    if not declared:
        return ZarrStoreWriter(str(tmp_path / "w.zarr"))
    return ZarrStoreWriter.create(
        path=str(tmp_path / "w.zarr"),
        key_shapes={"action": (4,), LOWDIM_KEY: (3,)},
        key_dtypes={"action": np.float32, LOWDIM_KEY: np.float32},
    )


class TestWriterContract:
    def test_happy_path(self, tmp_path):
        w = _zarr_writer(tmp_path)
        w.put_episode(_episode(5))
        w.put_episode(_episode(7))
        w.finalize()
        assert w.n_frames == 12

    def test_missing_key_on_later_episode_raises(self, tmp_path):
        w = _zarr_writer(tmp_path, declared=False)
        w.put_episode(_episode(5))
        ep = _episode(5)
        del ep["action"]
        with pytest.raises(ValueError, match=r"(?i)missing.*action"):
            w.put_episode(ep)

    def test_extra_key_on_later_episode_raises(self, tmp_path):
        w = _zarr_writer(tmp_path, declared=False)
        w.put_episode(_episode(5))
        ep = _episode(5, extra={"surprise": np.zeros((5, 2), np.float32)})
        with pytest.raises(ValueError, match=r"(?i)extra.*surprise"):
            w.put_episode(ep)

    def test_error_names_episode_index(self, tmp_path):
        w = _zarr_writer(tmp_path, declared=False)
        w.put_episode(_episode(5))
        w.put_episode(_episode(4))
        ep = _episode(5)
        del ep[LOWDIM_KEY]
        with pytest.raises(ValueError, match="Episode 2"):
            w.put_episode(ep)

    def test_frame_shape_mismatch_raises(self, tmp_path):
        w = _zarr_writer(tmp_path, declared=False)
        w.put_episode(_episode(5, lowdim_dim=3))
        with pytest.raises(ValueError, match="frame shape"):
            w.put_episode(_episode(5, lowdim_dim=4))

    def test_dtype_mismatch_raises(self, tmp_path):
        w = _zarr_writer(tmp_path, declared=False)
        w.put_episode(_episode(5, dtype=np.float32))
        with pytest.raises(ValueError, match="dtype"):
            w.put_episode(_episode(5, dtype=np.float64))

    def test_ragged_T_within_episode_raises(self, tmp_path):
        w = _zarr_writer(tmp_path, declared=False)
        ep = _episode(5)
        ep["action"] = ep["action"][:4]
        with pytest.raises(ValueError, match="length"):
            w.put_episode(ep)

    def test_declared_contract_wins_on_first_episode(self, tmp_path):
        # Declared shape (3,) for LOWDIM_KEY; first episode brings (4,) → raise.
        w = _zarr_writer(tmp_path, declared=True)
        with pytest.raises(ValueError, match="frame shape"):
            w.put_episode(_episode(5, lowdim_dim=4))

    def test_declared_keys_mismatch_raises(self, tmp_path):
        w = _zarr_writer(tmp_path, declared=True)
        ep = _episode(5)
        del ep["action"]
        with pytest.raises(ValueError, match="declared"):
            w.put_episode(ep)

    def test_declared_dtype_mismatch_raises(self, tmp_path):
        w = _zarr_writer(tmp_path, declared=True)
        with pytest.raises(ValueError, match="dtype"):
            w.put_episode(_episode(5, dtype=np.float64))


@requires_rlds
class TestRLDSContract:
    def test_declared_non_float32_lowdim_raises(self, tmp_path):
        from robo_dataset.backends.rlds import RLDSFrameStoreWriter
        with pytest.raises(ValueError, match="float32"):
            RLDSFrameStoreWriter.create(
                path=str(tmp_path / "w.rlds"),
                key_shapes={"action": (4,)},
                key_dtypes={"action": np.float64},
            )

    def test_declared_non_uint8_image_ok(self, tmp_path):
        """The float32 rule applies to non-image keys only."""
        from robo_dataset.backends.rlds import RLDSFrameStoreWriter
        RLDSFrameStoreWriter.create(
            path=str(tmp_path / "w.rlds"),
            key_shapes={"img": (8, 8, 3), "action": (4,)},
            key_dtypes={"img": np.uint8, "action": np.float32},
            image_keys=["img"],
        )

    def test_meta_records_float32_for_lowdim(self, tmp_path):
        """Persisted dtype must reflect the stored float32 bytes (H2)."""
        from robo_dataset.backends.rlds import RLDSFrameStore, RLDSFrameStoreWriter
        w = RLDSFrameStoreWriter(str(tmp_path / "w.rlds"), image_keys=[])
        w.put_episode({"action": np.random.default_rng(0).random((5, 4))})  # float64 data
        w.finalize()
        store = RLDSFrameStore(str(tmp_path / "w.rlds"))
        assert store.frame_dtype("action") == np.float32
        assert store.get_sequence("action", 0, 5).dtype == np.float32


# ── VideoStoreWriter image_keys/video_keys conflict (L1) ──────────────────────

class TestVideoWriterKeyConflict:
    def test_conflicting_key_sets_raise(self, tmp_path):
        pytest.importorskip("av")
        from robo_dataset.backends.video import VideoStoreWriter
        with pytest.raises(ValueError, match="video_keys"):
            VideoStoreWriter.create(
                path=str(tmp_path / "v"),
                key_shapes={OBS_KEYS[0]: (8, 8, 3)},
                key_dtypes={OBS_KEYS[0]: np.uint8},
                image_keys=[OBS_KEYS[0]],
                video_keys=["something_else"],
            )

    def test_identical_key_sets_ok(self, tmp_path):
        pytest.importorskip("av")
        from robo_dataset.backends.video import VideoStoreWriter
        VideoStoreWriter.create(
            path=str(tmp_path / "v"),
            key_shapes={OBS_KEYS[0]: (8, 8, 3)},
            key_dtypes={OBS_KEYS[0]: np.uint8},
            image_keys=[OBS_KEYS[0]],
            video_keys=[OBS_KEYS[0]],
        )


# ── store_factory RLDS detection (L3) ─────────────────────────────────────────

class TestDetectRLDS:
    @requires_rlds
    def test_detect_and_open_rlds(self, rlds_store_path):
        from robo_dataset.backends.rlds import RLDSFrameStore
        from robo_dataset.store_factory import detect_format, open_store
        assert detect_format(rlds_store_path) == "rlds"
        store = open_store(rlds_store_path)
        assert isinstance(store, RLDSFrameStore)
        assert store.n_frames > 0

    def test_zarr_still_detected(self, zarr_store_path):
        from robo_dataset.store_factory import detect_format
        assert detect_format(zarr_store_path) == "zarr"
