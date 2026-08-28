"""Tests for the ABC MCAP backend's supporting pieces.

The store itself needs real `episode.mcap` files carrying ABC's custom protobuf
schemas (RobotState, GripperState, Instructions) to exercise, which is more
fixture than it is worth here. What is tested is everything around it that can
go wrong silently:

* `detect_format` recognising the layout, which is pure directory structure;
* `ResizeImages(letterbox=True)`, which exists because ABC mixes 848x480 and
  640x480 episodes and stretching them to one shape distorts one group relative
  to the other;
* the zarr image compressor actually selecting JPEG when imagecodecs is present,
  since falling back to lossless lz4 costs ~10x the disk and does so quietly.
"""

import os

import numpy as np
import pytest

from robo_dataset.store_factory import detect_format
from robo_dataset.transforms import ResizeImages


def _touch(path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        f.write(b"")


# ── format detection ──────────────────────────────────────────────────────────

def test_detect_format_abc_mcap(tmp_path):
    root = tmp_path / "task"
    for name in ("episode_a", "episode_b"):
        _touch(str(root / name / "episode.mcap"))
    assert detect_format(str(root)) == "abc_mcap"


def test_detect_format_ignores_dir_without_episode_mcap(tmp_path):
    """A directory of subdirectories is not enough; episode.mcap is the signal.

    `detect_format` reports an unrecognised layout by raising, so the assertion
    is that it still does -- the ABC probe must not swallow the unknown case.
    """
    root = tmp_path / "task"
    _touch(str(root / "episode_a" / "something_else.bin"))
    with pytest.raises(ValueError, match="Cannot detect store format"):
        detect_format(str(root))


def test_detect_format_prefers_lerobot(tmp_path):
    """A store that is both must not be misread as ABC."""
    root = tmp_path / "task"
    _touch(str(root / "meta" / "info.json"))
    _touch(str(root / "episode_a" / "episode.mcap"))
    assert detect_format(str(root)) == "lerobot"


# ── letterboxed resize ────────────────────────────────────────────────────────

def _frames(h, w, t=3):
    """A frame with a distinguishable interior, not a flat fill."""
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    img = (40 + 200 * (yy / h) * (xx / w)).astype(np.uint8)
    return np.repeat(np.stack([img] * 3, -1)[None], t, axis=0)


@pytest.mark.parametrize("src", [(480, 640), (480, 848), (240, 320)])
def test_letterbox_output_shape_is_target(src):
    """Every source size, whatever its aspect, lands on exactly the target."""
    h, w = src
    tr = ResizeImages(["top"], target_h=240, target_w=320, letterbox=True)
    out = tr.apply_episode({"top": _frames(h, w)})["top"]
    assert out.shape == (3, 240, 320, 3)
    assert out.dtype == np.uint8


def test_letterbox_preserves_aspect_ratio():
    """16:9 content is scaled to fit and centred, not stretched to 4:3."""
    tr = ResizeImages(["top"], target_h=240, target_w=320, letterbox=True)
    sh, sw, y0, x0 = tr._fit(480, 848)
    # 320/848 is the binding scale, so height shrinks below the target.
    assert (sh, sw) == (181, 320)
    assert (y0, x0) == (29, 0)
    assert abs((sw / sh) - (848 / 480)) < 0.02

    # 4:3 fits exactly, so there is no padding at all.
    assert tr._fit(480, 640) == (240, 320, 0, 0)


def test_letterbox_pads_rather_than_stretches():
    tr = ResizeImages(["top"], target_h=240, target_w=320, letterbox=True,
                      pad_value=0)
    out = tr.apply_episode({"top": _frames(480, 848)})["top"][0]
    # Rows outside the fitted region are pad, rows inside are not.
    assert out[:29].max() == 0 and out[-29:].max() == 0
    assert out[29:-29].max() > 0


def test_letterbox_pad_value_is_honoured():
    tr = ResizeImages(["top"], target_h=240, target_w=320, letterbox=True,
                      pad_value=255)
    out = tr.apply_episode({"top": _frames(480, 848)})["top"][0]
    assert out[:29].min() == 255


def test_stretch_mode_is_unchanged_by_default():
    """letterbox defaults off, so existing callers keep exact-fill behaviour."""
    tr = ResizeImages(["top"], target_h=240, target_w=320)
    assert tr.letterbox is False
    out = tr.apply_episode({"top": _frames(480, 848)})["top"][0]
    assert out.shape == (240, 320, 3)
    # A stretch leaves no constant border.
    assert out[:29].max() > 0


def test_letterbox_changes_cache_key():
    """Otherwise switching the flag would silently reuse a stretched cache."""
    a = ResizeImages(["top"], target_h=240, target_w=320, letterbox=True)
    b = ResizeImages(["top"], target_h=240, target_w=320, letterbox=False)
    assert a.cache_key() != b.cache_key()


def test_resize_missing_key_raises():
    tr = ResizeImages(["top", "left"], target_h=240, target_w=320, letterbox=True)
    with pytest.raises(KeyError, match="left"):
        tr.apply_episode({"top": _frames(480, 640)})


def test_letterbox_mixed_sources_collate():
    """The point of the transform: two aspect ratios become one stackable shape."""
    tr = ResizeImages(["top"], target_h=240, target_w=320, letterbox=True)
    a = tr.apply_episode({"top": _frames(480, 640)})["top"]
    b = tr.apply_episode({"top": _frames(480, 848)})["top"]
    assert np.stack([a, b]).shape == (2, 3, 240, 320, 3)


# ── image compressor selection ────────────────────────────────────────────────

def _has_imagecodecs():
    import importlib.util
    return importlib.util.find_spec("imagecodecs") is not None


@pytest.mark.skipif(not _has_imagecodecs(), reason="imagecodecs not installed")
def test_jpeg_compressor_selects_jpeg_not_lz4():
    """imagecodecs does not self-register, and the kwarg was renamed; either
    mistake silently downgrades image keys to lossless lz4."""
    from robo_dataset.backends.zarr_backend import _jpeg_compressor
    codec = _jpeg_compressor()
    assert "jpeg" in type(codec).__name__.lower() or "jpeg" in codec.codec_id.lower()


@pytest.mark.skipif(not _has_imagecodecs(), reason="imagecodecs not installed")
def test_jpeg_compressor_round_trips_images():
    from robo_dataset.backends.zarr_backend import _jpeg_compressor
    codec = _jpeg_compressor()
    frame = _frames(240, 320, t=1)[0]
    back = codec.decode(codec.encode(frame))
    back = np.asarray(back).reshape(frame.shape)
    mse = np.mean((frame.astype(np.float64) - back.astype(np.float64)) ** 2)
    psnr = float("inf") if mse == 0 else 20 * np.log10(255.0 / np.sqrt(mse))
    assert psnr >= 30.0


# ── parallel disk cache ───────────────────────────────────────────────────────

class _FakeIndexedStore:
    """Minimal indexed EpisodeStore: enough for apply_disk_cached's fast path."""

    def __init__(self, n=7):
        self._eps = [
            {"img": (np.arange(4 * 6 * 8 * 3).reshape(4, 6, 8, 3) % 251 + i).astype(np.uint8),
             "state": np.full((4, 2), float(i), dtype=np.float32)}
            for i in range(n)
        ]

    @property
    def n_episodes(self):
        return len(self._eps)

    def get_episode(self, i):
        return {k: v.copy() for k, v in self._eps[i].items()}

    def iter_episodes(self):
        for i in range(self.n_episodes):
            yield self.get_episode(i)


def test_parallel_disk_cache_matches_serial(tmp_path):
    """workers>1 must not reorder episodes -- the writer appends, so order is data."""
    from robo_dataset.transforms import apply_disk_cached

    tr = ResizeImages(["img"], target_h=4, target_w=4, letterbox=True)
    a = apply_disk_cached(_FakeIndexedStore(), tr, str(tmp_path / "serial"),
                          image_keys=["img"], workers=1)
    b = apply_disk_cached(_FakeIndexedStore(), tr, str(tmp_path / "par"),
                          image_keys=["img"], workers=4)

    assert list(a.episode_ends) == list(b.episode_ends)
    # state is lossless on both sides, so episode identity is checkable exactly:
    # a scrambled order would still match on shape but not on values.
    np.testing.assert_array_equal(
        a.get_sequence("state", 0, a.n_frames),
        b.get_sequence("state", 0, b.n_frames),
    )


def test_parallel_disk_cache_falls_back_when_not_indexed(caplog):
    """A source without get_episode must still work, with a warning."""
    from robo_dataset.transforms import apply_disk_cached
    import tempfile

    class OnlyIterable:
        def iter_episodes(self):
            yield {"state": np.zeros((3, 2), dtype=np.float32)}

    with tempfile.TemporaryDirectory() as td:
        with caplog.at_level("WARNING"):
            out = apply_disk_cached(OnlyIterable(), ResizeImages([], 4, 4),
                                    td, workers=8)
    assert out.n_frames == 3
    assert any("not indexed" in r.getMessage() for r in caplog.records)
