"""Test fixtures: create small synthetic HDF5 and convert to other backends."""

import collections

import h5py
import numpy as np
import pytest

from .constants import N_EPISODES, EP_LENGTHS, H, W, C, ACTION_DIM, OBS_KEYS, LOWDIM_KEY, LOWDIM_DIM
from .optional_deps import OPTIONAL_DEPS, describe, missing_deps, strictly_required


# ── Optional capabilities: report always, enforce on request ──────────────────
# Without this a suite that skipped every LeRobot test reads the same as one
# that ran them. See tests/optional_deps.py.

def pytest_configure(config):
    missing = missing_deps()
    try:
        required = strictly_required()
    except ValueError as exc:  # a typo must not surface as INTERNALERROR
        raise pytest.UsageError(str(exc)) from None
    blocking = [p for p in required if p in missing]
    if blocking:
        raise pytest.UsageError(
            "ROBO_DATASET_STRICT_DEPS requires these capabilities, which are "
            "unavailable:\n"
            + "".join(f"  - {describe(p)}: {OPTIONAL_DEPS[p].description}\n"
                      for p in blocking)
            + "Install/upgrade them, or unset the variable to allow skipping."
        )
    config._missing_optional_deps = missing


def pytest_report_header(config):
    missing = getattr(config, "_missing_optional_deps", [])
    present = [p for p in OPTIONAL_DEPS if p not in missing]
    lines = [f"optional capabilities present: {', '.join(present) or 'none'}"]
    if missing:
        lines.append("optional capabilities MISSING: "
                     + "; ".join(describe(p) for p in missing)
                     + " (tests needing them will skip)")
    return lines


def pytest_terminal_summary(terminalreporter, exitstatus, config):
    """Attribute skips to the capability that caused them, so they stay visible."""
    missing = getattr(config, "_missing_optional_deps", [])
    if not missing:
        return
    counts = collections.Counter()
    for report in terminalreporter.stats.get("skipped", []):
        reason = str(getattr(report, "longrepr", ("", "", ""))[-1] or "")
        for name in missing:
            if OPTIONAL_DEPS[name].skip_token in reason:
                counts[name] += 1
                break
    terminalreporter.write_sep("=", "optional capabilities", yellow=True)
    for name in missing:
        terminalreporter.write_line(
            f"  {describe(name)} -> {counts[name]} test(s) skipped"
            f" | {OPTIONAL_DEPS[name].description}", yellow=True)
    terminalreporter.write_line(
        "  set ROBO_DATASET_STRICT_DEPS=1 to make these a hard error", yellow=True)


@pytest.fixture(scope="session")
def synthetic_hdf5(tmp_path_factory):
    """Create a small synthetic Robomimic-format HDF5 file."""
    path = str(tmp_path_factory.mktemp("data") / "test.hdf5")
    rng = np.random.default_rng(42)

    # Pre-build a smooth base image (gradient + blobs) that JPEGs well (PSNR > 30 dB)
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32) / max(H, W)
    base_gray = (128 + 60 * np.sin(4 * np.pi * yy) * np.cos(4 * np.pi * xx)).astype(np.uint8)
    base_img = np.stack([base_gray] * C, axis=-1)  # (H, W, C)

    with h5py.File(path, "w") as f:
        data = f.create_group("data")
        for i, T in enumerate(EP_LENGTHS):
            demo = data.create_group(f"demo_{i}")
            obs = demo.create_group("obs")
            for key in OBS_KEYS:
                # Smooth images: base + small per-frame perturbation (JPEG-friendly)
                perturbations = rng.integers(-10, 10, (T, H, W, C), dtype=np.int16)
                imgs = np.clip(base_img[None].astype(np.int16) + perturbations, 0, 255).astype(np.uint8)
                obs.create_dataset(key, data=imgs)
            obs.create_dataset(LOWDIM_KEY, data=rng.random((T, LOWDIM_DIM)).astype(np.float32))
            demo.create_dataset("actions", data=rng.random((T, ACTION_DIM)).astype(np.float32))
    return path


@pytest.fixture(scope="session")
def zarr_store_path(synthetic_hdf5, tmp_path_factory):
    """Convert synthetic HDF5 to zarr and return the path."""
    from robo_dataset.backends.hdf5 import RobomimicHDF5Store
    from robo_dataset.backends.zarr_backend import ZarrStoreWriter
    from robo_dataset.convert import convert

    hdf5 = RobomimicHDF5Store(synthetic_hdf5, obs_keys=OBS_KEYS + [LOWDIM_KEY])
    out_path = str(tmp_path_factory.mktemp("zarr") / "test.zarr")
    writer = ZarrStoreWriter.create(
        path=out_path,
        key_shapes={k: hdf5.frame_shape(k) for k in hdf5.keys},
        key_dtypes={k: hdf5.frame_dtype(k) for k in hdf5.keys},
        image_keys=OBS_KEYS,
    )
    convert(hdf5, writer, verbose=False)
    return out_path


@pytest.fixture(scope="session")
def rlds_store_path(synthetic_hdf5, tmp_path_factory):
    """Convert synthetic HDF5 to RLDS and return the path."""
    from robo_dataset.backends.hdf5 import RobomimicHDF5Store
    from robo_dataset.convert import convert_rlds

    hdf5 = RobomimicHDF5Store(synthetic_hdf5, obs_keys=OBS_KEYS + [LOWDIM_KEY])
    out_path = str(tmp_path_factory.mktemp("rlds") / "test.rlds")
    convert_rlds(hdf5, out_path, image_keys=OBS_KEYS, jpeg_quality=95, verbose=False)
    return out_path


@pytest.fixture(scope="session")
def zarr_sharded_path(synthetic_hdf5, tmp_path_factory):
    """Convert synthetic HDF5 to sharded zarr-v3 and return the path."""
    from robo_dataset.backends.hdf5 import RobomimicHDF5Store
    from robo_dataset.backends.zarr_sharded import ZarrShardedStoreWriter
    from robo_dataset.convert import convert

    hdf5 = RobomimicHDF5Store(synthetic_hdf5, obs_keys=OBS_KEYS + [LOWDIM_KEY])
    out_path = str(tmp_path_factory.mktemp("zarr_sharded") / "test_sharded.zarr")
    writer = ZarrShardedStoreWriter.create(
        path=out_path,
        key_shapes={k: hdf5.frame_shape(k) for k in hdf5.keys},
        key_dtypes={k: hdf5.frame_dtype(k) for k in hdf5.keys},
        chunk_frames=4,
        shard_frames=16,
    )
    convert(hdf5, writer, verbose=False)
    return out_path


@pytest.fixture(scope="session")
def video_store_path(synthetic_hdf5, tmp_path_factory):
    """Convert synthetic HDF5 to VideoStore (h264 kfi=1) and return the path."""
    pytest.importorskip("av")
    from robo_dataset.backends.hdf5 import RobomimicHDF5Store
    from robo_dataset.backends.video import VideoStoreWriter
    from robo_dataset.convert import convert

    hdf5 = RobomimicHDF5Store(synthetic_hdf5, obs_keys=OBS_KEYS + [LOWDIM_KEY])
    out_path = str(tmp_path_factory.mktemp("video") / "test.video")
    writer = VideoStoreWriter.create(
        path=out_path,
        n_episodes=hdf5.n_episodes,
        keys=hdf5.keys,
        key_shapes={k: hdf5.frame_shape(k) for k in hdf5.keys},
        key_dtypes={k: hdf5.frame_dtype(k) for k in hdf5.keys},
        image_keys=OBS_KEYS,
        kfi=1,
        fps=10,
    )
    convert(hdf5, writer, verbose=False)
    return out_path
