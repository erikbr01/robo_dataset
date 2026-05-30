"""Test fixtures: create small synthetic HDF5 and convert to other backends.

All tests use the synthetic fixture so they run without the real Robomimic dataset.
To run against real data, override HDF5_PATH env var.
"""

import os
import tempfile

import h5py
import numpy as np
import pytest

# ── Synthetic dataset parameters ──────────────────────────────────────────────

N_EPISODES = 5
EP_LENGTHS = [10, 8, 12, 7, 15]  # variable-length episodes
H, W, C = 32, 32, 3
ACTION_DIM = 7
OBS_KEYS = ["agentview_image", "robot0_eye_in_hand_image"]
LOWDIM_KEY = "robot0_eef_pos"
LOWDIM_DIM = 3


@pytest.fixture(scope="session")
def synthetic_hdf5(tmp_path_factory):
    """Create a small synthetic Robomimic-format HDF5 file."""
    path = str(tmp_path_factory.mktemp("data") / "test.hdf5")
    rng = np.random.default_rng(42)

    with h5py.File(path, "w") as f:
        data = f.create_group("data")
        for i, T in enumerate(EP_LENGTHS):
            demo = data.create_group(f"demo_{i}")
            obs = demo.create_group("obs")
            for key in OBS_KEYS:
                obs.create_dataset(key, data=rng.integers(0, 255, (T, H, W, C), dtype=np.uint8))
            obs.create_dataset(LOWDIM_KEY, data=rng.random((T, LOWDIM_DIM), dtype=np.float32))
            demo.create_dataset("actions", data=rng.random((T, ACTION_DIM), dtype=np.float32))
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
