"""open_store — open any TrajectoryStore from a path, with format auto-detection."""

import os
from typing import Optional

from .backends.base import TrajectoryStore


def detect_format(path: str) -> str:
    """Infer the store format from on-disk layout."""
    if os.path.isfile(path) and path.endswith((".hdf5", ".h5")):
        return "robomimic_hdf5"
    if os.path.isdir(path):
        if os.path.exists(os.path.join(path, "meta", "info.json")):
            return "lerobot"
        if os.path.exists(os.path.join(path, "meta.json")):
            import json
            with open(os.path.join(path, "meta.json")) as f:
                meta = json.load(f)
            if "video_keys" in meta:
                return "video"
            if "shard_frames" in meta:
                return "zarr_sharded"
            return "zarr"
        if os.path.exists(os.path.join(path, ".zgroup")) or os.path.exists(
            os.path.join(path, "zarr.json")
        ):
            return "zarr"
    raise ValueError(f"Cannot detect store format for {path!r}")


def open_store(path: str, format: str = "auto", obs_keys: Optional[list] = None,
               **kwargs) -> TrajectoryStore:
    """Open a read-only TrajectoryStore.

    Args:
        path:     store root (dir) or HDF5 file.
        format:   'auto' | 'video' | 'zarr' | 'zarr_sharded' | 'lerobot'
                  | 'robomimic_hdf5' | 'droid'
        obs_keys: required for robomimic_hdf5 (which obs keys to expose).
        kwargs:   forwarded to the backend constructor.
    """
    if format == "auto":
        format = detect_format(path)

    if format == "video":
        from .backends.video import VideoStore
        return VideoStore(path, **kwargs)
    if format == "zarr":
        from .backends.zarr_backend import ZarrStore
        return ZarrStore(path, **kwargs)
    if format == "zarr_sharded":
        from .backends.zarr_sharded import ZarrShardedStore
        return ZarrShardedStore(path, **kwargs)
    if format == "lerobot":
        from .backends.lerobot import LeRobotStore
        return LeRobotStore(path, **kwargs)
    if format == "robomimic_hdf5":
        from .backends.hdf5 import RobomimicHDF5Store
        if obs_keys is None:
            import h5py
            with h5py.File(path, "r") as f:
                obs_keys = list(f["data/demo_0/obs"].keys())
        return RobomimicHDF5Store(path, obs_keys=list(obs_keys), **kwargs)
    if format == "droid":
        from .backends.droid import DroidStore
        return DroidStore(path, **kwargs)
    raise ValueError(f"Unknown store format: {format!r}")
