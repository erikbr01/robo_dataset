"""Backend registry.

Core backends (HDF5, directory zarr, shm, video, zarr_episode) are imported eagerly.
Backends with optional heavy dependencies are imported lazily via PEP 562:

  - ZarrShardedStore / ZarrShardedStoreWriter  →  requires zarr >= 3.0
  - RLDSFrameStore / RLDSFrameStoreWriter      →  requires tfrecord + crc32c
"""

from .base import (
    EpisodeStore,
    WritableEpisodeStore,
    TrajectoryStore,
    WritableTrajectoryStore,
)
from .hdf5 import RobomimicHDF5Store
from .zarr_backend import ZarrStore, ZarrStoreWriter
from .zarr_episode import ZarrEpisodeStore, ZarrEpisodeStoreWriter
from .shm import ShmStore
from .video import VideoStore, VideoStoreWriter

_LAZY = {
    "ZarrShardedStore": "zarr_sharded",
    "ZarrShardedStoreWriter": "zarr_sharded",
    "RLDSFrameStore": "rlds",
    "RLDSFrameStoreWriter": "rlds",
    "LeRobotStore": "lerobot",
    "DroidStore": "droid",
}


def __getattr__(name):
    if name in _LAZY:
        import importlib
        module = importlib.import_module(f".{_LAZY[name]}", __name__)
        return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "EpisodeStore",
    "WritableEpisodeStore",
    "TrajectoryStore",
    "WritableTrajectoryStore",
    "RobomimicHDF5Store",
    "ZarrStore",
    "ZarrStoreWriter",
    "ZarrEpisodeStore",
    "ZarrEpisodeStoreWriter",
    "ZarrShardedStore",
    "ZarrShardedStoreWriter",
    "RLDSFrameStore",
    "RLDSFrameStoreWriter",
    "ShmStore",
    "VideoStore",
    "VideoStoreWriter",
    "LeRobotStore",
]
