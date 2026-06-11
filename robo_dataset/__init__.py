"""robo_dataset: unified trajectory dataset for robot learning.

Backends (zarr, HDF5, RLDS) and conversion utilities are importable directly.
Torch-dependent modules (TrajectoryDataset, transforms) are NOT eagerly imported
here to avoid crashes in environments where torch is unavailable or broken.
"""

from .backends import (
    TrajectoryStore,
    WritableTrajectoryStore,
    RobomimicHDF5Store,
    ZarrStore,
    ZarrStoreWriter,
    ShmStore,
    VideoStore,
    VideoStoreWriter,
)
from .sampler import SequenceSampler, get_val_mask
from .convert import convert, convert_rlds
from .schema import Field, VectorSchema

__all__ = [
    "TrajectoryStore",
    "WritableTrajectoryStore",
    "RobomimicHDF5Store",
    "ZarrStore",
    "ZarrStoreWriter",
    "ZarrShardedStore",
    "ZarrShardedStoreWriter",
    "RLDSFrameStore",
    "RLDSFrameStoreWriter",
    "ShmStore",
    "VideoStore",
    "VideoStoreWriter",
    "SequenceSampler",
    "get_val_mask",
    "convert",
    "convert_rlds",
    "Field",
    "VectorSchema",
    "TrajectoryDataset",
]


def __getattr__(name):
    if name == "TrajectoryDataset":
        from .dataset import TrajectoryDataset
        return TrajectoryDataset
    if name in ("DataPrefetcher", "CUDADataPrefetcher"):
        from .prefetch import DataPrefetcher
        return DataPrefetcher
    if name in ("ZarrShardedStore", "ZarrShardedStoreWriter",
                "RLDSFrameStore", "RLDSFrameStoreWriter"):
        from . import backends
        return getattr(backends, name)
    raise AttributeError(f"module 'robo_dataset' has no attribute {name!r}")
