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
    RLDSFrameStore,
    RLDSFrameStoreWriter,
    ShmStore,
)
from .sampler import SequenceSampler, get_val_mask
from .convert import convert, convert_rlds

__all__ = [
    "TrajectoryStore",
    "WritableTrajectoryStore",
    "RobomimicHDF5Store",
    "ZarrStore",
    "ZarrStoreWriter",
    "RLDSFrameStore",
    "RLDSFrameStoreWriter",
    "ShmStore",
    "SequenceSampler",
    "get_val_mask",
    "convert",
    "convert_rlds",
    "TrajectoryDataset",
]


def __getattr__(name):
    if name == "TrajectoryDataset":
        from .dataset import TrajectoryDataset
        return TrajectoryDataset
    if name in ("DataPrefetcher", "CUDADataPrefetcher"):
        from .prefetch import DataPrefetcher
        return DataPrefetcher
    raise AttributeError(f"module 'robo_dataset' has no attribute {name!r}")
