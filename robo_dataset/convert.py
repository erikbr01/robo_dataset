"""Streaming conversion between TrajectoryStore backends.

Conversion is RAM-efficient: one episode is in memory at a time.
Episodes from the source are written to the target one-by-one via put_episode().

Usage:
    from robo_dataset.convert import convert
    from robo_dataset.backends.hdf5 import RobomimicHDF5Store
    from robo_dataset.backends.zarr_backend import ZarrStoreWriter

    src = RobomimicHDF5Store(hdf5_path, obs_keys=["agentview_image", "robot0_eye_in_hand_image"])
    dst = ZarrStoreWriter.create(
        output_path,
        key_shapes={k: src.frame_shape(k) for k in src.keys},
        key_dtypes={k: src.frame_dtype(k) for k in src.keys},
        image_keys=["agentview_image", "robot0_eye_in_hand_image"],
    )
    convert(src, dst, verbose=True)
"""

from typing import Callable, Dict, List, Optional

import numpy as np
from tqdm import tqdm

from .backends.base import TrajectoryStore, WritableTrajectoryStore


def convert(
    source: TrajectoryStore,
    target: WritableTrajectoryStore,
    transform: Optional[Callable[[Dict[str, np.ndarray]], Dict[str, np.ndarray]]] = None,
    keys: Optional[List[str]] = None,
    verbose: bool = True,
) -> None:
    """
    Stream all episodes from source into target, optionally applying a per-episode transform.

    Args:
        source:    Source TrajectoryStore.
        target:    Writable target store (must be freshly created, not yet finalized).
        transform: Optional per-episode transform: Dict[str, ndarray] → Dict[str, ndarray].
                   Called once per episode before put_episode.
        keys:      Subset of source keys to copy. Defaults to all keys.
        verbose:   Show tqdm progress bar.
    """
    keys = keys or list(source.keys)
    it = source.iter_episodes()
    if verbose:
        it = tqdm(it, total=source.n_episodes, desc="Converting")

    for ep_data in it:
        ep_filtered = {k: ep_data[k] for k in keys}
        if transform is not None:
            ep_filtered = transform(ep_filtered)
        target.put_episode(ep_filtered)

    target.finalize()


def convert_rlds(
    source: TrajectoryStore,
    target_path: str,
    image_keys: List[str],
    jpeg_quality: int = 75,
    keys: Optional[List[str]] = None,
    transform: Optional[Callable] = None,
    verbose: bool = True,
) -> "RLDSFrameStore":
    """Convenience wrapper for HDF5/zarr → RLDS conversion."""
    from .backends.rlds import RLDSFrameStoreWriter, RLDSFrameStore

    keys = keys or list(source.keys)
    writer = RLDSFrameStoreWriter(target_path, image_keys=image_keys, jpeg_quality=jpeg_quality)

    it = source.iter_episodes()
    if verbose:
        it = tqdm(it, total=source.n_episodes, desc="Converting → RLDS")

    for ep_data in it:
        ep_filtered = {k: ep_data[k] for k in keys}
        if transform is not None:
            ep_filtered = transform(ep_filtered)
        writer.put_episode(ep_filtered)

    writer.finalize()
    return RLDSFrameStore(target_path)
