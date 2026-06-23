"""Streaming conversion between EpisodeStore backends.

Conversion is RAM-efficient: one episode is in memory at a time.
For WritableEpisodeStore targets, parallel workers are supported via
ThreadPoolExecutor when workers > 1.

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

from .backends.base import EpisodeStore, WritableEpisodeStore, WritableTrajectoryStore


def convert(
    source: EpisodeStore,
    target,
    transform: Optional[Callable[[Dict[str, np.ndarray]], Dict[str, np.ndarray]]] = None,
    keys: Optional[List[str]] = None,
    workers: int = 1,
    verbose: bool = True,
) -> None:
    """
    Stream all episodes from source into target, optionally applying a per-episode transform.

    If target is a WritableEpisodeStore, uses put_episode(ep_idx, data) and
    optionally parallelizes with ThreadPoolExecutor when workers > 1.
    Falls back to the old sequential put_episode(data) for legacy
    WritableTrajectoryStore targets.

    Args:
        source:    Source EpisodeStore.
        target:    Writable target store (must be freshly created, not yet finalized).
        transform: Optional per-episode transform: Dict[str, ndarray] → Dict[str, ndarray].
                   Called once per episode before put_episode.
        keys:      Subset of source keys to copy. Defaults to all keys.
        workers:   Number of parallel workers for WritableEpisodeStore targets (default 1).
        verbose:   Show tqdm progress bar.
    """
    keys = keys or list(source.keys)
    n = source.n_episodes
    is_new_writer = isinstance(target, WritableEpisodeStore)

    def _process(ep_idx: int) -> None:
        ep = {k: source.get_frames(ep_idx, k, 0, source.episode_length(ep_idx)) for k in keys}
        if transform is not None:
            ep = transform(ep)
        if is_new_writer:
            target.put_episode(ep_idx, ep)
        else:
            target.put_episode(ep)  # legacy sequential WritableTrajectoryStore

    if workers > 1 and is_new_writer:
        from concurrent.futures import ThreadPoolExecutor
        from tqdm import tqdm

        indices = list(range(n))
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(_process, i) for i in indices]
            if verbose:
                from tqdm import tqdm
                futures_iter = tqdm(futures, total=n, desc="Converting")
            else:
                futures_iter = futures
            for fut in futures_iter:
                fut.result()  # propagate exceptions
    else:
        indices = range(n)
        if verbose:
            from tqdm import tqdm
            indices = tqdm(indices, total=n, desc="Converting")
        for ep_idx in indices:
            _process(ep_idx)

    target.finalize()


def convert_rlds(
    source: EpisodeStore,
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

    if verbose:
        from tqdm import tqdm
        indices = tqdm(range(source.n_episodes), total=source.n_episodes, desc="Converting → RLDS")
    else:
        indices = range(source.n_episodes)

    for ep_idx in indices:
        ep_data = source.get_episode(ep_idx)
        ep_filtered = {k: ep_data[k] for k in keys}
        if transform is not None:
            ep_filtered = transform(ep_filtered)
        writer.put_episode(ep_filtered)

    writer.finalize()
    return RLDSFrameStore(target_path)
