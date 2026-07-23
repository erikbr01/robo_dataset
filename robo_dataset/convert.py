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

from __future__ import annotations

from typing import TYPE_CHECKING, Callable, Dict, List, Optional

import numpy as np

from .backends.base import EpisodeStore, WritableEpisodeStore, WritableTrajectoryStore

if TYPE_CHECKING:
    from .backends.rlds import RLDSFrameStore


def convert(
    source: EpisodeStore,
    target,
    transform: Optional[Callable[[Dict[str, np.ndarray]], Dict[str, np.ndarray]]] = None,
    keys: Optional[List[str]] = None,
    episode_indices: Optional[List[int]] = None,
    workers: int = 1,
    verbose: bool = True,
) -> None:
    """
    Stream episodes from source into target, optionally applying a per-episode transform.

    If target is a WritableEpisodeStore, uses put_episode(ep_idx, data) and
    optionally parallelizes with ThreadPoolExecutor when workers > 1.
    Falls back to the old sequential put_episode(data) for legacy
    WritableTrajectoryStore targets.

    Args:
        source:          Source EpisodeStore.
        target:          Writable target store (must be freshly created, not yet finalized).
        transform:       Optional per-episode transform: Dict[str, ndarray] → Dict[str, ndarray].
                         Called once per episode before put_episode.
        keys:            Subset of source keys to copy. Defaults to all keys.
        episode_indices: Source episode indices to convert, in this order —
                         everything else is skipped. Defaults to all of
                         source's episodes (range(source.n_episodes)). The
                         target always receives fresh contiguous indices
                         0..len(episode_indices)-1 regardless of the source
                         indices given, since every EpisodeStore backend
                         requires a contiguous 0-based layout; target's own
                         n_episodes must already reflect len(episode_indices)
                         (or len(range(source.n_episodes)) if omitted).
        workers:         Number of parallel workers for WritableEpisodeStore targets (default 1).
        verbose:         Show tqdm progress bar.
    """
    keys = keys or list(source.keys)
    src_indices = list(episode_indices) if episode_indices is not None else list(range(source.n_episodes))
    is_new_writer = isinstance(target, WritableEpisodeStore)

    def _process(pair: tuple) -> None:
        dst_idx, src_idx = pair
        ep = {k: source.get_frames(src_idx, k, 0, source.episode_length(src_idx)) for k in keys}
        if transform is not None:
            ep = transform(ep)
        if is_new_writer:
            target.put_episode(dst_idx, ep)
        else:
            target.put_episode(ep)  # legacy sequential WritableTrajectoryStore

    pairs = list(enumerate(src_indices))  # (dst_idx, src_idx) — dst is always contiguous from 0
    if workers > 1 and is_new_writer:
        from concurrent.futures import ThreadPoolExecutor
        from tqdm import tqdm

        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(_process, p) for p in pairs]
            if verbose:
                from tqdm import tqdm
                futures_iter = tqdm(futures, total=len(pairs), desc="Converting")
            else:
                futures_iter = futures
            for fut in futures_iter:
                fut.result()  # propagate exceptions
    else:
        it = pairs
        if verbose:
            from tqdm import tqdm
            it = tqdm(pairs, total=len(pairs), desc="Converting")
        for p in it:
            _process(p)

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
