"""Sequence sampler — builds a flat index of (start, end, pad_before, pad_after) tuples.

Adapted from diffusion_policy/common/sampler.py with the ReplayBuffer dependency
replaced by TrajectoryStore.
"""

from typing import Dict, List, Optional

import numba
import numpy as np

from .backends.base import TrajectoryStore


@numba.jit(nopython=True)
def _create_indices(
    episode_ends: np.ndarray,
    episode_mask: np.ndarray,
    sequence_length: int,
    pad_before: int,
    pad_after: int,
) -> np.ndarray:
    pad_before = min(max(pad_before, 0), sequence_length - 1)
    pad_after = min(max(pad_after, 0), sequence_length - 1)

    indices = []
    for i in range(len(episode_ends)):
        if not episode_mask[i]:
            continue
        ep_start = 0 if i == 0 else episode_ends[i - 1]
        ep_end = episode_ends[i]
        ep_len = ep_end - ep_start

        min_start = -pad_before
        max_start = ep_len - sequence_length + pad_after

        for idx in range(min_start, max_start + 1):
            buf_start = max(idx, 0) + ep_start
            buf_end = min(idx + sequence_length, ep_len) + ep_start
            sample_start = buf_start - (idx + ep_start)
            sample_end = sequence_length - ((idx + sequence_length + ep_start) - buf_end)
            indices.append([buf_start, buf_end, sample_start, sample_end])

    return np.array(indices)


def get_val_mask(n_episodes: int, val_ratio: float, seed: int = 0) -> np.ndarray:
    mask = np.zeros(n_episodes, dtype=bool)
    if val_ratio <= 0:
        return mask
    n_val = min(max(1, round(n_episodes * val_ratio)), n_episodes - 1)
    rng = np.random.default_rng(seed=seed)
    mask[rng.choice(n_episodes, size=n_val, replace=False)] = True
    return mask


class SequenceSampler:
    """
    Maps a flat integer index to a (horizon, *shape) sequence from a TrajectoryStore.

    Handles episode boundary padding (repeat first/last frame).
    """

    def __init__(
        self,
        store: TrajectoryStore,
        sequence_length: int,
        pad_before: int = 0,
        pad_after: int = 0,
        keys: Optional[List[str]] = None,
        key_first_k: Optional[Dict[str, int]] = None,
        episode_mask: Optional[np.ndarray] = None,
    ):
        assert sequence_length >= 1
        self.store = store
        self.sequence_length = sequence_length
        self.keys = list(keys) if keys is not None else list(store.keys)
        self.key_first_k = key_first_k or {}

        episode_ends = store.episode_ends
        if episode_mask is None:
            episode_mask = np.ones(len(episode_ends), dtype=bool)

        if np.any(episode_mask):
            self.indices = _create_indices(
                episode_ends,
                episode_mask,
                sequence_length,
                pad_before,
                pad_after,
            )
        else:
            self.indices = np.zeros((0, 4), dtype=np.int64)

    def __len__(self) -> int:
        return len(self.indices)

    def sample_sequence(self, idx: int) -> Dict[str, np.ndarray]:
        buf_start, buf_end, samp_start, samp_end = self.indices[idx]

        result = {}
        for key in self.keys:
            n_frames = buf_end - buf_start
            k_frames = min(self.key_first_k.get(key, n_frames), n_frames)

            fetched = self.store.get_sequence(key, int(buf_start), int(buf_start + k_frames))

            shape = self.store.frame_shape(key)
            dtype = self.store.frame_dtype(key)
            out = np.empty((self.sequence_length,) + shape, dtype=dtype)

            # Pad with boundary values
            if samp_start > 0:
                out[:samp_start] = fetched[0]
            if samp_end < self.sequence_length:
                out[samp_end:] = fetched[-1] if k_frames > 0 else np.zeros(shape, dtype=dtype)

            out[samp_start:samp_end] = fetched

            result[key] = out
        return result
