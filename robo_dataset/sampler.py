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


@numba.jit(nopython=True)
def _create_indices_with_sample_mask(
    episode_ends: np.ndarray,
    episode_mask: np.ndarray,
    sample_mask: np.ndarray,
    sequence_length: int,
    pad_before: int,
    pad_after: int,
    sample_mask_drop_threshold: float,
) -> np.ndarray:
    """Like _create_indices, but drops sequences whose excluded fraction
    (sample_mask[s] == False within the window) is >= the drop threshold.

    threshold 0.0 → drop if any frame excluded; 1.0 → drop only if all excluded.
    """
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

            n_excluded = 0
            window_len = buf_end - buf_start
            for s in range(buf_start, buf_end):
                if not sample_mask[s]:
                    n_excluded += 1
            if sample_mask_drop_threshold <= 0.0:
                drop = n_excluded > 0
            else:
                drop = window_len > 0 and (
                    n_excluded / window_len >= sample_mask_drop_threshold
                )
            if not drop:
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
        sample_mask: Optional[np.ndarray] = None,
        sample_mask_drop_threshold: float = 1.0,
    ):
        assert sequence_length >= 1
        self.store = store
        self.sequence_length = sequence_length
        self.keys = list(keys) if keys is not None else list(store.keys)
        self.key_first_k = key_first_k or {}

        episode_ends = store.episode_ends
        if episode_mask is None:
            episode_mask = np.ones(len(episode_ends), dtype=bool)

        if not np.any(episode_mask):
            self.indices = np.zeros((0, 4), dtype=np.int64)
        elif sample_mask is not None:
            self.indices = _create_indices_with_sample_mask(
                episode_ends,
                episode_mask,
                sample_mask,
                sequence_length,
                pad_before,
                pad_after,
                float(sample_mask_drop_threshold),
            )
        else:
            self.indices = _create_indices(
                episode_ends,
                episode_mask,
                sequence_length,
                pad_before,
                pad_after,
            )

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

            # key_first_k: only the first k frames are fetched; the remainder of
            # the window repeats the last fetched frame (consumers using
            # key_first_k discard frames past k anyway).
            out[samp_start : samp_start + k_frames] = fetched
            if k_frames < samp_end - samp_start:
                out[samp_start + k_frames : samp_end] = fetched[-1]

            result[key] = out
        return result
