"""Zero-copy logical views over trajectory stores."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from .backends.base import EpisodeStore


class EpisodeView(EpisodeStore):
    """Expose selected source episodes as a contiguous logical store.

    The view owns no frame bytes. Logical episode ``i`` forwards to source
    episode ``episode_ids[i]``; cumulative frame indices are recomputed for the
    logical ordering. This is suitable for overlapping research subsets over
    video stores because selection no longer requires decode/re-encode copies.
    """

    def __init__(self, source: EpisodeStore, episode_ids: Sequence[int]):
        ids = list(episode_ids)
        if not ids:
            raise ValueError("episode_ids must contain at least one episode")
        non_integer = [value for value in ids if isinstance(value, bool) or not isinstance(value, (int, np.integer))]
        if non_integer:
            raise TypeError(f"episode_ids must be integers, got {non_integer!r}")
        ids = [int(value) for value in ids]
        if len(ids) != len(set(ids)):
            raise ValueError("episode_ids must be unique")
        bad = [value for value in ids if value < 0 or value >= source.n_episodes]
        if bad:
            raise ValueError(
                f"episode_ids {bad} out of range for a {source.n_episodes}-episode store"
            )

        self.source = source
        self.episode_ids = tuple(ids)
        lengths = np.asarray(
            [source.episode_length(source_idx) for source_idx in ids],
            dtype=np.int64,
        )
        self._episode_ends = np.cumsum(lengths)

    @property
    def n_episodes(self) -> int:
        return len(self.episode_ids)

    @property
    def keys(self) -> list[str]:
        return list(self.source.keys)

    @property
    def episode_ends(self) -> np.ndarray:
        return self._episode_ends

    def episode_length(self, ep_idx: int) -> int:
        self._check_episode_index(ep_idx)
        end = int(self._episode_ends[ep_idx])
        start = 0 if ep_idx == 0 else int(self._episode_ends[ep_idx - 1])
        return end - start

    def frame_shape(self, key: str) -> tuple:
        return self.source.frame_shape(key)

    def frame_dtype(self, key: str) -> np.dtype:
        return self.source.frame_dtype(key)

    def get_frames(self, ep_idx: int, key: str, start: int, end: int) -> np.ndarray:
        self._check_episode_index(ep_idx)
        length = self.episode_length(ep_idx)
        if not (0 <= start < end <= length):
            raise IndexError(
                f"Requested logical episode {ep_idx} frames [{start}, {end}) "
                f"for an episode of length {length}"
            )
        return self.source.get_frames(self.episode_ids[ep_idx], key, start, end)

    def on_worker_init(self, worker_id: int) -> None:
        self.source.on_worker_init(worker_id)

    @property
    def schemas(self):
        return self.source.schemas

    def _check_episode_index(self, ep_idx: int) -> None:
        if not 0 <= ep_idx < self.n_episodes:
            raise IndexError(
                f"logical episode index {ep_idx} out of range for {self.n_episodes} episodes"
            )
