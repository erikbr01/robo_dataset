"""Abstract base class for trajectory stores."""

from abc import ABC, abstractmethod
from typing import Dict, Iterator, List

import numpy as np


class TrajectoryStore(ABC):
    """
    Flat-frame trajectory store: all episodes concatenated along axis 0.

    episode_ends[i] = cumulative frame count after episode i (exclusive end of ep i).

    Implementations must guarantee that get_sequence is safe to call concurrently
    from multiple DataLoader worker processes after on_worker_init() is called.
    """

    @property
    @abstractmethod
    def episode_ends(self) -> np.ndarray:
        """1-D int64 array of length n_episodes."""

    @property
    @abstractmethod
    def keys(self) -> List[str]:
        """All data keys available in this store."""

    @abstractmethod
    def get_sequence(self, key: str, start: int, end: int) -> np.ndarray:
        """Return array of shape (end-start, *frame_shape(key))."""

    @abstractmethod
    def frame_shape(self, key: str) -> tuple:
        """Shape of a single frame for key (excludes time axis)."""

    @abstractmethod
    def frame_dtype(self, key: str) -> np.dtype:
        """Dtype of a single frame for key."""

    def on_worker_init(self, worker_id: int) -> None:
        """Called once per DataLoader worker after fork. Override to reopen file handles."""

    # ── Derived helpers ───────────────────────────────────────────────────────

    @property
    def n_frames(self) -> int:
        return int(self.episode_ends[-1]) if len(self.episode_ends) else 0

    @property
    def n_episodes(self) -> int:
        return len(self.episode_ends)

    def episode_start(self, ep_idx: int) -> int:
        return 0 if ep_idx == 0 else int(self.episode_ends[ep_idx - 1])

    def episode_length(self, ep_idx: int) -> int:
        return int(self.episode_ends[ep_idx]) - self.episode_start(ep_idx)

    def iter_episodes(self) -> Iterator[Dict[str, np.ndarray]]:
        """Yield one dict per episode: {key: ndarray(T, *frame_shape)}."""
        for ep_idx in range(self.n_episodes):
            start = self.episode_start(ep_idx)
            end = int(self.episode_ends[ep_idx])
            yield {key: self.get_sequence(key, start, end) for key in self.keys}


class WritableTrajectoryStore(TrajectoryStore, ABC):
    """Store that accepts episode-by-episode writes (conversion target, disk cache)."""

    @classmethod
    @abstractmethod
    def create(
        cls,
        path: str,
        key_shapes: Dict[str, tuple],
        key_dtypes: Dict[str, np.dtype],
        image_keys: List[str] = (),
        **kwargs,
    ) -> "WritableTrajectoryStore":
        """Create a new empty writable store at path."""

    @abstractmethod
    def put_episode(self, data: Dict[str, np.ndarray]) -> None:
        """Write one episode. data[key] must have shape (T, *frame_shape(key))."""

    @abstractmethod
    def finalize(self) -> None:
        """Flush buffers, build indices. Call after all put_episode()."""
