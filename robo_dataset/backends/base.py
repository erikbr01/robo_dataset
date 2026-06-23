"""Abstract base class for trajectory stores.

Two hierarchies coexist:

EpisodeStore (new, primary)
  Episodes are the primary unit.  episode_ends is *derived* from per-episode
  array shapes — no episode_ends.npy on disk for new backends.
  write interface: put_episode(ep_idx, data) — episodes are independent,
  enabling trivially parallel writes.

TrajectoryStore = EpisodeStore   (backward-compat alias)
WritableEpisodeStore (new writable ABC)
WritableTrajectoryStore = WritableEpisodeStore  (backward-compat alias)
"""

from abc import ABC, abstractmethod
from typing import Dict, Iterator, List, Optional, Tuple

import numpy as np


def iter_episode_slices(
    episode_ends: np.ndarray, start: int, end: int
) -> Iterator[Tuple[int, int, int]]:
    """Yield (ep_idx, local_lo, local_hi) for each episode overlapping [start, end).

    The local indices are relative to the episode's own frame axis, so
    ``episode_data[local_lo:local_hi]`` concatenated over the yielded episodes
    reconstructs the flat range [start, end).

    Raises IndexError unless 0 <= start < end <= episode_ends[-1].
    """
    n_frames = int(episode_ends[-1]) if len(episode_ends) else 0
    if not (0 <= start < end <= n_frames):
        raise IndexError(
            f"Requested frame range [{start}, {end}) is invalid for a store "
            f"with {n_frames} frames (need 0 <= start < end <= {n_frames})."
        )
    ep_first = int(np.searchsorted(episode_ends, start, side="right"))
    ep_last = int(np.searchsorted(episode_ends, end - 1, side="right"))
    for ep_idx in range(ep_first, ep_last + 1):
        ep_offset = int(episode_ends[ep_idx - 1]) if ep_idx > 0 else 0
        ep_end_global = int(episode_ends[ep_idx])
        local_lo = max(start, ep_offset) - ep_offset
        local_hi = min(end, ep_end_global) - ep_offset
        yield ep_idx, local_lo, local_hi


def validate_episode_ends(
    episode_ends: np.ndarray,
    n_frames: Optional[int] = None,
    path: str = "",
) -> None:
    """Sanity-check a store's episode_ends array at open time.

    Raises ValueError (with a corrupt/truncated message) if the array is
    empty, not strictly increasing, or inconsistent with the metadata
    frame count.
    """
    where = f" at {path!r}" if path else ""
    if len(episode_ends) == 0:
        raise ValueError(
            f"Store{where} has an empty episode_ends array — "
            "store is corrupt/truncated."
        )
    if episode_ends[0] <= 0 or np.any(np.diff(episode_ends) <= 0):
        raise ValueError(
            f"Store{where} has a non-strictly-increasing episode_ends array "
            f"({episode_ends.tolist()}) — store is corrupt/truncated."
        )
    if n_frames is not None and int(episode_ends[-1]) != int(n_frames):
        raise ValueError(
            f"Store{where}: episode_ends[-1]={int(episode_ends[-1])} does not "
            f"match metadata n_frames={int(n_frames)} — store is corrupt/truncated."
        )


# ── New episode-indexed ABC ────────────────────────────────────────────────────

class EpisodeStore(ABC):
    """Episode-indexed trajectory store.

    Episodes are the primary unit.  episode_ends is DERIVED from
    episode_length() — it is not stored on disk for new backends.

    Required primitives
    -------------------
    n_episodes, keys, episode_length, frame_shape, frame_dtype, get_frames

    Everything else is derived and need not be overridden.

    Implementations must guarantee that get_frames is safe to call concurrently
    from multiple DataLoader worker processes after on_worker_init() is called.
    """

    # ── Required primitives ───────────────────────────────────────────────────

    @property
    @abstractmethod
    def n_episodes(self) -> int:
        """Number of episodes in this store."""

    @property
    @abstractmethod
    def keys(self) -> List[str]:
        """All data keys available in this store."""

    @abstractmethod
    def episode_length(self, ep_idx: int) -> int:
        """Number of frames in episode ep_idx."""

    @abstractmethod
    def frame_shape(self, key: str) -> tuple:
        """Shape of a single frame for key (excludes time axis)."""

    @abstractmethod
    def frame_dtype(self, key: str) -> np.dtype:
        """Dtype of a single frame for key."""

    @abstractmethod
    def get_frames(self, ep_idx: int, key: str, start: int, end: int) -> np.ndarray:
        """Read frames [start, end) from episode ep_idx for a single key.

        Returns ndarray of shape (end - start, *frame_shape(key)).
        """

    # ── Derived helpers ───────────────────────────────────────────────────────

    @property
    def episode_ends(self) -> np.ndarray:
        """Cumulative frame counts — derived from episode_length(), NOT stored on disk."""
        ends = np.array(
            [self.episode_length(i) for i in range(self.n_episodes)], dtype=np.int64
        )
        return np.cumsum(ends)

    @property
    def n_frames(self) -> int:
        return int(self.episode_ends[-1]) if self.n_episodes else 0

    def episode_start(self, ep_idx: int) -> int:
        return 0 if ep_idx == 0 else int(self.episode_ends[ep_idx - 1])

    def get_episode(self, ep_idx: int) -> Dict[str, np.ndarray]:
        """Return all frames for episode ep_idx as a dict of arrays."""
        n = self.episode_length(ep_idx)
        return {k: self.get_frames(ep_idx, k, 0, n) for k in self.keys}

    def get_sequence(self, key: str, start: int, end: int) -> np.ndarray:
        """Backward-compat flat-index access.  SequenceSampler calls this."""
        ends = self.episode_ends
        parts = []
        for ep_idx, lo, hi in iter_episode_slices(ends, start, end):
            parts.append(self.get_frames(ep_idx, key, lo, hi))
        return np.concatenate(parts, axis=0) if len(parts) > 1 else parts[0]

    def iter_episodes(self) -> Iterator[Dict[str, np.ndarray]]:
        """Yield one dict per episode: {key: ndarray(T, *frame_shape)}."""
        for i in range(self.n_episodes):
            yield self.get_episode(i)

    def on_worker_init(self, worker_id: int) -> None:
        """Called once per DataLoader worker after fork. Override to reopen file handles."""

    @property
    def schemas(self) -> Dict[str, "VectorSchema"]:  # noqa: F821
        """Vector field schemas per key (see robo_dataset.schema).

        Self-describing stores (written with ``schemas=``, or backends that
        know their layout) return one VectorSchema per vector-valued key;
        consumers should validate their expectations against these instead of
        hardcoding index ranges. Default: empty (no schema information).
        """
        return {}


# ── Backward-compat shim: TrajectoryStore ─────────────────────────────────────
#
# Old backends inherit from TrajectoryStore (= EpisodeStore) and implement
# episode_ends as a property and get_sequence directly.  They do not implement
# get_frames; to satisfy the ABC we provide a default get_frames that calls
# get_sequence.  n_episodes is derived from episode_ends.
#
# New backends should inherit from EpisodeStore and implement get_frames.

class TrajectoryStore(EpisodeStore):
    """Flat-frame trajectory store: backward-compat base for legacy backends.

    Legacy backends implement episode_ends (property) and get_sequence
    directly; n_episodes and episode_length are derived from episode_ends.
    get_frames defaults to a slice of get_sequence (correct but slightly
    inefficient for cross-episode slices — acceptable for legacy paths).

    New backends should inherit from EpisodeStore instead and implement
    get_frames natively.
    """

    # ── episode_ends: provided by legacy subclasses as a property ─────────────

    @property
    @abstractmethod
    def episode_ends(self) -> np.ndarray:
        """1-D int64 array of length n_episodes (stored or computed)."""

    # ── EpisodeStore primitives derived from episode_ends ─────────────────────

    @property
    def n_episodes(self) -> int:
        return len(self.episode_ends)

    def episode_length(self, ep_idx: int) -> int:
        return int(self.episode_ends[ep_idx]) - (
            0 if ep_idx == 0 else int(self.episode_ends[ep_idx - 1])
        )

    # ── get_sequence: provided by legacy subclasses ───────────────────────────

    @abstractmethod
    def get_sequence(self, key: str, start: int, end: int) -> np.ndarray:
        """Return array of shape (end-start, *frame_shape(key))."""

    # ── get_frames: default implementation via get_sequence ───────────────────

    def get_frames(self, ep_idx: int, key: str, start: int, end: int) -> np.ndarray:
        """Default: translate local indices to flat indices and call get_sequence."""
        ep_start = self.episode_start(ep_idx)
        return self.get_sequence(key, ep_start + start, ep_start + end)

    # ── episode_start: override EpisodeStore.episode_start to use stored ends ─

    def episode_start(self, ep_idx: int) -> int:
        return 0 if ep_idx == 0 else int(self.episode_ends[ep_idx - 1])

    # ── iter_episodes: flat read via get_sequence ─────────────────────────────

    def iter_episodes(self) -> Iterator[Dict[str, np.ndarray]]:
        """Yield one dict per episode: {key: ndarray(T, *frame_shape)}."""
        for ep_idx in range(self.n_episodes):
            start = self.episode_start(ep_idx)
            end = int(self.episode_ends[ep_idx])
            yield {key: self.get_sequence(key, start, end) for key in self.keys}

    # ── n_frames: override to use episode_ends directly ───────────────────────

    @property
    def n_frames(self) -> int:
        return int(self.episode_ends[-1]) if len(self.episode_ends) else 0

    def on_worker_init(self, worker_id: int) -> None:
        """Called once per DataLoader worker after fork. Override to reopen file handles."""

    @property
    def schemas(self) -> Dict[str, "VectorSchema"]:  # noqa: F821
        return {}


# ── New writable ABC ───────────────────────────────────────────────────────────

class WritableEpisodeStore(EpisodeStore, ABC):
    """Writable episode store — accepts put_episode(ep_idx, data) with no ordering
    constraint.  Different ep_idx values can be written concurrently.

    Episode contract
    ----------------
    Callers should ensure each ep_idx is written exactly once.
    Implementations are responsible for their own thread-safety.
    """

    def _init_contract(
        self,
        key_shapes: Optional[Dict[str, tuple]] = None,
        key_dtypes: Optional[Dict[str, np.dtype]] = None,
    ) -> None:
        """Store the declared episode contract (values may be None)."""
        self._contract_shapes: Optional[Dict[str, tuple]] = (
            {k: tuple(v) for k, v in key_shapes.items()} if key_shapes else None
        )
        self._contract_dtypes: Optional[Dict[str, np.dtype]] = (
            {k: np.dtype(v) for k, v in key_dtypes.items()} if key_dtypes else None
        )
        self._contract_keys: Optional[frozenset] = None  # locked on first episode

    def _validate_episode(self, data: Dict[str, np.ndarray], episode_idx: int) -> int:
        """Validate one episode against the (possibly declared) contract.

        On the first episode the key set is locked (a declared contract wins;
        a key-set mismatch against it raises).  Every episode is checked for:
        same key set, per-key frame shape (arr.shape[1:]), per-key dtype, and
        a single episode length T shared by all keys.  Returns T.
        """
        if not data:
            raise ValueError(f"Episode {episode_idx}: empty episode dict.")
        keys = frozenset(data)

        if self._contract_keys is None:
            # First episode: lock the key set (declared contract wins).
            if self._contract_shapes is not None:
                declared = frozenset(self._contract_shapes)
                if keys != declared:
                    missing = sorted(declared - keys)
                    extra = sorted(keys - declared)
                    raise ValueError(
                        f"Episode {episode_idx}: keys do not match the declared "
                        f"contract (missing: {missing}, extra: {extra})."
                    )
            shapes = {}
            dtypes = {}
            for key, arr in data.items():
                shapes[key] = (
                    self._contract_shapes[key]
                    if self._contract_shapes is not None
                    else tuple(arr.shape[1:])
                )
                dtypes[key] = (
                    self._contract_dtypes[key]
                    if self._contract_dtypes is not None and key in self._contract_dtypes
                    else np.dtype(arr.dtype)
                )
            self._contract_keys = keys
            self._contract_shapes = shapes
            self._contract_dtypes = dtypes
        elif keys != self._contract_keys:
            missing = sorted(self._contract_keys - keys)
            extra = sorted(keys - self._contract_keys)
            raise ValueError(
                f"Episode {episode_idx}: key set differs from the first episode "
                f"(missing: {missing}, extra: {extra})."
            )

        T: Optional[int] = None
        T_key: Optional[str] = None
        for key, arr in data.items():
            if tuple(arr.shape[1:]) != self._contract_shapes[key]:
                raise ValueError(
                    f"Episode {episode_idx}: key {key!r} has frame shape "
                    f"{tuple(arr.shape[1:])}, expected {self._contract_shapes[key]}."
                )
            if np.dtype(arr.dtype) != self._contract_dtypes[key]:
                raise ValueError(
                    f"Episode {episode_idx}: key {key!r} has dtype {arr.dtype}, "
                    f"expected {self._contract_dtypes[key]}."
                )
            if T is None:
                T, T_key = int(arr.shape[0]), key
            elif arr.shape[0] != T:
                raise ValueError(
                    f"Episode {episode_idx}: key {key!r} has length "
                    f"{arr.shape[0]}, but key {T_key!r} has length {T}."
                )
        return T

    @classmethod
    @abstractmethod
    def create(
        cls,
        path: str,
        n_episodes: int,
        keys: List[str],
        key_shapes: Dict[str, tuple],
        key_dtypes: Dict[str, np.dtype],
        **kwargs,
    ) -> "WritableEpisodeStore":
        """Create a new empty writable store at path."""

    @classmethod
    @abstractmethod
    def open_readonly(cls, path: str) -> EpisodeStore:
        """Open an existing store written by this class for reading."""

    @abstractmethod
    def put_episode(self, ep_idx: int, data: Dict[str, np.ndarray]) -> None:
        """Write episode ep_idx. Independent of all other episodes — parallelizable."""

    @abstractmethod
    def finalize(self) -> None:
        """Write meta.json / indices. Call after all put_episode() calls complete."""


# ── Backward-compat writable ABC ──────────────────────────────────────────────

class WritableTrajectoryStore(TrajectoryStore, ABC):
    """Store that accepts episode-by-episode writes (conversion target, disk cache).

    Episode contract
    ----------------
    Writers call :meth:`_init_contract` at construction (with the declared
    key_shapes/key_dtypes when the caller passed them, else None) and
    :meth:`_validate_episode` at the top of every ``put_episode``.  The first
    episode locks the key set; every episode is then checked for the same key
    set, per-key frame shape and dtype, and a single shared episode length T.

    Legacy interface: put_episode(data) with no ep_idx argument (sequential).
    """

    # ── Episode contract helpers ──────────────────────────────────────────────

    def _init_contract(
        self,
        key_shapes: Optional[Dict[str, tuple]] = None,
        key_dtypes: Optional[Dict[str, np.dtype]] = None,
    ) -> None:
        """Store the declared episode contract (values may be None)."""
        self._contract_shapes: Optional[Dict[str, tuple]] = (
            {k: tuple(v) for k, v in key_shapes.items()} if key_shapes else None
        )
        self._contract_dtypes: Optional[Dict[str, np.dtype]] = (
            {k: np.dtype(v) for k, v in key_dtypes.items()} if key_dtypes else None
        )
        self._contract_keys: Optional[frozenset] = None  # locked on first episode

    def _validate_episode(self, data: Dict[str, np.ndarray], episode_idx: int) -> int:
        """Validate one episode against the (possibly declared) contract.

        On the first episode the key set is locked (a declared contract wins;
        a key-set mismatch against it raises).  Every episode is checked for:
        same key set, per-key frame shape (arr.shape[1:]), per-key dtype, and
        a single episode length T shared by all keys.  Returns T.
        """
        if not data:
            raise ValueError(f"Episode {episode_idx}: empty episode dict.")
        keys = frozenset(data)

        if self._contract_keys is None:
            # First episode: lock the key set (declared contract wins).
            if self._contract_shapes is not None:
                declared = frozenset(self._contract_shapes)
                if keys != declared:
                    missing = sorted(declared - keys)
                    extra = sorted(keys - declared)
                    raise ValueError(
                        f"Episode {episode_idx}: keys do not match the declared "
                        f"contract (missing: {missing}, extra: {extra})."
                    )
            shapes = {}
            dtypes = {}
            for key, arr in data.items():
                shapes[key] = (
                    self._contract_shapes[key]
                    if self._contract_shapes is not None
                    else tuple(arr.shape[1:])
                )
                dtypes[key] = (
                    self._contract_dtypes[key]
                    if self._contract_dtypes is not None and key in self._contract_dtypes
                    else np.dtype(arr.dtype)
                )
            self._contract_keys = keys
            self._contract_shapes = shapes
            self._contract_dtypes = dtypes
        elif keys != self._contract_keys:
            missing = sorted(self._contract_keys - keys)
            extra = sorted(keys - self._contract_keys)
            raise ValueError(
                f"Episode {episode_idx}: key set differs from the first episode "
                f"(missing: {missing}, extra: {extra})."
            )

        T: Optional[int] = None
        T_key: Optional[str] = None
        for key, arr in data.items():
            if tuple(arr.shape[1:]) != self._contract_shapes[key]:
                raise ValueError(
                    f"Episode {episode_idx}: key {key!r} has frame shape "
                    f"{tuple(arr.shape[1:])}, expected {self._contract_shapes[key]}."
                )
            if np.dtype(arr.dtype) != self._contract_dtypes[key]:
                raise ValueError(
                    f"Episode {episode_idx}: key {key!r} has dtype {arr.dtype}, "
                    f"expected {self._contract_dtypes[key]}."
                )
            if T is None:
                T, T_key = int(arr.shape[0]), key
            elif arr.shape[0] != T:
                raise ValueError(
                    f"Episode {episode_idx}: key {key!r} has length "
                    f"{arr.shape[0]}, but key {T_key!r} has length {T}."
                )
        return T

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

    @classmethod
    @abstractmethod
    def open_readonly(cls, path: str) -> TrajectoryStore:
        """Open an existing store written by this class for reading."""

    @abstractmethod
    def put_episode(self, data: Dict[str, np.ndarray]) -> None:
        """Write one episode. data[key] must have shape (T, *frame_shape(key))."""

    @abstractmethod
    def finalize(self) -> None:
        """Flush buffers, build indices. Call after all put_episode()."""
