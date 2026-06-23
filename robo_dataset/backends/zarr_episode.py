"""Per-episode zarr backend (read + write).

Disk layout:
  <path>/
    meta.json            {"keys": [...], "shapes": {...}, "dtypes": {...}, "n_episodes": N}
    000000/
      action             zarr array (T, D)  lz4 compressed
      joint_positions    zarr array (T, 7)
      agentview_image    zarr array (T, H, W, 3) uint8
    000001/
      ...

Each episode is stored as an independent group (directory).  episode_ends is
derived from the shapes of stored arrays — no episode_ends.npy on disk.

Thread-safety: put_episode(ep_idx, data) writes to a separate directory per
episode, so concurrent calls for different ep_idx values are safe after the
one-time contract initialisation (guarded by _contract_lock).

Requires zarr >= 2 and numcodecs.
"""

from __future__ import annotations

import json
import os
import threading
from typing import Dict, List, Optional

import numpy as np
import numcodecs
import zarr

from .base import EpisodeStore, WritableEpisodeStore

_META_FILE = "meta.json"
_ZARR_MAJOR = int(zarr.__version__.split(".")[0])
_LZ4_COMPRESSOR = numcodecs.Blosc(cname="lz4", clevel=1)
_NON_IMAGE_CHUNK_FRAMES = 512


def _open_zarr_array(path: str, mode: str, **kwargs) -> zarr.Array:
    """Open/create zarr array, forcing zarr_format=2 in zarr v3 for compressor support."""
    if _ZARR_MAJOR >= 3 and mode in ("w", "w-", "a"):
        kwargs.setdefault("zarr_format", 2)
    return zarr.open(path, mode=mode, **kwargs)


def _ep_dir(store_path: str, ep_idx: int) -> str:
    return os.path.join(store_path, f"{ep_idx:06d}")


# ── Write-once store ──────────────────────────────────────────────────────────

class ZarrEpisodeStoreWriter(WritableEpisodeStore):
    """Write-once per-episode zarr store.

    put_episode(ep_idx, data) is thread-safe for distinct ep_idx values —
    each episode is written to its own directory.  The one-time contract
    initialisation is guarded by _contract_lock.

    Call finalize() once after all put_episode() calls to write meta.json.
    """

    def __init__(
        self,
        path: str,
        n_episodes: int,
        keys: List[str],
        key_shapes: Dict[str, tuple],
        key_dtypes: Dict[str, np.dtype],
        schemas=None,
    ):
        self.path = path
        self._n_episodes = n_episodes
        self._keys = list(keys)
        self._schemas = dict(schemas or {})
        # Shapes / dtypes used for writing
        self._shapes: Dict[str, tuple] = {k: tuple(v) for k, v in key_shapes.items()}
        self._dtypes: Dict[str, str] = {k: str(np.dtype(v)) for k, v in key_dtypes.items()}
        os.makedirs(path, exist_ok=True)

        # Guards the one-time contract mutation in _validate_episode
        self._contract_lock = threading.Lock()
        self._finalized = False

        # Initialise the contract checker with declared shapes/dtypes
        self._init_contract(key_shapes, key_dtypes)

    @classmethod
    def create(
        cls,
        path: str,
        n_episodes: int,
        keys: List[str],
        key_shapes: Dict[str, tuple],
        key_dtypes: Dict[str, np.dtype],
        schemas=None,
        **kwargs,
    ) -> "ZarrEpisodeStoreWriter":
        return cls(
            path=path,
            n_episodes=n_episodes,
            keys=keys,
            key_shapes=key_shapes,
            key_dtypes=key_dtypes,
            schemas=schemas,
        )

    @classmethod
    def open_readonly(cls, path: str) -> "ZarrEpisodeStore":
        return ZarrEpisodeStore(path)

    # ── EpisodeStore read interface ────────────────────────────────────────────

    @property
    def n_episodes(self) -> int:
        return self._n_episodes

    @property
    def keys(self) -> List[str]:
        return self._keys

    def episode_length(self, ep_idx: int) -> int:
        arr_path = os.path.join(self.path, f"{ep_idx:06d}", self._keys[0])
        arr = zarr.open(arr_path, mode="r")
        return int(arr.shape[0])

    def frame_shape(self, key: str) -> tuple:
        return self._shapes[key]

    def frame_dtype(self, key: str) -> np.dtype:
        return np.dtype(self._dtypes[key])

    def get_frames(self, ep_idx: int, key: str, start: int, end: int) -> np.ndarray:
        arr_path = os.path.join(self.path, f"{ep_idx:06d}", key)
        arr = zarr.open(arr_path, mode="r")
        return arr[start:end]

    # ── Write interface ────────────────────────────────────────────────────────

    def put_episode(self, ep_idx: int, data: Dict[str, np.ndarray]) -> None:
        """Write episode ep_idx.  Thread-safe for distinct ep_idx values."""
        assert not self._finalized, "Cannot write after finalize()"
        # _validate_episode mutates _contract_keys on the very first call;
        # guard that with a lock.  Once the contract is locked the body of
        # _validate_episode only reads, so subsequent calls don't contend.
        with self._contract_lock:
            T = self._validate_episode(data, episode_idx=ep_idx)

        ep_dir = _ep_dir(self.path, ep_idx)
        os.makedirs(ep_dir, exist_ok=True)

        for key, arr in data.items():
            arr_path = os.path.join(ep_dir, key)
            frame_shape = tuple(arr.shape[1:])
            chunk_t = min(T, _NON_IMAGE_CHUNK_FRAMES)
            chunks = (chunk_t,) + frame_shape

            zarr_arr = _open_zarr_array(
                arr_path,
                mode="w",
                shape=arr.shape,
                chunks=chunks,
                dtype=arr.dtype,
                compressor=_LZ4_COMPRESSOR,
            )
            zarr_arr[:] = arr

    def finalize(self) -> None:
        """Write meta.json.  Call after all put_episode() calls complete."""
        from ..schema import schemas_to_meta, validate_schemas_against_shapes

        validate_schemas_against_shapes(self._schemas, self._shapes)

        meta = {
            "keys": self._keys,
            "shapes": {k: list(v) for k, v in self._shapes.items()},
            "dtypes": self._dtypes,
            "n_episodes": self._n_episodes,
            "schemas": schemas_to_meta(self._schemas),
        }
        with open(os.path.join(self.path, _META_FILE), "w") as f:
            json.dump(meta, f, indent=2)
        self._finalized = True


# ── Read-only store ───────────────────────────────────────────────────────────

class ZarrEpisodeStore(EpisodeStore):
    """Read-only view of a per-episode zarr dataset.

    Each episode occupies its own subdirectory (six-digit zero-padded).
    episode_ends is derived from stored array shapes — not from disk.

    Fork-safe: zarr.open() is called per get_frames(); no persistent handles.
    """

    def __init__(self, path: str):
        self.path = path
        with open(os.path.join(path, _META_FILE)) as f:
            meta = json.load(f)

        self._keys: List[str] = meta["keys"]
        self._shapes: Dict[str, tuple] = {k: tuple(v) for k, v in meta["shapes"].items()}
        self._dtypes: Dict[str, str] = meta["dtypes"]
        self._n_episodes: int = int(meta["n_episodes"])
        from ..schema import schemas_from_meta
        self._schemas = schemas_from_meta(meta)

        # Cache episode lengths (read once per open — cheap, from zarr array .shape)
        self._lengths: Optional[List[int]] = None
        self._load_lengths()

    def _load_lengths(self) -> None:
        """Read episode lengths from the first key's array shape for each episode."""
        lengths = []
        first_key = self._keys[0]
        for ep_idx in range(self._n_episodes):
            arr_path = os.path.join(self.path, f"{ep_idx:06d}", first_key)
            arr = zarr.open(arr_path, mode="r")
            lengths.append(int(arr.shape[0]))
        self._lengths = lengths

    @property
    def schemas(self):
        return self._schemas

    # ── EpisodeStore interface ────────────────────────────────────────────────

    @property
    def n_episodes(self) -> int:
        return self._n_episodes

    @property
    def keys(self) -> List[str]:
        return self._keys

    def episode_length(self, ep_idx: int) -> int:
        return self._lengths[ep_idx]

    def frame_shape(self, key: str) -> tuple:
        return self._shapes[key]

    def frame_dtype(self, key: str) -> np.dtype:
        return np.dtype(self._dtypes[key])

    def get_frames(self, ep_idx: int, key: str, start: int, end: int) -> np.ndarray:
        arr_path = os.path.join(self.path, f"{ep_idx:06d}", key)
        arr = zarr.open(arr_path, mode="r")
        return arr[start:end]

    def on_worker_init(self, worker_id: int) -> None:
        # zarr.open() is called per get_frames() — no persistent handles to reopen.
        # Re-load lengths to ensure fresh state after fork.
        self._load_lengths()

    @classmethod
    def create(
        cls,
        path: str,
        n_episodes: int,
        keys: List[str],
        key_shapes: Dict[str, tuple],
        key_dtypes: Dict[str, np.dtype],
        schemas=None,
        **kwargs,
    ) -> ZarrEpisodeStoreWriter:
        """Create a new writable episode store.  Returns a ZarrEpisodeStoreWriter."""
        return ZarrEpisodeStoreWriter(
            path=path,
            n_episodes=n_episodes,
            keys=keys,
            key_shapes=key_shapes,
            key_dtypes=key_dtypes,
            schemas=schemas,
        )
