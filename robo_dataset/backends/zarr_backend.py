"""zarr DirectoryStore backend (read + write).

Layout on disk:
  <path>/data/<key>.zarr        one zarr array per key, (N, *frame_shape)
  <path>/meta/episode_ends.npy  int64 array

Image keys are stored with JPEG compression (numcodecs imagecodec or numcodecs Blosc+JPEG
via the imagecodecs_numcodecs codec if available, else standard Blosc+lz4 as fallback).
"""

import json
import os
from typing import Dict, List, Optional

import numpy as np
import zarr
import numcodecs

from .base import TrajectoryStore, WritableTrajectoryStore

_JPEG_QUALITY = 75
_META_FILE = "meta.json"


def _jpeg_compressor():
    """Return best available JPEG compressor, falling back to lz4."""
    try:
        from diffusion_policy.codecs.imagecodecs_numcodecs import register_codecs, Jpeg2k
        register_codecs()
        return Jpeg2k(level=50)
    except Exception:
        pass
    try:
        import imagecodecs
        import numcodecs
        # Try the imagecodecs_jpeg codec registered by imagecodecs
        return numcodecs.registry.get_codec({"id": "imagecodecs_jpeg", "quality": _JPEG_QUALITY})
    except Exception:
        pass
    # Fallback: Blosc lz4 (lossless but fast)
    return numcodecs.Blosc(cname="lz4", clevel=1)


class ZarrStore(TrajectoryStore):
    """Read-only view of an existing zarr dataset on disk."""

    def __init__(self, path: str):
        self.path = path
        meta_path = os.path.join(path, _META_FILE)
        with open(meta_path) as f:
            meta = json.load(f)

        self._keys: List[str] = meta["keys"]
        self._shapes: Dict[str, tuple] = {k: tuple(v) for k, v in meta["shapes"].items()}
        self._dtypes: Dict[str, str] = meta["dtypes"]
        self._episode_ends = np.load(os.path.join(path, "episode_ends.npy"))

        self._arrays: Optional[Dict[str, zarr.Array]] = None
        self._open_arrays()

    def _open_arrays(self):
        self._arrays = {}
        for key in self._keys:
            arr_path = os.path.join(self.path, "data", key)
            self._arrays[key] = zarr.open(arr_path, mode="r")

    def on_worker_init(self, worker_id: int) -> None:
        self._open_arrays()

    @property
    def episode_ends(self) -> np.ndarray:
        return self._episode_ends

    @property
    def keys(self) -> List[str]:
        return self._keys

    def frame_shape(self, key: str) -> tuple:
        return self._shapes[key]

    def frame_dtype(self, key: str) -> np.dtype:
        return np.dtype(self._dtypes[key])

    def get_sequence(self, key: str, start: int, end: int) -> np.ndarray:
        return self._arrays[key][start:end]


class ZarrStoreWriter(WritableTrajectoryStore):
    """Write-once zarr store that accumulates episodes and finalizes."""

    def __init__(self, path: str, image_keys: List[str] = ()):
        self.path = path
        self._image_keys = set(image_keys)
        os.makedirs(os.path.join(path, "data"), exist_ok=True)

        self._episode_ends_list: List[int] = []
        self._arrays: Dict[str, zarr.Array] = {}
        self._shapes: Dict[str, tuple] = {}
        self._dtypes: Dict[str, str] = {}
        self._frame_count = 0
        self._finalized = False

    @classmethod
    def create(
        cls,
        path: str,
        key_shapes: Dict[str, tuple],
        key_dtypes: Dict[str, np.dtype],
        image_keys: List[str] = (),
        **kwargs,
    ) -> "ZarrStoreWriter":
        writer = cls(path=path, image_keys=image_keys)
        # Pre-register shapes/dtypes (arrays created lazily on first put_episode)
        writer._pre_shapes = {k: tuple(v) for k, v in key_shapes.items()}
        writer._pre_dtypes = {k: np.dtype(v) for k, v in key_dtypes.items()}
        return writer

    def _get_or_create_array(self, key: str, frame: np.ndarray) -> zarr.Array:
        if key in self._arrays:
            return self._arrays[key]

        arr_path = os.path.join(self.path, "data", key)
        shape = (0,) + frame.shape
        chunks = (1,) + frame.shape

        compressor = _jpeg_compressor() if key in self._image_keys else numcodecs.Blosc(cname="lz4", clevel=1)

        arr = zarr.open(
            arr_path,
            mode="w",
            shape=shape,
            chunks=chunks,
            dtype=frame.dtype,
            compressor=compressor,
        )
        self._arrays[key] = arr
        self._shapes[key] = frame.shape
        self._dtypes[key] = str(frame.dtype)
        return arr

    def put_episode(self, data: Dict[str, np.ndarray]) -> None:
        assert not self._finalized, "Cannot write after finalize()"
        T = None
        for key, arr in data.items():
            if T is None:
                T = arr.shape[0]
            else:
                assert arr.shape[0] == T, f"Episode length mismatch on key {key}"
            zarr_arr = self._get_or_create_array(key, arr[0])
            zarr_arr.append(arr, axis=0)

        self._frame_count += T
        self._episode_ends_list.append(self._frame_count)

    def finalize(self) -> None:
        ep_ends = np.array(self._episode_ends_list, dtype=np.int64)
        np.save(os.path.join(self.path, "episode_ends.npy"), ep_ends)
        meta = {
            "keys": list(self._arrays.keys()),
            "shapes": {k: list(v) for k, v in self._shapes.items()},
            "dtypes": self._dtypes,
            "n_episodes": len(self._episode_ends_list),
            "n_frames": self._frame_count,
        }
        with open(os.path.join(self.path, _META_FILE), "w") as f:
            json.dump(meta, f, indent=2)
        self._finalized = True
        self._episode_ends = ep_ends

    # ── Read interface (available after finalize) ─────────────────────────────

    @property
    def episode_ends(self) -> np.ndarray:
        if not self._finalized:
            return np.array(self._episode_ends_list, dtype=np.int64)
        return self._episode_ends

    @property
    def keys(self) -> List[str]:
        return list(self._arrays.keys())

    def frame_shape(self, key: str) -> tuple:
        return self._shapes[key]

    def frame_dtype(self, key: str) -> np.dtype:
        return np.dtype(self._dtypes[key])

    def get_sequence(self, key: str, start: int, end: int) -> np.ndarray:
        return self._arrays[key][start:end]
