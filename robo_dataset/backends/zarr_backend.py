"""zarr DirectoryStore backend (read + write).

Layout on disk:
  <path>/data/<key>/             one zarr array per key
  <path>/episode_ends.npy        int64 episode boundary array
  <path>/meta.json               key shapes, dtypes, etc.

Compatible with zarr v2 and v3 (uses zarr_format=2 to keep numcodecs compressors working).
"""

import json
import os
from typing import Dict, List, Optional

import numpy as np
import zarr
import numcodecs

from .base import TrajectoryStore, WritableTrajectoryStore

_ZARR_MAJOR = int(zarr.__version__.split(".")[0])
_JPEG_QUALITY = 75
_META_FILE = "meta.json"


def _open_zarr_array(path: str, mode: str, **kwargs) -> zarr.Array:
    """Open/create zarr array, forcing zarr_format=2 in zarr v3 for compressor support."""
    if _ZARR_MAJOR >= 3 and mode == "w":
        kwargs.setdefault("zarr_format", 2)
    return zarr.open(path, mode=mode, **kwargs)


def _zarr_append(arr: zarr.Array, data: np.ndarray) -> None:
    """Append data along axis 0; works with zarr v2 and v3."""
    T = data.shape[0]
    old = arr.shape[0]
    new_shape = (old + T,) + tuple(arr.shape[1:])
    try:
        arr.resize(new_shape)
    except TypeError:
        arr.resize(old + T, *arr.shape[1:])
    arr[old : old + T] = data


def _jpeg_compressor() -> numcodecs.abc.Codec:
    """Return best available JPEG compressor, falling back to lz4."""
    try:
        from diffusion_policy.codecs.imagecodecs_numcodecs import register_codecs, Jpeg2k
        register_codecs()
        return Jpeg2k(level=50)
    except Exception:
        pass
    try:
        return numcodecs.registry.get_codec({"id": "imagecodecs_jpeg", "quality": _JPEG_QUALITY})
    except Exception:
        pass
    return numcodecs.Blosc(cname="lz4", clevel=1)


# ── Read-only store ───────────────────────────────────────────────────────────

class ZarrStore(TrajectoryStore):
    """Read-only view of an existing zarr dataset on disk.

    Pros
    ----
    - Vectorized slice: array[start:end] is a single decompression call for a
      contiguous chunk, avoiding per-frame syscall overhead for large horizons.
    - Flexible compression: lz4, zstd, JPEG, J2K, WebP via numcodecs (set at
      write time; read side is transparent).
    - Fork-safe: arrays are reopened in on_worker_init via fresh zarr.open calls.
    - Standard open format supported by many tools beyond this library.

    Cons
    ----
    - One file per chunk (default: chunks=(1, H, W, C)) → one open() syscall per
      frame per key. At 12 workers × 9666 frames × 2 keys ≈ 232k file opens per
      epoch — inode/VFS overhead dominates at high worker counts (see benchmarks).
    - On-disk directory tree scales linearly with frame count; stat() calls add up.
    - zarr v2/v3 API incompatibility: zarr_format=2 must be forced in v3 to keep
      numcodecs compressors working (see _open_zarr_array helper).
    """

    def __init__(self, path: str):
        self.path = path
        with open(os.path.join(path, _META_FILE)) as f:
            meta = json.load(f)

        self._keys: List[str] = meta["keys"]
        self._shapes: Dict[str, tuple] = {k: tuple(v) for k, v in meta["shapes"].items()}
        self._dtypes: Dict[str, str] = meta["dtypes"]
        from ..schema import schemas_from_meta
        self._schemas = schemas_from_meta(meta)
        self._episode_ends = np.load(os.path.join(path, "episode_ends.npy"))
        self._arrays: Optional[Dict[str, zarr.Array]] = None
        self._open_arrays()

    @property
    def schemas(self):
        return self._schemas

    def _open_arrays(self) -> None:
        self._arrays = {}
        for key in self._keys:
            self._arrays[key] = zarr.open(os.path.join(self.path, "data", key), mode="r")

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


# ── Write-once store ──────────────────────────────────────────────────────────

class ZarrStoreWriter(WritableTrajectoryStore):
    """Write-once zarr store that accumulates episodes and finalizes.

    Implements WritableTrajectoryStore so it can be used as a conversion target
    and as a disk-cache backend in apply_disk_cached().
    Same pros/cons as ZarrStore apply after finalize().
    """

    def __init__(self, path: str, image_keys: List[str] = (), schemas=None):
        self.path = path
        self._image_keys = set(image_keys)
        self._schemas = dict(schemas or {})
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
        schemas=None,
        **kwargs,
    ) -> "ZarrStoreWriter":
        return cls(path=path, image_keys=image_keys, schemas=schemas)

    @classmethod
    def open_readonly(cls, path: str) -> ZarrStore:
        return ZarrStore(path)

    def _get_or_create_array(self, key: str, frame: np.ndarray) -> zarr.Array:
        if key in self._arrays:
            return self._arrays[key]

        arr_path = os.path.join(self.path, "data", key)
        compressor = _jpeg_compressor() if key in self._image_keys else numcodecs.Blosc(cname="lz4", clevel=1)
        arr = _open_zarr_array(
            arr_path,
            mode="w",
            shape=(0,) + frame.shape,
            chunks=(1,) + frame.shape,
            dtype=frame.dtype,
            compressor=compressor,
        )
        self._arrays[key] = arr
        self._shapes[key] = tuple(frame.shape)
        self._dtypes[key] = str(frame.dtype)
        return arr

    def put_episode(self, data: Dict[str, np.ndarray]) -> None:
        assert not self._finalized, "Cannot write after finalize()"
        T = None
        for key, arr in data.items():
            if T is None:
                T = arr.shape[0]
            else:
                assert arr.shape[0] == T, f"Episode length mismatch on key {key!r}"
            _zarr_append(self._get_or_create_array(key, arr[0]), arr)
        self._frame_count += T
        self._episode_ends_list.append(self._frame_count)

    def finalize(self) -> None:
        from ..schema import schemas_to_meta, validate_schemas_against_shapes

        validate_schemas_against_shapes(self._schemas, self._shapes)
        ep_ends = np.array(self._episode_ends_list, dtype=np.int64)
        np.save(os.path.join(self.path, "episode_ends.npy"), ep_ends)
        meta = {
            "keys": list(self._arrays.keys()),
            "shapes": {k: list(v) for k, v in self._shapes.items()},
            "dtypes": self._dtypes,
            "n_episodes": len(self._episode_ends_list),
            "n_frames": self._frame_count,
            "schemas": schemas_to_meta(self._schemas),
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
