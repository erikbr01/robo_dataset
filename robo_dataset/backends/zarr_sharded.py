"""zarr-v3 ShardingCodec backend (read + write).

Layout on disk (identical to ZarrStore for tooling compatibility):
  <path>/data/<key>/             one zarr array per key
  <path>/episode_ends.npy        int64 episode boundary array
  <path>/meta.json               key shapes, dtypes, chunk_frames, shard_frames

Sharding collapses many inner chunks into one shard file, eliminating
per-frame open() syscall overhead that limits directory zarr at high
worker counts.  Each shard file holds shard_frames frames; each inner
chunk (the decompression unit) holds chunk_frames frames.

Vectorized advantage:  arr[start:end] decompresses only the inner chunks
that overlap [start, end].  The speedup over RLDS is largest when
horizon >= chunk_frames (all needed frames are in one decompression unit).

Fork-safe:  arrays are reopened in on_worker_init (same pattern as ZarrStore).
Requires zarr >= 3.0 (ShardingCodec is a v3-only feature).
"""

import json
import os
from typing import Dict, List, Optional

import numpy as np
import zarr
from zarr.codecs import ShardingCodec, BytesCodec, BloscCodec

from .base import TrajectoryStore, WritableTrajectoryStore, validate_episode_ends

_META_FILE = "meta.json"
_DEFAULT_CHUNK_FRAMES = 8
_DEFAULT_SHARD_FRAMES = 1024


_jpeg_zarr_codec_cls = None


def _get_jpeg_zarr_codec_cls():
    """Lazily build and register the zarr v3 JPEG ArrayBytesCodec class.

    Registers both the numcodecs side (imagecodecs_numcodecs.register_codecs)
    and the zarr registry side (zarr.registry.register_codec) so that arrays
    written with this codec can be reopened in the same or a fresh process
    (as long as this function is called first).
    """
    global _jpeg_zarr_codec_cls
    if _jpeg_zarr_codec_cls is not None:
        return _jpeg_zarr_codec_cls

    from zarr.codecs.numcodecs._codecs import _NumcodecsArrayBytesCodec, CODEC_PREFIX
    from zarr.registry import register_codec
    from diffusion_policy.codecs.imagecodecs_numcodecs import register_codecs
    register_codecs()

    codec_name = f"{CODEC_PREFIX}imagecodecs_jpeg"
    cls = type(
        "JpegZarrCodec",
        (_NumcodecsArrayBytesCodec,),
        {"codec_name": codec_name},
    )
    # Register under the codec's own name so zarr can look it up by name when
    # reading back stored arrays (get_codec_class('numcodecs.imagecodecs_jpeg')).
    register_codec(codec_name, cls)
    _jpeg_zarr_codec_cls = cls
    return cls


def _make_jpeg_zarr_codec(level: int = 75):
    """Return a zarr v3 ArrayBytesCodec wrapping imagecodecs JPEG.

    Requires imagecodecs and diffusion_policy to be importable.
    chunk_shape must have chunk_frames=1 so each inner chunk is a single
    (H, W, C) image that JPEG can encode.
    """
    return _get_jpeg_zarr_codec_cls()(level=level)


def _open_sharded_array(
    path: str,
    mode: str,
    frame_shape: tuple,
    dtype,
    chunk_frames: int,
    shard_frames: int,
    jpeg: bool = False,
    jpeg_level: int = 75,
) -> zarr.Array:
    inner = (chunk_frames,) + frame_shape
    outer = (shard_frames,) + frame_shape
    if jpeg:
        if chunk_frames != 1:
            raise ValueError(
                f"JPEG codec requires chunk_frames=1 (got {chunk_frames}); "
                "JPEG encodes one image at a time."
            )
        inner_codecs = [_make_jpeg_zarr_codec(level=jpeg_level)]
    else:
        inner_codecs = [BytesCodec(), BloscCodec(cname="lz4", clevel=1)]
    return zarr.open_array(
        path,
        mode=mode,
        shape=(0,) + frame_shape,
        chunks=outer,
        dtype=dtype,
        zarr_format=3,
        codecs=[
            ShardingCodec(
                chunk_shape=inner,
                codecs=inner_codecs,
            )
        ],
    )


# ── Read-only store ───────────────────────────────────────────────────────────

class ZarrShardedStore(TrajectoryStore):
    """Read-only view of a sharded zarr-v3 dataset.

    Each key's array uses ShardingCodec: shard_frames frames per file,
    chunk_frames frames per inner decompression unit.  Slicing [start:end]
    decompresses only the inner chunks that overlap the range — one numpy
    call regardless of how many frames are requested.

    Fork-safe: arrays reopened in on_worker_init.
    Requires zarr >= 3.0.
    """

    def __init__(self, path: str):
        self.path = path
        with open(os.path.join(path, _META_FILE)) as f:
            meta = json.load(f)

        self._keys: List[str] = meta["keys"]
        self._shapes: Dict[str, tuple] = {k: tuple(v) for k, v in meta["shapes"].items()}
        self._dtypes: Dict[str, str] = meta["dtypes"]
        self._chunk_frames: int = meta.get("chunk_frames", _DEFAULT_CHUNK_FRAMES)
        self._shard_frames: int = meta.get("shard_frames", _DEFAULT_SHARD_FRAMES)
        self._has_jpeg: bool = bool(meta.get("image_keys"))
        from ..schema import schemas_from_meta
        self._schemas = schemas_from_meta(meta)
        self._episode_ends = np.load(os.path.join(path, "episode_ends.npy"))
        validate_episode_ends(self._episode_ends, meta.get("n_frames"), path=path)
        self._arrays: Optional[Dict[str, zarr.Array]] = None
        self._open_arrays()

    @property
    def schemas(self):
        return self._schemas

    def _open_arrays(self) -> None:
        if self._has_jpeg:
            _get_jpeg_zarr_codec_cls()  # ensure codec registered before zarr opens the array
        self._arrays = {}
        for key in self._keys:
            self._arrays[key] = zarr.open_array(
                os.path.join(self.path, "data", key), mode="r"
            )

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

class ZarrShardedStoreWriter(WritableTrajectoryStore):
    """Write-once sharded zarr-v3 store.

    Creates arrays with ShardingCodec so that shard_frames frames are bundled
    into a single file, eliminating the per-frame open() overhead that limits
    directory zarr at high worker counts.

    Parameters
    ----------
    path:
        Output directory.  Will be created if absent.
    chunk_frames:
        Inner decompression unit (frames).  Smaller → finer random access;
        larger → better compression and fewer decompression calls per slice.
        Default 8 works well for most robotics use-cases (horizon ≤ 8).
    shard_frames:
        Outer shard file size (frames).  Must be a multiple of chunk_frames.
        Default 1024 → ~10 shard files for a 9k-frame dataset vs 9k+ for
        directory zarr.
    """

    def __init__(
        self,
        path: str,
        chunk_frames: int = _DEFAULT_CHUNK_FRAMES,
        shard_frames: int = _DEFAULT_SHARD_FRAMES,
        image_keys: List[str] = (),
        jpeg_level: int = 75,
        schemas=None,
    ):
        if shard_frames % chunk_frames != 0:
            raise ValueError(
                f"shard_frames ({shard_frames}) must be a multiple of chunk_frames ({chunk_frames})"
            )
        self.path = path
        self._chunk_frames = chunk_frames
        self._shard_frames = shard_frames
        self._image_keys = set(image_keys)
        self._jpeg_level = jpeg_level
        self._schemas = dict(schemas or {})
        os.makedirs(os.path.join(path, "data"), exist_ok=True)

        self._episode_ends_list: List[int] = []
        self._arrays: Dict[str, zarr.Array] = {}
        self._shapes: Dict[str, tuple] = {}
        self._dtypes: Dict[str, str] = {}
        self._frame_count = 0
        self._finalized = False
        self._init_contract()

    @classmethod
    def create(
        cls,
        path: str,
        key_shapes: Dict[str, tuple],
        key_dtypes: Dict[str, np.dtype],
        chunk_frames: int = _DEFAULT_CHUNK_FRAMES,
        shard_frames: int = _DEFAULT_SHARD_FRAMES,
        image_keys: List[str] = (),
        jpeg_level: int = 75,
        schemas=None,
        **kwargs,
    ) -> "ZarrShardedStoreWriter":
        writer = cls(
            path=path,
            chunk_frames=chunk_frames,
            shard_frames=shard_frames,
            image_keys=image_keys,
            jpeg_level=jpeg_level,
            schemas=schemas,
        )
        writer._init_contract(key_shapes, key_dtypes)
        return writer

    @classmethod
    def open_readonly(cls, path: str) -> ZarrShardedStore:
        return ZarrShardedStore(path)

    def _get_or_create_array(self, key: str, frame: np.ndarray) -> zarr.Array:
        if key in self._arrays:
            return self._arrays[key]

        use_jpeg = key in self._image_keys
        arr = _open_sharded_array(
            path=os.path.join(self.path, "data", key),
            mode="w",
            frame_shape=frame.shape,
            dtype=frame.dtype,
            chunk_frames=1 if use_jpeg else self._chunk_frames,
            shard_frames=self._shard_frames,
            jpeg=use_jpeg,
            jpeg_level=self._jpeg_level,
        )
        self._arrays[key] = arr
        self._shapes[key] = tuple(frame.shape)
        self._dtypes[key] = str(frame.dtype)
        return arr

    def put_episode(self, data: Dict[str, np.ndarray]) -> None:
        assert not self._finalized, "Cannot write after finalize()"
        T = self._validate_episode(data, episode_idx=len(self._episode_ends_list))
        for key, arr in data.items():
            zarr_arr = self._get_or_create_array(key, arr[0])
            old = zarr_arr.shape[0]
            zarr_arr.resize((old + T,) + tuple(zarr_arr.shape[1:]))
            zarr_arr[old : old + T] = arr
        self._frame_count += T
        self._episode_ends_list.append(self._frame_count)

    def finalize(self) -> None:
        from ..schema import schemas_to_meta, validate_schemas_against_shapes
        validate_schemas_against_shapes(self._schemas, self._shapes)
        for key, arr in self._arrays.items():
            assert arr.shape[0] == self._frame_count, (
                f"Key {key!r}: stored {arr.shape[0]} frames, expected "
                f"{self._frame_count} — store is inconsistent."
            )
        ep_ends = np.array(self._episode_ends_list, dtype=np.int64)
        np.save(os.path.join(self.path, "episode_ends.npy"), ep_ends)
        meta = {
            "keys": list(self._arrays.keys()),
            "shapes": {k: list(v) for k, v in self._shapes.items()},
            "dtypes": self._dtypes,
            "n_episodes": len(self._episode_ends_list),
            "n_frames": self._frame_count,
            "chunk_frames": self._chunk_frames,
            "shard_frames": self._shard_frames,
            "image_keys": sorted(self._image_keys),
            "jpeg_level": self._jpeg_level,
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
