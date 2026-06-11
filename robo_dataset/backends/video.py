"""Per-episode MP4 video backend (read + write).

Layout on disk:
  <path>/videos/<key>/episode_XXXXXX.mp4  one MP4 per episode per video key
  <path>/data/<key>/                       zarr lz4 arrays for non-video keys
  <path>/episode_ends.npy                  int64 episode boundary array
  <path>/meta.json                         keys, shapes, dtypes, video_keys, codec params

Read path uses torchcodec VideoDecoder with seek_mode='approximate'.
Decoders are cached per (key, episode_idx) per worker with an LRU cap
to bound peak memory.  Call on_worker_init() in DataLoader workers to
clear stale post-fork state.

Write path uses PyAV (libx264 by default) for video keys and zarr lz4
for non-video keys.
"""

import fractions
import json
import logging
import os
from collections import OrderedDict
from typing import Dict, List, Optional

import numpy as np

from .base import (
    TrajectoryStore,
    WritableTrajectoryStore,
    iter_episode_slices,
    validate_episode_ends,
)
from .zarr_backend import _open_zarr_array, _zarr_append

_META_FILE = "meta.json"
_DEFAULT_VCODEC = "libx264"
_DEFAULT_CRF = 23
_DEFAULT_KFI = 1
_DEFAULT_FPS = 10
_DEFAULT_MAX_DECODERS = 32
_AUTO_MAX_DECODER_CACHE = 1024  # cap for the auto-sized per-worker decoder cache
_NON_VIDEO_CHUNK_FRAMES = 512  # frames per zarr chunk for action/lowdim keys


# ── Decoder compatibility layer ───────────────────────────────────────────────
#
# Three decode paths, picked once per process at first use:
#   "modern"  torchcodec >= 0.1   VideoDecoder.get_frames_at(indices=[...])
#   "legacy"  torchcodec 0.0.3    SimpleVideoDecoder.get_frames_at(start, stop)
#                                 (contiguous ranges only — which is all
#                                  VideoStore ever requests per episode)
#   "pyav"    no torchcodec       PyAV sequential decode (slowest; correct)

_DECODER_BACKEND: Optional[str] = None


def _pick_decoder_backend() -> str:
    global _DECODER_BACKEND
    if _DECODER_BACKEND is not None:
        return _DECODER_BACKEND
    try:
        from torchcodec.decoders import VideoDecoder  # noqa: F401
        _DECODER_BACKEND = "modern"
    except ImportError:
        try:
            from torchcodec.decoders import SimpleVideoDecoder  # noqa: F401
            _DECODER_BACKEND = "legacy"
        except ImportError:
            import av  # noqa: F401  (raise here if neither stack is present)
            _DECODER_BACKEND = "pyav"
    logging.getLogger(__name__).info("video decode backend: %s", _DECODER_BACKEND)
    return _DECODER_BACKEND


class _ModernDecoder:
    """torchcodec >= 0.1: VideoDecoder with arbitrary-index batch API."""

    def __init__(self, path: str):
        from torchcodec.decoders import VideoDecoder
        self._dec = VideoDecoder(path, device="cpu", seek_mode="approximate")

    def get_frames(self, indices: List[int]) -> np.ndarray:
        result = self._dec.get_frames_at(indices=indices)
        return result.data.permute(0, 2, 3, 1).numpy()


class _LegacyDecoder:
    """torchcodec 0.0.3: SimpleVideoDecoder with range-based batch API."""

    def __init__(self, path: str):
        from torchcodec.decoders import SimpleVideoDecoder
        self._dec = SimpleVideoDecoder(path)

    def get_frames(self, indices: List[int]) -> np.ndarray:
        contiguous = len(indices) <= 1 or all(
            b - a == 1 for a, b in zip(indices[:-1], indices[1:])
        )
        if contiguous:
            batch = self._dec.get_frames_at(indices[0], indices[-1] + 1)
            return batch.data.permute(0, 2, 3, 1).numpy()
        frames = [self._dec.get_frame_at(i).data for i in indices]
        import torch
        return torch.stack(frames).permute(0, 2, 3, 1).numpy()


class _PyAVDecoder:
    """PyAV fallback: sequential decode of the requested range.

    With kfi=1 every frame is a keyframe, so seeking is exact; with larger
    kfi PyAV seeks to the previous keyframe and decodes forward.
    """

    def __init__(self, path: str):
        import av
        self._av = av
        self._path = path
        self._container = av.open(path)
        self._stream = self._container.streams.video[0]

    def get_frames(self, indices: List[int]) -> np.ndarray:
        # Decode sequentially from the nearest keyframe at/before indices[0].
        want = set(indices)
        first, last = min(indices), max(indices)
        stream = self._stream
        # Seek to the frame's timestamp (backward=True lands on a keyframe ≤ target).
        ts_per_frame = int(round(1 / (stream.average_rate * stream.time_base)))
        if ts_per_frame < 1:
            raise ValueError(
                f"{self._path}: cannot map frame indices to timestamps — "
                f"ts_per_frame={ts_per_frame} (average_rate={stream.average_rate}, "
                f"time_base={stream.time_base})."
            )
        self._container.seek(first * ts_per_frame, stream=stream, backward=True)
        got = {}
        n_seen = 0
        for frame in self._container.decode(stream):
            if frame.pts is None:
                raise ValueError(
                    f"{self._path}: decoded frame has no pts — cannot map it to "
                    "a frame index."
                )
            n_seen += 1
            idx = int(round(frame.pts / ts_per_frame))
            if idx in want:
                got[idx] = frame.to_ndarray(format="rgb24")
            if idx >= last:
                break
        missing = [i for i in indices if i not in got]
        if missing:
            raise ValueError(
                f"{self._path}: wanted frame indices {sorted(want)} but indices "
                f"{missing} were never decoded ({n_seen} frames seen after seek)."
            )
        return np.stack([got[i] for i in indices])


_DECODER_CLS = {"modern": _ModernDecoder, "legacy": _LegacyDecoder, "pyav": _PyAVDecoder}


class _LRUDecoderCache:
    def __init__(self, maxsize: int = _DEFAULT_MAX_DECODERS):
        self._cache: OrderedDict = OrderedDict()
        self._maxsize = maxsize

    def get(self, key):
        if key not in self._cache:
            return None
        self._cache.move_to_end(key)
        return self._cache[key]

    def put(self, key, decoder) -> None:
        if key in self._cache:
            self._cache.move_to_end(key)
        else:
            if len(self._cache) >= self._maxsize:
                self._cache.popitem(last=False)
            self._cache[key] = decoder

    def clear(self) -> None:
        self._cache.clear()


# ── Read-only store ───────────────────────────────────────────────────────────

class VideoStore(TrajectoryStore):
    """Read-only view of a per-episode MP4 video dataset.

    Video keys are stored as one MP4 per episode per key; non-video keys
    are stored as zarr lz4 arrays.  Uses torchcodec VideoDecoder with
    seek_mode='approximate' and a bounded LRU decoder cache per worker.

    max_decoders sizes the per-worker LRU decoder cache. The default (None)
    auto-sizes to n_episodes × n_video_keys (capped at 1024) — under uniform
    random sampling a cache smaller than the working set thrashes: every
    sample re-opens a decoder, costing ~3× throughput on small-image
    datasets. Pass an explicit value to bound memory on huge datasets.

    Requires torchcodec and torch (for decoding) and zarr (for non-video keys).
    """

    def __init__(self, path: str, max_decoders: Optional[int] = None):
        self.path = path
        with open(os.path.join(path, _META_FILE)) as f:
            meta = json.load(f)

        self._keys: List[str] = meta["keys"]
        self._shapes: Dict[str, tuple] = {k: tuple(v) for k, v in meta["shapes"].items()}
        self._dtypes: Dict[str, str] = meta["dtypes"]
        self._video_keys: List[str] = meta.get("video_keys", [])
        self._fps: float = float(meta.get("fps", _DEFAULT_FPS))
        from ..schema import schemas_from_meta
        self._schemas = schemas_from_meta(meta)
        self._episode_ends = np.load(os.path.join(path, "episode_ends.npy"))
        validate_episode_ends(self._episode_ends, meta.get("n_frames"), path=path)

        if max_decoders is None:
            working_set = max(1, len(self._episode_ends) * max(1, len(self._video_keys)))
            max_decoders = min(working_set, _AUTO_MAX_DECODER_CACHE)
        self._max_decoders = max_decoders
        self._decoder_cache = _LRUDecoderCache(maxsize=max_decoders)
        self._cache_pid = os.getpid()

        self._non_video_keys = [k for k in self._keys if k not in self._video_keys]
        self._zarr_arrays: Optional[Dict[str, object]] = None
        if self._non_video_keys:
            self._open_zarr_arrays()

    def _open_zarr_arrays(self) -> None:
        import zarr
        self._zarr_arrays = {}
        for key in self._non_video_keys:
            self._zarr_arrays[key] = zarr.open(
                os.path.join(self.path, "data", key), mode="r"
            )

    def on_worker_init(self, worker_id: int) -> None:
        self._decoder_cache.clear()
        self._cache_pid = os.getpid()
        if self._non_video_keys:
            self._open_zarr_arrays()

    @property
    def episode_ends(self) -> np.ndarray:
        return self._episode_ends

    @property
    def keys(self) -> List[str]:
        return self._keys

    @property
    def schemas(self):
        return self._schemas

    def frame_shape(self, key: str) -> tuple:
        return self._shapes[key]

    def frame_dtype(self, key: str) -> np.dtype:
        return np.dtype(self._dtypes[key])

    def _get_decoder(self, key: str, ep_idx: int):
        # Self-defending fork safety: decoders hold codec/file state that must
        # not be shared across processes. If we were forked without
        # on_worker_init(), drop the inherited cache.
        if os.getpid() != self._cache_pid:
            self._decoder_cache.clear()
            self._cache_pid = os.getpid()
        cache_key = (key, ep_idx)
        dec = self._decoder_cache.get(cache_key)
        if dec is None:
            mp4_path = os.path.join(
                self.path, "videos", key, f"episode_{ep_idx:06d}.mp4"
            )
            dec = _DECODER_CLS[_pick_decoder_backend()](mp4_path)
            self._decoder_cache.put(cache_key, dec)
        return dec

    def _decode_frames(self, key: str, ep_idx: int, local_indices: List[int]) -> np.ndarray:
        # (N, H, W, C) uint8
        return self._get_decoder(key, ep_idx).get_frames(local_indices)

    def get_sequence(self, key: str, start: int, end: int) -> np.ndarray:
        if key not in self._video_keys:
            return self._zarr_arrays[key][start:end]

        parts = [
            self._decode_frames(key, ep_idx, list(range(lo, hi)))
            for ep_idx, lo, hi in iter_episode_slices(self._episode_ends, start, end)
        ]
        return np.concatenate(parts, axis=0) if len(parts) > 1 else parts[0]


# ── Write-once store ──────────────────────────────────────────────────────────

class VideoStoreWriter(WritableTrajectoryStore):
    """Write-once MP4 video store.

    Video keys are encoded as one MP4 per episode using PyAV.
    Non-video keys are written as zarr lz4 arrays.

    Parameters
    ----------
    path : str
        Output directory.
    video_keys : list of str
        Keys whose data will be encoded as video.
    vcodec : str
        libav codec name (default 'libx264').
    crf : int
        Constant rate factor (default 23; lower = higher quality).
    kfi : int
        Keyframe interval in frames (default 1 for random-access decode).
    fps : float
        Video frame rate (default 10).
    """

    def __init__(
        self,
        path: str,
        video_keys: List[str] = (),
        vcodec: str = _DEFAULT_VCODEC,
        crf: int = _DEFAULT_CRF,
        kfi: int = _DEFAULT_KFI,
        fps: float = _DEFAULT_FPS,
        schemas=None,
    ):
        self.path = path
        self._video_keys_set = set(video_keys)
        self._video_keys_ordered = list(video_keys)
        self._schemas = dict(schemas or {})
        self._vcodec = vcodec
        self._crf = crf
        self._kfi = kfi
        self._fps = fps

        os.makedirs(os.path.join(path, "data"), exist_ok=True)
        for vk in video_keys:
            os.makedirs(os.path.join(path, "videos", vk), exist_ok=True)

        self._episode_ends_list: List[int] = []
        self._zarr_arrays: Dict[str, object] = {}
        self._all_keys: List[str] = []
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
        image_keys: List[str] = (),
        video_keys: List[str] = (),
        vcodec: str = _DEFAULT_VCODEC,
        crf: int = _DEFAULT_CRF,
        kfi: int = _DEFAULT_KFI,
        fps: float = _DEFAULT_FPS,
        schemas=None,
        **kwargs,
    ) -> "VideoStoreWriter":
        # image_keys is treated as video_keys for this backend when video_keys not given
        if image_keys and video_keys and set(image_keys) != set(video_keys):
            raise ValueError(
                f"VideoStoreWriter.create: image_keys {sorted(image_keys)} and "
                f"video_keys {sorted(video_keys)} were both passed but differ — "
                "pass only one (they are synonyms for this backend)."
            )
        vk = list(video_keys) if video_keys else list(image_keys)
        writer = cls(path=path, video_keys=vk, vcodec=vcodec, crf=crf, kfi=kfi, fps=fps,
                     schemas=schemas)
        writer._init_contract(key_shapes, key_dtypes)
        return writer

    @classmethod
    def open_readonly(cls, path: str) -> VideoStore:
        return VideoStore(path)

    def _get_or_create_zarr_array(self, key: str, frame: np.ndarray):
        if key in self._zarr_arrays:
            return self._zarr_arrays[key]
        import numcodecs
        # Use large chunks for non-video (action/lowdim) data to avoid per-frame
        # file overhead that zarr directory format suffers from with chunks=(1,...).
        arr = _open_zarr_array(
            os.path.join(self.path, "data", key),
            mode="w",
            shape=(0,) + frame.shape,
            chunks=(_NON_VIDEO_CHUNK_FRAMES,) + frame.shape,
            dtype=frame.dtype,
            compressor=numcodecs.Blosc(cname="lz4", clevel=1),
        )
        self._zarr_arrays[key] = arr
        return arr

    def _encode_episode_video(self, key: str, ep_idx: int, frames: np.ndarray) -> None:
        import av as av_mod
        mp4_path = os.path.join(self.path, "videos", key, f"episode_{ep_idx:06d}.mp4")
        codec_opts = {"crf": str(self._crf), "g": str(self._kfi), "preset": "fast"}
        if "svt" in self._vcodec or "av1" in self._vcodec:
            codec_opts = {"crf": str(self._crf), "g": str(self._kfi)}
        T, H, W = frames.shape[:3]
        c = av_mod.open(mp4_path, "w")
        # Fraction preserves non-integer rates (e.g. 29.97) that int() would truncate.
        rate = fractions.Fraction(self._fps).limit_denominator(1000)
        s = c.add_stream(self._vcodec, rate=rate)
        s.width, s.height, s.pix_fmt = W, H, "yuv420p"
        s.options = codec_opts
        for arr in frames:
            for p in s.encode(av_mod.VideoFrame.from_ndarray(arr, format="rgb24")):
                c.mux(p)
        for p in s.encode():
            c.mux(p)
        c.close()

    def put_episode(self, data: Dict[str, np.ndarray]) -> None:
        assert not self._finalized, "Cannot write after finalize()"
        ep_idx = len(self._episode_ends_list)
        T = self._validate_episode(data, episode_idx=ep_idx)
        for key, arr in data.items():
            if key not in self._all_keys:
                self._all_keys.append(key)
                self._shapes[key] = tuple(arr.shape[1:])
                self._dtypes[key] = str(arr.dtype)
            if key in self._video_keys_set:
                self._encode_episode_video(key, ep_idx, arr)
            else:
                _zarr_append(self._get_or_create_zarr_array(key, arr[0]), arr)

        self._frame_count += T
        self._episode_ends_list.append(self._frame_count)

    def finalize(self) -> None:
        from ..schema import schemas_to_meta, validate_schemas_against_shapes

        validate_schemas_against_shapes(self._schemas, self._shapes)
        n_episodes = len(self._episode_ends_list)
        for key, arr in self._zarr_arrays.items():
            assert arr.shape[0] == self._frame_count, (
                f"Key {key!r}: stored {arr.shape[0]} frames, expected "
                f"{self._frame_count} — store is inconsistent."
            )
        for key in self._all_keys:
            if key not in self._video_keys_set:
                continue
            vid_dir = os.path.join(self.path, "videos", key)
            n_files = len([f for f in os.listdir(vid_dir) if f.endswith(".mp4")])
            assert n_files == n_episodes, (
                f"Video key {key!r}: {n_files} episode files in {vid_dir!r}, "
                f"expected {n_episodes} — store is inconsistent."
            )
        ep_ends = np.array(self._episode_ends_list, dtype=np.int64)
        np.save(os.path.join(self.path, "episode_ends.npy"), ep_ends)
        meta = {
            "keys": self._all_keys,
            "shapes": {k: list(v) for k, v in self._shapes.items()},
            "dtypes": self._dtypes,
            "n_episodes": len(self._episode_ends_list),
            "n_frames": self._frame_count,
            "video_keys": self._video_keys_ordered,
            "vcodec": self._vcodec,
            "crf": self._crf,
            "kfi": self._kfi,
            "fps": self._fps,
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
        return self._all_keys

    def frame_shape(self, key: str) -> tuple:
        return self._shapes[key]

    def frame_dtype(self, key: str) -> np.dtype:
        return np.dtype(self._dtypes[key])

    def get_sequence(self, key: str, start: int, end: int) -> np.ndarray:
        if key not in self._video_keys_set:
            return self._zarr_arrays[key][start:end]
        assert self._finalized, "Cannot read video keys before finalize()"
        if not hasattr(self, "_video_reader") or self._video_reader is None:
            self._video_reader = VideoStore(self.path)
        return self._video_reader.get_sequence(key, start, end)
