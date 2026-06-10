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

import json
import os
from collections import OrderedDict
from typing import Dict, List, Optional

import numpy as np

from .base import TrajectoryStore, WritableTrajectoryStore
from .zarr_backend import _open_zarr_array, _zarr_append

_META_FILE = "meta.json"
_DEFAULT_VCODEC = "libx264"
_DEFAULT_CRF = 23
_DEFAULT_KFI = 1
_DEFAULT_FPS = 10
_DEFAULT_MAX_DECODERS = 32
_NON_VIDEO_CHUNK_FRAMES = 512  # frames per zarr chunk for action/lowdim keys


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

    Requires torchcodec and torch (for decoding) and zarr (for non-video keys).
    """

    def __init__(self, path: str, max_decoders: int = _DEFAULT_MAX_DECODERS):
        self.path = path
        with open(os.path.join(path, _META_FILE)) as f:
            meta = json.load(f)

        self._keys: List[str] = meta["keys"]
        self._shapes: Dict[str, tuple] = {k: tuple(v) for k, v in meta["shapes"].items()}
        self._dtypes: Dict[str, str] = meta["dtypes"]
        self._video_keys: List[str] = meta.get("video_keys", [])
        self._fps: float = float(meta.get("fps", _DEFAULT_FPS))
        self._episode_ends = np.load(os.path.join(path, "episode_ends.npy"))

        self._max_decoders = max_decoders
        self._decoder_cache = _LRUDecoderCache(maxsize=max_decoders)

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
        if self._non_video_keys:
            self._open_zarr_arrays()

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

    def _get_decoder(self, key: str, ep_idx: int):
        from torchcodec.decoders import VideoDecoder
        cache_key = (key, ep_idx)
        dec = self._decoder_cache.get(cache_key)
        if dec is None:
            mp4_path = os.path.join(
                self.path, "videos", key, f"episode_{ep_idx:06d}.mp4"
            )
            dec = VideoDecoder(mp4_path, device="cpu", seek_mode="approximate")
            self._decoder_cache.put(cache_key, dec)
        return dec

    def _decode_frames(self, key: str, ep_idx: int, local_indices: List[int]) -> np.ndarray:
        dec = self._get_decoder(key, ep_idx)
        result = dec.get_frames_at(indices=local_indices)
        # result.data: (N, C, H, W) uint8 → (N, H, W, C) uint8
        return result.data.permute(0, 2, 3, 1).numpy()

    def get_sequence(self, key: str, start: int, end: int) -> np.ndarray:
        if key not in self._video_keys:
            return self._zarr_arrays[key][start:end]

        ep_start = int(np.searchsorted(self._episode_ends, start, side="right"))
        ep_last = int(np.searchsorted(self._episode_ends, end - 1, side="right"))

        if ep_start == ep_last:
            ep_offset = int(self._episode_ends[ep_start - 1]) if ep_start > 0 else 0
            return self._decode_frames(
                key, ep_start, list(range(start - ep_offset, end - ep_offset))
            )

        # Cross-episode: decode each episode's slice and concatenate
        parts = []
        for ep_idx in range(ep_start, ep_last + 1):
            ep_offset = int(self._episode_ends[ep_idx - 1]) if ep_idx > 0 else 0
            ep_end_global = int(self._episode_ends[ep_idx])
            local_start = max(start, ep_offset) - ep_offset
            local_end = min(end, ep_end_global) - ep_offset
            parts.append(
                self._decode_frames(key, ep_idx, list(range(local_start, local_end)))
            )
        return np.concatenate(parts, axis=0)


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
    ):
        self.path = path
        self._video_keys_set = set(video_keys)
        self._video_keys_ordered = list(video_keys)
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
        **kwargs,
    ) -> "VideoStoreWriter":
        # image_keys is treated as video_keys for this backend when video_keys not given
        vk = list(video_keys) if video_keys else list(image_keys)
        return cls(path=path, video_keys=vk, vcodec=vcodec, crf=crf, kfi=kfi, fps=fps)

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
        s = c.add_stream(self._vcodec, rate=int(self._fps))
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
        T = None
        for key, arr in data.items():
            if T is None:
                T = arr.shape[0]
            else:
                assert arr.shape[0] == T, f"Episode length mismatch on key {key!r}"

        ep_idx = len(self._episode_ends_list)
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
