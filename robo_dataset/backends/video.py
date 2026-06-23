"""Per-episode MP4 video backend (read + write).

Layout on disk (new format):
  <path>/videos/<key>/episode_XXXXXX.mp4   one MP4 per episode per video key
  <path>/data/<ep_idx:06d>/<key>           zarr lz4 arrays for non-video keys
  <path>/meta.json                         keys, shapes, dtypes, video_keys, codec params, n_episodes

Old format (episode_ends.npy present) is detected and read transparently for
backward compatibility.  New stores written by VideoStoreWriter use the new layout.

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
    EpisodeStore,
    WritableEpisodeStore,
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

    def n_frames(self) -> int:
        return len(self._dec)


class _LegacyDecoder:
    """torchcodec 0.0.3: SimpleVideoDecoder with range-based batch API.

    The decoder stream is added through the core API with num_threads=1:
    SimpleVideoDecoder's default lets ffmpeg auto-thread (up to nproc threads
    PER DECODER), which exhausts the process/thread limit once many workers
    each hold a large decoder cache (observed as
    "ValueError: Resource temporarily unavailable" at 16 workers × ~600
    cached decoders on a 32-core box). Single-frame robot-cam decodes gain
    nothing from intra-frame threading.
    """

    def __init__(self, path: str):
        from torchcodec.decoders import SimpleVideoDecoder
        from torchcodec.decoders import _core as core
        from torchcodec.decoders._simple_video_decoder import (
            _get_and_validate_stream_metadata,
        )

        # Replicates SimpleVideoDecoder.__init__(path, "NCHW") with the one
        # change the public API does not expose: num_threads=1.
        dec = SimpleVideoDecoder.__new__(SimpleVideoDecoder)
        dec._decoder = core.create_from_file(path)
        core.scan_all_streams_to_update_metadata(dec._decoder)
        core.add_video_stream(dec._decoder, dimension_order="NCHW", num_threads=1)
        dec.metadata, dec._stream_index = _get_and_validate_stream_metadata(dec._decoder)
        if dec.metadata.num_frames_from_content is None:
            raise ValueError(f"Unknown frame count in {path!r}")
        dec._num_frames = dec.metadata.num_frames_from_content
        if dec.metadata.begin_stream_seconds is None or dec.metadata.end_stream_seconds is None:
            raise ValueError(f"Unknown stream pts bounds in {path!r}")
        dec._begin_stream_seconds = dec.metadata.begin_stream_seconds
        dec._end_stream_seconds = dec.metadata.end_stream_seconds
        self._dec = dec

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

    def n_frames(self) -> int:
        return self._dec._num_frames


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

    def n_frames(self) -> int:
        return self._stream.frames


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

class VideoStore(EpisodeStore):
    """Read-only view of a per-episode MP4 video dataset.

    Video keys are stored as one MP4 per episode per key; non-video keys
    are stored as zarr lz4 arrays.  Uses torchcodec VideoDecoder with
    seek_mode='approximate' and a bounded LRU decoder cache per worker.

    Supports both the new layout (per-episode zarr dirs under data/<ep_idx:06d>/)
    and the old layout (flat zarr arrays under data/<key>/ with episode_ends.npy).

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

        # Detect old layout: episode_ends.npy exists → old format
        ep_ends_path = os.path.join(path, "episode_ends.npy")
        self._old_format = os.path.exists(ep_ends_path)

        if self._old_format:
            self._episode_ends = np.load(ep_ends_path)
            validate_episode_ends(self._episode_ends, meta.get("n_frames"), path=path)
            self._n_episodes = len(self._episode_ends)
        else:
            self._n_episodes = int(meta["n_episodes"])
            # lengths loaded below after decoder cache init
            self._lengths: Optional[List[int]] = None

        if max_decoders is None:
            working_set = max(1, self._n_episodes * max(1, len(self._video_keys)))
            max_decoders = min(working_set, _AUTO_MAX_DECODER_CACHE)
        self._max_decoders = max_decoders
        self._decoder_cache = _LRUDecoderCache(maxsize=max_decoders)
        self._cache_pid = os.getpid()

        self._non_video_keys = [k for k in self._keys if k not in self._video_keys]

        if self._old_format:
            self._zarr_arrays: Optional[Dict[str, object]] = None
            if self._non_video_keys:
                self._open_zarr_arrays_flat()
        else:
            # New format: load episode lengths from MP4 frame counts or zarr shapes
            self._load_lengths()

    def _open_zarr_arrays_flat(self) -> None:
        """Old format: flat zarr arrays under data/<key>/."""
        import zarr
        self._zarr_arrays = {}
        for key in self._non_video_keys:
            self._zarr_arrays[key] = zarr.open(
                os.path.join(self.path, "data", key), mode="r"
            )

    def _load_lengths(self) -> None:
        """New format: read episode lengths from first video key's frame count
        (or first non-video key's zarr array shape if no video keys)."""
        lengths = []
        if self._video_keys:
            first_vk = self._video_keys[0]
            for ep_idx in range(self._n_episodes):
                mp4_path = os.path.join(
                    self.path, "videos", first_vk, f"episode_{ep_idx:06d}.mp4"
                )
                dec = _DECODER_CLS[_pick_decoder_backend()](mp4_path)
                lengths.append(dec.n_frames())
        elif self._non_video_keys:
            import zarr
            first_nk = self._non_video_keys[0]
            for ep_idx in range(self._n_episodes):
                arr_path = os.path.join(self.path, "data", f"{ep_idx:06d}", first_nk)
                arr = zarr.open(arr_path, mode="r")
                lengths.append(int(arr.shape[0]))
        else:
            lengths = [0] * self._n_episodes
        self._lengths = lengths

    def on_worker_init(self, worker_id: int) -> None:
        self._decoder_cache.clear()
        self._cache_pid = os.getpid()
        if self._old_format:
            if self._non_video_keys:
                self._open_zarr_arrays_flat()
        else:
            self._load_lengths()

    # ── EpisodeStore interface ────────────────────────────────────────────────

    @property
    def n_episodes(self) -> int:
        return self._n_episodes

    @property
    def episode_ends(self) -> np.ndarray:
        """Derived from episode_length() — or from stored array for old format."""
        if self._old_format:
            return self._episode_ends
        # New format: derive from cached lengths
        ends = np.array(self._lengths, dtype=np.int64)
        return np.cumsum(ends)

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

    def episode_length(self, ep_idx: int) -> int:
        if self._old_format:
            return int(self._episode_ends[ep_idx]) - (
                0 if ep_idx == 0 else int(self._episode_ends[ep_idx - 1])
            )
        return self._lengths[ep_idx]

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

    def get_frames(self, ep_idx: int, key: str, start: int, end: int) -> np.ndarray:
        """Read frames [start, end) from episode ep_idx for a single key."""
        if key in self._video_keys:
            return self._decode_frames(key, ep_idx, list(range(start, end)))
        # Non-video key
        if self._old_format:
            # Old format: flat zarr array — translate to flat indices
            ep_start = 0 if ep_idx == 0 else int(self._episode_ends[ep_idx - 1])
            return self._zarr_arrays[key][ep_start + start : ep_start + end]
        else:
            # New format: per-episode zarr directory
            import zarr
            arr_path = os.path.join(self.path, "data", f"{ep_idx:06d}", key)
            arr = zarr.open(arr_path, mode="r")
            return arr[start:end]

    def get_sequence(self, key: str, start: int, end: int) -> np.ndarray:
        """Backward-compat flat-index access."""
        if self._old_format:
            if key not in self._video_keys:
                return self._zarr_arrays[key][start:end]
            parts = [
                self._decode_frames(key, ep_idx, list(range(lo, hi)))
                for ep_idx, lo, hi in iter_episode_slices(self._episode_ends, start, end)
            ]
            return np.concatenate(parts, axis=0) if len(parts) > 1 else parts[0]
        else:
            # New format: use derived episode_ends from EpisodeStore
            return super().get_sequence(key, start, end)


# ── Write-once store (new EpisodeStore interface) ─────────────────────────────

class VideoStoreWriter(WritableEpisodeStore):
    """Write-once MP4 video store with episode-indexed layout.

    Video keys are encoded as one MP4 per episode using PyAV.
    Non-video keys are written as per-episode zarr lz4 arrays under
    data/<ep_idx:06d>/<key>/.

    put_episode(ep_idx, data) is thread-safe for distinct ep_idx values.

    Parameters
    ----------
    path : str
        Output directory.
    n_episodes : int
        Total number of episodes (required for episode-indexed layout).
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
        n_episodes: int,
        video_keys: List[str] = (),
        vcodec: str = _DEFAULT_VCODEC,
        crf: int = _DEFAULT_CRF,
        kfi: int = _DEFAULT_KFI,
        fps: float = _DEFAULT_FPS,
        schemas=None,
    ):
        self.path = path
        self._n_episodes = n_episodes
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

        self._all_keys: List[str] = []
        self._shapes: Dict[str, tuple] = {}
        self._dtypes: Dict[str, str] = {}
        self._finalized = False
        self._next_ep_idx: int = 0  # for legacy sequential put_episode(data)

        # Thread safety: _validate_episode mutates _contract_keys on first call
        import threading
        self._contract_lock = threading.Lock()
        self._init_contract()

    @classmethod
    def create(
        cls,
        path: str,
        key_shapes: Dict[str, tuple],
        key_dtypes: Dict[str, np.dtype],
        n_episodes: int = 0,
        keys: Optional[List[str]] = None,
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
        writer = cls(
            path=path, n_episodes=n_episodes, video_keys=vk,
            vcodec=vcodec, crf=crf, kfi=kfi, fps=fps, schemas=schemas,
        )
        writer._init_contract(key_shapes, key_dtypes)
        # Record keys/shapes/dtypes from declared contract
        resolved_keys = list(keys) if keys is not None else list(key_shapes.keys())
        writer._all_keys = resolved_keys
        writer._shapes = {k: tuple(v) for k, v in key_shapes.items()}
        writer._dtypes = {k: str(np.dtype(v)) for k, v in key_dtypes.items()}
        return writer

    @classmethod
    def open_readonly(cls, path: str) -> VideoStore:
        return VideoStore(path)

    # ── EpisodeStore read primitives (available immediately from contract) ─────

    @property
    def n_episodes(self) -> int:
        return self._n_episodes

    @property
    def keys(self) -> List[str]:
        return self._all_keys

    def frame_shape(self, key: str) -> tuple:
        return self._shapes[key]

    def frame_dtype(self, key: str) -> np.dtype:
        return np.dtype(self._dtypes[key])

    def episode_length(self, ep_idx: int) -> int:
        # Only available after put_episode for that ep_idx; not meaningful pre-write
        raise NotImplementedError("episode_length not available on writer before finalize")

    def get_frames(self, ep_idx: int, key: str, start: int, end: int) -> np.ndarray:
        raise NotImplementedError("get_frames not available on writer before finalize")

    def _update_schema_from_data(self, data: Dict[str, np.ndarray]) -> None:
        """Lock in keys/shapes/dtypes from first episode's data (no-op if already set)."""
        if not self._all_keys:
            self._all_keys = list(data.keys())
            self._shapes = {k: tuple(arr.shape[1:]) for k, arr in data.items()}
            self._dtypes = {k: str(arr.dtype) for k, arr in data.items()}

    # ── Video encoding ────────────────────────────────────────────────────────

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

    # ── Write interface ────────────────────────────────────────────────────────

    def put_episode(self, ep_idx_or_data, data: Optional[Dict[str, np.ndarray]] = None) -> None:
        """Write episode.

        Two calling conventions are accepted:
          put_episode(ep_idx, data)   — new episode-indexed interface
          put_episode(data)           — legacy sequential interface (ep_idx auto-assigned)
        """
        assert not self._finalized, "Cannot write after finalize()"

        # Detect calling convention
        if data is None:
            # Legacy: put_episode(data) — auto-assign next sequential ep_idx
            actual_data = ep_idx_or_data
            with self._contract_lock:
                ep_idx = self._next_ep_idx
                self._next_ep_idx += 1
                T = self._validate_episode(actual_data, episode_idx=ep_idx)
                self._update_schema_from_data(actual_data)
        else:
            ep_idx = ep_idx_or_data
            actual_data = data
            with self._contract_lock:
                T = self._validate_episode(actual_data, episode_idx=ep_idx)
                self._update_schema_from_data(actual_data)

        # Write video keys
        for key, arr in actual_data.items():
            if key in self._video_keys_set:
                self._encode_episode_video(key, ep_idx, arr)

        # Write non-video keys as per-episode zarr
        non_video = {k: v for k, v in actual_data.items() if k not in self._video_keys_set}
        if non_video:
            import numcodecs
            ep_dir = os.path.join(self.path, "data", f"{ep_idx:06d}")
            os.makedirs(ep_dir, exist_ok=True)
            for key, arr in non_video.items():
                arr_path = os.path.join(ep_dir, key)
                frame_shape = tuple(arr.shape[1:])
                chunk_t = min(T, _NON_VIDEO_CHUNK_FRAMES)
                chunks = (chunk_t,) + frame_shape
                zarr_arr = _open_zarr_array(
                    arr_path,
                    mode="w",
                    shape=arr.shape,
                    chunks=chunks,
                    dtype=arr.dtype,
                    compressor=numcodecs.Blosc(cname="lz4", clevel=1),
                )
                zarr_arr[:] = arr

    def finalize(self) -> None:
        """Write meta.json.  Call after all put_episode() calls complete."""
        from ..schema import schemas_to_meta, validate_schemas_against_shapes

        validate_schemas_against_shapes(self._schemas, self._shapes)

        # Verify video file counts (skip check if n_episodes was not declared)
        for key in self._all_keys:
            if key not in self._video_keys_set:
                continue
            vid_dir = os.path.join(self.path, "videos", key)
            if os.path.isdir(vid_dir):
                n_files = len([f for f in os.listdir(vid_dir) if f.endswith(".mp4")])
                if self._n_episodes > 0:
                    assert n_files == self._n_episodes, (
                        f"Video key {key!r}: {n_files} episode files in {vid_dir!r}, "
                        f"expected {self._n_episodes} — store is inconsistent."
                    )
                else:
                    # n_episodes was not declared; infer from file count
                    self._n_episodes = n_files

        meta = {
            "keys": self._all_keys,
            "shapes": {k: list(v) for k, v in self._shapes.items()},
            "dtypes": self._dtypes,
            "n_episodes": self._n_episodes,
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
