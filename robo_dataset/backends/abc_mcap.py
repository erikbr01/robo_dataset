"""ABC-130k MCAP backend — reads the released dataset in place, no conversion.

Layout on disk (as published in ``XDOF/ABC-130k``)::

    <path>/<episode_dir>/episode.mcap       three h264 camera streams + state/action
    <path>/<episode_dir>/annotation.mcap    optional subtask segment labels

``episode.mcap`` carries, per frame, one protobuf message per topic:

===========================  ===========================  ==========  ========
topic                        schema                       rate        width
===========================  ===========================  ==========  ========
``/top-camera``              ``foxglove.CompressedVideo``  ~60 Hz     h264 AU
``/left-wrist-camera``       ``foxglove.CompressedVideo``  ~60 Hz     h264 AU
``/right-wrist-camera``      ``foxglove.CompressedVideo``  ~60 Hz     h264 AU
``/{left,right}-arm-state``  ``RobotState``               ~276 Hz     6 joints
``/{left,right}-ee-state``   ``GripperState``             ~276 Hz     1 aperture
``/{left,right}-arm-action`` ``RobotState``               ~200 Hz     6 joints
``/{left,right}-ee-action``  ``GripperState``             ~200 Hz     1 aperture
``/instruction``             ``Instructions``             once        text
===========================  ===========================  ==========  ========

Two problems have to be solved to serve ``get_frames`` from this.

**Independent clocks.** Every stream has its own rate and message count -- the
four state topics in a single episode differ by tens of messages. The dataset
card's instruction is to nearest-neighbour match on the absolute timestamps, and
that is what happens here. The **top camera's frame times are the index**: an
episode's length is its number of top-camera frames, and every other key is
sampled onto those instants. That mirrors ABC's own export, where video frames
are the frame axis.

**MCAP has no seek index for video.** Serving a random ``[start, end)`` slice
would otherwise mean decoding the episode from frame zero every time. Rather
than convert the dataset, each camera's Annex-B payloads are concatenated into a
seekable MP4 sidecar under ``<episode_dir>/.robo_dataset/``, built once, lazily,
on first access. Decoding then reuses the same torchcodec/PyAV compatibility
layer and per-worker LRU decoder cache as :mod:`robo_dataset.backends.video`.

The sidecar is a pure **remux** (``ffmpeg -c copy``): container framing only, the
coded pictures are byte-identical -- verified against a straight sequential
decode -- costing ~0.2 s per camera per episode. If the sidecars are unwanted
(read-only media, or a shared cache), pass ``cache_dir=`` to place them
elsewhere.

**This store is good at sequential reads and bad at random ones.** ABC records
with a 250-frame GOP, so a random seek decodes ~125 frames to reach its target.
Measured on one 480x640 camera: sequential decode runs at 350-534 frames/s, but a
seek costs ~175 ms and thirty contiguous frames cost barely more than one, so the
cost is per-seek rather than per-frame. Whole-episode passes (feature extraction,
statistics, a conversion) are therefore cheap, while i.i.d. window sampling --
what a trainer does -- is not: ~460 ms per sample across three cameras.

So **do not train directly against this store.** Materialise it once with
:func:`robo_dataset.transforms.apply_disk_cached` and
:class:`~robo_dataset.transforms.ResizeImages`, which reads sequentially (fast),
resizes to the resolution the policy consumes, and writes a zarr store whose
random access is cheap and whose frame shape is uniform. Use
``ResizeImages(..., letterbox=True)`` here -- see the next paragraph.

**Native frame size is not uniform.** In the annotated bottles task, 28 of 40
episodes record at 848x480 (16:9) and 8 at 640x480 (4:3). ``frame_shape`` is one
shape per key for the whole store, so it can only report the first episode's, and
``get_frames`` will hand back arrays that disagree with it for the rest; a
mismatch is warned about once, naming the fix. Both problems are the resize
transform's job: it makes every episode one shape, and letterboxing rather than
stretching keeps objects the same shape across the two aspect-ratio groups
instead of distorting one relative to the other.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
import tempfile
from typing import Dict, List, Optional

import numpy as np

from ..schema import VectorSchema
from .base import EpisodeStore
from .video import _LRUDecoderCache, _pick_decoder_backend

log = logging.getLogger(__name__)

CAMERA_TOPICS = {
    "top": "/top-camera",
    "left": "/left-wrist-camera",
    "right": "/right-wrist-camera",
}
#: Concatenation order of the 14-dim state and action vectors. Fixed here so an
#: embedding extracted today matches one extracted later.
VECTOR_TOPICS = {
    "state": (
        ("/left-arm-state", 6),
        ("/left-ee-state", 1),
        ("/right-arm-state", 6),
        ("/right-ee-state", 1),
    ),
    "action": (
        ("/left-arm-action", 6),
        ("/left-ee-action", 1),
        ("/right-arm-action", 6),
        ("/right-ee-action", 1),
    ),
}
INSTRUCTION_TOPIC = "/instruction"
ANNOTATION_TOPIC = "/subtask-annotation"
_CACHE_DIRNAME = ".robo_dataset"
_INDEX_FILE = "index.json"
_DEFAULT_MAX_DECODERS = 32


def _require_mcap():
    try:
        from mcap.reader import make_reader
        from mcap_protobuf.decoder import DecoderFactory
    except ImportError as e:  # pragma: no cover - dependency hint
        raise ImportError(
            "The ABC MCAP backend needs `mcap` and `mcap-protobuf-support`; "
            "install with `pip install mcap mcap-protobuf-support`."
        ) from e
    return make_reader, DecoderFactory


class ABCMcapStore(EpisodeStore):
    """Read-only :class:`EpisodeStore` over a directory of ABC-130k episodes.

    Args:
        path: directory whose immediate children are episode directories
            containing ``episode.mcap``.
        cameras: which camera keys to expose. Defaults to all three.
        stride: keep every ``stride``-th top-camera frame. ``stride=2`` on a
            60 Hz recording gives a 30 Hz frame axis, which is what ABC's own
            export uses.
        cache_dir: where to write the MP4 sidecars and the per-episode index.
            Defaults to ``<episode_dir>/.robo_dataset``.
        max_decoders: per-worker LRU cap on open video decoders.
        require_annotations: skip episodes without ``annotation.mcap``. Useful
            because only 42,980 of the 130,703 published episodes are annotated.
    """

    def __init__(
        self,
        path: str,
        cameras: Optional[List[str]] = None,
        stride: int = 1,
        cache_dir: Optional[str] = None,
        max_decoders: int = _DEFAULT_MAX_DECODERS,
        require_annotations: bool = False,
    ):
        self.path = str(path)
        self.cameras = list(cameras) if cameras else list(CAMERA_TOPICS)
        unknown = [c for c in self.cameras if c not in CAMERA_TOPICS]
        if unknown:
            raise ValueError(f"unknown camera(s) {unknown}; known: {sorted(CAMERA_TOPICS)}")
        if stride < 1:
            raise ValueError(f"stride must be >= 1, got {stride}")
        self.stride = int(stride)
        self.cache_dir = cache_dir
        self.require_annotations = bool(require_annotations)

        # Set on the first index seen; a later index with a different native
        # size means `frame_shape` cannot describe the whole store (see
        # `_check_uniform_size`).
        self._native_size: Optional[tuple] = None
        self._warned_mixed = False

        self._episodes = [
            d
            for d in sorted(
                os.path.join(self.path, n)
                for n in os.listdir(self.path)
                if os.path.isdir(os.path.join(self.path, n))
            )
            if os.path.exists(os.path.join(d, "episode.mcap"))
            and (
                not require_annotations
                or os.path.exists(os.path.join(d, "annotation.mcap"))
            )
        ]
        if not self._episodes:
            raise ValueError(
                f"no episodes with episode.mcap under {self.path!r}"
                + (" that also have annotation.mcap" if require_annotations else "")
            )
        self._index: Dict[int, dict] = {}
        self._vectors: Dict[int, Dict[str, np.ndarray]] = {}
        self._frameidx_cache: Dict[tuple, np.ndarray] = {}
        self._decoder_cache = _LRUDecoderCache(maxsize=max_decoders)
        _pick_decoder_backend()
        log.info("ABCMcapStore: %d episodes at %s", len(self._episodes), self.path)

    # ── required primitives ───────────────────────────────────────────────────

    @property
    def n_episodes(self) -> int:
        return len(self._episodes)

    @property
    def keys(self) -> List[str]:
        return list(VECTOR_TOPICS) + list(self.cameras)

    def episode_length(self, ep_idx: int) -> int:
        return int(self._ensure_index(ep_idx)["n_frames"])

    def frame_shape(self, key: str) -> tuple:
        if key in VECTOR_TOPICS:
            return (sum(w for _, w in VECTOR_TOPICS[key]),)
        idx = self._ensure_index(0)
        return (idx["height"], idx["width"], 3)

    def frame_dtype(self, key: str) -> np.dtype:
        return np.dtype(np.float32 if key in VECTOR_TOPICS else np.uint8)

    def get_frames(self, ep_idx: int, key: str, start: int, end: int) -> np.ndarray:
        n = self.episode_length(ep_idx)
        if not (0 <= start < end <= n):
            raise IndexError(
                f"frame range [{start}, {end}) invalid for episode {ep_idx} "
                f"with {n} frames"
            )
        if key in VECTOR_TOPICS:
            return self._ensure_vectors(ep_idx)[key][start:end]
        if key not in self.cameras:
            raise KeyError(f"unknown key {key!r}; have {self.keys}")
        return self._decode(ep_idx, key, list(range(start, end)))

    def on_worker_init(self, worker_id: int) -> None:
        # Video decoders hold file descriptors that must not be shared across a
        # fork; the vector cache is plain numpy and is safe to keep.
        self._decoder_cache.clear()

    @property
    def schemas(self) -> Dict[str, VectorSchema]:
        """Declare the bimanual layout so no consumer hardcodes index ranges."""
        fields = [
            {"name": "left_arm", "dim": 6, "kind": "joints", "representation": "abs"},
            {"name": "left_gripper", "dim": 1, "kind": "gripper",
             "representation": "continuous"},
            {"name": "right_arm", "dim": 6, "kind": "joints", "representation": "abs"},
            {"name": "right_gripper", "dim": 1, "kind": "gripper",
             "representation": "continuous"},
        ]
        return {k: VectorSchema.from_config(fields) for k in VECTOR_TOPICS}

    # ── ABC-specific extras ───────────────────────────────────────────────────

    def episode_dir(self, ep_idx: int) -> str:
        return self._episodes[ep_idx]

    def instruction(self, ep_idx: int) -> str:
        return str(self._ensure_index(ep_idx).get("instruction") or "")

    def frame_times_ns(self, ep_idx: int) -> np.ndarray:
        """Absolute top-camera frame timestamps, the store's own frame axis."""
        return np.asarray(self._ensure_index(ep_idx)["times_ns"], dtype=np.int64)

    def subtask_segments(self, ep_idx: int) -> List[dict]:
        """``[{start_sec, end_sec, subtask}]`` from ``annotation.mcap``, or [].

        ABC stores annotations as timestamped *start markers*, so segment *i*
        runs from marker *i* to marker *i+1* and the last runs to the end of the
        frame axis. Times are seconds relative to the first frame.
        """
        make_reader, DecoderFactory = _require_mcap()
        ann_path = os.path.join(self._episodes[ep_idx], "annotation.mcap")
        if not os.path.exists(ann_path):
            return []
        times = self.frame_times_ns(ep_idx)
        t0, t1 = int(times[0]), int(times[-1])
        marks = []
        with open(ann_path, "rb") as fh:
            reader = make_reader(fh, decoder_factories=[DecoderFactory()])
            for _s, ch, msg, dec in reader.iter_decoded_messages():
                if ch.topic == ANNOTATION_TOPIC:
                    marks.append((msg.log_time, str(dec.data)))
        marks.sort(key=lambda x: x[0])
        out = []
        for i, (t, text) in enumerate(marks):
            end_ns = marks[i + 1][0] if i + 1 < len(marks) else t1
            if end_ns > t:
                out.append(
                    {
                        "start_sec": (t - t0) / 1e9,
                        "end_sec": (end_ns - t0) / 1e9,
                        "subtask": text,
                    }
                )
        return out

    # ── internals ─────────────────────────────────────────────────────────────

    def _cache_root(self, ep_idx: int) -> str:
        if self.cache_dir:
            return os.path.join(self.cache_dir, os.path.basename(self._episodes[ep_idx]))
        return os.path.join(self._episodes[ep_idx], _CACHE_DIRNAME)

    def _ensure_index(self, ep_idx: int) -> dict:
        """Frame axis + per-camera sidecars, built once and cached on disk."""
        if ep_idx in self._index:
            return self._index[ep_idx]
        root = self._cache_root(ep_idx)
        idx_path = os.path.join(root, _INDEX_FILE)
        if os.path.exists(idx_path):
            with open(idx_path) as f:
                idx = json.load(f)
            if idx.get("stride") == self.stride and set(self.cameras).issubset(
                idx.get("cameras", [])
            ):
                self._index[ep_idx] = idx
                self._check_uniform_size(ep_idx, idx)
                return idx
        idx = self._build_episode_cache(ep_idx, root)
        self._index[ep_idx] = idx
        self._check_uniform_size(ep_idx, idx)
        return idx

    def _check_uniform_size(self, ep_idx: int, idx: dict) -> None:
        """Warn once if episodes disagree on frame size.

        ABC is not uniform: in the annotated bottles task, 28 of 40 episodes are
        848x480 and 8 are 640x480. `frame_shape` is a single per-key shape for
        the whole store, so it can only report the first episode's -- and
        `get_frames` would then hand back arrays that do not match it, which
        collates into a confusing failure much further away. The resize transform
        removes the problem, so point at it rather than just warn.
        """
        size = (idx["height"], idx["width"])
        if self._native_size is None:
            self._native_size = size
        elif size != self._native_size and not self._warned_mixed:
            self._warned_mixed = True
            log.warning(
                "ABCMcapStore: episode %d is %dx%d but an earlier episode was "
                "%dx%d. frame_shape() cannot describe a store with mixed sizes. "
                "Materialise the store through transforms.apply_disk_cached with "
                "ResizeImages(..., letterbox=True) to get one shape for every "
                "episode.",
                ep_idx, size[0], size[1], *self._native_size,
            )

    def _build_episode_cache(self, ep_idx: int, root: str) -> dict:
        make_reader, DecoderFactory = _require_mcap()
        ep_dir = self._episodes[ep_idx]
        os.makedirs(root, exist_ok=True)

        want_cam = {CAMERA_TOPICS[c]: c for c in self.cameras}
        want_vec = {t: (k, w) for k, tw in VECTOR_TOPICS.items() for t, w in tw}
        cam_pk: Dict[str, list] = {c: [] for c in self.cameras}
        vec_msgs: Dict[str, list] = {t: [] for t in want_vec}
        instruction = ""

        with open(os.path.join(ep_dir, "episode.mcap"), "rb") as fh:
            reader = make_reader(fh, decoder_factories=[DecoderFactory()])
            for _s, ch, msg, dec in reader.iter_decoded_messages():
                if ch.topic in want_cam:
                    cam_pk[want_cam[ch.topic]].append((msg.log_time, bytes(dec.data)))
                elif ch.topic in want_vec:
                    vec_msgs[ch.topic].append((msg.log_time, list(dec.position)))
                elif ch.topic == INSTRUCTION_TOPIC and not instruction:
                    instruction = str(dec.data)
        for v in cam_pk.values():
            v.sort(key=lambda x: x[0])
        for v in vec_msgs.values():
            v.sort(key=lambda x: x[0])

        if not cam_pk.get("top") and "top" in self.cameras:
            raise ValueError(f"{ep_dir}: no /top-camera messages; cannot build a frame axis")
        anchor = "top" if "top" in self.cameras else self.cameras[0]
        anchor_times = np.asarray([t for t, _ in cam_pk[anchor]], dtype=np.int64)
        keep = np.arange(0, len(anchor_times), self.stride, dtype=np.int64)
        times = anchor_times[keep]

        width = height = None
        cam_files = {}
        for cam in self.cameras:
            if not cam_pk[cam]:
                raise ValueError(f"{ep_dir}: camera {cam!r} has no messages")
            mp4, w, h = self._remux(cam_pk[cam], root, cam)
            cam_files[cam] = os.path.basename(mp4)
            width, height = w, h
            # Each camera has its own clock; map the shared frame axis onto this
            # camera's own frame indices by nearest timestamp.
            cam_times = np.asarray([t for t, _ in cam_pk[cam]], dtype=np.int64)
            cam_idx = _nearest_index(cam_times, times)
            np.save(os.path.join(root, f"{cam}_frameidx.npy"), cam_idx)

        # Vector keys onto the same axis, nearest-neighbour per the dataset card.
        for key, topic_widths in VECTOR_TOPICS.items():
            cols = []
            for topic, w in topic_widths:
                cols.append(_resample(vec_msgs[topic], times, w))
            np.save(os.path.join(root, f"{key}.npy"),
                    np.concatenate(cols, axis=1).astype(np.float32))

        idx = {
            "n_frames": int(len(times)),
            "times_ns": [int(t) for t in times],
            "width": int(width),
            "height": int(height),
            "cameras": list(self.cameras),
            "camera_files": cam_files,
            "stride": self.stride,
            "instruction": instruction,
            "source": os.path.basename(ep_dir),
        }
        tmp = os.path.join(root, _INDEX_FILE + ".tmp")
        with open(tmp, "w") as f:
            json.dump(idx, f)
        os.replace(tmp, os.path.join(root, _INDEX_FILE))
        return idx

    def _remux(self, packets: list, root: str, cam: str):
        """Annex-B payloads -> a seekable MP4 via `ffmpeg -c copy` (no transcode).

        Returns ``(path, width, height)``; the size is probed from the result
        rather than assumed, because ABC episodes are not all one resolution.
        """
        out = os.path.join(root, f"{cam}.mp4")
        if os.path.exists(out) and os.path.getsize(out) > 0:
            w, h = _probe_size(out)
            return out, w, h
        if shutil.which("ffmpeg") is None:  # pragma: no cover
            raise RuntimeError("ffmpeg is required to index ABC video streams")
        with tempfile.TemporaryDirectory() as td:
            raw = os.path.join(td, "s.h264")
            with open(raw, "wb") as f:
                for _t, payload in packets:
                    f.write(payload)
            tmp_out = out + ".tmp.mp4"
            proc = subprocess.run(
                ["ffmpeg", "-v", "error", "-y", "-f", "h264", "-i", raw,
                 "-c", "copy", tmp_out],
                capture_output=True,
            )
            if proc.returncode != 0 or not os.path.exists(tmp_out):
                raise RuntimeError(
                    f"remux of {cam} failed: "
                    f"{proc.stderr.decode('utf-8', 'replace')[:400]}"
                )
            os.replace(tmp_out, out)
        w, h = _probe_size(out)
        return out, w, h

    def _ensure_vectors(self, ep_idx: int) -> Dict[str, np.ndarray]:
        if ep_idx not in self._vectors:
            root = self._cache_root(ep_idx)
            self._ensure_index(ep_idx)
            self._vectors[ep_idx] = {
                k: np.load(os.path.join(root, f"{k}.npy"), mmap_mode="r")
                for k in VECTOR_TOPICS
            }
        return self._vectors[ep_idx]

    def _decode(self, ep_idx: int, cam: str, frame_indices: List[int]) -> np.ndarray:
        idx = self._ensure_index(ep_idx)
        root = self._cache_root(ep_idx)
        cache_key = (ep_idx, cam)
        dec = self._decoder_cache.get(cache_key)
        if dec is None:
            backend = _pick_decoder_backend()
            path = os.path.join(root, idx["camera_files"][cam])
            if backend == "modern":
                from .video import _ModernDecoder as D
                # Exact seeking, not the default approximate. The sidecar is
                # remuxed from raw Annex-B h264, which carries no timestamps, so
                # ffmpeg synthesises them and index->timestamp estimation from
                # the average frame rate breaks down at the tail: episode 8 of
                # the bottles task reports 4370 frames and ffmpeg decodes all
                # 4370, yet approximate mode raises on frame 4368. Exact builds
                # a real index and measured just as cheap to open here.
                dec = D(path, seek_mode="exact")
            else:
                if backend == "legacy":
                    from .video import _LegacyDecoder as D
                else:
                    from .video import _PyAVDecoder as D
                dec = D(path)
            self._decoder_cache.put(cache_key, dec)
        mapping = self._frameidx(ep_idx, cam, root)
        native = [int(mapping[i]) for i in frame_indices]
        return dec.get_frames(native)

    def _frameidx(self, ep_idx: int, cam: str, root: str) -> np.ndarray:
        """Frame-axis -> this camera's own frame numbers, cached in memory.

        Small (one int64 per frame) but read on every sample, so re-loading it
        per `get_frames` call would put a file open in the hot path.
        """
        key = (ep_idx, cam)
        arr = self._frameidx_cache.get(key)
        if arr is None:
            arr = np.load(os.path.join(root, f"{cam}_frameidx.npy"))
            self._frameidx_cache[key] = arr
        return arr


def _probe_size(path: str):
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
         "stream=width,height", "-of", "csv=p=0", path],
        capture_output=True, text=True,
    ).stdout.strip()
    w, h = (int(x) for x in out.split(",")[:2])
    return w, h


def _nearest_index(src_times: np.ndarray, want_times: np.ndarray) -> np.ndarray:
    """Index into ``src_times`` of the nearest entry for each ``want_times``."""
    if len(src_times) == 1:
        return np.zeros(len(want_times), dtype=np.int64)
    pos = np.clip(np.searchsorted(src_times, want_times), 1, len(src_times) - 1)
    left, right = src_times[pos - 1], src_times[pos]
    take_left = (want_times - left) <= (right - want_times)
    return np.where(take_left, pos - 1, pos).astype(np.int64)


def _resample(messages: list, times: np.ndarray, width: int) -> np.ndarray:
    """Nearest-neighbour resample (time, vector) messages onto ``times``.

    Nearest rather than interpolated: a gripper aperture mid-transition should
    not be smeared across the transition, since the transition is exactly the
    event downstream consumers are looking for.
    """
    if not messages:
        return np.zeros((len(times), width), dtype=np.float32)
    src = np.asarray([t for t, _ in messages], dtype=np.int64)
    vals = np.asarray(
        [(list(v) + [0.0] * width)[:width] for _, v in messages], dtype=np.float32
    )
    return vals[_nearest_index(src, times)]
