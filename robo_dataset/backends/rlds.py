"""RLDS/TFRecord backend (read + write).

Layout on disk:
  <path>/frames.tfrecord     one tf.train.Example per frame
  <path>/frames.index        text file: one byte offset per line
  <path>/meta.json           key shapes, dtypes, image_keys, n_frames, episode_ends

Each Example has one bytes_list feature per key:
  - Image keys: JPEG bytes
  - Other keys: raw float32 bytes (shape/dtype recorded in meta.json)
"""

import json
import os
import struct
import threading
from typing import Dict, List, Optional

import cv2
import numpy as np

from .base import TrajectoryStore, WritableTrajectoryStore

_TFRECORD_FILE = "frames.tfrecord"
_INDEX_FILE = "frames.index"
_META_FILE = "meta.json"


# ── TFRecord low-level I/O ────────────────────────────────────────────────────

def _masked_crc32(data: bytes) -> int:
    import crc32c
    crc = crc32c.crc32c(data)
    return (((crc >> 15) | (crc << 17)) + 0xA282EAD8) & 0xFFFFFFFF


def _read_record(f, offset: int) -> bytes:
    f.seek(offset)
    len_bytes = f.read(8)
    if len(len_bytes) != 8:
        raise ValueError(
            f"Truncated TFRecord header at byte offset {offset}: got "
            f"{len(len_bytes)} of 8 length bytes."
        )
    _len_crc = f.read(4)
    length = struct.unpack("<Q", len_bytes)[0]
    data = f.read(length)
    if len(data) != length:
        raise ValueError(
            f"Truncated TFRecord at byte offset {offset}: record declares "
            f"{length} bytes but only {len(data)} could be read."
        )
    _data_crc = f.read(4)
    return data


def _write_record(f, data: bytes) -> int:
    offset = f.tell()
    length = len(data)
    len_bytes = struct.pack("<Q", length)
    f.write(len_bytes)
    f.write(struct.pack("<I", _masked_crc32(len_bytes)))
    f.write(data)
    f.write(struct.pack("<I", _masked_crc32(data)))
    return offset


# ── Example proto helpers ─────────────────────────────────────────────────────

def _make_example(features: Dict[str, bytes]) -> bytes:
    from tfrecord import example_pb2
    ex = example_pb2.Example()
    for key, blob in features.items():
        ex.features.feature[key].bytes_list.value.append(blob)
    return ex.SerializeToString()


def _parse_example(raw: bytes, keys: List[str]) -> Dict[str, bytes]:
    from tfrecord import example_pb2
    ex = example_pb2.Example()
    ex.ParseFromString(raw)
    return {key: bytes(ex.features.feature[key].bytes_list.value[0]) for key in keys}


def _encode_jpeg(img: np.ndarray, quality: int) -> bytes:
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
    _, buf = cv2.imencode(".jpg", img, encode_param)
    return buf.tobytes()


# ── Reader ────────────────────────────────────────────────────────────────────

class RLDSFrameStore(TrajectoryStore):
    """Read-only random-access RLDS/TFRecord per-frame store.

    Pros
    ----
    - Single file: one open() per worker regardless of dataset size → no inode
      overhead. At 12 workers this is 12 opens vs. zarr's ~232k per epoch.
    - O(1) random access via the companion .index file (byte offsets, one per frame).
    - Highest DataLoader throughput on local SSD (≈ 2× zarr_disk_jpeg at 12 workers,
      see benchmarks/rlds_bench_results.json).
    - TFRecord format is compatible with tf.data for TensorFlow-side consumers.
    - JPEG compression is applied per frame at write time; no additional codec needed.

    Cons
    ----
    - No vectorized access: get_sequence(key, start, end) issues (end - start)
      individual seeks — one per frame. For a horizon-16 window that is 16 seeks
      vs. zarr's single array slice. CPU cost (protobuf + JPEG decode) dominates on
      fast SSDs, but the seek count matters for slow storage.
    - Two-file layout: .tfrecord + .index must be kept together. Deleting or
      misplacing the index makes random access impossible.
    - Write-once: appending to an existing TFRecord is not supported.
    - Requires protobuf and crc32c as runtime dependencies.
    - JPEG is lossy; not appropriate when bit-exact fidelity is required.
    """

    def __init__(self, path: str):
        self.path = path
        with open(os.path.join(path, _META_FILE)) as f:
            meta = json.load(f)

        self._all_keys: List[str] = list(meta["shapes"].keys())
        self._image_keys: set = set(meta["image_keys"])
        self._shapes: Dict[str, tuple] = {k: tuple(v) for k, v in meta["shapes"].items()}
        self._dtypes: Dict[str, str] = meta["dtypes"]
        self._episode_ends = np.array(meta["episode_ends"], dtype=np.int64)

        self._tfr_path = os.path.join(path, _TFRECORD_FILE)
        with open(os.path.join(path, _INDEX_FILE)) as f:
            self._offsets = np.array([int(l.strip()) for l in f if l.strip()], dtype=np.int64)
        n_frames = int(self._episode_ends[-1]) if len(self._episode_ends) else 0
        if len(self._offsets) != n_frames:
            raise ValueError(
                f"RLDS store at {path!r}: index has {len(self._offsets)} offsets "
                f"but episode_ends declares {n_frames} frames — "
                "store is corrupt/truncated."
            )

        self._local = threading.local()

    def _fh(self):
        if not hasattr(self._local, "f") or self._local.f is None:
            self._local.f = open(self._tfr_path, "rb")
        return self._local.f

    def on_worker_init(self, worker_id: int) -> None:
        if hasattr(self._local, "f") and self._local.f is not None:
            self._local.f.close()
        self._local.f = open(self._tfr_path, "rb")

    @property
    def episode_ends(self) -> np.ndarray:
        return self._episode_ends

    @property
    def keys(self) -> List[str]:
        return self._all_keys

    def frame_shape(self, key: str) -> tuple:
        return self._shapes[key]

    def frame_dtype(self, key: str) -> np.dtype:
        return np.dtype(self._dtypes[key])

    def get_sequence(self, key: str, start: int, end: int) -> np.ndarray:
        fh = self._fh()
        frames = []
        for i in range(start, end):
            raw = _read_record(fh, int(self._offsets[i]))
            blob = _parse_example(raw, [key])[key]
            if key in self._image_keys:
                img = cv2.imdecode(np.frombuffer(blob, np.uint8), cv2.IMREAD_COLOR)
                frames.append(img)
            else:
                arr = np.frombuffer(blob, dtype=np.float32).reshape(self._shapes[key])
                frames.append(arr)
        return np.stack(frames)

    def __del__(self):
        if hasattr(self._local, "f") and self._local.f is not None:
            try:
                self._local.f.close()
            except Exception:
                pass


# ── Writer ────────────────────────────────────────────────────────────────────

class RLDSFrameStoreWriter(WritableTrajectoryStore):
    """Streaming writer for the RLDS frame format.

    Implements WritableTrajectoryStore so it can be used as a conversion target
    and as a disk-cache backend in apply_disk_cached().

    After finalize(), the read interface delegates to an internal RLDSFrameStore.
    Before finalize(), read calls raise RuntimeError.
    Same pros/cons as RLDSFrameStore apply after finalize().
    """

    def __init__(self, path: str, image_keys: List[str], jpeg_quality: int = 75):
        self.path = path
        self.image_keys = set(image_keys)
        self.jpeg_quality = jpeg_quality
        os.makedirs(path, exist_ok=True)

        self._tfr_path = os.path.join(path, _TFRECORD_FILE)
        self._idx_path = os.path.join(path, _INDEX_FILE)
        self._f = open(self._tfr_path, "wb")
        self._offsets: List[int] = []
        self._episode_ends_list: List[int] = []
        self._n_frames = 0
        self._shapes: Dict[str, tuple] = {}
        self._dtypes: Dict[str, str] = {}
        self._reader: Optional[RLDSFrameStore] = None
        self._init_contract()

    def _init_contract(self, key_shapes=None, key_dtypes=None) -> None:
        # RLDS stores all non-image data as raw float32 bytes; a declared
        # non-float32 dtype could never round-trip, so refuse it up front.
        if key_dtypes:
            bad = {
                k: str(np.dtype(v)) for k, v in key_dtypes.items()
                if k not in self.image_keys and np.dtype(v) != np.float32
            }
            if bad:
                raise ValueError(
                    f"RLDSFrameStoreWriter stores non-image data as float32; "
                    f"declared non-float32 dtypes are not supported: {bad}. "
                    "Cast the data (or declare float32) before writing."
                )
        super()._init_contract(key_shapes, key_dtypes)

    @classmethod
    def create(
        cls,
        path: str,
        key_shapes: Dict[str, tuple],
        key_dtypes: Dict[str, np.dtype],
        image_keys: List[str] = (),
        **kwargs,
    ) -> "RLDSFrameStoreWriter":
        writer = cls(path=path, image_keys=image_keys, jpeg_quality=kwargs.get("jpeg_quality", 75))
        writer._init_contract(key_shapes, key_dtypes)
        # Pre-populate shapes/dtypes for non-image keys (image shapes come from first frame)
        for k, shape in key_shapes.items():
            if k not in writer.image_keys:
                writer._shapes[k] = tuple(shape)
                writer._dtypes[k] = str(np.dtype(key_dtypes[k]))
        return writer

    @classmethod
    def open_readonly(cls, path: str) -> RLDSFrameStore:
        return RLDSFrameStore(path)

    def _write_frame(self, frame_data: Dict[str, np.ndarray]) -> None:
        features = {}
        for key, arr in frame_data.items():
            if key in self.image_keys:
                if key not in self._shapes:
                    self._shapes[key] = arr.shape
                    self._dtypes[key] = str(arr.dtype)
                features[key] = _encode_jpeg(arr, self.jpeg_quality)
            else:
                if key not in self._shapes:
                    self._shapes[key] = arr.shape
                # Non-image data is stored (and read back) as float32 — record
                # that, not the incoming dtype, so meta.json reflects reality.
                self._dtypes[key] = "float32"
                features[key] = arr.astype(np.float32).tobytes()
        offset = _write_record(self._f, _make_example(features))
        self._offsets.append(offset)
        self._n_frames += 1

    def put_episode(self, data: Dict[str, np.ndarray]) -> None:
        assert self._reader is None, "Cannot write after finalize()"
        T = self._validate_episode(data, episode_idx=len(self._episode_ends_list))
        for t in range(T):
            self._write_frame({k: data[k][t] for k in data})
        self._episode_ends_list.append(self._n_frames)

    def finalize(self) -> None:
        total = self._episode_ends_list[-1] if self._episode_ends_list else 0
        assert len(self._offsets) == total == self._n_frames, (
            f"RLDS writer: {len(self._offsets)} record offsets vs "
            f"{total} frames in episode_ends (n_frames={self._n_frames}) — "
            "store is inconsistent."
        )
        self._f.close()
        with open(self._idx_path, "w") as f:
            for offset in self._offsets:
                f.write(f"{offset}\n")
        meta = {
            "n_frames": self._n_frames,
            "episode_ends": self._episode_ends_list,
            "shapes": {k: list(v) for k, v in self._shapes.items()},
            "dtypes": self._dtypes,
            "image_keys": list(self.image_keys),
        }
        with open(os.path.join(self.path, _META_FILE), "w") as f:
            json.dump(meta, f, indent=2)
        self._reader = RLDSFrameStore(self.path)

    # ── Read interface — delegates to RLDSFrameStore after finalize ───────────

    def _require_finalized(self):
        if self._reader is None:
            raise RuntimeError("RLDSFrameStoreWriter: call finalize() before reading")

    @property
    def episode_ends(self) -> np.ndarray:
        if self._reader is not None:
            return self._reader.episode_ends
        return np.array(self._episode_ends_list, dtype=np.int64)

    @property
    def keys(self) -> List[str]:
        return list(self._shapes.keys())

    def frame_shape(self, key: str) -> tuple:
        return self._shapes[key]

    def frame_dtype(self, key: str) -> np.dtype:
        return np.dtype(self._dtypes[key])

    def get_sequence(self, key: str, start: int, end: int) -> np.ndarray:
        self._require_finalized()
        return self._reader.get_sequence(key, start, end)

    def on_worker_init(self, worker_id: int) -> None:
        self._require_finalized()
        self._reader.on_worker_init(worker_id)
