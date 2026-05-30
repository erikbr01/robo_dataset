"""RLDS/TFRecord backend (read-only for now).

Layout on disk:
  <path>/frames.tfrecord     one tf.train.Example per frame
  <path>/frames.index        text file: one row per frame, "offset length"
  <path>/meta.json           key shapes, dtypes, image_keys, n_frames, episode_ends

Each Example has one feature per key:
  - Image keys: JPEG bytes     → bytes_list
  - Other keys: raw float32    → bytes_list (raw bytes, shape in meta)
"""

import json
import os
import struct
import threading
from typing import Dict, List, Optional

import cv2
import numpy as np

from .base import TrajectoryStore

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
    _len_crc = f.read(4)
    length = struct.unpack("<Q", len_bytes)[0]
    data = f.read(length)
    _data_crc = f.read(4)
    return data


def _write_record(f, data: bytes) -> int:
    """Write one TFRecord record, return the byte offset where it was written."""
    offset = f.tell()
    length = len(data)
    len_bytes = struct.pack("<Q", length)
    f.write(len_bytes)
    f.write(struct.pack("<I", _masked_crc32(len_bytes)))
    f.write(data)
    f.write(struct.pack("<I", _masked_crc32(data)))
    return offset


# ── Example proto helpers ─────────────────────────────────────────────────────

def _make_example(features: Dict[str, bytes]):
    """Build a tf.train.Example with one bytes_list feature per key."""
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


# ── Writer ────────────────────────────────────────────────────────────────────

class RLDSFrameStoreWriter:
    """Streaming writer for the RLDS frame format."""

    def __init__(self, path: str, image_keys: List[str], jpeg_quality: int = 75):
        self.path = path
        self.image_keys = set(image_keys)
        self.jpeg_quality = jpeg_quality
        os.makedirs(path, exist_ok=True)

        self._tfr_path = os.path.join(path, _TFRECORD_FILE)
        self._idx_path = os.path.join(path, _INDEX_FILE)
        self._f = open(self._tfr_path, "wb")
        self._offsets: List[int] = []
        self._episode_ends: List[int] = []
        self._n_frames = 0

        self._shapes: Dict[str, tuple] = {}
        self._dtypes: Dict[str, str] = {}

    def write_frame(self, frame_data: Dict[str, np.ndarray]) -> None:
        features = {}
        for key, arr in frame_data.items():
            if key in self.image_keys:
                features[key] = _encode_jpeg(arr, self.jpeg_quality)
            else:
                if key not in self._shapes:
                    self._shapes[key] = arr.shape
                    self._dtypes[key] = str(arr.dtype)
                features[key] = arr.astype(np.float32).tobytes()
        offset = _write_record(self._f, _make_example(features))
        self._offsets.append(offset)
        self._n_frames += 1

    def end_episode(self) -> None:
        self._episode_ends.append(self._n_frames)

    def finalize(self, key_shapes: Dict[str, tuple] = None, key_dtypes: Dict[str, np.dtype] = None):
        self._f.close()
        # Write index
        with open(self._idx_path, "w") as f:
            for offset in self._offsets:
                # length = next offset - current - 16 header bytes; easier to re-read
                f.write(f"{offset}\n")
        # Merge shapes/dtypes
        if key_shapes:
            for k, v in key_shapes.items():
                if k not in self._shapes:
                    self._shapes[k] = tuple(v)
        if key_dtypes:
            for k, v in key_dtypes.items():
                if k not in self._dtypes:
                    self._dtypes[k] = str(np.dtype(v))
        meta = {
            "n_frames": self._n_frames,
            "episode_ends": self._episode_ends,
            "shapes": {k: list(v) for k, v in self._shapes.items()},
            "dtypes": self._dtypes,
            "image_keys": list(self.image_keys),
        }
        with open(os.path.join(self.path, _META_FILE), "w") as f:
            json.dump(meta, f, indent=2)


def _encode_jpeg(img: np.ndarray, quality: int) -> bytes:
    """Encode HWC uint8 image as JPEG bytes."""
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
    _, buf = cv2.imencode(".jpg", img, encode_param)
    return buf.tobytes()


# ── Reader ────────────────────────────────────────────────────────────────────

class RLDSFrameStore(TrajectoryStore):
    """Read-only random-access RLDS/TFRecord per-frame store."""

    def __init__(self, path: str):
        self.path = path
        meta_path = os.path.join(path, _META_FILE)
        with open(meta_path) as f:
            meta = json.load(f)

        self._all_keys: List[str] = list(meta["shapes"].keys())
        self._image_keys: set = set(meta["image_keys"])
        self._shapes: Dict[str, tuple] = {k: tuple(v) for k, v in meta["shapes"].items()}
        self._dtypes: Dict[str, str] = meta["dtypes"]
        self._episode_ends = np.array(meta["episode_ends"], dtype=np.int64)
        self._n_frames = meta["n_frames"]

        self._tfr_path = os.path.join(path, _TFRECORD_FILE)
        self._idx_path = os.path.join(path, _INDEX_FILE)

        # Load offset index
        with open(self._idx_path) as f:
            self._offsets = np.array([int(line.strip()) for line in f if line.strip()], dtype=np.int64)

        # Per-thread file handles (DataLoader workers + main thread)
        self._local = threading.local()

    def _fh(self):
        if not hasattr(self._local, "f") or self._local.f is None:
            self._local.f = open(self._tfr_path, "rb")
        return self._local.f

    def on_worker_init(self, worker_id: int) -> None:
        # Close any inherited handle and reopen fresh
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
            parsed = _parse_example(raw, [key])
            blob = parsed[key]
            if key in self._image_keys:
                img = cv2.imdecode(np.frombuffer(blob, np.uint8), cv2.IMREAD_COLOR)
                frames.append(img)
            else:
                dtype = np.dtype(self._dtypes[key])
                shape = self._shapes[key]
                arr = np.frombuffer(blob, dtype=np.float32).reshape(shape)
                frames.append(arr)
        return np.stack(frames)

    def __del__(self):
        if hasattr(self._local, "f") and self._local.f is not None:
            try:
                self._local.f.close()
            except Exception:
                pass
