"""Read-only LeRobot v2 backend (parquet + per-episode MP4).

LeRobot v2 layout:
  <root>/meta/info.json        features, fps, data_path / video_path templates
  <root>/meta/episodes.jsonl   one line per episode: {"episode_index", "length", ...}
  <root>/meta/modality.json    (optional, GR00T/robocasa) named slices into the
                               concatenated state/action vectors and video key map
  <root>/data/chunk-XXX/episode_XXXXXX.parquet
  <root>/videos/chunk-XXX/<video_key>/episode_XXXXXX.mp4

Exposed keys:
  - video features:  "video.<name>" when modality.json exists, else the raw
    lerobot feature name (e.g. "observation.images.robot0_eye_in_hand")
  - parquet vectors:  raw column names (e.g. "observation.state", "action")
  - modality slices:  "state.<name>" / "action.<name>" when modality.json exists

No dependency on lerobot / robocasa / GR00T — only pyarrow (or pandas) for
parquet and the shared decoder compat layer (torchcodec or PyAV) for video.

Primarily a *conversion source* (sequential iter_episodes); random access works
but per-frame parquet reads go through a small per-episode LRU cache.
"""

import json
import os
from collections import OrderedDict
from typing import Dict, List, Optional, Tuple

import numpy as np

from .base import TrajectoryStore
from .video import _DECODER_CLS, _LRUDecoderCache, _pick_decoder_backend

_DEFAULT_MAX_DECODERS = 32
_EPISODE_CACHE_SIZE = 4  # parquet episodes kept in RAM per worker


class LeRobotStore(TrajectoryStore):
    """Read-only flat-frame view over a LeRobot v2 dataset."""

    def __init__(self, path: str, max_decoders: int = _DEFAULT_MAX_DECODERS):
        self.path = path
        meta_dir = os.path.join(path, "meta")

        with open(os.path.join(meta_dir, "info.json")) as f:
            self._info = json.load(f)

        self._episode_lengths: List[int] = []
        with open(os.path.join(meta_dir, "episodes.jsonl")) as f:
            episodes = [json.loads(line) for line in f if line.strip()]
        episodes.sort(key=lambda e: e["episode_index"])
        self._episode_lengths = [int(e["length"]) for e in episodes]
        self._episode_ends = np.cumsum(self._episode_lengths).astype(np.int64)

        modality_path = os.path.join(meta_dir, "modality.json")
        self._modality: Optional[dict] = None
        if os.path.exists(modality_path):
            with open(modality_path) as f:
                self._modality = json.load(f)

        self._chunks_size = int(self._info.get("chunks_size", 1000))
        self._data_path_tpl: str = self._info["data_path"]
        self._video_path_tpl: str = self._info.get("video_path", "")

        features = self._info["features"]
        # key → (lerobot feature name) for videos
        self._video_features = {
            name: attr for name, attr in features.items()
            if attr.get("dtype") == "video"
        }
        self._parquet_features = {
            name: attr for name, attr in features.items()
            if attr.get("dtype") not in ("video", "image")
        }

        self._shapes: Dict[str, tuple] = {}
        self._dtypes: Dict[str, np.dtype] = {}
        # video key → lerobot feature name
        self._video_key_map: Dict[str, str] = {}
        # parquet key → (column, start, end) — end=None means full column
        self._parquet_key_map: Dict[str, Tuple[str, int, Optional[int]]] = {}

        if self._modality is not None:
            for name, spec in self._modality.get("video", {}).items():
                feat = spec["original_key"]
                if feat not in self._video_features:
                    continue
                key = f"video.{name}"
                self._video_key_map[key] = feat
                h, w, c = self._video_shape(self._video_features[feat])
                self._shapes[key] = (h, w, c)
                self._dtypes[key] = np.dtype(np.uint8)
            for group in ("state", "action"):
                for name, spec in self._modality.get(group, {}).items():
                    col = spec["original_key"]
                    if col not in self._parquet_features:
                        continue
                    key = f"{group}.{name}"
                    start, end = int(spec["start"]), int(spec["end"])
                    self._parquet_key_map[key] = (col, start, end)
                    self._shapes[key] = (end - start,)
                    self._dtypes[key] = np.dtype(np.float32)
        else:
            for feat, attr in self._video_features.items():
                self._video_key_map[feat] = feat
                h, w, c = self._video_shape(attr)
                self._shapes[feat] = (h, w, c)
                self._dtypes[feat] = np.dtype(np.uint8)

        # Full parquet vector columns are always exposed (skip scalar
        # bookkeeping columns like episode_index / frame_index / timestamp).
        for col, attr in self._parquet_features.items():
            shape = tuple(attr.get("shape") or ())
            if len(shape) != 1 or col in ("timestamp",) or col.endswith("_index"):
                continue
            self._parquet_key_map[col] = (col, 0, None)
            self._shapes[col] = shape
            self._dtypes[col] = np.dtype(np.float32)

        self._keys = list(self._video_key_map) + list(self._parquet_key_map)

        # Best-effort schemas from modality.json: each named span of a full
        # parquet vector column becomes a field (kind generic — the modality
        # file does not carry representations).
        from ..schema import Field, VectorSchema
        self._schemas: Dict[str, VectorSchema] = {}
        if self._modality is not None:
            for group, col_default in (("state", "observation.state"), ("action", "action")):
                spans = [
                    (name, spec) for name, spec in self._modality.get(group, {}).items()
                    if spec["original_key"] in self._parquet_key_map
                ]
                spans.sort(key=lambda x: int(x[1]["start"]))
                if spans:
                    col = spans[0][1]["original_key"]
                    fields = tuple(
                        Field(name, int(s["end"]) - int(s["start"]), "generic")
                        for name, s in spans
                    )
                    schema = VectorSchema(fields)
                    col_dim = self._shapes.get(col, (None,))[0]
                    if col_dim == schema.dim:
                        self._schemas[col] = schema

        self._decoder_cache = _LRUDecoderCache(maxsize=max_decoders)
        self._episode_cache: OrderedDict = OrderedDict()  # ep_idx → {col: ndarray}

    @property
    def schemas(self):
        return self._schemas

    @staticmethod
    def _video_shape(attr: dict) -> Tuple[int, int, int]:
        shape = list(attr["shape"])
        names = attr.get("names") or []
        # LeRobot stores either (H, W, C) or (C, H, W) depending on exporter.
        if len(shape) == 3 and (names[:1] == ["channels"] or shape[0] in (1, 3, 4)) \
                and shape[0] < shape[1] and shape[0] < shape[2]:
            c, h, w = shape
        else:
            h, w, c = shape
        return int(h), int(w), int(c)

    # ── TrajectoryStore interface ─────────────────────────────────────────────

    @property
    def episode_ends(self) -> np.ndarray:
        return self._episode_ends

    @property
    def keys(self) -> List[str]:
        return self._keys

    def frame_shape(self, key: str) -> tuple:
        return self._shapes[key]

    def frame_dtype(self, key: str) -> np.dtype:
        return self._dtypes[key]

    def on_worker_init(self, worker_id: int) -> None:
        self._decoder_cache.clear()
        self._episode_cache.clear()

    # ── Path resolution ───────────────────────────────────────────────────────

    def _episode_chunk(self, ep_idx: int) -> int:
        return ep_idx // self._chunks_size

    def _parquet_path(self, ep_idx: int) -> str:
        rel = self._data_path_tpl.format(
            episode_chunk=self._episode_chunk(ep_idx), episode_index=ep_idx
        )
        return os.path.join(self.path, rel)

    def _video_path(self, ep_idx: int, feat: str) -> str:
        rel = self._video_path_tpl.format(
            episode_chunk=self._episode_chunk(ep_idx),
            episode_index=ep_idx,
            video_key=feat,
        )
        return os.path.join(self.path, rel)

    # ── Parquet reads ─────────────────────────────────────────────────────────

    def _episode_table(self, ep_idx: int) -> Dict[str, np.ndarray]:
        cached = self._episode_cache.get(ep_idx)
        if cached is not None:
            self._episode_cache.move_to_end(ep_idx)
            return cached
        import pyarrow.parquet as pq
        table = pq.read_table(self._parquet_path(ep_idx))
        data: Dict[str, np.ndarray] = {}
        for col, _, _ in set(self._parquet_key_map.values()):
            if col not in table.column_names:
                continue
            cells = table.column(col).to_pylist()
            data[col] = np.asarray(cells, dtype=np.float32)
        if len(self._episode_cache) >= _EPISODE_CACHE_SIZE:
            self._episode_cache.popitem(last=False)
        self._episode_cache[ep_idx] = data
        return data

    # ── Video reads ───────────────────────────────────────────────────────────

    def _decode_frames(self, key: str, ep_idx: int, local_indices: List[int]) -> np.ndarray:
        cache_key = (key, ep_idx)
        dec = self._decoder_cache.get(cache_key)
        if dec is None:
            feat = self._video_key_map[key]
            dec = _DECODER_CLS[_pick_decoder_backend()](self._video_path(ep_idx, feat))
            self._decoder_cache.put(cache_key, dec)
        return dec.get_frames(local_indices)

    # ── get_sequence ──────────────────────────────────────────────────────────

    def get_sequence(self, key: str, start: int, end: int) -> np.ndarray:
        ep_first = int(np.searchsorted(self._episode_ends, start, side="right"))
        ep_last = int(np.searchsorted(self._episode_ends, end - 1, side="right"))

        parts = []
        for ep_idx in range(ep_first, ep_last + 1):
            ep_offset = int(self._episode_ends[ep_idx - 1]) if ep_idx > 0 else 0
            ep_end_global = int(self._episode_ends[ep_idx])
            lo = max(start, ep_offset) - ep_offset
            hi = min(end, ep_end_global) - ep_offset
            if key in self._video_key_map:
                parts.append(self._decode_frames(key, ep_idx, list(range(lo, hi))))
            else:
                col, s, e = self._parquet_key_map[key]
                arr = self._episode_table(ep_idx)[col][lo:hi]
                parts.append(arr if e is None else arr[:, s:e])
        return np.concatenate(parts, axis=0) if len(parts) > 1 else parts[0]
