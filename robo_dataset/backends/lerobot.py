"""Read-only LeRobot v2 / v3 backend (parquet + MP4).

LeRobot v2 layout:
  <root>/meta/info.json        features, fps, data_path / video_path templates
  <root>/meta/episodes.jsonl   one line per episode: {"episode_index", "length", ...}
  <root>/meta/modality.json    (optional, GR00T/robocasa) named slices into the
                               concatenated state/action vectors and video key map
  <root>/data/chunk-XXX/episode_XXXXXX.parquet
  <root>/videos/chunk-XXX/<video_key>/episode_XXXXXX.mp4

LeRobot v3 layout (auto-detected) packs many episodes per file instead:
  <root>/meta/episodes/chunk-XXX/file-XXXXX.parquet   one ROW per episode, with
      episode_index, length, data/{chunk,file}_index, dataset_{from,to}_index,
      videos/<key>/{chunk,file}_index, videos/<key>/{from,to}_timestamp
  <root>/data/chunk-XXX/file-XXXXX.parquet            episodes concatenated
  <root>/videos/<video_key>/chunk-XXX/file-XXXXX.mp4  episodes concatenated;
      an episode starts at round(from_timestamp * fps) within the file.
The two differ only in how an episode is addressed, so the v3 support here is
confined to episode metadata, path resolution, and a frame/row offset.

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

from .base import TrajectoryStore, iter_episode_slices
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

        # v2 keeps one JSON line per episode; v3 replaced that with a parquet
        # table under meta/episodes/ (one row per episode).
        episodes_jsonl = os.path.join(meta_dir, "episodes.jsonl")
        episodes_dir = os.path.join(meta_dir, "episodes")
        self._v3 = not os.path.exists(episodes_jsonl) and os.path.isdir(episodes_dir)
        if self._v3:
            self._ep_meta = self._read_v3_episode_meta(episodes_dir)
        else:
            with open(episodes_jsonl) as f:
                episodes = [json.loads(line) for line in f if line.strip()]
            episodes.sort(key=lambda e: e["episode_index"])
            self._ep_meta = episodes
        self._episode_lengths: List[int] = [int(e["length"]) for e in self._ep_meta]
        self._episode_ends = np.cumsum(self._episode_lengths).astype(np.int64)
        self._fps = float(self._info.get("fps") or 0.0)
        if self._v3 and not self._fps:
            raise ValueError(
                f"{path}: LeRobot v3 needs 'fps' in meta/info.json to locate an "
                "episode's first frame inside a packed video file"
            )
        # v3 only: (path -> {column: ndarray}) for the packed parquet currently in
        # use. Episodes are read in order, so a single slot avoids re-reading a
        # multi-hundred-megabyte file once per episode.
        self._v3_file_cache: Optional[Tuple[str, Dict[str, np.ndarray]]] = None

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
            bad_entries: List[str] = []
            for name, spec in self._modality.get("video", {}).items():
                feat = spec["original_key"]
                if feat not in self._video_features:
                    bad_entries.append(f"video.{name} → {feat!r}")
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
                        bad_entries.append(f"{group}.{name} → {col!r}")
                        continue
                    key = f"{group}.{name}"
                    start, end = int(spec["start"]), int(spec["end"])
                    self._parquet_key_map[key] = (col, start, end)
                    self._shapes[key] = (end - start,)
                    self._dtypes[key] = np.dtype(np.float32)
            if bad_entries:
                raise ValueError(
                    f"modality.json at {modality_path!r} references features "
                    f"absent from info.json features: {bad_entries}"
                )
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
            if len(shape) != 1 or col in ("timestamp", "index") or col.endswith("_index"):
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
        self._cache_pid = os.getpid()

    def _check_fork(self) -> None:
        """Self-defending fork safety: drop caches inherited across fork."""
        if os.getpid() != self._cache_pid:
            self._decoder_cache.clear()
            self._episode_cache.clear()
            self._cache_pid = os.getpid()

    @property
    def schemas(self):
        return self._schemas

    @staticmethod
    def _missing_cols(table, cols, parquet_path: str) -> None:
        missing = [c for c in cols if c not in table.column_names]
        if missing:
            raise KeyError(
                f"Columns {missing!r} (mapped from store keys) are missing from "
                f"parquet file {parquet_path!r}"
            )

    @staticmethod
    def _read_v3_episode_meta(episodes_dir: str) -> List[dict]:
        """One dict per episode from meta/episodes/**.parquet, ordered by index."""
        import glob as _glob

        import pyarrow.parquet as pq

        files = sorted(_glob.glob(os.path.join(episodes_dir, "**", "*.parquet"), recursive=True))
        if not files:
            raise FileNotFoundError(f"no episode metadata parquet under {episodes_dir!r}")
        rows: List[dict] = []
        for f in files:
            rows.extend(pq.read_table(f).to_pylist())
        rows.sort(key=lambda e: int(e["episode_index"]))
        return rows

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
        self._cache_pid = os.getpid()

    # ── Path resolution ───────────────────────────────────────────────────────

    def _episode_chunk(self, ep_idx: int) -> int:
        return ep_idx // self._chunks_size

    def _parquet_path(self, ep_idx: int) -> str:
        if self._v3:
            row = self._ep_meta[ep_idx]
            rel = self._data_path_tpl.format(
                chunk_index=int(row["data/chunk_index"]),
                file_index=int(row["data/file_index"]),
            )
            return os.path.join(self.path, rel)
        rel = self._data_path_tpl.format(
            episode_chunk=self._episode_chunk(ep_idx), episode_index=ep_idx
        )
        return os.path.join(self.path, rel)

    def _video_path(self, ep_idx: int, feat: str) -> str:
        if self._v3:
            row = self._ep_meta[ep_idx]
            rel = self._video_path_tpl.format(
                video_key=feat,
                chunk_index=int(row[f"videos/{feat}/chunk_index"]),
                file_index=int(row[f"videos/{feat}/file_index"]),
            )
            return os.path.join(self.path, rel)
        rel = self._video_path_tpl.format(
            episode_chunk=self._episode_chunk(ep_idx),
            episode_index=ep_idx,
            video_key=feat,
        )
        return os.path.join(self.path, rel)

    def _video_frame_offset(self, ep_idx: int, feat: str) -> int:
        """Index of the episode's first frame inside its (v3-packed) video file."""
        if not self._v3:
            return 0
        return int(round(float(self._ep_meta[ep_idx][f"videos/{feat}/from_timestamp"]) * self._fps))

    # ── Parquet reads ─────────────────────────────────────────────────────────

    def _episode_table(self, ep_idx: int) -> Dict[str, np.ndarray]:
        self._check_fork()
        cached = self._episode_cache.get(ep_idx)
        if cached is not None:
            self._episode_cache.move_to_end(ep_idx)
            return cached
        import pyarrow.parquet as pq
        parquet_path = self._parquet_path(ep_idx)
        cols = {col for col, _, _ in self._parquet_key_map.values()}
        if self._v3:
            # The file holds many episodes back to back; read it once, then take
            # this episode's contiguous row range.
            if self._v3_file_cache is None or self._v3_file_cache[0] != parquet_path:
                table = pq.read_table(parquet_path)
                self._missing_cols(table, cols, parquet_path)
                cached = {c: np.asarray(table.column(c).to_pylist(), dtype=np.float32)
                          for c in cols}
                cached["__episode_index__"] = np.asarray(
                    table.column("episode_index").to_pylist()
                ).reshape(-1)
                self._v3_file_cache = (parquet_path, cached)
            file_cols = self._v3_file_cache[1]
            rows = np.flatnonzero(file_cols["__episode_index__"] == ep_idx)
            if rows.size != self._episode_lengths[ep_idx]:
                raise ValueError(
                    f"{parquet_path!r}: episode {ep_idx} has {rows.size} rows but "
                    f"metadata says {self._episode_lengths[ep_idx]}"
                )
            lo, hi = int(rows[0]), int(rows[-1]) + 1
            data = {c: file_cols[c][lo:hi] for c in cols}
        else:
            table = pq.read_table(parquet_path)
            self._missing_cols(table, cols, parquet_path)
            data = {c: np.asarray(table.column(c).to_pylist(), dtype=np.float32) for c in cols}
        if len(self._episode_cache) >= _EPISODE_CACHE_SIZE:
            self._episode_cache.popitem(last=False)
        self._episode_cache[ep_idx] = data
        return data

    # ── Video reads ───────────────────────────────────────────────────────────

    def _decode_frames(self, key: str, ep_idx: int, local_indices: List[int]) -> np.ndarray:
        self._check_fork()
        feat = self._video_key_map[key]
        video_path = self._video_path(ep_idx, feat)
        # Keyed per (key, episode) even under v3, where many episodes share one
        # file: callers decode several episodes CONCURRENTLY (rgrl's
        # EpisodeStore.load fans out over ~16 threads), and the decoder cache is
        # only documented thread-safe when each episode has its own decoder.
        # Sharing one per file segfaults every worker. Episodes on the same file
        # therefore open separate decoders over it, bounded by max_decoders.
        cache_key = (key, ep_idx)
        dec = self._decoder_cache.get(cache_key)
        if dec is None:
            dec = _DECODER_CLS[_pick_decoder_backend()](video_path)
            self._decoder_cache.put(cache_key, dec)
        offset = self._video_frame_offset(ep_idx, feat)
        if offset:
            local_indices = [i + offset for i in local_indices]
        return dec.get_frames(local_indices)

    # ── get_sequence ──────────────────────────────────────────────────────────

    def get_sequence(self, key: str, start: int, end: int) -> np.ndarray:
        parts = []
        for ep_idx, lo, hi in iter_episode_slices(self._episode_ends, start, end):
            if key in self._video_key_map:
                parts.append(self._decode_frames(key, ep_idx, list(range(lo, hi))))
            else:
                col, s, e = self._parquet_key_map[key]
                arr = self._episode_table(ep_idx)[col][lo:hi]
                parts.append(arr if e is None else arr[:, s:e])
        return np.concatenate(parts, axis=0) if len(parts) > 1 else parts[0]
