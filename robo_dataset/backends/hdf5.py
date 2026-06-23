"""Read-only Robomimic HDF5 backend.

Robomimic HDF5 layout:
  data/demo_{i}/obs/{key}        (T_i, H, W, C) uint8 for images
  data/demo_{i}/obs/{key}        (T_i, D) float32 for lowdim
  data/demo_{i}/actions          (T_i, action_dim) float32

Episodes are stored separately; no flat stitching required for the new
EpisodeStore interface.  get_sequence() is still available (derived).
"""

import os
from typing import Dict, List, Optional

import h5py
import numpy as np

from .base import EpisodeStore


class RobomimicHDF5Store(EpisodeStore):
    """Read-only episode-indexed view over a Robomimic HDF5 dataset.

    Pros
    ----
    - Zero setup: train directly from the original Robomimic HDF5 without conversion.
    - Episode-indexed: get_frames() reads from data/demo_{ep_idx}/... directly.
    - get_sequence() works via the derived EpisodeStore implementation.
    - Supports arbitrary h5py compression codecs (depends on how the file was created).

    Cons
    ----
    - Not fork-safe: h5py file handles must be reopened per DataLoader worker
      (handled by on_worker_init, but adds latency on first access after fork).
    - No vectorized slice: a get_frames call issues one HDF5 read per call,
      each with its own decompression step.
    - Images are typically stored uncompressed in Robomimic files → large disk footprint.
    - Read-only: no write path (use ZarrStoreWriter or RLDSFrameStoreWriter to convert).
    """

    def __init__(
        self,
        path: str,
        obs_keys: List[str],
        abs_action: bool = False,
        rotation_rep: str = "axis_angle",
    ):
        self.path = path
        self.obs_keys = list(obs_keys)
        self.abs_action = abs_action
        self.rotation_rep = rotation_rep

        # Build metadata: episode lengths, shapes, dtypes
        with h5py.File(path, "r") as f:
            demos = f["data"]
            self._n_episodes = len(demos)

            self._ep_lengths: List[int] = []
            for i in range(self._n_episodes):
                T = demos[f"demo_{i}"]["actions"].shape[0]
                self._ep_lengths.append(int(T))

            demo0 = demos["demo_0"]
            self._shapes: Dict[str, tuple] = {}
            self._dtypes: Dict[str, np.dtype] = {}
            for key in obs_keys:
                arr = demo0["obs"][key]
                self._shapes[key] = arr.shape[1:]
                self._dtypes[key] = np.dtype(arr.dtype)
            action_shape = demo0["actions"].shape[1:]
            self._shapes["action"] = action_shape
            self._dtypes["action"] = np.dtype(np.float32)

        # File handle (opened lazily per worker)
        self._h5file: Optional[h5py.File] = None
        self._h5file_pid: Optional[int] = None
        self._worker_id: Optional[int] = None

    # ── EpisodeStore primitives ───────────────────────────────────────────────

    @property
    def n_episodes(self) -> int:
        return self._n_episodes

    @property
    def keys(self) -> List[str]:
        return self.obs_keys + ["action"]

    def episode_length(self, ep_idx: int) -> int:
        return self._ep_lengths[ep_idx]

    def frame_shape(self, key: str) -> tuple:
        return self._shapes[key]

    def frame_dtype(self, key: str) -> np.dtype:
        return self._dtypes[key]

    def get_frames(self, ep_idx: int, key: str, start: int, end: int) -> np.ndarray:
        """Read frames [start, end) from episode ep_idx for a single key."""
        f = self._file()
        demo = f["data"][f"demo_{ep_idx}"]
        if key == "action":
            arr = demo["actions"][start:end].astype(np.float32)
            if self.abs_action:
                arr = _convert_actions(arr, self.rotation_rep)
        else:
            arr = demo["obs"][key][start:end]
        return arr

    # ── Fork safety ───────────────────────────────────────────────────────────

    def on_worker_init(self, worker_id: int) -> None:
        if self._h5file is not None:
            self._h5file.close()
        self._h5file = h5py.File(self.path, "r")
        self._h5file_pid = os.getpid()
        self._worker_id = worker_id

    def _file(self) -> h5py.File:
        # Self-defending fork safety: an h5py handle inherited across fork is
        # unsafe to use; reopen if the pid changed since the handle was opened.
        if self._h5file is not None and self._h5file_pid != os.getpid():
            self._h5file = None
        if self._h5file is None:
            self._h5file = h5py.File(self.path, "r")
            self._h5file_pid = os.getpid()
        return self._h5file

    def __del__(self):
        if self._h5file is not None:
            try:
                self._h5file.close()
            except Exception:
                pass


def _convert_actions(raw: np.ndarray, rotation_rep: str) -> np.ndarray:
    """Convert axis_angle actions to rotation_rep."""
    if rotation_rep == "axis_angle":
        return raw

    from robo_dataset.transforms.rotation_transformer import RotationTransformer
    rt = RotationTransformer(from_rep="axis_angle", to_rep=rotation_rep)

    is_dual_arm = raw.shape[-1] == 14
    if is_dual_arm:
        raw = raw.reshape(-1, 2, 7)

    pos = raw[..., :3]
    rot = raw[..., 3:6]
    gripper = raw[..., 6:]
    rot = rt.forward(rot)
    out = np.concatenate([pos, rot, gripper], axis=-1).astype(np.float32)
    if is_dual_arm:
        out = out.reshape(-1, out.shape[-1] * 2)
    return out
