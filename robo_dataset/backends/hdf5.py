"""Read-only Robomimic HDF5 backend.

Robomimic HDF5 layout:
  data/demo_{i}/obs/{key}        (T_i, H, W, C) uint8 for images
  data/demo_{i}/obs/{key}        (T_i, D) float32 for lowdim
  data/demo_{i}/actions          (T_i, action_dim) float32

Episodes are stored separately; this backend stitches them into a flat view.
"""

import os
from typing import Dict, List, Optional

import h5py
import numpy as np

from .base import TrajectoryStore, iter_episode_slices


class RobomimicHDF5Store(TrajectoryStore):
    """Read-only flat-frame view over a Robomimic HDF5 dataset.

    Pros
    ----
    - Zero setup: train directly from the original Robomimic HDF5 without conversion.
    - Cross-episode sequences stitched automatically via searchsorted + multi-read.
    - Supports arbitrary h5py compression codecs (depends on how the file was created).

    Cons
    ----
    - Not fork-safe: h5py file handles must be reopened per DataLoader worker
      (handled by on_worker_init, but adds latency on first access after fork).
    - No vectorized slice: a single get_sequence call spanning E episodes issues E
      separate HDF5 reads, each with its own decompression step.
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

        # Build metadata: episode boundaries, shapes, dtypes
        with h5py.File(path, "r") as f:
            demos = f["data"]
            n_demos = len(demos)

            ends = []
            cumsum = 0
            for i in range(n_demos):
                T = demos[f"demo_{i}"]["actions"].shape[0]
                cumsum += T
                ends.append(cumsum)
            self._episode_ends = np.array(ends, dtype=np.int64)

            demo0 = demos["demo_0"]
            self._shapes: Dict[str, tuple] = {}
            self._dtypes: Dict[str, np.dtype] = {}
            for key in obs_keys:
                arr = demo0["obs"][key]
                self._shapes[key] = arr.shape[1:]
                self._dtypes[key] = np.dtype(arr.dtype)
            action_shape = demo0["actions"].shape[1:]
            self._shapes["action"] = action_shape
            self._dtypes["action"] = np.float32

        # Flat-to-episode mapping: for each flat frame i, which episode + intra index
        self._ep_starts = np.concatenate([[0], self._episode_ends[:-1]])

        # File handle (opened lazily per worker)
        self._h5file: Optional[h5py.File] = None
        self._h5file_pid: Optional[int] = None
        self._worker_id: Optional[int] = None

    @property
    def episode_ends(self) -> np.ndarray:
        return self._episode_ends

    @property
    def keys(self) -> List[str]:
        return self.obs_keys + ["action"]

    def frame_shape(self, key: str) -> tuple:
        return self._shapes[key]

    def frame_dtype(self, key: str) -> np.dtype:
        return self._dtypes[key]

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

    def get_sequence(self, key: str, start: int, end: int) -> np.ndarray:
        """Fetch frames[start:end] across episode boundaries."""
        f = self._file()
        demos = f["data"]

        parts = []
        for ep_idx, lo, hi in iter_episode_slices(self._episode_ends, start, end):
            demo = demos[f"demo_{ep_idx}"]
            if key == "action":
                arr = demo["actions"][lo:hi].astype(np.float32)
                if self.abs_action:
                    arr = _convert_actions(arr, self.rotation_rep)
            else:
                arr = demo["obs"][key][lo:hi]  # h5py slicing yields an ndarray
            parts.append(arr)

        return np.concatenate(parts, axis=0) if len(parts) > 1 else parts[0]

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
