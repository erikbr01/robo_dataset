"""Read-only DROID backend — raw DROID session folders as a TrajectoryStore.

A DROID dataset root contains session folders (possibly nested, e.g.
``success/<date>/<timestamp>/``), each with:

  trajectory.h5                  robot states + actions (per raw step)
  recordings/MP4/<serial>.mp4    per-camera videos  (optional)
  recordings/SVO/<serial>.svo*   ZED native recordings (optional)

Episodes are the session folders, sorted by path. Raw steps where the
controller had movement disabled are dropped using the
``observation/controller_info/movement_enabled`` mask — camera frame i
corresponds to raw step i, so the same mask aligns SVO/MP4 frames with the
kept robot states (verified equivalent to the official droid loader's
``remove_skipped_steps``).

Exposed keys:
  joint_positions (T, 7) · cartesian_position (T, 6) · gripper_position (T, 1)
  action (T, 8): <action_space> arm action (7) + gripper_position (1)
  reward (T, 1): success flag on the last step (from trajectory.h5 attrs)
  + one (T, H, W, 3) uint8 key per requested camera

Cameras:
  cameras={"hand_camera_image": "14313307"} maps obs keys to ZED serials.
  camera_source picks the decode path:
    "svo"   decode recordings/SVO/<serial>.svo* via pyzed (needs the ZED SDK
            shared libs on LD_LIBRARY_PATH)
    "mp4"   decode recordings/MP4 via the official ``droid`` package loader
            (lazy import; pass droid_package_path if not installed)
    "none"  lowdim only (default)

Camera decode failures and frame-count mismatches raise — a session with
broken recordings must be excluded explicitly, never silently dropped or
zero-filled.

Random access loads whole episodes through a small per-worker LRU cache —
fine for conversion (sequential) and clustering; for high-throughput training
convert to a VideoStore first.
"""

import logging
import os
import sys
from collections import OrderedDict
from typing import Dict, Iterator, List, Optional

import numpy as np

from .base import TrajectoryStore, iter_episode_slices

_log = logging.getLogger(__name__)

_EPISODE_CACHE_SIZE = 2
_LOWDIM_KEYS = {
    "joint_positions": (7,),
    "cartesian_position": (6,),
    "gripper_position": (1,),
}
_MOVEMENT_KEY = "observation/controller_info/movement_enabled"


class DroidStore(TrajectoryStore):
    """Flat-frame view over a tree of raw DROID session folders."""

    def __init__(
        self,
        path: str,
        cameras: Optional[Dict[str, str]] = None,
        camera_source: str = "none",   # "none" | "svo" | "mp4"
        image_size: tuple = (180, 320),
        action_space: str = "joint_velocity",
        max_episodes: Optional[int] = None,
        droid_package_path: str = "~/src_droid",
        zed_settings_path: Optional[str] = None,
        skip_empty: bool = False,
        skip_corrupt: bool = False,
        session_offset: int = 0,
    ):
        if camera_source not in ("none", "svo", "mp4"):
            raise ValueError(f"camera_source must be none|svo|mp4, got {camera_source!r}")
        if camera_source != "none" and not cameras:
            raise ValueError("cameras={obs_key: zed_serial} required with camera_source")

        self.path = path
        self.cameras = dict(cameras or {})
        self.camera_source = camera_source
        self.image_h, self.image_w = image_size
        self.action_space = action_space
        self._droid_package_path = droid_package_path
        # Dir with per-camera ZED calibration files (SN<serial>.conf); without
        # it, offline SVO decode fails with CALIBRATION FILE NOT AVAILABLE.
        self.zed_settings_path = zed_settings_path
        self._skip_empty = skip_empty
        self._skip_corrupt = skip_corrupt
        self._session_offset = session_offset

        # Discover session folders (any dir containing trajectory.h5).
        sessions: List[str] = []
        for root, _dirs, files in sorted(os.walk(path, followlinks=True)):
            if "trajectory.h5" in files:
                sessions.append(root)
        sessions.sort()
        if session_offset:
            sessions = sessions[session_offset:]
        if max_episodes is not None:
            sessions = sessions[:max_episodes]
        if not sessions:
            raise ValueError(f"No trajectory.h5 sessions found under {path!r}")

        # Episode lengths from the movement mask — no camera decode needed.
        import h5py
        lengths = []
        kept_sessions = []
        for s in sessions:
            with h5py.File(os.path.join(s, "trajectory.h5"), "r") as f:
                if _MOVEMENT_KEY in f:
                    T = int(f[_MOVEMENT_KEY][:].astype(bool).sum())
                else:
                    T = int(f[f"action/{self.action_space}"].shape[0])
            if T == 0:
                if self._skip_empty:
                    _log.warning("skipping zero-step session: %s", s)
                    continue
                raise ValueError(
                    f"Session {s!r} has zero kept steps — exclude it explicitly."
                )
            lengths.append(T)
            kept_sessions.append(s)
        self.sessions = kept_sessions
        self._episode_ends = np.cumsum(lengths).astype(np.int64)

        self._shapes: Dict[str, tuple] = dict(_LOWDIM_KEYS)
        self._shapes["action"] = (8,)
        self._shapes["reward"] = (1,)
        self._dtypes: Dict[str, np.dtype] = {
            k: np.dtype(np.float32) for k in self._shapes
        }
        if camera_source != "none":
            for key in self.cameras:
                self._shapes[key] = (self.image_h, self.image_w, 3)
                self._dtypes[key] = np.dtype(np.uint8)

        self._keys = list(self._shapes)
        self._episode_cache: OrderedDict = OrderedDict()

        # DROID layouts are fixed — describe them so converted stores are
        # self-describing (see robo_dataset.schema).
        from ..schema import Field, VectorSchema
        arm_kind, arm_rep = (
            ("joints", "velocity") if "velocity" in self.action_space else
            ("joints", "abs") if "joint" in self.action_space else
            ("generic", None)
        )
        self._schemas = {
            "action": VectorSchema((
                Field(self.action_space, 7, arm_kind, arm_rep),
                Field("gripper", 1, "gripper", "continuous"),
            )),
            "joint_positions": VectorSchema((Field("joints", 7, "joints", "abs"),)),
            # DROID cartesian_position is xyz + euler
            "cartesian_position": VectorSchema((
                Field("pos", 3, "pos", "abs"),
                Field("rot", 3, "rot", "euler"),
            )),
            "gripper_position": VectorSchema((Field("gripper", 1, "gripper", "continuous"),)),
        }

    @property
    def schemas(self):
        return self._schemas

    # ── lowdim (direct trajectory.h5 read, movement-mask filtered) ───────────

    def _load_lowdim(self, session: str) -> Dict[str, np.ndarray]:
        import h5py

        with h5py.File(os.path.join(session, "trajectory.h5"), "r") as f:
            n_raw = f[f"action/{self.action_space}"].shape[0]
            if _MOVEMENT_KEY in f:
                mask = f[_MOVEMENT_KEY][:].astype(bool)
                if len(mask) != n_raw:
                    raise ValueError(
                        f"{session!r}: movement mask length {len(mask)} != "
                        f"raw steps {n_raw}"
                    )
            else:
                mask = np.ones(n_raw, dtype=bool)

            arm = f[f"action/{self.action_space}"][:][mask].astype(np.float32)
            grip = f["action/gripper_position"][:][mask].astype(np.float32)
            if grip.ndim == 1:
                grip = grip[:, None]
            data: Dict[str, np.ndarray] = {
                "action": np.concatenate([arm, grip], axis=-1),
                "joint_positions": f["observation/robot_state/joint_positions"][:][mask].astype(np.float32),
                "cartesian_position": f["observation/robot_state/cartesian_position"][:][mask].astype(np.float32),
            }
            gp = f["observation/robot_state/gripper_position"][:][mask].astype(np.float32)
            data["gripper_position"] = gp[:, None] if gp.ndim == 1 else gp

            success = bool(f.attrs.get("success", False))

        T = data["action"].shape[0]
        reward = np.zeros((T, 1), dtype=np.float32)
        reward[-1, 0] = float(success)
        data["reward"] = reward
        data["_mask"] = mask
        return data

    # ── camera decode ─────────────────────────────────────────────────────────

    def _read_svo_frames(self, session: str, serial: str, valid_mask: np.ndarray) -> np.ndarray:
        """Decode left-view frames from recordings/SVO/<serial>.svo*, keeping
        frames where valid_mask is True (frame i ↔ raw step i)."""
        try:
            import pyzed.sl as sl
        except ImportError as e:
            raise ImportError(
                "camera_source='svo' needs pyzed + the ZED SDK shared libs on "
                "LD_LIBRARY_PATH (e.g. LD_LIBRARY_PATH=<zed_sdk>/lib:<env>/lib)."
            ) from e
        import cv2

        svo_dir = os.path.join(session, "recordings", "SVO")
        svo_path = None
        if os.path.isdir(svo_dir):
            for f in sorted(os.listdir(svo_dir)):
                if f.startswith(str(serial)) and ".svo" in f:
                    svo_path = os.path.join(svo_dir, f)
                    break
        if svo_path is None:
            raise FileNotFoundError(
                f"No SVO recording for serial {serial!r} under {svo_dir!r}"
            )

        init = sl.InitParameters()
        init.set_from_svo_file(svo_path)
        init.svo_real_time_mode = False
        if self.zed_settings_path:
            init.optional_settings_path = self.zed_settings_path
        cam = sl.Camera()
        status = cam.open(init)
        if status != sl.ERROR_CODE.SUCCESS:
            hint = (
                " (pass zed_settings_path=<dir with SN<serial>.conf>)"
                if "CALIBRATION" in str(status) else ""
            )
            raise RuntimeError(f"SVO open failed ({status}): {svo_path}{hint}")

        rt = sl.RuntimeParameters()
        mat = sl.Mat()
        frames = []
        frame_idx = 0
        while True:
            status = cam.grab(rt)
            if status == sl.ERROR_CODE.END_OF_SVOFILE_REACHED:
                break
            if status != sl.ERROR_CODE.SUCCESS:
                cam.close()
                raise RuntimeError(f"SVO grab failed ({status}) at frame {frame_idx}: {svo_path}")
            if frame_idx < len(valid_mask) and valid_mask[frame_idx]:
                cam.retrieve_image(mat, sl.VIEW.LEFT)
                img = cv2.cvtColor(mat.get_data(), cv2.COLOR_BGRA2RGB)
                if img.shape[:2] != (self.image_h, self.image_w):
                    img = cv2.resize(img, (self.image_w, self.image_h),
                                     interpolation=cv2.INTER_AREA)
                frames.append(img)
            frame_idx += 1
        cam.close()

        n_expected = int(valid_mask.sum())
        if len(frames) != n_expected:
            raise ValueError(
                f"{svo_path}: decoded {len(frames)} masked frames, expected "
                f"{n_expected} (raw SVO frames seen: {frame_idx}, "
                f"mask length: {len(valid_mask)})"
            )
        return np.stack(frames).astype(np.uint8)

    def _read_mp4_frames(self, session: str, T: int) -> Dict[str, np.ndarray]:
        """Decode all requested cameras via the official droid loader."""
        try:
            import droid  # noqa: F401
        except ImportError:
            pkg = os.path.expanduser(self._droid_package_path)
            if os.path.isdir(pkg) and pkg not in sys.path:
                sys.path.insert(0, pkg)
        try:
            from droid.trajectory_utils.misc import load_trajectory
        except ImportError as e:
            raise ImportError(
                "camera_source='mp4' needs the official 'droid' package "
                f"(tried droid_package_path={self._droid_package_path!r})."
            ) from e
        import cv2

        mp4_dir = os.path.join(session, "recordings", "MP4")
        if not os.path.isdir(mp4_dir):
            raise FileNotFoundError(f"No recordings/MP4 dir in {session!r}")
        steps = load_trajectory(
            filepath=os.path.join(session, "trajectory.h5"),
            read_cameras=True,
            recording_folderpath=mp4_dir,
            remove_skipped_steps=True,
        )
        if len(steps) != T:
            raise ValueError(
                f"{session!r}: droid loader returned {len(steps)} steps, "
                f"movement mask kept {T}"
            )
        out: Dict[str, np.ndarray] = {}
        for obs_key, serial in self.cameras.items():
            frames = []
            for i, s in enumerate(steps):
                img = s["observation"].get("image", {}).get(f"{serial}_left")
                if img is None:
                    raise ValueError(
                        f"{session!r}: camera {serial}_left missing at step {i}"
                    )
                if img.ndim == 3 and img.shape[2] == 4:
                    img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
                elif img.ndim == 3 and img.shape[2] == 3:
                    img = img[..., ::-1].copy()  # BGR → RGB
                if img.shape[:2] != (self.image_h, self.image_w):
                    img = cv2.resize(img, (self.image_w, self.image_h),
                                     interpolation=cv2.INTER_AREA)
                frames.append(img)
            out[obs_key] = np.stack(frames).astype(np.uint8)
        return out

    # ── episode assembly ──────────────────────────────────────────────────────

    def _load_episode(self, ep_idx: int) -> Dict[str, np.ndarray]:
        cached = self._episode_cache.get(ep_idx)
        if cached is not None:
            self._episode_cache.move_to_end(ep_idx)
            return cached

        session = self.sessions[ep_idx]
        data = self._load_lowdim(session)
        mask = data.pop("_mask")
        T = data["action"].shape[0]

        if self.camera_source == "svo":
            for obs_key, serial in self.cameras.items():
                data[obs_key] = self._read_svo_frames(session, serial, mask)
        elif self.camera_source == "mp4":
            data.update(self._read_mp4_frames(session, T))

        if len(self._episode_cache) >= _EPISODE_CACHE_SIZE:
            self._episode_cache.popitem(last=False)
        self._episode_cache[ep_idx] = data
        return data

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
        self._episode_cache.clear()

    def get_sequence(self, key: str, start: int, end: int) -> np.ndarray:
        parts = [
            self._load_episode(ep_idx)[key][lo:hi]
            for ep_idx, lo, hi in iter_episode_slices(self._episode_ends, start, end)
        ]
        return np.concatenate(parts, axis=0) if len(parts) > 1 else parts[0]

    def iter_episodes(self) -> Iterator[Dict[str, np.ndarray]]:
        for ep_idx in range(self.n_episodes):
            start = self.episode_start(ep_idx)
            end = int(self.episode_ends[ep_idx])
            try:
                yield {key: self.get_sequence(key, start, end) for key in self.keys}
            except (ValueError, RuntimeError, FileNotFoundError) as exc:
                if not self._skip_corrupt:
                    raise
                _log.warning(
                    "skipping corrupt episode %d (%s): %s",
                    ep_idx, self.sessions[ep_idx], exc,
                )
