"""Read-only DROID backend — raw DROID session folders as a TrajectoryStore.

A DROID dataset root contains session folders (possibly nested, e.g.
``success/<date>/<timestamp>/``), each with:

  trajectory.h5            robot states + actions (per-step)
  recordings/MP4/<serial>.mp4   per-camera videos  (optional)
  recordings/SVO/<serial>.svo*  ZED native recordings (optional)

Episodes are the session folders, sorted by path. Step alignment
(remove_skipped_steps) is delegated to the official ``droid`` package, which
must be importable — pass ``droid_package_path`` (default ``~/src_droid``) if
it is not installed.

Exposed keys:
  joint_positions (T, 7) · cartesian_position (T, 6) · gripper_position (T, 1)
  action (T, 8): <action_space> arm action (7) + gripper_position (1)
  reward (T, 1): success flag on the last step (from trajectory.h5 attrs)
  + one (T, H, W, 3) uint8 key per requested camera

Cameras:
  cameras={"hand_camera_image": "14313307"} maps obs keys to ZED serials.
  camera_source picks the decode path: "mp4" uses the droid loader,
  "svo" decodes recordings/SVO via pyzed (requires the ZED SDK),
  "none" exposes lowdim only (default).

Random access loads whole episodes through a small per-worker LRU cache —
fine for conversion (sequential) and clustering; for high-throughput training
convert to a VideoStore first.
"""

import os
import sys
from collections import OrderedDict
from typing import Dict, List, Optional

import numpy as np

from .base import TrajectoryStore

_EPISODE_CACHE_SIZE = 2
_LOWDIM_KEYS = {
    "joint_positions": (7,),
    "cartesian_position": (6,),
    "gripper_position": (1,),
}


def _ensure_droid_importable(droid_package_path: str) -> None:
    try:
        import droid  # noqa: F401
        return
    except ImportError:
        pass
    pkg = os.path.expanduser(droid_package_path)
    if os.path.isdir(pkg) and pkg not in sys.path:
        sys.path.insert(0, pkg)
    try:
        import droid  # noqa: F401
    except ImportError as e:
        raise ImportError(
            f"The 'droid' package is required for DroidStore. Install it or "
            f"pass droid_package_path (tried {pkg!r})."
        ) from e


class DroidStore(TrajectoryStore):
    """Flat-frame view over a tree of raw DROID session folders."""

    def __init__(
        self,
        path: str,
        cameras: Optional[Dict[str, str]] = None,
        camera_source: str = "none",   # "none" | "mp4" | "svo"
        image_size: tuple = (180, 320),
        action_space: str = "joint_velocity",
        max_episodes: Optional[int] = None,
        droid_package_path: str = "~/src_droid",
    ):
        assert camera_source in ("none", "mp4", "svo")
        if camera_source != "none" and not cameras:
            raise ValueError("cameras={obs_key: zed_serial} required with camera_source")
        _ensure_droid_importable(droid_package_path)

        self.path = path
        self.cameras = dict(cameras or {})
        self.camera_source = camera_source
        self.image_h, self.image_w = image_size
        self.action_space = action_space

        # Discover session folders (any dir containing trajectory.h5).
        sessions: List[str] = []
        for root, _dirs, files in sorted(os.walk(path)):
            if "trajectory.h5" in files:
                sessions.append(root)
        sessions.sort()
        if max_episodes is not None:
            sessions = sessions[:max_episodes]
        if not sessions:
            raise ValueError(f"No trajectory.h5 sessions found under {path!r}")

        # Episode lengths: load each trajectory once without cameras (cheap;
        # remove_skipped_steps changes T, so the h5 row count is not enough).
        lengths = []
        kept_sessions = []
        for s in sessions:
            try:
                T = len(self._load_steps(s, with_cameras=False))
            except Exception as e:
                print(f"[DroidStore] skip {s}: {e}")
                continue
            if T > 0:
                kept_sessions.append(s)
                lengths.append(T)
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

    # ── droid loading ─────────────────────────────────────────────────────────

    def _load_steps(self, session: str, with_cameras: bool):
        from droid.trajectory_utils.misc import load_trajectory

        recording_dir = None
        if with_cameras and self.camera_source == "mp4":
            mp4_dir = os.path.join(session, "recordings", "MP4")
            recording_dir = mp4_dir if os.path.isdir(mp4_dir) else None
        return load_trajectory(
            filepath=os.path.join(session, "trajectory.h5"),
            read_cameras=(recording_dir is not None),
            recording_folderpath=recording_dir,
            remove_skipped_steps=True,
        )

    def _read_svo_frames(self, session: str, serial: str, frame_indices) -> Optional[np.ndarray]:
        """Decode left-view frames from recordings/SVO/<serial>.svo* via pyzed."""
        try:
            import pyzed.sl as sl
        except ImportError as e:
            raise ImportError(
                "camera_source='svo' needs pyzed + the ZED SDK on "
                "LD_LIBRARY_PATH (see scripts/setup or the ZED docs)."
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
            return None

        init = sl.InitParameters()
        init.set_from_svo_file(svo_path)
        init.svo_real_time_mode = False
        init.camera_disable_self_calib = True
        cam = sl.Camera()
        if cam.open(init) != sl.ERROR_CODE.SUCCESS:
            return None
        rt = sl.RuntimeParameters()
        mat = sl.Mat()
        frames = []
        while True:
            status = cam.grab(rt)
            if status == sl.ERROR_CODE.END_OF_SVOFILE_REACHED:
                break
            if status != sl.ERROR_CODE.SUCCESS:
                break
            cam.retrieve_image(mat, sl.VIEW.LEFT)
            img = mat.get_data()  # BGRA
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
            if img.shape[:2] != (self.image_h, self.image_w):
                img = cv2.resize(img, (self.image_w, self.image_h),
                                 interpolation=cv2.INTER_AREA)
            frames.append(img)
        cam.close()
        if not frames:
            return None
        arr = np.stack(frames).astype(np.uint8)
        return arr[np.asarray(frame_indices)] if frame_indices is not None else arr

    def _load_episode(self, ep_idx: int) -> Dict[str, np.ndarray]:
        cached = self._episode_cache.get(ep_idx)
        if cached is not None:
            self._episode_cache.move_to_end(ep_idx)
            return cached

        import cv2
        import h5py

        session = self.sessions[ep_idx]
        with_cams = self.camera_source == "mp4"
        steps = self._load_steps(session, with_cameras=with_cams)
        T = len(steps)

        arm = np.stack([s["action"][self.action_space] for s in steps])
        grip = np.stack([np.atleast_1d(s["action"]["gripper_position"]) for s in steps])
        data: Dict[str, np.ndarray] = {
            "action": np.concatenate([arm, grip], axis=-1).astype(np.float32),
            "joint_positions": np.stack(
                [s["observation"]["robot_state"]["joint_positions"] for s in steps]
            ).astype(np.float32),
            "cartesian_position": np.stack(
                [s["observation"]["robot_state"]["cartesian_position"] for s in steps]
            ).astype(np.float32),
            "gripper_position": np.stack(
                [np.atleast_1d(s["observation"]["robot_state"]["gripper_position"]) for s in steps]
            ).astype(np.float32),
        }

        success = False
        try:
            with h5py.File(os.path.join(session, "trajectory.h5"), "r") as f:
                success = bool(f.attrs.get("success", False))
        except Exception:
            pass
        reward = np.zeros((T, 1), dtype=np.float32)
        reward[-1, 0] = float(success)
        data["reward"] = reward

        if self.camera_source == "mp4":
            for obs_key, serial in self.cameras.items():
                frames = []
                for s in steps:
                    img = s["observation"].get("image", {}).get(f"{serial}_left")
                    if img is None:
                        break
                    if img.ndim == 3 and img.shape[2] == 4:
                        img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
                    elif img.ndim == 3 and img.shape[2] == 3:
                        img = img[..., ::-1].copy()  # BGR → RGB
                    if img.shape[:2] != (self.image_h, self.image_w):
                        img = cv2.resize(img, (self.image_w, self.image_h),
                                         interpolation=cv2.INTER_AREA)
                    frames.append(img)
                if len(frames) == T:
                    data[obs_key] = np.stack(frames).astype(np.uint8)
        elif self.camera_source == "svo":
            # SVO frame index alignment: load_trajectory's step dicts carry the
            # camera frame index per step when recordings were synced; fall
            # back to the first T decoded frames.
            for obs_key, serial in self.cameras.items():
                arr = self._read_svo_frames(session, serial, None)
                if arr is not None and len(arr) >= T:
                    data[obs_key] = arr[:T]

        for key in self.cameras if self.camera_source != "none" else ():
            if key not in data:
                raise ValueError(
                    f"Camera {key!r} missing/incomplete in {session!r} — "
                    "exclude the session or drop the camera key."
                )

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
        ep_first = int(np.searchsorted(self._episode_ends, start, side="right"))
        ep_last = int(np.searchsorted(self._episode_ends, end - 1, side="right"))
        parts = []
        for ep_idx in range(ep_first, ep_last + 1):
            ep_offset = int(self._episode_ends[ep_idx - 1]) if ep_idx > 0 else 0
            lo = max(start, ep_offset) - ep_offset
            hi = min(end, int(self._episode_ends[ep_idx])) - ep_offset
            parts.append(self._load_episode(ep_idx)[key][lo:hi])
        return np.concatenate(parts, axis=0) if len(parts) > 1 else parts[0]
