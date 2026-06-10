"""LeRobotStore tests against a synthetic LeRobot v2 dataset.

The fixture writes the same layout robocasa's GR00T exporter produces:
parquet per episode, MP4 per (episode, camera), info.json / episodes.jsonl /
modality.json. Ground-truth arrays are kept in RAM for comparison.
"""

import json
import os

import numpy as np
import pytest

pytest.importorskip("pyarrow")
pytest.importorskip("av")

N_EPISODES = 3
EP_LENGTHS = [12, 9, 15]
H, W, C = 32, 32, 3
STATE_DIM = 5
ACTION_DIM = 4


def _smooth_frames(ep: int, T: int) -> np.ndarray:
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float64) / max(H, W)
    frames = []
    for t in range(T):
        phase = (t / T + ep * 0.3) * 2 * np.pi
        r = 127 + 110 * np.sin(2 * np.pi * xx + phase)
        g = 127 + 110 * np.cos(2 * np.pi * yy + phase)
        b = 127 + 110 * np.sin(2 * np.pi * (xx + yy) - phase)
        frames.append(np.stack([r, g, b], axis=-1))
    return np.clip(np.stack(frames), 0, 255).astype(np.uint8)


@pytest.fixture(scope="session")
def lerobot_dataset(tmp_path_factory):
    import av
    import pyarrow as pa
    import pyarrow.parquet as pq

    root = tmp_path_factory.mktemp("lerobot")
    rng = np.random.default_rng(7)

    video_feat = "observation.images.cam"
    ref = {"video.cam": [], "observation.state": [], "action": []}

    os.makedirs(root / "meta")
    for ep, T in enumerate(EP_LENGTHS):
        state = rng.standard_normal((T, STATE_DIM)).astype(np.float32)
        action = rng.standard_normal((T, ACTION_DIM)).astype(np.float32)
        frames = _smooth_frames(ep, T)
        ref["observation.state"].append(state)
        ref["action"].append(action)
        ref["video.cam"].append(frames)

        # parquet
        pq_dir = root / "data" / "chunk-000"
        os.makedirs(pq_dir, exist_ok=True)
        table = pa.table({
            "observation.state": [row.tolist() for row in state],
            "action": [row.tolist() for row in action],
            "episode_index": [ep] * T,
            "frame_index": list(range(T)),
            "timestamp": (np.arange(T) / 10.0).tolist(),
        })
        pq.write_table(table, pq_dir / f"episode_{ep:06d}.parquet")

        # video
        vid_dir = root / "videos" / "chunk-000" / video_feat
        os.makedirs(vid_dir, exist_ok=True)
        container = av.open(str(vid_dir / f"episode_{ep:06d}.mp4"), "w")
        stream = container.add_stream("libx264", rate=10)
        stream.width, stream.height, stream.pix_fmt = W, H, "yuv420p"
        stream.options = {"crf": "18", "g": "1", "preset": "fast"}
        for arr in frames:
            for pkt in stream.encode(av.VideoFrame.from_ndarray(arr, format="rgb24")):
                container.mux(pkt)
        for pkt in stream.encode():
            container.mux(pkt)
        container.close()

    info = {
        "codebase_version": "v2.0",
        "fps": 10,
        "total_episodes": N_EPISODES,
        "total_frames": int(sum(EP_LENGTHS)),
        "chunks_size": 1000,
        "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
        "features": {
            video_feat: {"dtype": "video", "shape": [H, W, C],
                         "names": ["height", "width", "channels"]},
            "observation.state": {"dtype": "float32", "shape": [STATE_DIM]},
            "action": {"dtype": "float32", "shape": [ACTION_DIM]},
            "episode_index": {"dtype": "int64", "shape": [1]},
            "frame_index": {"dtype": "int64", "shape": [1]},
            "timestamp": {"dtype": "float32", "shape": [1]},
        },
    }
    with open(root / "meta" / "info.json", "w") as f:
        json.dump(info, f)

    with open(root / "meta" / "episodes.jsonl", "w") as f:
        for ep, T in enumerate(EP_LENGTHS):
            f.write(json.dumps({"episode_index": ep, "length": T, "tasks": ["t"]}) + "\n")

    modality = {
        "state": {
            "eef_pos": {"original_key": "observation.state", "start": 0, "end": 3},
            "gripper_qpos": {"original_key": "observation.state", "start": 3, "end": 5},
        },
        "action": {
            "all": {"original_key": "action", "start": 0, "end": ACTION_DIM},
        },
        "video": {
            "cam": {"original_key": video_feat},
        },
    }
    with open(root / "meta" / "modality.json", "w") as f:
        json.dump(modality, f)

    cat = {k: np.concatenate(v) for k, v in ref.items()}
    return str(root), cat


def _psnr(a, b):
    mse = np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2)
    return float("inf") if mse == 0 else 20 * np.log10(255.0 / np.sqrt(mse))


class TestLeRobotStore:
    def test_metadata(self, lerobot_dataset):
        from robo_dataset.backends.lerobot import LeRobotStore
        root, ref = lerobot_dataset
        store = LeRobotStore(root)
        assert store.n_episodes == N_EPISODES
        np.testing.assert_array_equal(store.episode_ends, np.cumsum(EP_LENGTHS))
        assert "video.cam" in store.keys
        assert "observation.state" in store.keys
        assert "action" in store.keys
        assert "state.eef_pos" in store.keys
        assert store.frame_shape("video.cam") == (H, W, C)
        assert store.frame_shape("state.eef_pos") == (3,)
        assert store.frame_shape("action") == (ACTION_DIM,)

    def test_parquet_exact(self, lerobot_dataset):
        from robo_dataset.backends.lerobot import LeRobotStore
        root, ref = lerobot_dataset
        store = LeRobotStore(root)
        n = store.n_frames
        np.testing.assert_allclose(
            store.get_sequence("observation.state", 0, n), ref["observation.state"])
        np.testing.assert_allclose(store.get_sequence("action", 0, n), ref["action"])
        # modality slice == columns 0:3
        np.testing.assert_allclose(
            store.get_sequence("state.eef_pos", 0, n), ref["observation.state"][:, :3])

    def test_cross_episode_read(self, lerobot_dataset):
        from robo_dataset.backends.lerobot import LeRobotStore
        root, ref = lerobot_dataset
        store = LeRobotStore(root)
        start, end = EP_LENGTHS[0] - 3, EP_LENGTHS[0] + 3
        np.testing.assert_allclose(
            store.get_sequence("action", start, end), ref["action"][start:end])
        frames = store.get_sequence("video.cam", start, end)
        assert frames.shape == (6, H, W, C)

    def test_video_psnr(self, lerobot_dataset):
        from robo_dataset.backends.lerobot import LeRobotStore
        root, ref = lerobot_dataset
        store = LeRobotStore(root)
        got = store.get_sequence("video.cam", 0, EP_LENGTHS[0])
        psnr = _psnr(got, ref["video.cam"][: EP_LENGTHS[0]])
        assert psnr >= 28.0, f"video decode PSNR {psnr:.1f} dB too low"

    def test_convert_to_videostore(self, lerobot_dataset, tmp_path):
        from robo_dataset.backends.lerobot import LeRobotStore
        from robo_dataset.backends.video import VideoStore, VideoStoreWriter
        from robo_dataset.convert import convert
        root, ref = lerobot_dataset
        src = LeRobotStore(root)
        keys = ["video.cam", "state.eef_pos", "state.gripper_qpos", "action"]
        writer = VideoStoreWriter.create(
            str(tmp_path / "out.video"),
            key_shapes={k: src.frame_shape(k) for k in keys},
            key_dtypes={k: src.frame_dtype(k) for k in keys},
            image_keys=["video.cam"],
            crf=18,
        )
        convert(src, writer, keys=keys, verbose=False)
        out = VideoStore(str(tmp_path / "out.video"))
        n = out.n_frames
        assert n == sum(EP_LENGTHS)
        np.testing.assert_allclose(
            out.get_sequence("state.eef_pos", 0, n), ref["observation.state"][:, :3])
        np.testing.assert_allclose(out.get_sequence("action", 0, n), ref["action"])
        psnr = _psnr(out.get_sequence("video.cam", 0, n), ref["video.cam"])
        assert psnr >= 26.0, f"re-encoded PSNR {psnr:.1f} dB too low"
