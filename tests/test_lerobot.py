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


# ── LeRobot v3 ────────────────────────────────────────────────────────────────
# v3 packs every episode into shared parquet/mp4 files and addresses them via
# meta/episodes/*.parquet (dataset_{from,to}_index + per-camera from_timestamp),
# so these cover the offset arithmetic v2 never exercises. No modality.json here
# (the common non-GR00T case) => keys are the raw feature names.

V3_FPS = 10


@pytest.fixture(scope="session")
def lerobot_v3_dataset(tmp_path_factory):
    import av
    import pyarrow as pa
    import pyarrow.parquet as pq

    root = tmp_path_factory.mktemp("lerobot_v3")
    rng = np.random.default_rng(11)
    video_feat = "top_camera-images-rgb"
    ref = {video_feat: [], "state": [], "actions": []}

    os.makedirs(root / "meta" / "episodes" / "chunk-000")
    rows, ep_rows, cursor = [], [], 0
    for ep, T in enumerate(EP_LENGTHS):
        state = rng.standard_normal((T, STATE_DIM)).astype(np.float32)
        action = rng.standard_normal((T, ACTION_DIM)).astype(np.float32)
        frames = _smooth_frames(ep, T)
        ref["state"].append(state)
        ref["actions"].append(action)
        ref[video_feat].append(frames)
        rows.append((state, action, ep, T))
        ep_rows.append({
            "episode_index": ep, "length": T,
            "data/chunk_index": 0, "data/file_index": 0,
            "dataset_from_index": cursor, "dataset_to_index": cursor + T,
            f"videos/{video_feat}/chunk_index": 0,
            f"videos/{video_feat}/file_index": 0,
            f"videos/{video_feat}/from_timestamp": cursor / V3_FPS,
            f"videos/{video_feat}/to_timestamp": (cursor + T) / V3_FPS,
        })
        cursor += T

    # one parquet holding every episode, back to back
    os.makedirs(root / "data" / "chunk-000")
    pq.write_table(
        pa.table({
            "state": [r.tolist() for s, _, _, _ in rows for r in s],
            "actions": [r.tolist() for _, a, _, _ in rows for r in a],
            "episode_index": [ep for _, _, ep, T in rows for _ in range(T)],
            "frame_index": [t for _, _, _, T in rows for t in range(T)],
            "timestamp": [t / V3_FPS for _, _, _, T in rows for t in range(T)],
        }),
        root / "data" / "chunk-000" / "file-00000.parquet",
    )

    # one mp4 holding every episode, back to back
    vid_dir = root / "videos" / video_feat / "chunk-000"
    os.makedirs(vid_dir)
    container = av.open(str(vid_dir / "file-00000.mp4"), "w")
    stream = container.add_stream("libx264", rate=V3_FPS)
    stream.width, stream.height, stream.pix_fmt = W, H, "yuv420p"
    stream.options = {"crf": "18", "g": "1", "preset": "fast"}
    for frames in ref[video_feat]:
        for arr in frames:
            for pkt in stream.encode(av.VideoFrame.from_ndarray(arr, format="rgb24")):
                container.mux(pkt)
    for pkt in stream.encode():
        container.mux(pkt)
    container.close()

    pq.write_table(
        pa.Table.from_pylist(ep_rows),
        root / "meta" / "episodes" / "chunk-000" / "file-00000.parquet",
    )
    with open(root / "meta" / "info.json", "w") as f:
        json.dump({
            "codebase_version": "v3.0",
            "fps": V3_FPS,
            "total_episodes": N_EPISODES,
            "total_frames": int(sum(EP_LENGTHS)),
            "chunks_size": 1000,
            "data_path": "data/chunk-{chunk_index:03d}/file-{file_index:05d}.parquet",
            "video_path": "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:05d}.mp4",
            "features": {
                video_feat: {"dtype": "video", "shape": [H, W, C],
                             "names": ["height", "width", "channels"]},
                "state": {"dtype": "float32", "shape": [STATE_DIM]},
                "actions": {"dtype": "float32", "shape": [ACTION_DIM]},
                "episode_index": {"dtype": "int64", "shape": [1]},
                "frame_index": {"dtype": "int64", "shape": [1]},
                "timestamp": {"dtype": "float32", "shape": [1]},
            },
        }, f)

    return str(root), {k: np.concatenate(v) for k, v in ref.items()}, video_feat


class TestLeRobotStoreV3:
    def test_detected_and_metadata(self, lerobot_v3_dataset):
        from robo_dataset.store_factory import detect_format, open_store
        root, ref, video_feat = lerobot_v3_dataset
        assert detect_format(root) == "lerobot"
        store = open_store(root)
        assert store._v3 is True
        assert store.n_episodes == N_EPISODES
        np.testing.assert_array_equal(store.episode_ends, np.cumsum(EP_LENGTHS))
        assert store.frame_shape(video_feat) == (H, W, C)
        assert store.frame_shape("state") == (STATE_DIM,)
        # bookkeeping columns stay out of the key list
        assert "index" not in store.keys and "timestamp" not in store.keys

    def test_packed_parquet_exact(self, lerobot_v3_dataset):
        from robo_dataset.backends.lerobot import LeRobotStore
        root, ref, _ = lerobot_v3_dataset
        store = LeRobotStore(root)
        n = store.n_frames
        np.testing.assert_allclose(store.get_sequence("state", 0, n), ref["state"])
        np.testing.assert_allclose(store.get_sequence("actions", 0, n), ref["actions"])

    def test_packed_video_offsets(self, lerobot_v3_dataset):
        """Each episode must start at its own from_timestamp inside the shared mp4."""
        from robo_dataset.backends.lerobot import LeRobotStore
        root, ref, video_feat = lerobot_v3_dataset
        store = LeRobotStore(root)
        starts = np.concatenate([[0], np.cumsum(EP_LENGTHS)[:-1]])
        for ep, T in enumerate(EP_LENGTHS):
            a = int(starts[ep])
            got = store.get_sequence(video_feat, a, a + T)
            psnr = _psnr(got, ref[video_feat][a:a + T])
            assert psnr >= 28.0, f"ep{ep} decode PSNR {psnr:.1f} dB (wrong offset?)"

    def test_cross_episode_read(self, lerobot_v3_dataset):
        from robo_dataset.backends.lerobot import LeRobotStore
        root, ref, video_feat = lerobot_v3_dataset
        store = LeRobotStore(root)
        start, end = EP_LENGTHS[0] - 3, EP_LENGTHS[0] + 3
        np.testing.assert_allclose(
            store.get_sequence("actions", start, end), ref["actions"][start:end])
        assert store.get_sequence(video_feat, start, end).shape == (6, H, W, C)
