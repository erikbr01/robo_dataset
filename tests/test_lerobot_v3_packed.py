"""LeRobot v3 packed-parquet reads: concurrency, addressing, cache behaviour.

Complements test_lerobot.py, which covers the v3 video path and therefore needs
PyAV. Everything here is parquet-only so it runs in a bare environment, and it
targets the parts of the packed-file reader that a single-threaded, contiguous,
single-file fixture cannot exercise: several data files, episode_index labels
that do not equal their position, and ~16 concurrent readers (what
convert(workers=N) and rgrl's EpisodeStore.load do).
"""

import json
import os
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest

pytest.importorskip("pyarrow")

FPS = 10
STATE_DIM, ACTION_DIM = 5, 4
LENGTHS = [7, 11, 5, 9, 13, 6, 8, 10]
EPS_PER_FILE = 3  # => 3 data files


def _write_v3(root, lengths, labels, eps_per_file):
    """Minimal video-less v3 layout; returns {label: (state, action)}."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    os.makedirs(os.path.join(root, "meta", "episodes", "chunk-000"))
    os.makedirs(os.path.join(root, "data", "chunk-000"))
    rng = np.random.default_rng(23)

    ref, ep_rows, files = {}, [], {}
    for pos, (label, T) in enumerate(zip(labels, lengths)):
        state = rng.standard_normal((T, STATE_DIM)).astype(np.float32)
        action = rng.standard_normal((T, ACTION_DIM)).astype(np.float32)
        ref[label] = (state, action)
        file_index = pos // eps_per_file
        files.setdefault(file_index, []).append((label, state, action))
        ep_rows.append({
            "episode_index": label, "length": T,
            "data/chunk_index": 0, "data/file_index": file_index,
            # stats/* is the bulk of a real meta table and must not be read
            "stats/state/mean": state.mean(0).tolist(),
            "stats/actions/mean": action.mean(0).tolist(),
        })

    for file_index, eps in files.items():
        state = np.concatenate([e[1] for e in eps])
        action = np.concatenate([e[2] for e in eps])
        pq.write_table(
            pa.table({
                "state": pa.array([r.tolist() for r in state], pa.list_(pa.float32())),
                "actions": pa.array([r.tolist() for r in action], pa.list_(pa.float32())),
                "episode_index": [e[0] for e in eps for _ in range(len(e[1]))],
                "frame_index": list(range(len(state))),
                "timestamp": (np.arange(len(state)) / FPS).tolist(),
            }),
            os.path.join(root, "data", "chunk-000", f"file-{file_index:05d}.parquet"),
        )

    pq.write_table(
        pa.Table.from_pylist(ep_rows),
        os.path.join(root, "meta", "episodes", "chunk-000", "file-00000.parquet"),
    )
    with open(os.path.join(root, "meta", "info.json"), "w") as f:
        json.dump({
            "codebase_version": "v3.0", "fps": FPS, "chunks_size": 1000,
            "data_path": "data/chunk-{chunk_index:03d}/file-{file_index:05d}.parquet",
            "features": {
                "state": {"dtype": "float32", "shape": [STATE_DIM]},
                "actions": {"dtype": "float32", "shape": [ACTION_DIM]},
                "episode_index": {"dtype": "int64", "shape": [1]},
                "frame_index": {"dtype": "int64", "shape": [1]},
                "timestamp": {"dtype": "float32", "shape": [1]},
            },
        }, f)
    return ref


@pytest.fixture(scope="module")
def packed_v3(tmp_path_factory):
    root = str(tmp_path_factory.mktemp("lerobot_v3_packed"))
    labels = list(range(len(LENGTHS)))
    ref = _write_v3(root, LENGTHS, labels, EPS_PER_FILE)
    return root, ref, labels, LENGTHS


def _episode_bounds(lengths):
    ends = np.cumsum(lengths)
    return [(0 if i == 0 else int(ends[i - 1]), int(ends[i])) for i in range(len(lengths))]


class TestPackedParquetReads:
    def test_every_episode_exact_across_files(self, packed_v3):
        from robo_dataset.backends.lerobot import LeRobotStore
        root, ref, labels, lengths = packed_v3
        store = LeRobotStore(root)
        for pos, (lo, hi) in enumerate(_episode_bounds(lengths)):
            state, action = ref[labels[pos]]
            np.testing.assert_array_equal(store.get_sequence("state", lo, hi), state)
            np.testing.assert_array_equal(store.get_sequence("actions", lo, hi), action)

    def test_episode_arrays_are_copies_not_file_views(self, packed_v3):
        """A view would pin the whole packed file and alias its neighbours."""
        from robo_dataset.backends.lerobot import LeRobotStore
        root, ref, labels, lengths = packed_v3
        store = LeRobotStore(root)
        bounds = _episode_bounds(lengths)

        first = store._episode_table(0)["actions"]
        assert first.base is None
        first[:] = 999.0  # same file as episodes 1 and 2

        lo, hi = bounds[1]
        np.testing.assert_array_equal(
            store.get_sequence("actions", lo, hi), ref[labels[1]][1])

    def test_returned_arrays_stay_writable(self, packed_v3):
        """Arrow buffers are read-only; get_sequence must not hand those out."""
        from robo_dataset.backends.lerobot import LeRobotStore
        root, _, _, lengths = packed_v3
        store = LeRobotStore(root)
        assert store.get_sequence("state", 0, 3).flags.writeable
        # a range spanning two episodes goes through concatenate
        boundary = lengths[0]
        assert store.get_sequence("state", boundary - 2, boundary + 2).flags.writeable

    def test_file_cache_is_bounded(self, packed_v3):
        from robo_dataset.backends.lerobot import LeRobotStore
        from robo_dataset.backends.lerobot import _V3_FILE_CACHE_SIZE
        root, _, _, lengths = packed_v3
        store = LeRobotStore(root)
        for lo, hi in _episode_bounds(lengths):
            store.get_sequence("state", lo, hi)
        assert len(store._v3_file_cache) <= _V3_FILE_CACHE_SIZE

    def test_caches_cleared_on_fork_and_worker_init(self, packed_v3):
        from robo_dataset.backends.lerobot import LeRobotStore
        root, _, _, _ = packed_v3
        store = LeRobotStore(root)
        store.get_sequence("state", 0, 3)
        assert store._v3_file_cache and store._episode_cache

        store._cache_pid = -1  # pretend we were inherited across a fork
        store._check_fork()
        assert not store._v3_file_cache and not store._episode_cache

        store.get_sequence("state", 0, 3)
        store.on_worker_init(0)
        assert not store._v3_file_cache and not store._episode_cache

    def test_missing_episode_index_column_reports_the_file(self, tmp_path):
        """episode_index goes through the same friendly check as mapped keys."""
        import pyarrow.parquet as pq
        from robo_dataset.backends.lerobot import LeRobotStore
        root = str(tmp_path / "bad")
        _write_v3(root, [4, 4], [0, 1], eps_per_file=2)
        path = os.path.join(root, "data", "chunk-000", "file-00000.parquet")
        pq.write_table(pq.read_table(path).drop_columns(["episode_index"]), path)

        with pytest.raises(KeyError, match="missing from parquet file"):
            LeRobotStore(root).get_sequence("actions", 0, 2)

    def test_row_count_mismatch_is_reported(self, tmp_path):
        import pyarrow.parquet as pq
        from robo_dataset.backends.lerobot import LeRobotStore
        root = str(tmp_path / "short")
        _write_v3(root, [4, 4], [0, 1], eps_per_file=2)
        path = os.path.join(root, "data", "chunk-000", "file-00000.parquet")
        pq.write_table(pq.read_table(path).slice(0, 6), path)

        with pytest.raises(ValueError, match="rows but"):
            LeRobotStore(root).get_sequence("actions", 4, 8)


class TestNonContiguousEpisodeIndex:
    """episode_ends is positional; episode_index is a label. A partially
    materialized or filtered meta/ makes the two diverge."""

    def test_v3_labels_are_not_positions(self, tmp_path):
        from robo_dataset.backends.lerobot import LeRobotStore
        root = str(tmp_path / "v3nc")
        lengths, labels = [6, 4, 8, 5], [0, 3, 7, 11]
        ref = _write_v3(root, lengths, labels, eps_per_file=2)
        store = LeRobotStore(root)
        for pos, (lo, hi) in enumerate(_episode_bounds(lengths)):
            np.testing.assert_array_equal(
                store.get_sequence("actions", lo, hi), ref[labels[pos]][1])

    def test_v2_labels_are_not_positions(self, tmp_path):
        import pyarrow as pa
        import pyarrow.parquet as pq
        from robo_dataset.backends.lerobot import LeRobotStore

        root = tmp_path / "v2nc"
        os.makedirs(root / "meta")
        os.makedirs(root / "data" / "chunk-000")
        rng = np.random.default_rng(5)
        lengths, labels, ref = [5, 7, 4], [2, 5, 9], {}
        for label, T in zip(labels, lengths):
            action = rng.standard_normal((T, ACTION_DIM)).astype(np.float32)
            ref[label] = action
            pq.write_table(
                pa.table({
                    "actions": pa.array([r.tolist() for r in action],
                                        pa.list_(pa.float32())),
                    "episode_index": [label] * T,
                    "frame_index": list(range(T)),
                }),
                root / "data" / "chunk-000" / f"episode_{label:06d}.parquet",
            )
        with open(root / "meta" / "info.json", "w") as f:
            json.dump({
                "codebase_version": "v2.0", "fps": FPS, "chunks_size": 1000,
                "data_path":
                    "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
                "features": {
                    "actions": {"dtype": "float32", "shape": [ACTION_DIM]},
                    "episode_index": {"dtype": "int64", "shape": [1]},
                    "frame_index": {"dtype": "int64", "shape": [1]},
                },
            }, f)
        with open(root / "meta" / "episodes.jsonl", "w") as f:
            for label, T in zip(labels, lengths):
                f.write(json.dumps({"episode_index": label, "length": T}) + "\n")

        store = LeRobotStore(str(root))
        for pos, (lo, hi) in enumerate(_episode_bounds(lengths)):
            got = store.get_sequence("actions", lo, hi)
            np.testing.assert_array_equal(got, ref[labels[pos]])
            assert got.flags.writeable  # v2 arrays were writable before v3 support


class TestConcurrentReads:
    """convert(workers=N) fans get_frames out over a thread pool; rgrl uses ~16."""

    def test_threads_get_correct_data_without_rereading_every_time(self, packed_v3):
        import pyarrow.parquet as pq
        from robo_dataset.backends.lerobot import LeRobotStore
        root, ref, labels, lengths = packed_v3
        bounds = _episode_bounds(lengths)
        store = LeRobotStore(root)

        reads, original = [], pq.read_table

        def counting_read_table(*args, **kwargs):
            reads.append(args[0] if args else kwargs.get("source"))
            return original(*args, **kwargs)

        errors = []

        def fetch(i):
            pos = i % len(lengths)
            lo, hi = bounds[pos]
            try:
                got = store.get_sequence("state", lo, hi)
                if not np.array_equal(got, ref[labels[pos]][0]):
                    errors.append(f"wrong rows for episode {labels[pos]}")
            except Exception as exc:  # noqa: BLE001 — a race must not surface here
                errors.append(f"{type(exc).__name__}: {exc}")

        pq.read_table = counting_read_table
        try:
            with ThreadPoolExecutor(max_workers=16) as pool:
                list(pool.map(fetch, range(320)))
        finally:
            pq.read_table = original

        assert not errors, errors[:3]
        # A single unsynchronized cache slot re-read the file on nearly every
        # fetch (~270/320 measured). Cache + per-path load lock keeps it low.
        assert len(reads) < 40, f"{len(reads)} full-file reads for 320 fetches"
