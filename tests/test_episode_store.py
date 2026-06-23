"""Equivalence between ZarrEpisodeStore (new hierarchical) and ZarrStore (legacy flat).

ZarrEpisodeStore stores each episode as an independent zarr group — no
episode_ends.npy on disk.  It derives episode_ends from group array shapes,
so SequenceSampler / TrajectoryDataset require no changes.

The write interface is put_episode(ep_idx, data): episodes are independent
and can be written in any order, including concurrently.
"""

import numpy as np
import pytest
from concurrent.futures import ThreadPoolExecutor

from .constants import N_EPISODES, EP_LENGTHS, OBS_KEYS, LOWDIM_KEY, H, W, C, ACTION_DIM


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture(scope="session")
def episode_store(synthetic_hdf5, tmp_path_factory):
    from robo_dataset.backends.hdf5 import RobomimicHDF5Store
    from robo_dataset.backends.zarr_episode import ZarrEpisodeStore

    hdf5 = RobomimicHDF5Store(synthetic_hdf5, obs_keys=OBS_KEYS + [LOWDIM_KEY])
    out_path = str(tmp_path_factory.mktemp("ep_store") / "sequential.zarr")

    writer = ZarrEpisodeStore.create(
        path=out_path,
        n_episodes=hdf5.n_episodes,
        keys=list(hdf5.keys),
        key_shapes={k: hdf5.frame_shape(k) for k in hdf5.keys},
        key_dtypes={k: hdf5.frame_dtype(k) for k in hdf5.keys},
    )
    for i, ep_data in enumerate(hdf5.iter_episodes()):
        writer.put_episode(i, ep_data)
    writer.finalize()
    return ZarrEpisodeStore(out_path)


# ── Data equivalence ──────────────────────────────────────────────────────────

class TestEpisodeStoreEquivalence:

    def test_n_episodes(self, episode_store):
        assert episode_store.n_episodes == N_EPISODES

    def test_episode_lengths(self, episode_store):
        for i, expected in enumerate(EP_LENGTHS):
            assert episode_store.episode_length(i) == expected

    def test_keys(self, synthetic_hdf5, episode_store):
        from robo_dataset.backends.hdf5 import RobomimicHDF5Store
        hdf5 = RobomimicHDF5Store(synthetic_hdf5, obs_keys=OBS_KEYS + [LOWDIM_KEY])
        assert set(episode_store.keys) == set(hdf5.keys)

    def test_episode_ends_derived_not_stored(self, episode_store):
        """episode_ends must be derived from group shapes — no episode_ends.npy on disk."""
        import os
        assert not os.path.exists(os.path.join(episode_store.path, "episode_ends.npy"))
        expected = np.cumsum(EP_LENGTHS).astype(np.int64)
        np.testing.assert_array_equal(episode_store.episode_ends, expected)

    def test_action_exact_per_episode(self, synthetic_hdf5, episode_store):
        from robo_dataset.backends.hdf5 import RobomimicHDF5Store
        hdf5 = RobomimicHDF5Store(synthetic_hdf5, obs_keys=OBS_KEYS + [LOWDIM_KEY])
        for i, ep in enumerate(hdf5.iter_episodes()):
            np.testing.assert_array_equal(episode_store.get_episode(i)["action"], ep["action"])

    def test_lowdim_exact_per_episode(self, synthetic_hdf5, episode_store):
        from robo_dataset.backends.hdf5 import RobomimicHDF5Store
        hdf5 = RobomimicHDF5Store(synthetic_hdf5, obs_keys=OBS_KEYS + [LOWDIM_KEY])
        for i, ep in enumerate(hdf5.iter_episodes()):
            np.testing.assert_array_equal(
                episode_store.get_episode(i)[LOWDIM_KEY], ep[LOWDIM_KEY]
            )

    def test_images_exact_per_episode(self, synthetic_hdf5, episode_store):
        from robo_dataset.backends.hdf5 import RobomimicHDF5Store
        hdf5 = RobomimicHDF5Store(synthetic_hdf5, obs_keys=OBS_KEYS + [LOWDIM_KEY])
        for i, ep in enumerate(hdf5.iter_episodes()):
            result = episode_store.get_episode(i)
            for key in OBS_KEYS:
                np.testing.assert_array_equal(result[key], ep[key])

    def test_get_sequence_matches_flat_store(self, zarr_store_path, episode_store):
        """get_sequence with flat indices must return identical data to ZarrStore."""
        from robo_dataset.backends.zarr_backend import ZarrStore
        flat = ZarrStore(zarr_store_path)
        for key in flat.keys:
            expected = flat.get_sequence(key, 0, flat.n_frames)
            actual = episode_store.get_sequence(key, 0, episode_store.n_frames)
            np.testing.assert_array_equal(actual, expected, err_msg=f"mismatch for {key!r}")


# ── Parallel write independence ────────────────────────────────────────────────

class TestParallelWrite:

    def test_reverse_order_equivalent(self, synthetic_hdf5, tmp_path):
        """Writing episodes in reverse order must produce the same logical store."""
        from robo_dataset.backends.hdf5 import RobomimicHDF5Store
        from robo_dataset.backends.zarr_episode import ZarrEpisodeStore

        hdf5 = RobomimicHDF5Store(synthetic_hdf5, obs_keys=OBS_KEYS + [LOWDIM_KEY])
        episodes = list(hdf5.iter_episodes())

        writer = ZarrEpisodeStore.create(
            path=str(tmp_path / "reversed.zarr"),
            n_episodes=len(episodes),
            keys=list(hdf5.keys),
            key_shapes={k: hdf5.frame_shape(k) for k in hdf5.keys},
            key_dtypes={k: hdf5.frame_dtype(k) for k in hdf5.keys},
        )
        for i in reversed(range(len(episodes))):
            writer.put_episode(i, episodes[i])
        writer.finalize()

        store = ZarrEpisodeStore(str(tmp_path / "reversed.zarr"))
        for i, ep in enumerate(episodes):
            result = store.get_episode(i)
            for key in hdf5.keys:
                np.testing.assert_array_equal(result[key], ep[key])

    def test_threaded_write_equivalent(self, synthetic_hdf5, tmp_path):
        """Concurrent put_episode calls must be safe and produce correct data."""
        from robo_dataset.backends.hdf5 import RobomimicHDF5Store
        from robo_dataset.backends.zarr_episode import ZarrEpisodeStore

        hdf5 = RobomimicHDF5Store(synthetic_hdf5, obs_keys=OBS_KEYS + [LOWDIM_KEY])
        episodes = list(hdf5.iter_episodes())

        writer = ZarrEpisodeStore.create(
            path=str(tmp_path / "threaded.zarr"),
            n_episodes=len(episodes),
            keys=list(hdf5.keys),
            key_shapes={k: hdf5.frame_shape(k) for k in hdf5.keys},
            key_dtypes={k: hdf5.frame_dtype(k) for k in hdf5.keys},
        )
        with ThreadPoolExecutor(max_workers=4) as pool:
            list(pool.map(lambda i: writer.put_episode(i, episodes[i]), range(len(episodes))))
        writer.finalize()

        store = ZarrEpisodeStore(str(tmp_path / "threaded.zarr"))
        for i, ep in enumerate(episodes):
            result = store.get_episode(i)
            for key in hdf5.keys:
                np.testing.assert_array_equal(result[key], ep[key])


# ── DataLoader equivalence ─────────────────────────────────────────────────────

class TestDataLoaderEquivalence:

    def test_sampler_index_identical(self, zarr_store_path, episode_store):
        """SequenceSampler builds the same flat index from derived episode_ends."""
        from robo_dataset.backends.zarr_backend import ZarrStore
        from robo_dataset.sampler import SequenceSampler

        flat = ZarrStore(zarr_store_path)
        flat_sampler = SequenceSampler(flat, sequence_length=4, pad_before=1, pad_after=1)
        ep_sampler = SequenceSampler(episode_store, sequence_length=4, pad_before=1, pad_after=1)

        assert len(flat_sampler) == len(ep_sampler)
        np.testing.assert_array_equal(flat_sampler.indices, ep_sampler.indices)

    def test_batches_identical(self, zarr_store_path, episode_store):
        """Deterministic DataLoader (shuffle=False, workers=0) must yield identical batches."""
        from robo_dataset.backends.zarr_backend import ZarrStore
        from robo_dataset.dataset import TrajectoryDataset

        flat = ZarrStore(zarr_store_path)
        flat_ds = TrajectoryDataset(flat, horizon=4, pad_before=1, pad_after=1)
        ep_ds = TrajectoryDataset(episode_store, horizon=4, pad_before=1, pad_after=1)

        flat_loader = flat_ds.make_dataloader(
            batch_size=4, num_workers=0, shuffle=False, pin_memory=False, persistent_workers=False
        )
        ep_loader = ep_ds.make_dataloader(
            batch_size=4, num_workers=0, shuffle=False, pin_memory=False, persistent_workers=False
        )

        flat_batches = list(flat_loader)
        ep_batches = list(ep_loader)
        assert len(flat_batches) == len(ep_batches)

        for flat_batch, ep_batch in zip(flat_batches, ep_batches):
            for key in flat_batch:
                np.testing.assert_array_equal(
                    flat_batch[key].numpy(), ep_batch[key].numpy(),
                    err_msg=f"batch mismatch for key {key!r}",
                )

    def test_multiworker_dataloader(self, episode_store):
        """ZarrEpisodeStore must work correctly with multiple DataLoader workers."""
        from robo_dataset.dataset import TrajectoryDataset
        ds = TrajectoryDataset(episode_store, horizon=4, pad_before=1, pad_after=1)
        loader = ds.make_dataloader(
            batch_size=4, num_workers=2, shuffle=False, pin_memory=False
        )
        batch = next(iter(loader))
        assert batch[OBS_KEYS[0]].shape == (4, 4, H, W, C)
        assert batch["action"].shape == (4, 4, ACTION_DIM)
