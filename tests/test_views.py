import numpy as np
import pytest

from robo_dataset.backends.base import EpisodeStore
from robo_dataset.views import EpisodeView


class MemoryEpisodes(EpisodeStore):
    def __init__(self):
        self.episodes = [
            np.arange(0, 3, dtype=np.float32)[:, None],
            np.arange(10, 12, dtype=np.float32)[:, None],
            np.arange(20, 24, dtype=np.float32)[:, None],
        ]
        self.initialized_worker = None

    @property
    def n_episodes(self):
        return len(self.episodes)

    @property
    def keys(self):
        return ["state"]

    def episode_length(self, ep_idx):
        return len(self.episodes[ep_idx])

    def frame_shape(self, key):
        assert key == "state"
        return (1,)

    def frame_dtype(self, key):
        assert key == "state"
        return np.dtype(np.float32)

    def get_frames(self, ep_idx, key, start, end):
        assert key == "state"
        return self.episodes[ep_idx][start:end]

    def on_worker_init(self, worker_id):
        self.initialized_worker = worker_id

    @property
    def schemas(self):
        return {"state": "schema-marker"}


def test_episode_view_maps_logical_order_without_copying():
    source = MemoryEpisodes()
    view = EpisodeView(source, [2, 0])

    assert view.source is source
    assert view.episode_ids == (2, 0)
    assert view.n_episodes == 2
    np.testing.assert_array_equal(view.episode_ends, [4, 7])
    np.testing.assert_array_equal(
        view.get_frames(0, "state", 1, 3),
        source.episodes[2][1:3],
    )
    np.testing.assert_array_equal(
        view.get_sequence("state", 3, 6).ravel(),
        [23, 0, 1],
    )
    assert view.schemas == source.schemas
    view.on_worker_init(4)
    assert source.initialized_worker == 4


@pytest.mark.parametrize(
    ("episode_ids", "error"),
    [
        ([], ValueError),
        ([0, 0], ValueError),
        ([-1], ValueError),
        ([3], ValueError),
        ([True], TypeError),
        ([0.5], TypeError),
    ],
)
def test_episode_view_rejects_invalid_selections(episode_ids, error):
    with pytest.raises(error):
        EpisodeView(MemoryEpisodes(), episode_ids)
