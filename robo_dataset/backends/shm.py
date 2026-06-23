"""Shared-memory RAM store.

Loads an entire EpisodeStore into ``torch.Tensor``s backed by POSIX shared
memory (``tensor.share_memory_()``).  After the initial load all DataLoader
workers access the data zero-copy without any disk I/O, zarr decompression, or
file-system calls — they simply read from the same physical memory pages.

Pros
----
- Zero disk I/O after the first load: completely eliminates the disk / inode
  bottleneck that limits zarr and RLDS at high worker counts.
- Worker access is a single shared-memory slice — O(1), zero decompression.
- Fork-safe: shared memory pages survive ``fork()``; ``on_worker_init`` is a
  no-op.
- Works with any source backend (zarr, HDF5, RLDS, …) via ``from_store``.

Cons
----
- Requires enough RAM for the full uncompressed dataset.  A typical 2-camera
  robotics dataset at 84×84 is ~200–500 MB; large datasets may not fit.
- Startup cost: the first load reads and decompresses the whole source into RAM
  (one-time, amortised over the training run).
- No persistence: data is lost when the Python process exits.  Use a disk
  backend as the source; convert once per experiment.
"""

from typing import Dict, List, Optional

import numpy as np

try:
    from tqdm import tqdm as _tqdm
except ImportError:
    def _tqdm(it, **kwargs):
        return it

import torch

from .base import EpisodeStore


class ShmStore(EpisodeStore):
    """Read-only in-RAM store backed by torch shared memory.

    Workers forked by ``torch.utils.data.DataLoader`` access the tensors
    zero-copy.  ``on_worker_init`` is a no-op since no file handles need to be
    reopened.

    Construct via :meth:`from_store`; direct ``__init__`` is for internal use.
    """

    def __init__(
        self,
        data: List[Dict[str, torch.Tensor]],
        dtypes: Dict[str, np.dtype],
        shapes: Dict[str, tuple],
        keys: List[str],
    ):
        self._data = data  # List[Dict[str, Tensor]], one per episode
        self._dtypes = dtypes
        self._shapes = shapes
        self._keys: List[str] = keys

    @classmethod
    def from_store(
        cls,
        source: EpisodeStore,
        keys: Optional[List[str]] = None,
        verbose: bool = True,
    ) -> "ShmStore":
        """Load all episodes from *source* into shared memory.

        Args:
            source:  Any :class:`~robo_dataset.backends.base.EpisodeStore`.
                     ``get_episode`` is called once per episode; this is the
                     one-time startup cost.
            keys:    Subset of keys to load.  Defaults to all keys.
            verbose: Show a tqdm progress bar while loading.

        Returns:
            An :class:`ShmStore` ready for use as a DataLoader dataset backend.
        """
        keys = keys or list(source.keys)

        dtypes: Dict[str, np.dtype] = {k: source.frame_dtype(k) for k in keys}
        shapes: Dict[str, tuple] = {k: source.frame_shape(k) for k in keys}

        episodes_range = range(source.n_episodes)
        it = _tqdm(episodes_range, desc="ShmStore: loading to shared memory") if verbose else episodes_range

        data: List[Dict[str, torch.Tensor]] = []
        for ep_idx in it:
            ep = source.get_episode(ep_idx)
            ep_tensors: Dict[str, torch.Tensor] = {}
            for k in keys:
                arr = np.ascontiguousarray(ep[k])
                ep_tensors[k] = torch.from_numpy(arr).share_memory_()
            data.append(ep_tensors)

        return cls(data, dtypes, shapes, keys)

    # ── EpisodeStore interface ─────────────────────────────────────────────────

    @property
    def n_episodes(self) -> int:
        return len(self._data)

    @property
    def keys(self) -> List[str]:
        return self._keys

    def episode_length(self, ep_idx: int) -> int:
        return int(self._data[ep_idx][self._keys[0]].shape[0])

    def frame_shape(self, key: str) -> tuple:
        return self._shapes[key]

    def frame_dtype(self, key: str) -> np.dtype:
        return self._dtypes[key]

    def get_frames(self, ep_idx: int, key: str, start: int, end: int) -> np.ndarray:
        return self._data[ep_idx][key][start:end].numpy()

    def on_worker_init(self, worker_id: int) -> None:
        pass  # shared memory survives fork — nothing to reopen
