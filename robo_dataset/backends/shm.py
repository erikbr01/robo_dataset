"""Shared-memory RAM store.

Loads an entire TrajectoryStore into ``torch.Tensor``s backed by POSIX shared
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

from .base import TrajectoryStore


class ShmStore(TrajectoryStore):
    """Read-only in-RAM store backed by torch shared memory.

    Workers forked by ``torch.utils.data.DataLoader`` access the tensors
    zero-copy.  ``on_worker_init`` is a no-op since no file handles need to be
    reopened.

    Construct via :meth:`from_store`; direct ``__init__`` is for internal use.
    """

    def __init__(
        self,
        data: Dict[str, torch.Tensor],
        episode_ends: np.ndarray,
        dtypes: Dict[str, np.dtype],
        shapes: Dict[str, tuple],
    ):
        self._data = data
        self._episode_ends = episode_ends
        self._dtypes = dtypes
        self._shapes = shapes
        self._keys: List[str] = list(data.keys())

    @classmethod
    def from_store(
        cls,
        source: TrajectoryStore,
        keys: Optional[List[str]] = None,
        verbose: bool = True,
    ) -> "ShmStore":
        """Load all frames from *source* into shared memory.

        Args:
            source:  Any :class:`~robo_dataset.backends.base.TrajectoryStore`.
                     ``get_sequence`` is called once per key to load the full
                     dataset; this is the one-time startup cost.
            keys:    Subset of keys to load.  Defaults to all keys.
            verbose: Show a tqdm progress bar while loading.

        Returns:
            An :class:`ShmStore` ready for use as a DataLoader dataset backend.
        """
        keys = keys or list(source.keys)
        ep_ends = source.episode_ends
        n_frames = int(ep_ends[-1]) if len(ep_ends) > 0 else 0

        dtypes: Dict[str, np.dtype] = {}
        shapes: Dict[str, tuple] = {}
        data: Dict[str, torch.Tensor] = {}

        it = _tqdm(keys, desc="ShmStore: loading to shared memory") if verbose else keys

        for key in it:
            arr = source.get_sequence(key, 0, n_frames)
            arr = np.ascontiguousarray(arr)  # some backends return non-contiguous views
            dtypes[key] = arr.dtype
            shapes[key] = source.frame_shape(key)
            data[key] = torch.from_numpy(arr).share_memory_()

        return cls(data, ep_ends.copy(), dtypes, shapes)

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

    def get_sequence(self, key: str, start: int, end: int) -> np.ndarray:
        return self._data[key][start:end].numpy()

    def on_worker_init(self, worker_id: int) -> None:
        pass  # shared memory survives fork — nothing to reopen
