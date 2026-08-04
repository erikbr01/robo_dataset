"""Accelerator-agnostic data prefetcher.

Strategy per device type:

  cuda / xpu  — **side-stream**: H2D on a dedicated stream; overlap is zero-copy
                and nearly free.  Requires ``pin_memory=True`` on the DataLoader.
  mps / other — **background-thread**: H2D in a daemon thread with a 1-batch
                look-ahead queue.  MPS has no public stream API, so a thread
                provides the same overlap at the cost of one extra Python object.
  cpu         — **sync**: no H2D needed; yields batches unchanged.

The chosen strategy is exposed as ``prefetcher.strategy`` (``"stream"``,
``"thread"``, or ``"sync"``).

Usage::

    from robo_dataset.prefetch import DataPrefetcher

    loader = DataLoader(dataset, batch_size=32, num_workers=4,
                        pin_memory=(device.type == "cuda"))
    prefetcher = DataPrefetcher(loader, device=device)

    for batch in prefetcher:           # batch already on device
        loss = model(batch)
        ...

**Bandwidth context** (PCIe 3.0 x16, batch=32, T=2, 84×84×3):

- float32 images: 5.42 MB, ~0.41 ms H2D
- uint8  images: 1.35 MB, ~0.10 ms H2D  (3.8× cheaper)

Pair with ``ImageToTensorDevice`` to keep images as uint8 through the
DataLoader and convert on-device after transfer.
"""

import queue
import threading
import warnings
from typing import Any

import torch

# ── module-level sentinel — cheaper than checking isinstance ──────────────────
_DONE = object()


# ── device-transfer helpers ───────────────────────────────────────────────────

def _to_device(batch: Any, device: torch.device, non_blocking: bool) -> Any:
    """Recursively move tensors to *device*; leaves non-tensors untouched."""
    if isinstance(batch, torch.Tensor):
        return batch.to(device, non_blocking=non_blocking)
    if isinstance(batch, dict):
        return {k: _to_device(v, device, non_blocking) for k, v in batch.items()}
    if isinstance(batch, (list, tuple)):
        out = [_to_device(v, device, non_blocking) for v in batch]
        return type(batch)(out)
    return batch


def _make_stream(device: torch.device):
    """Return a side stream for *device*, or None if unsupported."""
    if device.type == "cuda":
        return torch.cuda.Stream(device)
    if device.type == "xpu" and hasattr(torch, "xpu") and hasattr(torch.xpu, "Stream"):
        return torch.xpu.Stream(device)
    return None


def _wait_stream(device: torch.device, stream) -> None:
    """Block the current stream on *device* until *stream* has caught up."""
    if device.type == "cuda":
        torch.cuda.current_stream(device).wait_stream(stream)
    elif device.type == "xpu":
        torch.xpu.current_stream(device).wait_stream(stream)


def _record_current_stream(batch: Any, device: torch.device) -> None:
    """Keep side-stream allocations alive through main-stream consumption."""
    if device.type == "cuda":
        stream = torch.cuda.current_stream(device)
    elif device.type == "xpu":
        stream = torch.xpu.current_stream(device)
    else:
        return
    if isinstance(batch, torch.Tensor):
        batch.record_stream(stream)
    elif isinstance(batch, dict):
        for value in batch.values():
            _record_current_stream(value, device)
    elif isinstance(batch, (list, tuple)):
        for value in batch:
            _record_current_stream(value, device)


def _stream_context(device: torch.device, stream):
    """Return the stream context-manager for *device*."""
    if device.type == "cuda":
        return torch.cuda.stream(stream)
    if device.type == "xpu":
        return torch.xpu.stream(stream)
    raise RuntimeError(f"No stream context for device type {device.type!r}")


# ── main class ────────────────────────────────────────────────────────────────

class DataPrefetcher:
    """Accelerator-agnostic DataLoader prefetcher.

    Args:
        loader:       A DataLoader returning dict (or list/tuple) batches.
        device:       Target device — any of ``"cuda"``, ``"cuda:N"``, ``"mps"``,
                      ``"xpu"``, ``"cpu"``, or a ``torch.device`` instance.
        non_blocking: Use non-blocking H2D copies (default True).  Effective
                      only when ``pin_memory=True`` on the DataLoader; has no
                      impact on MPS (MPS copies are internally synchronous).
    """

    _STREAM_DEVICES = frozenset({"cuda", "xpu"})

    def __init__(
        self,
        loader: torch.utils.data.DataLoader,
        device: Any,
        non_blocking: bool = True,
    ):
        self.loader = loader
        self.device = torch.device(device)
        self.non_blocking = non_blocking

        if self.device.type in self._STREAM_DEVICES:
            stream = _make_stream(self.device)
            if stream is not None:
                self._stream = stream
                self.strategy: str = "stream"
            else:
                # xpu detected but torch.xpu.Stream not available
                self._stream = None
                self.strategy = "thread"
        elif self.device.type == "cpu":
            self._stream = None
            self.strategy = "sync"
        else:
            # mps, hpu, npu, and any future accelerator without a stream API
            self._stream = None
            self.strategy = "thread"

        if self.strategy == "stream" and not getattr(loader, "pin_memory", False):
            warnings.warn(
                "DataPrefetcher strategy='stream' requires pin_memory=True on the "
                "DataLoader for non-blocking H2D transfers to overlap with compute. "
                "Pass pin_memory=True when constructing your DataLoader.",
                stacklevel=2,
            )

    def __len__(self) -> int:
        return len(self.loader)

    def __iter__(self):
        if self.strategy == "stream":
            yield from self._stream_iter()
        elif self.strategy == "thread":
            yield from self._thread_iter()
        else:
            yield from self._sync_iter()

    # ── strategy implementations ──────────────────────────────────────────────

    def _stream_iter(self):
        """Side-stream prefetch for CUDA and XPU.

        Kicks off H2D for batch N+1 on a dedicated side stream while the main
        stream processes batch N.  ``wait_stream`` ensures the main stream only
        reads a batch after its H2D is complete — no full device sync needed.

        A ``num_workers=0`` DataLoader normally constructs the next CPU batch
        synchronously when this generator resumes, leaving the accelerator
        idle before the side-stream copy can begin. In that mode a small
        producer thread supplies pinned CPU batches one step ahead. This avoids
        multiprocessing/IPC for very large image batches while overlapping the
        vectorized host gather with the current model update.
        """
        it = (
            self._threaded_loader_iter()
            if getattr(self.loader, "num_workers", 0) == 0
            else iter(self.loader)
        )
        try:
            first = next(it)
        except StopIteration:
            return

        with _stream_context(self.device, self._stream):
            next_gpu = _to_device(first, self.device, self.non_blocking)

        for raw in it:
            _wait_stream(self.device, self._stream)
            batch_gpu = next_gpu
            _record_current_stream(batch_gpu, self.device)

            with _stream_context(self.device, self._stream):
                next_gpu = _to_device(raw, self.device, self.non_blocking)

            yield batch_gpu

        _wait_stream(self.device, self._stream)
        _record_current_stream(next_gpu, self.device)
        yield next_gpu

    def _threaded_loader_iter(self):
        """Iterate a single-process DataLoader on a bounded producer thread."""
        batches: queue.Queue = queue.Queue(maxsize=2)
        stop = threading.Event()

        def put(item: Any) -> bool:
            while not stop.is_set():
                try:
                    batches.put(item, timeout=0.1)
                    return True
                except queue.Full:
                    continue
            return False

        def producer():
            try:
                for batch in self.loader:
                    if not put(batch):
                        return
            except Exception as exc:  # noqa: BLE001
                put(exc)
                return
            put(_DONE)

        thread = threading.Thread(
            target=producer,
            daemon=True,
            name="robo-dataset-prefetch",
        )
        thread.start()
        try:
            while True:
                item = batches.get()
                if item is _DONE:
                    return
                if isinstance(item, Exception):
                    raise item
                yield item
        finally:
            stop.set()

    def _thread_iter(self):
        """Background-thread prefetch for MPS and other devices.

        A daemon thread moves the next batch to device while the main thread
        processes the current one.  Queue maxsize=2 lets one batch sit ready
        plus room for an error or the done sentinel without deadlocking.
        """
        q: queue.Queue = queue.Queue(maxsize=2)

        def worker():
            try:
                for batch in self.loader:
                    q.put(_to_device(batch, self.device, self.non_blocking))
            except Exception as exc:  # noqa: BLE001
                q.put(exc)
                return
            q.put(_DONE)

        t = threading.Thread(target=worker, daemon=True)
        t.start()

        while True:
            item = q.get()
            if item is _DONE:
                break
            if isinstance(item, Exception):
                raise item
            yield item

    def _sync_iter(self):
        """Synchronous transfer — used for CPU targets."""
        for batch in self.loader:
            yield _to_device(batch, self.device, self.non_blocking)


# ── backward-compatible alias ─────────────────────────────────────────────────

#: Alias kept for backward compatibility.  New code should use :class:`DataPrefetcher`.
CUDADataPrefetcher = DataPrefetcher
