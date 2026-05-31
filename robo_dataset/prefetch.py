"""CUDA data prefetcher — overlaps H2D transfer with model forward pass.

Usage::

    from robo_dataset.prefetch import CUDADataPrefetcher

    loader = DataLoader(dataset, batch_size=32, num_workers=4, pin_memory=True)
    prefetcher = CUDADataPrefetcher(loader, device="cuda:0")

    for batch in prefetcher:
        # batch is already on GPU; H2D for the *next* batch is in flight
        loss = model(batch)
        ...

**Bandwidth context** (PCIe 3.0 x16, batch=32, T=2, 84×84×3):

- float32 images: 5.42 MB, ~0.41 ms H2D latency
- uint8  images: 1.35 MB, ~0.10 ms H2D latency (3.8× faster)

When step time > transfer time the prefetcher completely hides the H2D cost.
Pair with ``ImageToTensorGPU`` to keep images as uint8 through the DataLoader,
deferring the uint8→float32 conversion to the GPU side.
"""

from typing import Any, Dict, Iterator

import torch


class CUDADataPrefetcher:
    """Wraps a DataLoader and prefetches the next batch to GPU on a side CUDA stream.

    The prefetch overlaps the H2D transfer for batch N+1 with the model forward
    pass on batch N.  At model-step times > 0.4 ms (almost always true) the full
    H2D latency is hidden.

    Requires ``pin_memory=True`` on the DataLoader for non-blocking transfers to
    work correctly.

    Args:
        loader:       A PyTorch DataLoader returning dict batches.
        device:       Target CUDA device (must be a CUDA device).
        non_blocking: Use non-blocking H2D transfers (default True; requires
                      pin_memory=True on the loader).
    """

    def __init__(
        self,
        loader: torch.utils.data.DataLoader,
        device: Any,
        non_blocking: bool = True,
    ):
        device = torch.device(device)
        if device.type != "cuda":
            raise ValueError(
                f"CUDADataPrefetcher requires a CUDA device, got '{device}'."
            )
        self.loader = loader
        self.device = device
        self.non_blocking = non_blocking
        self._stream = torch.cuda.Stream(device)

    def __len__(self) -> int:
        return len(self.loader)

    def _to_device(self, batch: Any) -> Any:
        """Recursively move tensors to self.device (non-blocking on side stream)."""
        if isinstance(batch, torch.Tensor):
            return batch.to(self.device, non_blocking=self.non_blocking)
        if isinstance(batch, dict):
            return {k: self._to_device(v) for k, v in batch.items()}
        if isinstance(batch, (list, tuple)):
            converted = [self._to_device(v) for v in batch]
            return type(batch)(converted)
        return batch

    def __iter__(self) -> Iterator[Any]:
        it = iter(self.loader)

        # Prefetch first batch
        try:
            next_batch = next(it)
        except StopIteration:
            return

        with torch.cuda.stream(self._stream):
            next_batch_gpu = self._to_device(next_batch)

        for raw_batch in it:
            # Wait for the prefetch of next_batch_gpu to complete before yielding it
            torch.cuda.current_stream(self.device).wait_stream(self._stream)
            batch_gpu = next_batch_gpu

            # Kick off prefetch of the following batch on the side stream
            with torch.cuda.stream(self._stream):
                next_batch_gpu = self._to_device(raw_batch)

            yield batch_gpu

        # Yield the last prefetched batch
        torch.cuda.current_stream(self.device).wait_stream(self._stream)
        yield next_batch_gpu
