"""Disk-cached transform: apply an offline transform and cache in a ZarrStoreWriter.

The cache is stored under `<cache_root>/<cache_key>/` using the same storage backend
as the target (ZarrStoreWriter by default). If the directory already exists and
contains a valid `meta.json`, the transform is skipped and the cached store is returned.

Cache key = sha256(source_code(apply_episode) + repr(config_dict))[:16]
Limitation: the hash only covers the class source and explicit config, not imported helpers.
Set `version` in the config to bust the cache when deps change.
"""

import logging
import os
import shutil
from typing import Any, Dict, List, Optional, Type

import numpy as np

from ..backends.base import TrajectoryStore, WritableTrajectoryStore
from ..backends.zarr_backend import ZarrStore, ZarrStoreWriter
from ..convert import convert
from .base import DiskCachedTransform

log = logging.getLogger(__name__)


def apply_disk_cached(
    source: TrajectoryStore,
    transform: DiskCachedTransform,
    cache_root: str,
    image_keys: List[str] = (),
    writer_cls: Type[WritableTrajectoryStore] = ZarrStoreWriter,
    force: bool = False,
) -> TrajectoryStore:
    """
    Apply transform to source and cache result under cache_root/<cache_key>/.

    If cache already exists (and force=False), opens and returns it immediately.
    The returned store is a read-only ZarrStore (or writer_cls's read counterpart).
    """
    cache_key = transform.cache_key()
    cache_path = os.path.join(cache_root, cache_key)
    meta_file = os.path.join(cache_path, "meta.json")

    if not force and os.path.exists(meta_file):
        log.info(f"[disk_cache] HIT  {cache_key}  {cache_path}")
        return ZarrStore(cache_path)

    if force and os.path.exists(cache_path):
        shutil.rmtree(cache_path)

    log.info(f"[disk_cache] MISS {cache_key}  → building {cache_path}")

    # Probe shapes/dtypes from a single episode to set up writer
    first_ep = next(source.iter_episodes())
    transformed_first = transform.apply_episode(first_ep)
    key_shapes = {k: v.shape[1:] for k, v in transformed_first.items()}
    key_dtypes = {k: v.dtype for k, v in transformed_first.items()}

    writer = ZarrStoreWriter.create(
        path=cache_path,
        key_shapes=key_shapes,
        key_dtypes=key_dtypes,
        image_keys=list(image_keys),
    )
    # Write first episode (already transformed above)
    writer.put_episode(transformed_first)

    # Stream remaining episodes
    for ep_idx, ep_data in enumerate(source.iter_episodes()):
        if ep_idx == 0:
            continue  # already written
        writer.put_episode(transform.apply_episode(ep_data))

    writer.finalize()
    log.info(f"[disk_cache] DONE {cache_key}")
    return ZarrStore(cache_path)


# ── Built-in disk transforms ──────────────────────────────────────────────────

class ResizeImages(DiskCachedTransform):
    """Resize all image keys to target_h × target_w using cv2.INTER_AREA."""

    def __init__(self, image_keys: List[str], target_h: int, target_w: int, version: str = "1"):
        self.image_keys = image_keys
        self.target_h = target_h
        self.target_w = target_w
        self.version = version

    def apply_episode(self, ep_data: Dict[str, Any]) -> Dict[str, Any]:
        import cv2
        result = dict(ep_data)
        for key in self.image_keys:
            if key not in ep_data:
                continue
            frames = ep_data[key]  # (T, H, W, C) uint8
            T = frames.shape[0]
            out = np.empty((T, self.target_h, self.target_w, frames.shape[3]), dtype=frames.dtype)
            for t in range(T):
                out[t] = cv2.resize(frames[t], (self.target_w, self.target_h), interpolation=cv2.INTER_AREA)
            result[key] = out
        return result
