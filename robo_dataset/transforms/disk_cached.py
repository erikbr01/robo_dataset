"""Disk-cached transform: apply an offline transform and cache the result.

Cache lives under `<cache_root>/<cache_key>/` as a ZarrStore (default) or any
WritableTrajectoryStore subclass passed via writer_cls.  On a cache hit (meta.json
already exists), the transform is skipped entirely and the cached store is opened.

Cache key = sha256(apply_episode source code + repr(config))[:16]
Limitation: the hash covers the class source and __dict__ values, but not imported
helpers or library versions.  Use the `version` field on your transform class to
bust the cache when dependencies change.
"""

import logging
import os
import shutil
from typing import Any, Dict, List, Optional, Type

import numpy as np

from ..backends.base import TrajectoryStore, WritableTrajectoryStore
from ..backends.zarr_backend import ZarrStoreWriter
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

    Uses writer_cls to write the cache and writer_cls.open_readonly() to re-open it.
    The writer is created lazily from the first transformed episode — no double iteration.
    """
    cache_key = transform.cache_key()
    cache_path = os.path.join(cache_root, cache_key)
    meta_file = os.path.join(cache_path, "meta.json")

    if not force and os.path.exists(meta_file):
        log.info(f"[disk_cache] HIT  {cache_key}  {cache_path}")
        return writer_cls.open_readonly(cache_path)

    if force and os.path.exists(cache_path):
        shutil.rmtree(cache_path)

    log.info(f"[disk_cache] MISS {cache_key}  → building {cache_path}")

    writer = None
    for ep_data in source.iter_episodes():
        transformed = transform.apply_episode(ep_data)
        if writer is None:
            writer = writer_cls.create(
                path=cache_path,
                key_shapes={k: v.shape[1:] for k, v in transformed.items()},
                key_dtypes={k: v.dtype for k, v in transformed.items()},
                image_keys=list(image_keys),
            )
        writer.put_episode(transformed)

    if writer is None:
        raise ValueError("Source store is empty — nothing to cache.")
    writer.finalize()
    log.info(f"[disk_cache] DONE {cache_key}")
    return writer_cls.open_readonly(cache_path)


# ── Built-in disk transforms ──────────────────────────────────────────────────

class ResizeImages(DiskCachedTransform):
    """Resize all image keys to target_h × target_w using cv2.INTER_AREA.

    By default the frame is stretched to exactly the target, which is fine when
    every source frame already shares one aspect ratio. When a store mixes aspect
    ratios it is not: stretching then distorts one group of episodes relative to
    another, giving a policy a cue that has nothing to do with the task. ABC-130k
    is such a store -- in its annotated bottles task, 28 of 40 episodes record at
    848x480 (16:9) and 8 at 640x480 (4:3). Pass ``letterbox=True`` to scale to fit
    and pad instead, which keeps objects the same shape everywhere and confines
    the difference to the padding.
    """

    def __init__(self, image_keys: List[str], target_h: int, target_w: int,
                 letterbox: bool = False, pad_value: int = 0, version: str = "1"):
        self.image_keys = image_keys
        self.target_h = target_h
        self.target_w = target_w
        self.letterbox = bool(letterbox)
        self.pad_value = int(pad_value)
        self.version = version

    def _fit(self, h: int, w: int) -> tuple:
        """Scaled size and top-left offset for a letterboxed frame."""
        scale = min(self.target_h / h, self.target_w / w)
        # round, not floor: floor can lose a pixel row on exact-fit sources.
        sh = max(1, min(self.target_h, int(round(h * scale))))
        sw = max(1, min(self.target_w, int(round(w * scale))))
        return sh, sw, (self.target_h - sh) // 2, (self.target_w - sw) // 2

    def apply_episode(self, ep_data: Dict[str, Any]) -> Dict[str, Any]:
        import cv2
        result = dict(ep_data)
        for key in self.image_keys:
            if key not in ep_data:
                # Disk-cached output is permanent — silently skipping a
                # configured key would bake the omission into the cache.
                raise KeyError(
                    f"ResizeImages: configured image key {key!r} is missing from "
                    f"the episode (available keys: {sorted(ep_data)})."
                )
            frames = ep_data[key]  # (T, H, W, C) uint8
            T = frames.shape[0]
            out = np.empty((T, self.target_h, self.target_w, frames.shape[3]), dtype=frames.dtype)
            if not self.letterbox:
                for t in range(T):
                    out[t] = cv2.resize(frames[t], (self.target_w, self.target_h), interpolation=cv2.INTER_AREA)
            else:
                sh, sw, y0, x0 = self._fit(frames.shape[1], frames.shape[2])
                out.fill(self.pad_value)
                for t in range(T):
                    out[t, y0:y0 + sh, x0:x0 + sw] = cv2.resize(
                        frames[t], (sw, sh), interpolation=cv2.INTER_AREA)
            result[key] = out
        return result
