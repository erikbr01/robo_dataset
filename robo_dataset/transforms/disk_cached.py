"""Disk-cached transform: apply an offline transform and cache the result.

Cache lives under `<cache_root>/<cache_key>/` as a ZarrStore (default) or any
WritableTrajectoryStore subclass passed via writer_cls.  On a cache hit (meta.json
already exists), the transform is skipped entirely and the cached store is opened.

Cache key = sha256(apply_episode source code + repr(config))[:16]
Limitation: the hash covers the class source and __dict__ values, but not imported
helpers or library versions.  Use the `version` field on your transform class to
bust the cache when dependencies change.
"""

import collections
import logging
import os
import shutil
from typing import Any, Dict, List, Optional, Type

import numpy as np

from ..backends.base import TrajectoryStore, WritableTrajectoryStore
from ..backends.zarr_backend import ZarrStoreWriter
from .base import DiskCachedTransform

log = logging.getLogger(__name__)


class _EpisodeFailed:
    """A sentinel carrying the failure, so the writer can skip in order."""

    def __init__(self, index: int, error: BaseException):
        self.index = index
        self.error = error


def _episode_pipeline(source, transform, n_episodes: int, workers: int,
                      on_error: str = "raise"):
    """Yield transformed episodes in order, reading/transforming ahead in threads.

    The write side cannot be parallelised: `ZarrStoreWriter.put_episode` appends,
    so out-of-order writes would scramble `episode_ends`. (That is why
    `convert()`'s thread pool only engages for an indexed `WritableEpisodeStore`.)
    The read side is where the time goes, though -- decoding video and resizing --
    and both release the GIL, so threads are enough and avoid shipping
    multi-gigabyte episodes through pickle.

    In-flight work is bounded to `workers + 1` episodes rather than submitting
    everything: a source episode at native resolution can be many GB, so an
    unbounded pool would queue the entire dataset into RAM.
    """
    from concurrent.futures import ThreadPoolExecutor

    def one(i):
        if on_error == "skip":
            try:
                return i, transform.apply_episode(source.get_episode(i))
            except Exception as e:  # noqa: BLE001 - reported, then skipped
                return i, _EpisodeFailed(i, e)
        return i, transform.apply_episode(source.get_episode(i))

    with ThreadPoolExecutor(max_workers=workers) as pool:
        pending = collections.deque()
        nxt = 0
        while nxt < min(workers + 1, n_episodes):
            pending.append(pool.submit(one, nxt))
            nxt += 1
        while pending:
            result = pending.popleft().result()
            if nxt < n_episodes:
                pending.append(pool.submit(one, nxt))
                nxt += 1
            yield result


def apply_disk_cached(
    source: TrajectoryStore,
    transform: DiskCachedTransform,
    cache_root: str,
    image_keys: List[str] = (),
    writer_cls: Type[WritableTrajectoryStore] = ZarrStoreWriter,
    force: bool = False,
    workers: int = 1,
    on_error: str = "raise",
    kept_out: Optional[List[int]] = None,
) -> TrajectoryStore:
    """
    Apply transform to source and cache result under cache_root/<cache_key>/.

    Uses writer_cls to write the cache and writer_cls.open_readonly() to re-open it.
    The writer is created lazily from the first transformed episode — no double iteration.

    workers > 1 reads and transforms up to `workers + 1` episodes concurrently
    while writing them in order. Worth it when the source decodes video (an ABC
    MCAP store spends most of the pass in decode), but it costs roughly
    `workers + 1` whole episodes of RAM at *source* resolution, so raise it with
    an eye on memory rather than on core count. Requires an indexed source
    (`n_episodes` + `get_episode`); anything else falls back to serial.

    on_error="skip" logs and skips an episode the source cannot produce, instead
    of aborting. Real datasets are ragged -- ABC-130k has episodes missing a
    camera stream entirely -- and one of those should not destroy a thousand
    episodes of completed work. It is off by default because a cache silently
    missing episodes is its own hazard: pass `kept_out` to receive the source
    indices actually written, in order, which is what any positional mapping
    (an episode hold-out, say) must be built from rather than from
    `range(n_episodes)`.
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

    if on_error not in ("raise", "skip"):
        raise ValueError(f"on_error must be 'raise' or 'skip', got {on_error!r}")

    indexed = hasattr(source, "get_episode") and hasattr(source, "n_episodes")
    if workers > 1 and indexed:
        log.info("[disk_cache] reading %d episodes ahead in %d threads",
                 workers + 1, workers)
        episodes = _episode_pipeline(source, transform, source.n_episodes,
                                     workers, on_error=on_error)
    else:
        if workers > 1:
            log.warning("[disk_cache] workers=%d ignored: source is not indexed",
                        workers)
        episodes = (
            (i, transform.apply_episode(ep))
            for i, ep in enumerate(source.iter_episodes())
        )

    writer = None
    skipped = 0
    for index, transformed in episodes:
        if isinstance(transformed, _EpisodeFailed):
            skipped += 1
            log.warning("[disk_cache] skipping episode %d: %s: %s",
                        transformed.index, type(transformed.error).__name__,
                        str(transformed.error)[:200])
            continue
        if writer is None:
            writer = writer_cls.create(
                path=cache_path,
                key_shapes={k: v.shape[1:] for k, v in transformed.items()},
                key_dtypes={k: v.dtype for k, v in transformed.items()},
                image_keys=list(image_keys),
            )
        writer.put_episode(transformed)
        if kept_out is not None:
            kept_out.append(index)

    if skipped:
        log.warning("[disk_cache] %d episode(s) skipped; the cache has fewer "
                    "episodes than the source", skipped)
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
