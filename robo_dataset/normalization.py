"""Normalization — per-key, schema-field-aware affine normalizers.

One place computes normalization parameters from store data; consumers
(training stacks) wrap the resulting (scale, offset, stats) in whatever
container their policy expects. Everything is an elementwise affine map:

    normalized = x * scale + offset          unnormalize(x) inverts it.

Methods (``KNOWN_METHODS``):
    none         identity (scale 1, offset 0); stats still recorded
    zscore       (x - mean) / std            (std < eps dims pass through)
    minmax       affine map [min, max] → out_range (default [-1, 1]);
                 near-constant dims (range < eps) are centered, not blown up
    minmax_p99   like minmax but using the 1st/99th percentile instead of
                 min/max — robust to outlier frames
    max_abs      x / max(|min|, |max|) over ALL dims (one shared scalar);
                 the classic diffusion-policy lowdim-obs normalizer
    fixed_range  data-independent affine from a declared input range
                 (e.g. images in [0, 1] → [-1, 1]); never reads the data

Per-field granularity: when a key carries a VectorSchema, a spec may map
field *kinds* (pos / rot / gripper / joints / generic) or field *names* to
methods — e.g. the robomimic abs-action convention is ``{"pos": "minmax"}``
with default ``none``: every pos field is range-normalized, everything else
passes through. Index ranges always come from the schema; never hardcoded.

Specs are plain dicts (Hydra-friendly), e.g.::

    normalization:
      action: {per_kind: {pos: minmax}, default: none}
      obs:    max_abs
      agentview_image: {method: fixed_range, in_range: [0.0, 1.0]}

Fitted normalizers serialize to/from plain dicts (``to_dict``/``from_dict``)
so they can be persisted next to a store or inside a checkpoint.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Mapping, Optional, Sequence, Tuple, Union

import numpy as np

KNOWN_METHODS = ("none", "zscore", "minmax", "minmax_p99", "max_abs", "fixed_range")

_RANGE_EPS = 1e-7
_STD_EPS = 1e-7


def compute_stats(arr: np.ndarray) -> Dict[str, np.ndarray]:
    """Per-dimension stats over all leading axes (last axis = feature dim)."""
    flat = np.asarray(arr, dtype=np.float64).reshape(-1, arr.shape[-1])
    return {
        "min": flat.min(axis=0).astype(np.float32),
        "max": flat.max(axis=0).astype(np.float32),
        "mean": flat.mean(axis=0).astype(np.float32),
        "std": flat.std(axis=0).astype(np.float32),
        "p01": np.percentile(flat, 1, axis=0).astype(np.float32),
        "p99": np.percentile(flat, 99, axis=0).astype(np.float32),
    }


def _affine_from_range(
    lo: np.ndarray, hi: np.ndarray, out_range: Tuple[float, float]
) -> Tuple[np.ndarray, np.ndarray]:
    """[lo, hi] → out_range, with diffusion_policy's near-constant-dim rule:
    dims whose input range < eps are shifted to the output midpoint."""
    out_min, out_max = out_range
    input_range = (hi - lo).astype(np.float32)
    ignore = input_range < _RANGE_EPS
    input_range = input_range.copy()
    input_range[ignore] = out_max - out_min
    scale = (out_max - out_min) / input_range
    offset = out_min - scale * lo
    offset[ignore] = (out_max + out_min) / 2 - lo[ignore]
    return scale.astype(np.float32), offset.astype(np.float32)


def fit_affine(
    stats: Mapping[str, np.ndarray],
    method: str,
    out_range: Tuple[float, float] = (-1.0, 1.0),
    in_range: Optional[Tuple[float, float]] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """(scale, offset) for one method from per-dim stats."""
    example = np.asarray(stats["min"], dtype=np.float32)
    if method == "none":
        return np.ones_like(example), np.zeros_like(example)
    if method == "zscore":
        std = np.asarray(stats["std"], dtype=np.float32).copy()
        std[std < _STD_EPS] = 1.0
        scale = 1.0 / std
        return scale, (-np.asarray(stats["mean"], np.float32) * scale)
    if method == "minmax":
        return _affine_from_range(
            np.asarray(stats["min"], np.float32),
            np.asarray(stats["max"], np.float32), out_range)
    if method == "minmax_p99":
        return _affine_from_range(
            np.asarray(stats["p01"], np.float32),
            np.asarray(stats["p99"], np.float32), out_range)
    if method == "max_abs":
        max_abs = float(np.maximum(
            np.asarray(stats["max"]).max(), np.abs(np.asarray(stats["min"])).max()))
        max_abs = max_abs if max_abs > _STD_EPS else 1.0
        return (np.full_like(example, 1.0 / max_abs), np.zeros_like(example))
    if method == "fixed_range":
        if in_range is None:
            raise ValueError("fixed_range requires in_range=(lo, hi)")
        lo, hi = float(in_range[0]), float(in_range[1])
        out_min, out_max = out_range
        s = (out_max - out_min) / (hi - lo)
        return (np.full_like(example, s, dtype=np.float32),
                np.full_like(example, out_min - s * lo, dtype=np.float32))
    raise ValueError(f"Unknown normalization method {method!r} (known: {KNOWN_METHODS})")


@dataclass
class KeyNormalizer:
    """Elementwise affine normalizer for one key: x*scale + offset."""

    scale: np.ndarray
    offset: np.ndarray
    stats: Dict[str, np.ndarray] = field(default_factory=dict)
    method: str = "none"

    def normalize(self, x):
        scale, offset = self._as_backend(x)
        return x * scale + offset

    def unnormalize(self, x):
        scale, offset = self._as_backend(x)
        return (x - offset) / scale

    def _as_backend(self, x):
        try:
            import torch
            if isinstance(x, torch.Tensor):
                return (torch.as_tensor(self.scale, dtype=x.dtype, device=x.device),
                        torch.as_tensor(self.offset, dtype=x.dtype, device=x.device))
        except ImportError:
            pass
        return self.scale, self.offset

    def to_dict(self) -> dict:
        return {
            "method": self.method,
            "scale": np.asarray(self.scale).tolist(),
            "offset": np.asarray(self.offset).tolist(),
            "stats": {k: np.asarray(v).tolist() for k, v in self.stats.items()},
        }

    @classmethod
    def from_dict(cls, d: Mapping) -> "KeyNormalizer":
        return cls(
            scale=np.asarray(d["scale"], dtype=np.float32),
            offset=np.asarray(d["offset"], dtype=np.float32),
            stats={k: np.asarray(v, dtype=np.float32) for k, v in d.get("stats", {}).items()},
            method=str(d.get("method", "none")),
        )


# ── spec handling ─────────────────────────────────────────────────────────────
#
# A key spec is either a method string, or a dict:
#   {method: minmax, out_range: [-1, 1]}                      whole key
#   {method: fixed_range, in_range: [0, 1]}                   data-independent
#   {per_kind: {pos: minmax}, default: none}                  schema-field kinds
#   {per_field: {eef_pos: minmax}, default: none}             schema-field names


def _normalize_spec(spec: Union[str, Mapping]) -> Mapping:
    if isinstance(spec, str):
        return {"method": spec}
    return dict(spec)


def fit_key(
    arr: Optional[np.ndarray],
    spec: Union[str, Mapping],
    schema=None,
) -> KeyNormalizer:
    """Fit one key's normalizer from data (or from a fixed range, dataless).

    With ``per_kind``/``per_field`` the key must carry a VectorSchema; each
    field's slice gets its own method and the results are concatenated —
    no index ranges appear anywhere outside the schema.
    """
    spec = _normalize_spec(spec)
    out_range = tuple(spec.get("out_range", (-1.0, 1.0)))

    if "per_kind" in spec or "per_field" in spec:
        if schema is None:
            raise ValueError(
                f"Spec {spec} uses per-field granularity but the key has no "
                "VectorSchema — re-convert the store with a schemas: block."
            )
        if arr is None:
            raise ValueError("per-field normalization requires data to fit on")
        schema.validate_dim(arr.shape[-1], context="normalization input")
        stats = compute_stats(arr)
        per_kind = spec.get("per_kind", {}) or {}
        per_field = spec.get("per_field", {}) or {}
        default = str(spec.get("default", "none"))
        scale = np.ones(arr.shape[-1], dtype=np.float32)
        offset = np.zeros(arr.shape[-1], dtype=np.float32)
        slices = schema.slices()
        methods = []
        for f in schema.fields:
            method = str(per_field.get(f.name, per_kind.get(f.kind, default)))
            sl = slices[f.name]
            sub_stats = {k: v[sl] for k, v in stats.items()}
            s, o = fit_affine(sub_stats, method, out_range=out_range)
            scale[sl], offset[sl] = s, o
            methods.append(f"{f.name}:{method}")
        return KeyNormalizer(scale=scale, offset=offset, stats=stats,
                             method="per_field(" + ",".join(methods) + ")")

    method = str(spec.get("method", "none"))
    if method == "fixed_range":
        in_range = spec.get("in_range")
        if in_range is None:
            raise ValueError("fixed_range spec requires in_range: [lo, hi]")
        lo, hi = float(in_range[0]), float(in_range[1])
        example = np.zeros(1, dtype=np.float32)
        scale, offset = fit_affine(
            {"min": example}, "fixed_range", out_range=out_range,
            in_range=(lo, hi))
        stats = {
            "min": np.array([lo], dtype=np.float32),
            "max": np.array([hi], dtype=np.float32),
            "mean": np.array([(lo + hi) / 2], dtype=np.float32),
            "std": np.array([np.sqrt((hi - lo) ** 2 / 12)], dtype=np.float32),
        }
        return KeyNormalizer(scale=scale, offset=offset, stats=stats, method=method)

    if arr is None:
        raise ValueError(f"Method {method!r} requires data to fit on")
    stats = compute_stats(arr)
    scale, offset = fit_affine(stats, method, out_range=out_range)
    return KeyNormalizer(scale=scale, offset=offset, stats=stats, method=method)


def fit_store_normalizers(
    store,
    spec: Mapping[str, Union[str, Mapping]],
    keys: Optional[Sequence[str]] = None,
) -> Dict[str, KeyNormalizer]:
    """Fit normalizers for store keys per spec.

    ``spec`` maps key → key-spec (see module docstring). Keys without a spec
    entry are skipped. ``fixed_range`` entries never read data; everything
    else reads the key's full array once (lowdim/action-sized reads — do not
    point data-dependent methods at video keys).
    """
    out: Dict[str, KeyNormalizer] = {}
    for key in keys if keys is not None else list(spec):
        if key not in spec:
            continue
        key_spec = _normalize_spec(spec[key])
        needs_data = key_spec.get("method") != "fixed_range"
        arr = None
        if needs_data:
            arr = np.asarray(store.get_sequence(key, 0, store.n_frames))
        out[key] = fit_key(arr, key_spec, schema=store.schemas.get(key))
    return out
