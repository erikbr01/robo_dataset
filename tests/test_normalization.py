"""Normalization: methods, schema-field granularity, round-trips, parity.

Parity targets are the diffusion-policy conventions this module replaces:
minmax == get_range_normalizer_from_stat (incl. the near-constant-dim rule),
max_abs == normalizer_from_stat, fixed_range([0,1]) == get_image_range_normalizer.
"""

import numpy as np
import pytest

from robo_dataset.normalization import (
    KNOWN_METHODS,
    KeyNormalizer,
    compute_stats,
    fit_affine,
    fit_key,
    fit_store_normalizers,
)
from robo_dataset.schema import Field, VectorSchema

RNG = np.random.default_rng(0)


def _arr(n=500, d=4, scale=10.0):
    return (RNG.standard_normal((n, d)) * scale).astype(np.float32)


class TestMethods:
    def test_none_is_identity(self):
        x = _arr()
        kn = fit_key(x, "none")
        np.testing.assert_array_equal(kn.normalize(x), x)

    def test_zscore(self):
        x = _arr()
        kn = fit_key(x, "zscore")
        z = kn.normalize(x.astype(np.float64))
        np.testing.assert_allclose(z.mean(axis=0), 0, atol=1e-5)
        np.testing.assert_allclose(z.std(axis=0), 1, atol=1e-4)

    def test_minmax_maps_to_range(self):
        x = _arr()
        kn = fit_key(x, "minmax")
        z = kn.normalize(x)
        np.testing.assert_allclose(z.min(axis=0), -1, atol=1e-6)
        np.testing.assert_allclose(z.max(axis=0), 1, atol=1e-6)

    def test_minmax_constant_dim_centered(self):
        x = _arr()
        x[:, 1] = 3.14  # constant dim
        kn = fit_key(x, "minmax")
        z = kn.normalize(x)
        # dp convention: constant dims are shifted to the output midpoint
        np.testing.assert_allclose(z[:, 1], 0.0, atol=1e-6)

    def test_minmax_p99_clips_outliers_influence(self):
        x = _arr(n=1000, d=1, scale=1.0)
        x[0, 0] = 1e6  # single outlier
        plain = fit_key(x, "minmax")
        robust = fit_key(x, "minmax_p99")
        # plain: outlier compresses everything; robust: body still spans ~[-1,1]
        body = x[1:, 0]
        assert plain.normalize(body).max() < -0.9
        assert robust.normalize(body).max() > 0.8
        # and the p99 bounds map to exactly [-1, 1]
        p01, p99 = np.percentile(x[:, 0], [1, 99])
        np.testing.assert_allclose(
            robust.normalize(np.array([p01, p99], dtype=np.float32)),
            [-1.0, 1.0], atol=1e-4)

    def test_max_abs_shared_scalar(self):
        x = _arr()
        kn = fit_key(x, "max_abs")
        max_abs = max(x.max(), -x.min())
        np.testing.assert_allclose(kn.scale, 1.0 / max_abs, rtol=1e-6)
        np.testing.assert_array_equal(kn.offset, np.zeros(x.shape[-1]))

    def test_fixed_range_dataless(self):
        kn = fit_key(None, {"method": "fixed_range", "in_range": [0.0, 1.0]})
        # dp get_image_range_normalizer parity: scale 2, offset -1
        np.testing.assert_allclose(kn.scale, [2.0])
        np.testing.assert_allclose(kn.offset, [-1.0])
        np.testing.assert_allclose(kn.stats["std"], [np.sqrt(1 / 12)], rtol=1e-6)

    def test_unknown_method_raises(self):
        with pytest.raises(ValueError, match="Unknown normalization method"):
            fit_key(_arr(), "standard")

    def test_roundtrip_all_methods(self):
        x = _arr()
        for method in ("none", "zscore", "minmax", "minmax_p99", "max_abs"):
            kn = fit_key(x, method)
            np.testing.assert_allclose(
                kn.unnormalize(kn.normalize(x.astype(np.float64))), x, atol=1e-4,
                err_msg=method)

    def test_torch_tensors(self):
        import torch
        x = _arr()
        kn = fit_key(x, "minmax")
        z_np = kn.normalize(x)
        z_t = kn.normalize(torch.from_numpy(x))
        np.testing.assert_allclose(z_t.numpy(), z_np, atol=1e-6)


class TestDpParity:
    """Bit-parity with the diffusion-policy formulas this module replaces."""

    def test_minmax_matches_get_range_normalizer_from_stat(self):
        x = _arr()
        x[:, 2] = 7.0  # exercise the near-constant rule
        stats = compute_stats(x)
        scale, offset = fit_affine(stats, "minmax")
        # reference implementation (dp normalize_util)
        input_max, input_min = stats["max"].copy(), stats["min"].copy()
        input_range = input_max - input_min
        ignore = input_range < 1e-7
        input_range[ignore] = 2.0
        ref_scale = 2.0 / input_range
        ref_offset = -1 - ref_scale * input_min
        ref_offset[ignore] = -input_min[ignore]
        np.testing.assert_allclose(scale, ref_scale, rtol=1e-7)
        np.testing.assert_allclose(offset, ref_offset, rtol=1e-7, atol=1e-7)

    def test_max_abs_matches_normalizer_from_stat(self):
        x = _arr()
        stats = compute_stats(x)
        scale, offset = fit_affine(stats, "max_abs")
        max_abs = np.maximum(stats["max"].max(), np.abs(stats["min"]).max())
        np.testing.assert_allclose(scale, np.full(x.shape[-1], 1 / max_abs), rtol=1e-7)
        np.testing.assert_array_equal(offset, np.zeros(x.shape[-1]))


class TestSchemaFieldGranularity:
    SCHEMA = VectorSchema((
        Field("pos", 3, "pos", "abs"),
        Field("rot", 6, "rot", "rotation_6d"),
        Field("gripper", 1, "gripper", "continuous"),
    ))

    def test_per_kind(self):
        x = _arr(d=10)
        kn = fit_key(x, {"per_kind": {"pos": "minmax"}, "default": "none"},
                     schema=self.SCHEMA)
        z = kn.normalize(x)
        # pos dims mapped to [-1, 1]; rot/gripper untouched
        np.testing.assert_allclose(z[:, :3].min(axis=0), -1, atol=1e-6)
        np.testing.assert_allclose(z[:, :3].max(axis=0), 1, atol=1e-6)
        np.testing.assert_array_equal(z[:, 3:], x[:, 3:])

    def test_per_field_name_overrides_kind(self):
        x = _arr(d=10)
        kn = fit_key(x, {"per_kind": {"pos": "minmax"},
                         "per_field": {"pos": "zscore"}, "default": "none"},
                     schema=self.SCHEMA)
        z = kn.normalize(x.astype(np.float64))
        np.testing.assert_allclose(z[:, :3].mean(axis=0), 0, atol=1e-5)

    def test_missing_schema_raises(self):
        with pytest.raises(ValueError, match="no VectorSchema"):
            fit_key(_arr(d=10), {"per_kind": {"pos": "minmax"}}, schema=None)

    def test_dim_mismatch_raises(self):
        with pytest.raises(ValueError, match="declares dim 10"):
            fit_key(_arr(d=7), {"per_kind": {"pos": "minmax"}}, schema=self.SCHEMA)


class TestSerialization:
    def test_roundtrip(self):
        kn = fit_key(_arr(), "minmax")
        kn2 = KeyNormalizer.from_dict(kn.to_dict())
        np.testing.assert_allclose(kn2.scale, kn.scale)
        np.testing.assert_allclose(kn2.offset, kn.offset)
        np.testing.assert_allclose(kn2.stats["p99"], kn.stats["p99"])


class TestFitStoreNormalizers:
    def test_fit_from_store(self, tmp_path):
        from robo_dataset.backends.zarr_backend import ZarrStore, ZarrStoreWriter
        schema = VectorSchema((
            Field("pos", 3, "pos", "abs"),
            Field("gripper", 1, "gripper"),
        ))
        writer = ZarrStoreWriter.create(
            str(tmp_path / "s.zarr"),
            key_shapes={"action": (4,), "obs": (5,)},
            key_dtypes={"action": np.float32, "obs": np.float32},
            schemas={"action": schema},
        )
        for _ in range(3):
            writer.put_episode({
                "action": _arr(50, 4), "obs": _arr(50, 5),
            })
        writer.finalize()
        store = ZarrStore(str(tmp_path / "s.zarr"))
        norms = fit_store_normalizers(store, {
            "action": {"per_kind": {"pos": "minmax"}, "default": "none"},
            "obs": "max_abs",
        })
        assert set(norms) == {"action", "obs"}
        assert norms["obs"].method == "max_abs"
        # gripper dim untouched
        np.testing.assert_allclose(norms["action"].scale[3], 1.0)
