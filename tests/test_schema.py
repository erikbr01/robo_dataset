"""VectorSchema: derived slices, validation, persistence through stores."""

import numpy as np
import pytest

from robo_dataset.schema import (
    Field,
    VectorSchema,
    schemas_from_meta,
    schemas_to_meta,
    validate_schemas_against_shapes,
)

ABS_ACTION = VectorSchema((
    Field("pos", 3, "pos", "abs"),
    Field("rot", 6, "rot", "rotation_6d"),
    Field("gripper", 1, "gripper", "continuous"),
))


class TestVectorSchema:
    def test_dim_and_slices_are_derived(self):
        assert ABS_ACTION.dim == 10
        s = ABS_ACTION.slices()
        assert s["pos"] == slice(0, 3)
        assert s["rot"] == slice(3, 9)
        assert s["gripper"] == slice(9, 10)

    def test_validate_dim_loud(self):
        with pytest.raises(ValueError, match="declares dim 10, data has dim 7"):
            ABS_ACTION.validate_dim(7, context="action")

    def test_rot_dim_must_match_representation(self):
        with pytest.raises(ValueError, match="rotation_6d.*6-dim"):
            Field("rot", 3, "rot", "rotation_6d")

    def test_unknown_kind_and_representation_rejected(self):
        with pytest.raises(ValueError, match="unknown kind"):
            Field("x", 3, "position")
        with pytest.raises(ValueError, match="unknown representation"):
            Field("pos", 3, "pos", "absolute")

    def test_duplicate_names_rejected(self):
        with pytest.raises(ValueError, match="Duplicate"):
            VectorSchema((Field("pos", 3, "pos"), Field("pos", 3, "pos")))

    def test_with_rotation_representation(self):
        delta = VectorSchema((
            Field("pos", 3, "pos", "delta"),
            Field("rot", 3, "rot", "axis_angle"),
            Field("gripper", 1, "gripper"),
        ))
        conv = delta.with_rotation_representation("rotation_6d")
        assert conv.dim == 10
        assert conv.field("rot").representation == "rotation_6d"
        assert conv.field("pos") == delta.field("pos")

    def test_config_roundtrip(self):
        cfg = ABS_ACTION.to_config()
        assert VectorSchema.from_config(cfg) == ABS_ACTION

    def test_fields_of_kind(self):
        assert [f.name for f in ABS_ACTION.fields_of_kind("pos")] == ["pos"]

    def test_meta_roundtrip(self):
        meta = {"schemas": schemas_to_meta({"action": ABS_ACTION})}
        out = schemas_from_meta(meta)
        assert out["action"] == ABS_ACTION
        assert schemas_from_meta({}) == {}

    def test_validate_against_shapes(self):
        validate_schemas_against_shapes({"action": ABS_ACTION}, {"action": (10,)})
        with pytest.raises(ValueError, match="not in the store"):
            validate_schemas_against_shapes({"action": ABS_ACTION}, {"obs": (10,)})
        with pytest.raises(ValueError, match="schema .* declares dim 10"):
            validate_schemas_against_shapes({"action": ABS_ACTION}, {"action": (7,)})
        with pytest.raises(ValueError, match="1-D vector key"):
            validate_schemas_against_shapes({"action": ABS_ACTION}, {"action": (84, 84, 3)})


class TestSchemaPersistence:
    def _write_store(self, writer_cls, path):
        rng = np.random.default_rng(0)
        writer = writer_cls.create(
            str(path),
            key_shapes={"action": (10,), "obs": (5,)},
            key_dtypes={"action": np.float32, "obs": np.float32},
            schemas={"action": ABS_ACTION},
        )
        for _ in range(2):
            writer.put_episode({
                "action": rng.standard_normal((7, 10)).astype(np.float32),
                "obs": rng.standard_normal((7, 5)).astype(np.float32),
            })
        writer.finalize()

    def test_zarr_roundtrip(self, tmp_path):
        from robo_dataset.backends.zarr_backend import ZarrStore, ZarrStoreWriter
        self._write_store(ZarrStoreWriter, tmp_path / "s.zarr")
        store = ZarrStore(str(tmp_path / "s.zarr"))
        assert store.schemas["action"] == ABS_ACTION
        assert "obs" not in store.schemas

    def test_video_roundtrip(self, tmp_path):
        from robo_dataset.backends.video import VideoStore, VideoStoreWriter
        self._write_store(VideoStoreWriter, tmp_path / "s.video")
        store = VideoStore(str(tmp_path / "s.video"))
        assert store.schemas["action"] == ABS_ACTION

    def test_mismatched_schema_blocks_finalize(self, tmp_path):
        from robo_dataset.backends.zarr_backend import ZarrStoreWriter
        writer = ZarrStoreWriter.create(
            str(tmp_path / "bad.zarr"),
            key_shapes={"action": (7,)},
            key_dtypes={"action": np.float32},
            schemas={"action": ABS_ACTION},  # declares 10, data is 7
        )
        writer.put_episode({"action": np.zeros((4, 7), dtype=np.float32)})
        with pytest.raises(ValueError, match="declares dim 10, data has dim 7"):
            writer.finalize()

    def test_stores_without_schema_return_empty(self, tmp_path):
        from robo_dataset.backends.zarr_backend import ZarrStore, ZarrStoreWriter
        writer = ZarrStoreWriter.create(
            str(tmp_path / "ns.zarr"),
            key_shapes={"action": (7,)},
            key_dtypes={"action": np.float32},
        )
        writer.put_episode({"action": np.zeros((4, 7), dtype=np.float32)})
        writer.finalize()
        assert ZarrStore(str(tmp_path / "ns.zarr")).schemas == {}
