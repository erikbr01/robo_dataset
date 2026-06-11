"""Vector field schemas — named, ordered, dimensioned fields for vector keys.

A VectorSchema declares what the components of a vector-valued dataset key
(action, concatenated state, ...) mean, so that NO consumer ever hardcodes
index ranges:

    action = VectorSchema.from_config([
        {"name": "pos",     "dim": 3, "kind": "pos",     "representation": "abs"},
        {"name": "rot",     "dim": 6, "kind": "rot",     "representation": "rotation_6d"},
        {"name": "gripper", "dim": 1, "kind": "gripper", "representation": "continuous"},
    ])
    action.dim                # 10  (derived — never stated separately)
    action.slices()["rot"]    # slice(3, 9)  (derived from order + dims)

Index ranges are always derived from (order, dim); they are not part of the
schema and cannot drift out of sync with it.

Field kinds (drive downstream behavior, e.g. normalizer family):
    pos      cartesian position            representation: abs | delta
    rot      rotation                      representation: euler | quat |
                                           axis_angle | rotation_6d
    gripper  gripper command/state         representation: continuous | binary
    joints   joint positions/velocities    representation: abs | delta | velocity
    generic  anything else                 representation: free-form

Schemas are persisted in store metadata by the writers (``schemas=`` argument
to ``create()``) and exposed by the readers (``store.schemas``), making a
converted dataset self-describing: a consumer can verify at load time that
the vector layout it expects is the layout the data actually has — loudly.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

KNOWN_KINDS = ("pos", "rot", "gripper", "joints", "generic")
KNOWN_REPRESENTATIONS = {
    "pos": ("abs", "delta"),
    "rot": ("euler", "quat", "axis_angle", "rotation_6d"),
    "gripper": ("continuous", "binary"),
    "joints": ("abs", "delta", "velocity"),
}
ROTATION_DIMS = {"euler": 3, "quat": 4, "axis_angle": 3, "rotation_6d": 6}


@dataclass(frozen=True)
class Field:
    """One named span of a vector: (name, dim, kind, representation)."""

    name: str
    dim: int
    kind: str = "generic"
    representation: Optional[str] = None

    def __post_init__(self):
        if self.dim < 1:
            raise ValueError(f"Field {self.name!r}: dim must be >= 1, got {self.dim}")
        if self.kind not in KNOWN_KINDS:
            raise ValueError(
                f"Field {self.name!r}: unknown kind {self.kind!r} "
                f"(known: {KNOWN_KINDS})"
            )
        allowed = KNOWN_REPRESENTATIONS.get(self.kind)
        if allowed and self.representation is not None and self.representation not in allowed:
            raise ValueError(
                f"Field {self.name!r} (kind={self.kind}): unknown representation "
                f"{self.representation!r} (known: {allowed})"
            )
        if self.kind == "rot" and self.representation is not None:
            expected = ROTATION_DIMS[self.representation]
            if self.dim != expected:
                raise ValueError(
                    f"Field {self.name!r}: rot representation "
                    f"{self.representation!r} is {expected}-dim, got dim={self.dim}"
                )


@dataclass(frozen=True)
class VectorSchema:
    """Ordered fields of one vector-valued key. Index ranges are derived."""

    fields: Tuple[Field, ...]

    def __post_init__(self):
        names = [f.name for f in self.fields]
        if len(names) != len(set(names)):
            raise ValueError(f"Duplicate field names in schema: {names}")
        if not self.fields:
            raise ValueError("VectorSchema needs at least one field")

    @property
    def dim(self) -> int:
        return sum(f.dim for f in self.fields)

    def slices(self) -> Dict[str, slice]:
        out, start = {}, 0
        for f in self.fields:
            out[f.name] = slice(start, start + f.dim)
            start += f.dim
        return out

    def field(self, name: str) -> Field:
        for f in self.fields:
            if f.name == name:
                return f
        raise KeyError(f"No field {name!r} in schema (have: {[f.name for f in self.fields]})")

    def fields_of_kind(self, kind: str) -> List[Field]:
        return [f for f in self.fields if f.kind == kind]

    def validate_dim(self, actual_dim: int, context: str = "") -> None:
        """Raise loudly when the data's vector width does not match the schema."""
        if actual_dim != self.dim:
            raise ValueError(
                f"Vector schema mismatch{' for ' + context if context else ''}: "
                f"schema {self.describe()} declares dim {self.dim}, "
                f"data has dim {actual_dim}"
            )

    def describe(self) -> str:
        return "[" + ", ".join(
            f"{f.name}({f.dim}:{f.kind}"
            + (f"/{f.representation}" if f.representation else "")
            + ")"
            for f in self.fields
        ) + "]"

    def replace_field(self, name: str, **changes) -> "VectorSchema":
        """Return a new schema with one field replaced (e.g. after a rotation
        representation conversion changes its dim)."""
        from dataclasses import replace as dc_replace
        new_fields = tuple(
            dc_replace(f, **changes) if f.name == name else f for f in self.fields
        )
        if all(f is g for f, g in zip(new_fields, self.fields)):
            raise KeyError(f"No field {name!r} in schema")
        return VectorSchema(new_fields)

    def with_rotation_representation(self, to_rep: str) -> "VectorSchema":
        """Schema after converting every rot field to ``to_rep`` (dim follows)."""
        fields = []
        for f in self.fields:
            if f.kind == "rot":
                from dataclasses import replace as dc_replace
                fields.append(
                    dc_replace(f, dim=ROTATION_DIMS[to_rep], representation=to_rep)
                )
            else:
                fields.append(f)
        return VectorSchema(tuple(fields))

    # ── (de)serialisation ─────────────────────────────────────────────────────

    def to_config(self) -> List[dict]:
        return [
            {
                "name": f.name,
                "dim": f.dim,
                "kind": f.kind,
                **({"representation": f.representation} if f.representation else {}),
            }
            for f in self.fields
        ]

    @classmethod
    def from_config(cls, fields: Sequence[Mapping]) -> "VectorSchema":
        return cls(tuple(
            Field(
                name=str(f["name"]),
                dim=int(f["dim"]),
                kind=str(f.get("kind", "generic")),
                representation=f.get("representation"),
            )
            for f in fields
        ))


def schemas_to_meta(schemas: Optional[Mapping[str, VectorSchema]]) -> Dict[str, list]:
    """key → field-list dicts, for embedding in store meta.json."""
    if not schemas:
        return {}
    return {key: schema.to_config() for key, schema in schemas.items()}


def schemas_from_meta(meta: Mapping) -> Dict[str, VectorSchema]:
    """Parse the 'schemas' entry of a store meta.json (missing → empty)."""
    raw = meta.get("schemas") or {}
    return {key: VectorSchema.from_config(fields) for key, fields in raw.items()}


def validate_schemas_against_shapes(
    schemas: Mapping[str, VectorSchema],
    key_shapes: Mapping[str, tuple],
) -> None:
    """Raise when a schema names a missing key or its dim mismatches the data.

    Called by writers at finalize() — a schema that does not describe the
    data must never be persisted.
    """
    for key, schema in (schemas or {}).items():
        if key not in key_shapes:
            raise ValueError(
                f"Schema given for key {key!r} which is not in the store "
                f"(keys: {sorted(key_shapes)})"
            )
        shape = tuple(key_shapes[key])
        if len(shape) != 1:
            raise ValueError(
                f"Schema for key {key!r} requires a 1-D vector key, "
                f"got frame shape {shape}"
            )
        schema.validate_dim(shape[0], context=key)
