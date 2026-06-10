"""Action-space InferenceTransforms.

Applied to the action tensor AFTER policy.predict_action() — identically at
train time and inference time.

The policy's internal unnormalisation (LinearNormalizer.unnormalize) already
runs inside predict_action(); this module covers structural conversions such
as rotation-representation changes.
"""

from typing import List, Optional, Union

import numpy as np
import torch

from robo_dataset.transforms.base import InferenceTransform
from robo_dataset.transforms.rotation_transformer import RotationTransformer


class ConvertRotation(InferenceTransform):
    """Convert one or more named rotation fields in the action vector.

    The single source of truth for field→indices lives in the data_source
    config (action.fields); this transform receives that mapping and derives
    slices internally so index logic never appears in pipeline configs.

    When converting multiple rotation fields (e.g. bimanual tasks), fields are
    processed in descending index order so earlier-field indices remain valid
    after higher-field slices shrink or grow.

    Example — single arm, rotation_6d (10-dim) → axis_angle (7-dim)::

        ConvertRotation(
            rotation_fields=['rotation'],
            action_fields={'pos': [0,1,2], 'rotation': [3,4,5,6,7,8], 'gripper': [9]},
            from_rep='rotation_6d',
            to_rep='axis_angle',
        )

    Example — bimanual transport, rotation_6d (20-dim) → axis_angle (14-dim)::

        ConvertRotation(
            rotation_fields=['robot0_rotation', 'robot1_rotation'],
            action_fields={..., 'robot0_rotation': [3,4,5,6,7,8], ...,
                           'robot1_rotation': [13,14,15,16,17,18], ...},
            from_rep='rotation_6d',
            to_rep='axis_angle',
        )

    Args:
        rotation_fields: field names in action_fields to convert.  Use the
            ``action_field`` shorthand to pass a single field name as a string.
        action_fields:   dict mapping field names to lists of contiguous indices
                         (loaded from data_source config via Hydra interpolation).
        from_rep:        source rotation representation.
        to_rep:          target rotation representation.
        action_field:    convenience alias — sets rotation_fields to [action_field].
    """

    def __init__(
        self,
        action_fields: dict,
        from_rep: str = 'rotation_6d',
        to_rep: str = 'axis_angle',
        rotation_fields: Optional[List[str]] = None,
        action_field: Optional[str] = None,
    ):
        if action_field is not None and rotation_fields is None:
            rotation_fields = [action_field]
        if not rotation_fields:
            raise ValueError("Provide rotation_fields or action_field")

        # Derive (start, end) for each field; validate contiguous.
        parsed = []
        for name in rotation_fields:
            indices = list(action_fields[name])
            expected = list(range(indices[0], indices[0] + len(indices)))
            if indices != expected:
                raise ValueError(
                    f"action_fields['{name}'] must be contiguous, got {indices}"
                )
            parsed.append((name, indices[0], indices[-1] + 1))

        # Process highest-index fields first so earlier indices stay correct.
        self._fields = sorted(parsed, key=lambda x: x[1], reverse=True)
        self.transformer = RotationTransformer(from_rep=from_rep, to_rep=to_rep)

    def __call__(
        self,
        action: Union[np.ndarray, torch.Tensor],
    ) -> Union[np.ndarray, torch.Tensor]:
        is_torch = isinstance(action, torch.Tensor)
        cat = torch.cat if is_torch else np.concatenate

        for _name, start, end in self._fields:
            pre = action[..., :start]
            rot = action[..., start:end]
            post = action[..., end:]
            converted = self.transformer.forward(rot)
            parts = [p for p in (pre, converted, post) if p.shape[-1] > 0]
            action = cat(parts, dim=-1) if is_torch else cat(parts, axis=-1)

        return action


class IdentityAction(InferenceTransform):
    """No-op action postprocess — pass-through."""

    def __call__(
        self,
        action: Union[np.ndarray, torch.Tensor],
    ) -> Union[np.ndarray, torch.Tensor]:
        return action
