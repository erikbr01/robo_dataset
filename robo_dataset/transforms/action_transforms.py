"""Action-space InferenceTransforms.

Applied to the action tensor AFTER policy.predict_action() — identically at
train time and inference time.

The policy's internal unnormalisation (LinearNormalizer.unnormalize) already
runs inside predict_action(); this module covers structural conversions such
as rotation-representation changes.
"""

from typing import Union

import numpy as np
import torch

from robo_dataset.transforms.base import InferenceTransform
from robo_dataset.transforms.rotation_transformer import RotationTransformer


class ConvertRotation(InferenceTransform):
    """Convert a rotation slice of the action vector between representations.

    Splices the converted slice back with the surrounding parts of the
    action; the output action dimension may therefore differ from the input.

    Example — rotation_6d (10-dim abs action) → axis_angle (7-dim)::

        ConvertRotation(rotation_start=3, rotation_end=9,
                        from_rep='rotation_6d', to_rep='axis_angle')

        input:  [..., pos(3) | rot6d(6) | gripper(1)]  → dim 10
        output: [..., pos(3) | axisangle(3) | gripper(1)] → dim 7

    Args:
        rotation_start: first index of the rotation slice (inclusive).
        rotation_end:   last index  of the rotation slice (exclusive).
        from_rep:       source rotation representation.
        to_rep:         target rotation representation.
    """

    def __init__(
        self,
        rotation_start: int,
        rotation_end: int,
        from_rep: str = 'rotation_6d',
        to_rep: str = 'axis_angle',
    ):
        self.rotation_start = int(rotation_start)
        self.rotation_end = int(rotation_end)
        self.transformer = RotationTransformer(from_rep=from_rep, to_rep=to_rep)

    def __call__(
        self,
        action: Union[np.ndarray, torch.Tensor],
    ) -> Union[np.ndarray, torch.Tensor]:
        pre  = action[..., :self.rotation_start]
        rot  = action[..., self.rotation_start:self.rotation_end]
        post = action[..., self.rotation_end:]

        converted = self.transformer.forward(rot)

        if isinstance(action, torch.Tensor):
            parts = [p for p in (pre, converted, post) if p.shape[-1] > 0]
            return torch.cat(parts, dim=-1)
        else:
            parts = [p for p in (pre, converted, post) if p.shape[-1] > 0]
            return np.concatenate(parts, axis=-1)


class IdentityAction(InferenceTransform):
    """No-op action postprocess — pass-through."""

    def __call__(
        self,
        action: Union[np.ndarray, torch.Tensor],
    ) -> Union[np.ndarray, torch.Tensor]:
        return action
