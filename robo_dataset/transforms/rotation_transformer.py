"""Rotation representation conversions — pure PyTorch, no pytorch3d dependency.

Supports: axis_angle, quaternion (wxyz), rotation_6d, matrix.

All conversion functions accept arbitrarily-batched inputs (..., dim) and
return (..., out_dim).  Numpy arrays are accepted and returned as numpy arrays.

RotationTransformer provides forward() / inverse() just like the version in
diffusion_policy, so it can be a drop-in replacement when only NumPy/PyTorch
is available.
"""

from typing import List, Optional, Union

import numpy as np
import torch
import torch.nn.functional as F


# ─── primitive conversion functions ────────────────────────────────────────

def axis_angle_to_matrix(aa: torch.Tensor) -> torch.Tensor:
    """(..., 3) axis-angle → (..., 3, 3) SO(3) via Rodrigues formula."""
    angle = torch.norm(aa, p=2, dim=-1, keepdim=True)          # (..., 1)
    safe_angle = torch.where(angle < 1e-8, torch.ones_like(angle), angle)
    axis = aa / safe_angle                                       # (..., 3)

    x, y, z = axis[..., 0], axis[..., 1], axis[..., 2]
    zeros = torch.zeros_like(x)
    K = torch.stack([                                            # skew-symmetric
        zeros, -z,  y,
        z,  zeros, -x,
        -y,  x,  zeros,
    ], dim=-1).reshape(*aa.shape[:-1], 3, 3)

    I = torch.eye(3, dtype=aa.dtype, device=aa.device)
    sin_a = torch.sin(angle)[..., None]                          # (..., 1, 1)
    cos_a = torch.cos(angle)[..., None]
    R = I + sin_a * K + (1.0 - cos_a) * (K @ K)

    # Zero-angle edge case: identity matrix
    near_zero = (angle < 1e-8)[..., None]
    return torch.where(near_zero, I.expand_as(R), R)


def matrix_to_axis_angle(R: torch.Tensor) -> torch.Tensor:
    """(..., 3, 3) SO(3) → (..., 3) axis-angle."""
    trace = R[..., 0, 0] + R[..., 1, 1] + R[..., 2, 2]
    cos_a = ((trace - 1.0) / 2.0).clamp(-1.0 + 1e-7, 1.0 - 1e-7)
    angle = torch.arccos(cos_a)                                  # (...,)

    unnorm = torch.stack([
        R[..., 2, 1] - R[..., 1, 2],
        R[..., 0, 2] - R[..., 2, 0],
        R[..., 1, 0] - R[..., 0, 1],
    ], dim=-1)                                                   # (..., 3)

    denom = 2.0 * torch.sin(angle).unsqueeze(-1) + 1e-8
    axis = unnorm / denom
    return axis * angle.unsqueeze(-1)


def rotation_6d_to_matrix(d6: torch.Tensor) -> torch.Tensor:
    """(..., 6) 6D representation → (..., 3, 3) SO(3).

    The 6D representation uses the first two ROWS of the rotation matrix,
    following Zhou et al. "On the Continuity of Rotation Representations
    in Neural Networks" (CVPR 2019).  Gram-Schmidt orthonormalization.
    """
    a1, a2 = d6[..., :3], d6[..., 3:]
    b1 = F.normalize(a1, dim=-1)
    b2 = F.normalize(a2 - (b1 * a2).sum(dim=-1, keepdim=True) * b1, dim=-1)
    b3 = torch.cross(b1, b2, dim=-1)
    return torch.stack([b1, b2, b3], dim=-2)                    # (..., 3, 3)


def matrix_to_rotation_6d(R: torch.Tensor) -> torch.Tensor:
    """(..., 3, 3) SO(3) → (..., 6) 6D representation (first two rows)."""
    return R[..., :2, :].reshape(*R.shape[:-2], 6)


def quaternion_to_matrix(q: torch.Tensor) -> torch.Tensor:
    """(..., 4) quaternion (w, x, y, z) → (..., 3, 3) SO(3)."""
    q = F.normalize(q, dim=-1)
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    R = torch.stack([
        1 - 2*(y*y + z*z),  2*(x*y - z*w),      2*(x*z + y*w),
        2*(x*y + z*w),      1 - 2*(x*x + z*z),  2*(y*z - x*w),
        2*(x*z - y*w),      2*(y*z + x*w),      1 - 2*(x*x + y*y),
    ], dim=-1).reshape(*q.shape[:-1], 3, 3)
    return R


def matrix_to_quaternion(R: torch.Tensor) -> torch.Tensor:
    """(..., 3, 3) SO(3) → (..., 4) quaternion (w, x, y, z)."""
    trace = R[..., 0, 0] + R[..., 1, 1] + R[..., 2, 2]

    def case0(R, trace):
        s = torch.sqrt((trace + 1.0).clamp(min=1e-10)) * 2  # 4w
        w = 0.25 * s
        x = (R[..., 2, 1] - R[..., 1, 2]) / s
        y = (R[..., 0, 2] - R[..., 2, 0]) / s
        z = (R[..., 1, 0] - R[..., 0, 1]) / s
        return torch.stack([w, x, y, z], dim=-1)

    def case1(R):
        s = torch.sqrt((1.0 + R[..., 0, 0] - R[..., 1, 1] - R[..., 2, 2]).clamp(min=1e-10)) * 2
        w = (R[..., 2, 1] - R[..., 1, 2]) / s
        x = 0.25 * s
        y = (R[..., 0, 1] + R[..., 1, 0]) / s
        z = (R[..., 0, 2] + R[..., 2, 0]) / s
        return torch.stack([w, x, y, z], dim=-1)

    def case2(R):
        s = torch.sqrt((1.0 + R[..., 1, 1] - R[..., 0, 0] - R[..., 2, 2]).clamp(min=1e-10)) * 2
        w = (R[..., 0, 2] - R[..., 2, 0]) / s
        x = (R[..., 0, 1] + R[..., 1, 0]) / s
        y = 0.25 * s
        z = (R[..., 1, 2] + R[..., 2, 1]) / s
        return torch.stack([w, x, y, z], dim=-1)

    def case3(R):
        s = torch.sqrt((1.0 + R[..., 2, 2] - R[..., 0, 0] - R[..., 1, 1]).clamp(min=1e-10)) * 2
        w = (R[..., 1, 0] - R[..., 0, 1]) / s
        x = (R[..., 0, 2] + R[..., 2, 0]) / s
        y = (R[..., 1, 2] + R[..., 2, 1]) / s
        z = 0.25 * s
        return torch.stack([w, x, y, z], dim=-1)

    c0 = trace > 0
    c1 = (~c0) & (R[..., 0, 0] > R[..., 1, 1]) & (R[..., 0, 0] > R[..., 2, 2])
    c2 = (~c0) & (~c1) & (R[..., 1, 1] > R[..., 2, 2])

    q = case0(R, trace)
    q = torch.where(c0.unsqueeze(-1), q, case1(R))
    q = torch.where((c0 | c1).unsqueeze(-1), q, case2(R))
    q = torch.where((c0 | c1 | c2).unsqueeze(-1), q, case3(R))
    return F.normalize(q, dim=-1)


# ─── conversion routing table ───────────────────────────────────────────────

_TO_MATRIX = {
    'axis_angle':  axis_angle_to_matrix,
    'quaternion':  quaternion_to_matrix,
    'rotation_6d': rotation_6d_to_matrix,
    'matrix':      lambda x: x,
}

_FROM_MATRIX = {
    'axis_angle':  matrix_to_axis_angle,
    'quaternion':  matrix_to_quaternion,
    'rotation_6d': matrix_to_rotation_6d,
    'matrix':      lambda x: x,
}


# ─── RotationTransformer ────────────────────────────────────────────────────

class RotationTransformer:
    """Convert between rotation representations via SO(3) matrix as intermediate.

    API mirrors the diffusion_policy version (forward / inverse) so this
    is a drop-in replacement that works without pytorch3d.

    Supported representations: axis_angle, quaternion, rotation_6d, matrix.

    Example::

        tf = RotationTransformer('axis_angle', 'rotation_6d')
        rot6d = tf.forward(axis_angle_array)   # np or torch input/output
        back  = tf.inverse(rot6d)
    """

    valid_reps = ['axis_angle', 'quaternion', 'rotation_6d', 'matrix']

    def __init__(self, from_rep: str = 'axis_angle', to_rep: str = 'rotation_6d'):
        assert from_rep in self.valid_reps, f"from_rep must be one of {self.valid_reps}"
        assert to_rep in self.valid_reps, f"to_rep must be one of {self.valid_reps}"
        assert from_rep != to_rep, "from_rep and to_rep must differ"
        self.from_rep = from_rep
        self.to_rep = to_rep

    @staticmethod
    def _convert(x: Union[np.ndarray, torch.Tensor], from_rep: str, to_rep: str
                 ) -> Union[np.ndarray, torch.Tensor]:
        was_numpy = isinstance(x, np.ndarray)
        t = torch.from_numpy(x) if was_numpy else x
        mat = _TO_MATRIX[from_rep](t)
        out = _FROM_MATRIX[to_rep](mat)
        return out.numpy() if was_numpy else out

    def forward(self, x: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
        """Convert from from_rep to to_rep."""
        return self._convert(x, self.from_rep, self.to_rep)

    def inverse(self, x: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
        """Convert from to_rep back to from_rep."""
        return self._convert(x, self.to_rep, self.from_rep)
