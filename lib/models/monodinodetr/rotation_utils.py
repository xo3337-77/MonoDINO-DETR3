"""
lib/models/monodinodetr/rotation_utils.py
─────────────────────────────────────────
Drop this file into lib/models/monodinodetr/ and import from it.

Usage in monodinodetr.py:
    from lib.models.monodinodetr.rotation_utils import (
        euler_to_rotmat, geodesic_loss, rotmat_to_euler
    )

Usage in criterion.py:
    from lib.models.monodinodetr.rotation_utils import (
        euler_to_rotmat, geodesic_loss
    )
"""

import torch
import torch.nn.functional as F
from scipy.spatial.transform import Rotation as ScipyRotation
import numpy as np


# ── PyTorch (differentiable) ─────────────────────────────────────────────

def euler_to_rotmat(euler_angles: torch.Tensor) -> torch.Tensor:
    """
    Convert Euler angles (XYZ extrinsic convention) → rotation matrices.

    Args:
        euler_angles: (N, 3) float tensor, columns = [rx, ry, rz] in radians

    Returns:
        (N, 3, 3) rotation matrices  R = Rz @ Ry @ Rx
    """
    assert euler_angles.ndim == 2 and euler_angles.shape[1] == 3

    rx = euler_angles[:, 0]
    ry = euler_angles[:, 1]
    rz = euler_angles[:, 2]

    ones  = torch.ones_like(rx)
    zeros = torch.zeros_like(rx)

    cos_x, sin_x = torch.cos(rx), torch.sin(rx)
    cos_y, sin_y = torch.cos(ry), torch.sin(ry)
    cos_z, sin_z = torch.cos(rz), torch.sin(rz)

    # Each rotation matrix: row-major, shape (N, 3, 3)
    Rx = torch.stack([
        torch.stack([ones,  zeros,  zeros ], dim=1),
        torch.stack([zeros, cos_x, -sin_x ], dim=1),
        torch.stack([zeros, sin_x,  cos_x ], dim=1),
    ], dim=1)

    Ry = torch.stack([
        torch.stack([ cos_y, zeros, sin_y], dim=1),
        torch.stack([ zeros, ones,  zeros], dim=1),
        torch.stack([-sin_y, zeros, cos_y], dim=1),
    ], dim=1)

    Rz = torch.stack([
        torch.stack([cos_z, -sin_z, zeros], dim=1),
        torch.stack([sin_z,  cos_z, zeros], dim=1),
        torch.stack([zeros,  zeros, ones ], dim=1),
    ], dim=1)

    return torch.bmm(Rz, torch.bmm(Ry, Rx))  # (N, 3, 3)


def geodesic_loss(R_pred: torch.Tensor,
                  R_tgt:  torch.Tensor,
                  reduction: str = 'none') -> torch.Tensor:
    """
    Geodesic (angular) distance between predicted and target rotation matrices.

    Superior to L1 on Euler angles because:
      - No gimbal lock
      - No discontinuity at ±π
      - Directly measures true angular error

    Args:
        R_pred:    (N, 3, 3) predicted rotation matrices
        R_tgt:     (N, 3, 3) target rotation matrices
        reduction: 'none' | 'mean' | 'sum'

    Returns:
        (N,) or scalar angular distances in radians  ∈ [0, π]
    """
    assert R_pred.shape == R_tgt.shape and R_pred.ndim == 3

    R_diff = torch.bmm(R_pred, R_tgt.transpose(1, 2))  # R_pred @ R_tgt^T
    trace  = R_diff[:, 0, 0] + R_diff[:, 1, 1] + R_diff[:, 2, 2]

    # Clamp strictly inside (-1, 1) for numerical stability of acos
    cos_angle = torch.clamp((trace - 1.0) / 2.0,
                            min=-1.0 + 1e-6,
                            max= 1.0 - 1e-6)
    loss = torch.acos(cos_angle)  # (N,) in radians

    if reduction == 'mean':
        return loss.mean()
    elif reduction == 'sum':
        return loss.sum()
    return loss  # 'none'


def rotation_loss(pred_euler: torch.Tensor,
                  tgt_euler:  torch.Tensor,
                  num_boxes:  int,
                  lambda_geo: float = 1.0,
                  lambda_l1:  float = 0.1) -> dict:
    """
    Combined rotation loss:
      - Geodesic loss  (primary,  robust to all 3 angles)
      - Per-axis L1    (auxiliary, helps early convergence)

    Args:
        pred_euler: (M, 3) predicted [rx, ry, rz] in radians
        tgt_euler:  (M, 3) target    [rx, ry, rz] in radians
        num_boxes:  normalisation factor
        lambda_geo: weight for geodesic term
        lambda_l1:  weight for L1 auxiliary term

    Returns:
        dict of named losses ready to add to the criterion loss dict
    """
    # Convert to rotation matrices
    R_pred = euler_to_rotmat(pred_euler)   # (M, 3, 3)
    R_tgt  = euler_to_rotmat(tgt_euler)   # (M, 3, 3)

    # Primary: geodesic
    geo = geodesic_loss(R_pred, R_tgt)    # (M,)
    loss_geo = lambda_geo * geo.sum() / num_boxes

    # Auxiliary: per-axis L1  (helps with convergence speed)
    l1 = F.l1_loss(pred_euler, tgt_euler, reduction='none')   # (M, 3)
    loss_rx_l1 = lambda_l1 * l1[:, 0].sum() / num_boxes
    loss_ry_l1 = lambda_l1 * l1[:, 1].sum() / num_boxes
    loss_rz_l1 = lambda_l1 * l1[:, 2].sum() / num_boxes

    return {
        'loss_rotation': loss_geo,       # weighted in weight_dict
        'loss_rx_l1':    loss_rx_l1,     # for logging only (weight=0)
        'loss_ry_l1':    loss_ry_l1,     # for logging only (weight=0)
        'loss_rz_l1':    loss_rz_l1,     # for logging only (weight=0)
    }


# ── NumPy helpers (for label generation, not training) ───────────────────

def rotmat_to_euler_np(R: np.ndarray,
                       convention: str = 'xyz') -> np.ndarray:
    """
    Convert a (3,3) rotation matrix to Euler angles.
    Used during pseudo-label generation from Open3D OBB.

    Args:
        R:          (3, 3) rotation matrix
        convention: scipy Rotation convention string (default 'xyz')

    Returns:
        (3,) array [rx, ry, rz] in radians
    """
    rot = ScipyRotation.from_matrix(R)
    return rot.as_euler(convention, degrees=False)


def obb_to_euler(obb_R: np.ndarray) -> tuple:
    """
    Convenience wrapper: Open3D OBB rotation matrix → (rx, ry, rz).

    Args:
        obb_R: (3, 3) numpy rotation matrix from open3d OBB

    Returns:
        (rx, ry, rz) tuple in radians
    """
    angles = rotmat_to_euler_np(obb_R, convention='xyz')
    return float(angles[0]), float(angles[1]), float(angles[2])


# ── Quick sanity test ─────────────────────────────────────────────────────

if __name__ == '__main__':
    import math

    print("Running rotation_utils sanity check...")

    # Test round-trip: Euler → R → geodesic loss = 0
    angles = torch.tensor([[0.1, 0.5, -0.3],
                            [math.pi / 4, 0.0, math.pi / 6]])
    R = euler_to_rotmat(angles)

    # Geodesic distance with itself should be ~0
    loss = geodesic_loss(R, R)
    assert (loss < 1e-5).all(), f"Self-geodesic should be 0, got {loss}"

    # Geodesic distance with 90° rotation around Y = π/2
    angles_a = torch.tensor([[0.0, 0.0,       0.0]])
    angles_b = torch.tensor([[0.0, math.pi/2, 0.0]])
    Ra = euler_to_rotmat(angles_a)
    Rb = euler_to_rotmat(angles_b)
    d  = geodesic_loss(Ra, Rb)
    assert abs(d.item() - math.pi / 2) < 1e-4, \
        f"90° geodesic should be π/2, got {d.item():.4f}"

    print("  ✓ Self-geodesic = 0")
    print(f"  ✓ 90° geodesic = {d.item():.4f} ≈ π/2 = {math.pi/2:.4f}")
    print("All checks passed.")
