"""The canonical camera format consumed by the fused-PRoPE model.

Both producers emit the same thing: ``[F, 12]`` float32 rows holding
``[:3, :4]`` of the camera-to-anchor transform ``inv(c2w_anchor) @ c2w_i``,
row-major.  The anchor frame - the overlap tail for AR windows, the first
frame for T2V/I2V - is the identity row.  Training reaches it by interpolating
sparse SpatialVID pose7 sidecars; inference reaches it by sampling an action
vocabulary.  Only this module defines the arithmetic that gets there.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np


EPS = 1e-8
FRAME_TOLERANCE = 1e-7
ROTATION_TOLERANCE = 1e-5
IDENTITY_ROW = np.array(
    [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0], dtype=np.float32,
)


def finite_array(value: Any, name: str) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64)
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{name} contains non-finite values")
    return result


def normalize_quaternion_xyzw(value: Any, eps: float = EPS) -> np.ndarray:
    """Normalize ``xyzw`` quaternions and reject degenerate rotations."""

    quaternion = finite_array(value, "quaternion")
    if quaternion.ndim == 0 or quaternion.shape[-1] != 4:
        raise ValueError(
            f"expected quaternion shape (..., 4), got {quaternion.shape}"
        )
    norm = np.linalg.norm(quaternion, axis=-1, keepdims=True)
    if np.any(norm < eps) or not np.all(np.isfinite(norm)):
        raise ValueError("quaternion norm is zero or non-finite")
    return quaternion / norm


def quaternion_xyzw_to_matrix(value: Any) -> np.ndarray:
    """Convert ``xyzw`` quaternions to rotation matrices."""

    x, y, z, w = np.moveaxis(normalize_quaternion_xyzw(value), -1, 0)
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    xw, yw, zw = x * w, y * w, z * w
    entries = np.stack(
        (
            1.0 - 2.0 * (yy + zz),
            2.0 * (xy - zw),
            2.0 * (xz + yw),
            2.0 * (xy + zw),
            1.0 - 2.0 * (xx + zz),
            2.0 * (yz - xw),
            2.0 * (xz - yw),
            2.0 * (yz + xw),
            1.0 - 2.0 * (xx + yy),
        ),
        axis=-1,
    )
    return entries.reshape((*x.shape, 3, 3))


def pose7_to_c2w(value: Any) -> np.ndarray:
    """Convert pose7 world-to-camera rows to camera-to-world matrices.

    The translation column of the w2c matrix locates the camera, so the c2w
    center is ``-R.T @ t`` and the c2w rotation is ``R.T``.
    """

    pose = finite_array(value, name="pose7")
    if pose.ndim != 2 or pose.shape[1] != 7:
        raise ValueError(f"expected pose7 shape (N, 7), got {pose.shape}")
    rotation_w2c = quaternion_xyzw_to_matrix(pose[:, 3:])
    rotation_c2w = np.swapaxes(rotation_w2c, 1, 2)
    center = -np.einsum("nij,nj->ni", rotation_c2w, pose[:, :3])
    result = np.broadcast_to(
        np.eye(4, dtype=np.float64), (len(pose), 4, 4)
    ).copy()
    result[:, :3, :3] = rotation_c2w
    result[:, :3, 3] = center
    return result


def _matrix_to_quaternion_xyzw(rotation: np.ndarray) -> np.ndarray:
    """Convert a single proper rotation matrix to an ``xyzw`` quaternion."""

    matrix = np.asarray(rotation, dtype=np.float64)
    if matrix.shape != (3, 3):
        raise ValueError(f"expected a 3x3 rotation, got {matrix.shape}")
    trace = float(np.trace(matrix))
    if trace > 0.0:
        scale = math.sqrt(max(trace + 1.0, 0.0)) * 2.0
        w = 0.25 * scale
        x = (matrix[2, 1] - matrix[1, 2]) / scale
        y = (matrix[0, 2] - matrix[2, 0]) / scale
        z = (matrix[1, 0] - matrix[0, 1]) / scale
    else:
        diagonal = np.diag(matrix)
        index = int(np.argmax(diagonal))
        if index == 0:
            scale = math.sqrt(max(1.0 + matrix[0, 0] - matrix[1, 1] - matrix[2, 2], 0.0)) * 2.0
            x = 0.25 * scale
            y = (matrix[0, 1] + matrix[1, 0]) / scale
            z = (matrix[0, 2] + matrix[2, 0]) / scale
            w = (matrix[2, 1] - matrix[1, 2]) / scale
        elif index == 1:
            scale = math.sqrt(max(1.0 - matrix[0, 0] + matrix[1, 1] - matrix[2, 2], 0.0)) * 2.0
            x = (matrix[0, 1] + matrix[1, 0]) / scale
            y = 0.25 * scale
            z = (matrix[1, 2] + matrix[2, 1]) / scale
            w = (matrix[0, 2] - matrix[2, 0]) / scale
        else:
            scale = math.sqrt(max(1.0 - matrix[0, 0] - matrix[1, 1] + matrix[2, 2], 0.0)) * 2.0
            x = (matrix[0, 2] + matrix[2, 0]) / scale
            y = (matrix[1, 2] + matrix[2, 1]) / scale
            z = 0.25 * scale
            w = (matrix[1, 0] - matrix[0, 1]) / scale
    quaternion = np.asarray((x, y, z, w), dtype=np.float64)
    norm = float(np.linalg.norm(quaternion))
    if not math.isfinite(norm) or norm < EPS:
        raise ValueError("rotation produced an invalid quaternion")
    return quaternion / norm


def _c2w_quaternions(matrices: np.ndarray) -> np.ndarray:
    return np.stack(
        [_matrix_to_quaternion_xyzw(matrix[:3, :3]) for matrix in matrices],
        axis=0,
    )


def c2w_to_pose7(c2w_matrices: Any) -> np.ndarray:
    """Convert camera-to-world matrices to pose7 world-to-camera rows.

    The exact inverse of :func:`pose7_to_c2w`.  An offline producer uses it to
    fold a c2w source into the shared pose7 sidecar contract without leaving
    the arithmetic owned by this module.
    """

    matrices = finite_array(c2w_matrices, name="c2w matrices")
    if matrices.ndim != 3 or matrices.shape[1:] != (4, 4) or len(matrices) == 0:
        raise ValueError(f"expected non-empty c2w matrices (N, 4, 4), got {matrices.shape}")
    rotation_w2c = np.swapaxes(matrices[:, :3, :3], 1, 2)
    translation_w2c = -np.einsum("nij,nj->ni", rotation_w2c, matrices[:, :3, 3])
    quaternions = np.stack(
        [_matrix_to_quaternion_xyzw(matrix) for matrix in rotation_w2c],
        axis=0,
    )
    return np.concatenate([translation_w2c, quaternions], axis=1)


def _slerp_xyzw(left: np.ndarray, right: np.ndarray, amount: float) -> np.ndarray:
    """Spherical linear interpolation with shortest-path sign handling."""

    first = normalize_quaternion_xyzw(left)
    second = normalize_quaternion_xyzw(right)
    dot = float(np.dot(first, second))
    if dot < 0.0:
        second = -second
        dot = -dot
    dot = min(1.0, max(-1.0, dot))
    if dot > 1.0 - 1e-7:
        return normalize_quaternion_xyzw(first + amount * (second - first))
    angle = math.acos(dot)
    sine = math.sin(angle)
    if abs(sine) < EPS:
        return first
    left_weight = math.sin((1.0 - amount) * angle) / sine
    right_weight = math.sin(amount * angle) / sine
    return normalize_quaternion_xyzw(left_weight * first + right_weight * second)


def interpolate_c2w(
    source_frame_indices: Any,
    source_pose7: Any,
    target_frame_indices: Any,
) -> np.ndarray:
    """Interpolate sparse pose7 annotations at target frame positions.

    Translation is interpolated as the canonical c2w camera center. Rotation
    uses quaternion SLERP. Targets outside the sparse annotation interval are
    rejected; in particular, this function never silently extrapolates.
    """

    source_frames = finite_array(source_frame_indices, name="source frame indices").reshape(-1)
    target_frames = finite_array(target_frame_indices, name="target frame indices").reshape(-1)
    poses = finite_array(source_pose7, name="source pose7")
    if poses.ndim != 2 or poses.shape[1] != 7:
        raise ValueError(f"expected source pose7 shape (N, 7), got {poses.shape}")
    if len(source_frames) != len(poses) or len(source_frames) == 0:
        raise ValueError("source frame indices and poses must have equal non-zero length")
    if np.any(np.diff(source_frames) <= 0.0):
        raise ValueError("source frame indices must be strictly increasing")
    if len(target_frames) == 0:
        return np.empty((0, 4, 4), dtype=np.float64)
    if (
        np.min(target_frames) < source_frames[0] - FRAME_TOLERANCE
        or np.max(target_frames) > source_frames[-1] + FRAME_TOLERANCE
    ):
        raise ValueError(
            "target frame indices fall outside sparse pose support; extrapolation is disabled"
        )
    target_frames = np.clip(target_frames, source_frames[0], source_frames[-1])
    sparse_c2w = pose7_to_c2w(poses)
    centers = sparse_c2w[:, :3, 3]
    rotations = _c2w_quaternions(sparse_c2w)
    output = np.broadcast_to(
        np.eye(4, dtype=np.float64), (len(target_frames), 4, 4)
    ).copy()
    for target_index, frame in enumerate(target_frames):
        right = int(np.searchsorted(source_frames, frame, side="left"))
        if right == 0:
            left = 0
            right = 0
            amount = 0.0
        elif right >= len(source_frames):
            left = len(source_frames) - 1
            right = left
            amount = 0.0
        elif abs(source_frames[right] - frame) <= FRAME_TOLERANCE:
            left = right
            amount = 0.0
        else:
            left = right - 1
            denominator = source_frames[right] - source_frames[left]
            amount = float((frame - source_frames[left]) / denominator)
        output[target_index, :3, 3] = (
            (1.0 - amount) * centers[left] + amount * centers[right]
        )
        output[target_index, :3, :3] = quaternion_xyzw_to_matrix(
            _slerp_xyzw(rotations[left], rotations[right], amount)
        )
    return output


def relative_camera_matrices(
    c2w_matrices: Any,
    anchor_index: int = 0,
) -> np.ndarray:
    """Rebase c2w matrices to the anchor camera and return flattened 3x4 rows.

    The anchor frame (overlap tail for AR windows, the first frame for
    T2V/I2V) becomes the identity row.
    """

    matrices = finite_array(c2w_matrices, name="c2w matrices")
    if matrices.ndim != 3 or matrices.shape[1:] != (4, 4) or len(matrices) == 0:
        raise ValueError(f"expected non-empty c2w matrices (N, 4, 4), got {matrices.shape}")
    if not 0 <= anchor_index < len(matrices):
        raise ValueError(
            f"anchor index {anchor_index} out of range for {len(matrices)} cameras"
        )
    anchor = matrices[anchor_index]
    rotation = anchor[:3, :3]
    anchor_inverse = np.eye(4, dtype=np.float64)
    anchor_inverse[:3, :3] = rotation.T
    anchor_inverse[:3, 3] = -rotation.T @ anchor[:3, 3]
    relative = np.matmul(anchor_inverse[None], matrices)
    return np.ascontiguousarray(relative[:, :3, :4].reshape(len(matrices), 12), dtype=np.float32)


def interpolate_pose7(
    source_frame_indices: Any,
    source_pose7: Any,
    target_frame_indices: Any,
    anchor_index: int = 0,
) -> np.ndarray:
    """Return rebased, flattened ``[N, 12]`` camera rows."""

    return relative_camera_matrices(
        interpolate_c2w(source_frame_indices, source_pose7, target_frame_indices),
        anchor_index=anchor_index,
    )


def validate_camera_rows(
    rows: Any,
    *,
    anchor_index: int | None = None,
    expected_length: int | None = None,
) -> dict[str, float | int]:
    """Check one producer's ``[F, 12]`` output and return its statistics.

    Every producer of camera rows calls this at its boundary, so a malformed
    window fails where it was built instead of inside attention.  The anchor
    row is reported rather than asserted: both producers write it exactly, but
    a tolerance-free claim about a float32 row belongs in the test that pins
    the behaviour, not in a runtime guard.
    """

    value = finite_array(rows, name="camera rows")
    if value.ndim != 2 or value.shape[1] != 12:
        raise ValueError(f"camera rows must have shape [F, 12], got {value.shape}")
    if len(value) < 1:
        raise ValueError("camera rows must hold at least one frame")
    if anchor_index is not None and not 0 <= int(anchor_index) < len(value):
        raise ValueError(
            f"anchor index {anchor_index} out of range for {len(value)} camera rows"
        )
    if expected_length is not None and len(value) != int(expected_length):
        raise ValueError(
            f"camera rows hold {len(value)} frames, expected {int(expected_length)}"
        )
    rotations = value.reshape(-1, 3, 4)[:, :, :3]
    orthogonality = np.max(
        np.abs(np.matmul(np.swapaxes(rotations, -1, -2), rotations) - np.eye(3)),
        axis=(1, 2),
    )
    determinants = np.linalg.det(rotations)
    if np.max(orthogonality) > ROTATION_TOLERANCE or np.max(np.abs(determinants - 1.0)) > ROTATION_TOLERANCE:
        raise ValueError("camera rows contain a rotation that is not in SO(3)")
    statistics: dict[str, float | int] = {
        "count": int(len(value)),
        "rotation_orthogonality_max_error": float(np.max(orthogonality)),
        "rotation_det_min": float(np.min(determinants)),
        "rotation_det_max": float(np.max(determinants)),
    }
    if anchor_index is not None:
        statistics["anchor_identity_max_abs_error"] = float(
            np.max(np.abs(value[int(anchor_index)] - IDENTITY_ROW))
        )
    return statistics


__all__ = [
    "EPS",
    "FRAME_TOLERANCE",
    "IDENTITY_ROW",
    "ROTATION_TOLERANCE",
    "c2w_to_pose7",
    "finite_array",
    "interpolate_c2w",
    "interpolate_pose7",
    "normalize_quaternion_xyzw",
    "pose7_to_c2w",
    "quaternion_xyzw_to_matrix",
    "relative_camera_matrices",
    "validate_camera_rows",
]
