"""Camera trajectories for fused-PRoPE inference.

A trajectory is the camera's absolute camera-to-world path, one 4x4 matrix per
video frame, read from an externally authored file. Everything downstream
consumes that array; the pipeline never sees an action name.

``window_rows`` slices the path for one AR window. Row ``r`` of chunk ``c``
comes from latent ``1 + history_frames(c) + r``: latent ``j`` covers pixel
frames ``[4j - 3, 4j]`` and the camera row stands for the last frame of that
group, matching the training dataset's ``source_indices[::4]`` sampling. That
start is the same expression the training side uses for the frames a
continuation window covers, so both ends agree by construction.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping

import numpy as np

from example.recam.config.config import GeometryConfig, VAE_TEMPORAL_FACTOR
from example.recam.datasets.camera.format import (
    relative_camera_matrices,
    validate_camera_rows,
)


def _orthonormalize(matrices: np.ndarray) -> np.ndarray:
    """Project the rotation block onto SO(3).

    Externally authored paths are only approximately orthonormal - real-world
    trajectories can carry up to 6e-4 of error, well past what
    ``validate_camera_rows`` tolerates.
    """

    rotations = matrices[:, :3, :3]
    left, _, right = np.linalg.svd(rotations)
    fixed = left @ right
    flipped = np.linalg.det(fixed) < 0.0
    if np.any(flipped):
        left = left.copy()
        left[flipped, :, -1] *= -1.0
        fixed = left @ right
    result = matrices.copy()
    result[:, :3, :3] = fixed
    return result


def load_trajectory_file(path: str | Path) -> np.ndarray:
    """Read a per-frame c2w path from a ``{frame: {extrinsic, K}}`` JSON."""

    location = Path(path)
    payload = json.loads(location.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping) or not payload:
        raise ValueError(f"{location} must hold a non-empty frame mapping")
    frames = sorted(int(key) for key in payload)
    if frames != list(range(len(frames))):
        raise ValueError(f"{location} frame keys must be contiguous from 0")
    entries = [payload[str(frame)] for frame in frames]
    intrinsics = np.asarray(
        [entry["K"] for entry in entries], dtype=np.float64,
    )
    if not np.allclose(intrinsics, intrinsics[0]):
        raise ValueError(
            f"{location} varies its intrinsics per frame; the model carries a "
            "single focal length"
        )
    extrinsics = np.asarray(
        [entry["extrinsic"] for entry in entries], dtype=np.float64,
    )
    if extrinsics.shape != (len(frames), 4, 4):
        raise ValueError(
            f"{location} extrinsics must be 4x4, got {extrinsics.shape}"
        )
    if not np.all(np.isfinite(extrinsics)):
        raise ValueError(f"{location} contains non-finite extrinsics")
    return _orthonormalize(extrinsics)


def window_latent_indices(
    geometry: GeometryConfig,
    *,
    chunk_index: int,
    is_initial: bool,
) -> np.ndarray:
    """Return the global latent indices one chunk's camera rows cover."""

    if is_initial:
        first = 0
        count = 1 + geometry.new_frames
    else:
        first = 1 + geometry.history_frames(chunk_index)
        count = geometry.main_frames
    if first < 0:
        raise ValueError(
            f"chunk {chunk_index} starts at latent {first}; the trajectory is "
            "shorter than the geometry needs"
        )
    return first + np.arange(count, dtype=np.int64)


def window_rows(
    trajectory: np.ndarray,
    *,
    chunk_index: int,
    geometry: GeometryConfig,
    is_initial: bool,
) -> np.ndarray:
    """Slice one chunk's ``[F, 12]`` rows out of an absolute camera path."""

    latents = window_latent_indices(
        geometry, chunk_index=chunk_index, is_initial=is_initial,
    )
    frames = latents * VAE_TEMPORAL_FACTOR
    if frames[-1] >= len(trajectory):
        raise ValueError(
            f"chunk {chunk_index} needs trajectory frame {int(frames[-1])}, "
            f"but the path holds {len(trajectory)}"
        )
    anchor = 0 if is_initial else geometry.overlap - 1
    rows = relative_camera_matrices(trajectory[frames], anchor_index=anchor)
    validate_camera_rows(rows, anchor_index=anchor, expected_length=len(frames))
    return rows


__all__ = [
    "load_trajectory_file",
    "window_latent_indices",
    "window_rows",
]
