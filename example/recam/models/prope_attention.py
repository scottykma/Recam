"""Torch implementation of the fused PRoPE transform for the ReCam
packed attention layout.

Arithmetic uses the reference ordering: native RoPE first, then this Q/K/V
basis transform, one attention call, and the inverse output-basis transform.
The intrinsics matrix is the diagonal ``diag(fx, fy, 1)`` form the reference
transform path actually uses (its principal-point template never enters the
projection math).
"""

from __future__ import annotations

from collections.abc import Callable
from functools import partial

import torch

DEFAULT_FOCAL_X = 969.6969696969696 / 1920.0
DEFAULT_FOCAL_Y = 969.6969696969696 / 1080.0


def normalize_translation_transform(value: object = "linear") -> str:
    selected = str(value).strip().lower()
    if selected not in {"linear", "logd4"}:
        raise ValueError("camera translation transform must be linear or logd4")
    return selected


def transform_relative_viewmats(
    viewmats: torch.Tensor,
    transform: object = "linear",
) -> torch.Tensor:
    """Apply the model-only linear/logd4 translation transform."""

    selected = normalize_translation_transform(transform)
    if not isinstance(viewmats, torch.Tensor) or tuple(viewmats.shape[-2:]) != (4, 4):
        raise ValueError("camera viewmats must be floating tensors ending in [4,4]")
    if not torch.is_floating_point(viewmats):
        raise TypeError("camera viewmats must be floating point")
    if selected == "linear":
        return viewmats
    translation = viewmats[..., :3, 3]
    compute_dtype = torch.float64 if viewmats.dtype == torch.float64 else torch.float32
    value = translation.to(dtype=compute_dtype)
    norm = torch.linalg.vector_norm(value, dim=-1, keepdim=True)
    safe_norm = norm.clamp_min(torch.finfo(compute_dtype).tiny)
    scale = torch.where(
        norm > 0,
        torch.log1p(norm) / (4.0 * safe_norm),
        torch.zeros_like(norm),
    )
    output = viewmats.clone()
    output[..., :3, 3] = (value * scale).to(dtype=viewmats.dtype)
    return output


def relative_rows_to_c2w(rows: torch.Tensor) -> torch.Tensor:
    """Build [B,F,4,4] camera-to-world matrices from flattened 3x4 rows.

    ``rows`` are rebased relative c2w rows in row-major ``[:3,:4]`` flattening,
    with the anchor frame (overlap tail / initial first frame) as identity.
    """
    if not isinstance(rows, torch.Tensor) or rows.ndim != 3 or rows.shape[2] != 12:
        raise ValueError(f"camera rows must have shape [B,F,12]; got {tuple(rows.shape)}")
    batch, frames, _ = rows.shape
    c2w = torch.zeros(
        batch, frames, 4, 4, device=rows.device, dtype=torch.float32,
    )
    c2w[:, :, :3, :4] = rows.to(dtype=torch.float32).reshape(batch, frames, 3, 4)
    c2w[:, :, 3, 3] = 1.0
    return c2w


def build_prope_matrices(
    rows: torch.Tensor,
    *,
    focal_x: float = DEFAULT_FOCAL_X,
    focal_y: float = DEFAULT_FOCAL_Y,
    camera_translation_transform: str = "linear",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build per-frame projection matrices from rows.

    Returns ``(P, P^-1)`` as ``[B,F,4,4]`` float32 tensors to be broadcast to
    per-token shape by the caller.  ``P = K @ w2c`` and
    ``P^-1 = w2c^-1 @ K^-1`` (SE(3) inverse on the pure camera matrix,
    then the diagonal intrinsics inverse, as in the reference).  The input rows are
    c2w, so they are inverted to the w2c convention the transform consumes before
    the translation transform, which is defined on W2C translations.
    """
    viewmats = transform_relative_viewmats(
        _invert_se3(relative_rows_to_c2w(rows)),
        camera_translation_transform,
    )
    k_lift = torch.zeros(
        (*viewmats.shape[:-2], 4, 4), device=viewmats.device, dtype=torch.float32,
    )
    k_lift[..., 0, 0] = focal_x
    k_lift[..., 1, 1] = focal_y
    k_lift[..., 2, 2] = 1.0
    k_lift[..., 3, 3] = 1.0
    k_inv_lift = torch.zeros_like(k_lift)
    k_inv_lift[..., 0, 0] = 1.0 / focal_x
    k_inv_lift[..., 1, 1] = 1.0 / focal_y
    k_inv_lift[..., 2, 2] = 1.0
    k_inv_lift[..., 3, 3] = 1.0
    matrices = torch.einsum("...ij,...jk->...ik", k_lift, viewmats)
    inverse_matrices = torch.einsum(
        "...ij,...jk->...ik",
        _invert_se3(viewmats),
        k_inv_lift,
    )
    return matrices, inverse_matrices


def prope_qkv(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    matrices: torch.Tensor,
    inverse_matrices: torch.Tensor,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    Callable[[torch.Tensor], torch.Tensor],
]:
    """Transform packed [B,S,D] Q/K/V into camera coordinates.

    ``matrices`` / ``inverse_matrices`` are token-aligned [B,S,4,4]
    projections ``P = K @ viewmat`` and their exact inverses; the returned
    callable applies the matching output transform to attention rows before
    the native output projection.
    """
    if q.ndim != 3 or q.shape != k.shape or q.shape != v.shape:
        raise ValueError(
            f"fused PRoPE requires equal [B,S,D] Q/K/V tensors; got "
            f"{tuple(q.shape)}, {tuple(k.shape)}, {tuple(v.shape)}"
        )
    batch, sequence, feature_dim = q.shape
    if tuple(matrices.shape) != (batch, sequence, 4, 4):
        raise ValueError("PRoPE matrices must be token-aligned [B,S,4,4]")
    if tuple(inverse_matrices.shape) != (batch, sequence, 4, 4):
        raise ValueError("PRoPE inverse matrices must be token-aligned [B,S,4,4]")
    if feature_dim % 4:
        raise ValueError("PRoPE feature width must be divisible by four")
    return (
        _apply_tiled_projection(q, matrix=matrices.transpose(-1, -2)),
        _apply_tiled_projection(k, matrix=inverse_matrices),
        _apply_tiled_projection(v, matrix=inverse_matrices),
        partial(_apply_tiled_projection, matrix=matrices),
    )


def _apply_tiled_projection(
    features: torch.Tensor,
    *,
    matrix: torch.Tensor,
) -> torch.Tensor:
    batch, sequence, feature_dim = features.shape
    dimension = int(matrix.shape[-1])
    if feature_dim % dimension:
        raise ValueError("PRoPE feature width is not divisible by its basis dimension")
    if tuple(matrix.shape) != (batch, sequence, dimension, dimension):
        raise ValueError("PRoPE projection matrix is not token aligned")
    tiled = features.view(batch, sequence, feature_dim // dimension, dimension)
    matrix = matrix.to(dtype=features.dtype)
    return torch.einsum("bsij,bspj->bspi", matrix, tiled).reshape(features.shape)


def _invert_se3(transforms: torch.Tensor) -> torch.Tensor:
    rotation_inv = transforms[..., :3, :3].transpose(-1, -2)
    output = torch.zeros_like(transforms)
    output[..., :3, :3] = rotation_inv
    output[..., :3, 3] = -torch.einsum(
        "...ij,...j->...i",
        rotation_inv,
        transforms[..., :3, 3],
    )
    output[..., 3, 3] = 1.0
    return output


__all__ = [
    "DEFAULT_FOCAL_X",
    "DEFAULT_FOCAL_Y",
    "build_prope_matrices",
    "normalize_translation_transform",
    "prope_qkv",
    "relative_rows_to_c2w",
    "transform_relative_viewmats",
]
