"""3D RoPE precomputation and application."""

import torch
from einops import rearrange

from example.recam.models.flash_backends import (
    FLASH_ROTARY_AVAILABLE,
    _use_flash_rope,
    flash_apply_rotary_emb,
)


def precompute_freqs_cis_3d(dim: int, end: int = 1024, theta: float = 10000.0):
    f_freqs_cis = precompute_freqs_cis(dim - 2 * (dim // 3), end, theta)
    h_freqs_cis = precompute_freqs_cis(dim // 3, end, theta)
    w_freqs_cis = precompute_freqs_cis(dim // 3, end, theta)
    return f_freqs_cis, h_freqs_cis, w_freqs_cis


def precompute_freqs_cis(dim: int, end: int = 1024, theta: float = 10000.0):
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)
                   [: (dim // 2)].double() / dim))
    freqs = torch.outer(torch.arange(end, device=freqs.device), freqs)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
    return freqs_cis


def _apply_rope_torch(x, freqs, num_heads):
    x = rearrange(x, "b s (n d) -> b s n d", n=num_heads)
    x_out = torch.view_as_complex(x.to(torch.float64).reshape(
        x.shape[0], x.shape[1], x.shape[2], -1, 2))
    x_out = torch.view_as_real(x_out * freqs).flatten(2)
    return x_out.to(x.dtype)


def _flash_rope_cos_sin(freqs):
    freqs = freqs.squeeze(1) if freqs.dim() == 3 and freqs.shape[1] == 1 else freqs
    return freqs.real.to(torch.float32), freqs.imag.to(torch.float32)


def apply_rope(x, freqs, num_heads):
    if _use_flash_rope() and FLASH_ROTARY_AVAILABLE and x.is_cuda:
        x = rearrange(x, "b s (n d) -> b s n d", n=num_heads).contiguous()
        cos, sin = _flash_rope_cos_sin(freqs)
        x = flash_apply_rotary_emb(x, cos, sin, interleaved=True, inplace=False)
        return rearrange(x, "b s n d -> b s (n d)")
    return _apply_rope_torch(x, freqs, num_heads)


def apply_qk_rope(q, k, freqs, num_heads):
    if _use_flash_rope() and FLASH_ROTARY_AVAILABLE and q.is_cuda and k.is_cuda:
        q = rearrange(q, "b s (n d) -> b s n d", n=num_heads)
        k = rearrange(k, "b s (n d) -> b s n d", n=num_heads)
        qk = torch.cat([q, k], dim=2).contiguous()
        cos, sin = _flash_rope_cos_sin(freqs)
        qk = flash_apply_rotary_emb(qk, cos, sin, interleaved=True, inplace=False)
        q, k = qk.split(num_heads, dim=2)
        return (
            rearrange(q, "b s n d -> b s (n d)"),
            rearrange(k, "b s n d -> b s (n d)"),
        )
    return apply_rope(q, freqs, num_heads), apply_rope(k, freqs, num_heads)


def freqs_cis_at(
    dim: int, positions: torch.Tensor, theta: float = 10000.0,
) -> torch.Tensor:
    """Compute RoPE frequencies at arbitrary floating-point coordinates."""
    inv_freq = 1.0 / (theta ** (
        torch.arange(
            0, dim, 2, dtype=torch.float64, device=positions.device,
        )[: dim // 2] / dim
    ))
    phase = positions.to(torch.float64).unsqueeze(-1) * inv_freq
    return torch.polar(torch.ones_like(phase), phase)

