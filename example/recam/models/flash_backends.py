"""FlashAttention / SageAttention probing and dispatch."""

import os

import torch
import torch.nn.functional as F
from einops import rearrange


try:
    import flash_attn_interface
    FLASH_ATTN_3_AVAILABLE = True
except ModuleNotFoundError:
    FLASH_ATTN_3_AVAILABLE = False

print("FLASH_ATTN_3_AVAILABLE:", FLASH_ATTN_3_AVAILABLE)

try:
    import flash_attn
    FLASH_ATTN_2_AVAILABLE = True
except ModuleNotFoundError:
    FLASH_ATTN_2_AVAILABLE = False

print("FLASH_ATTN_2_AVAILABLE:", FLASH_ATTN_2_AVAILABLE)

try:
    from flash_attn.layers.rotary import apply_rotary_emb as flash_apply_rotary_emb
    FLASH_ROTARY_AVAILABLE = True
except ImportError:
    flash_apply_rotary_emb = None
    FLASH_ROTARY_AVAILABLE = False

try:
    from flash_attn.ops.rms_norm import rms_norm as flash_rms_norm
    FLASH_RMS_NORM_AVAILABLE = True
    FLASH_RMS_NORM_BACKEND = "flash_attn.ops.rms_norm"
except (ImportError, RuntimeError):
    try:
        from flash_attn.ops.triton.layer_norm import rms_norm_fn as _flash_rms_norm_fn

        def flash_rms_norm(x, weight, eps):
            return _flash_rms_norm_fn(x, weight, None, eps=eps)

        FLASH_RMS_NORM_AVAILABLE = True
        FLASH_RMS_NORM_BACKEND = "flash_attn.ops.triton.layer_norm"
    except (ImportError, RuntimeError):
        flash_rms_norm = None
        FLASH_RMS_NORM_AVAILABLE = False
        FLASH_RMS_NORM_BACKEND = None


def _use_flash_rms_norm() -> bool:
    value = os.environ.get("FLOW_WAN_RMSNORM_IMPL", "").strip().lower()
    return value in {"1", "true", "yes", "flash", "flash_attn", "flash_attention"}


def _use_flash_rope() -> bool:
    value = os.environ.get("FLOW_WAN_ROPE_IMPL", "").strip().lower()
    return value in {"1", "true", "yes", "flash", "flash_attn", "flash_attention"}

try:
    from sageattention import sageattn
    SAGE_ATTN_AVAILABLE = True
except ModuleNotFoundError:
    SAGE_ATTN_AVAILABLE = False

print("SAGE_ATTN_AVAILABLE:", SAGE_ATTN_AVAILABLE)


def flash_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, head_dim: int, compatibility_mode=False):
    if compatibility_mode:
        q = rearrange(q, "b s (n d) -> b n s d", d=head_dim)
        k = rearrange(k, "b s (n d) -> b n s d", d=head_dim)
        v = rearrange(v, "b s (n d) -> b n s d", d=head_dim)
        x = F.scaled_dot_product_attention(q, k, v)
        x = rearrange(x, "b n s d -> b s (n d)", d=head_dim)
    elif FLASH_ATTN_3_AVAILABLE and q.is_cuda:
        q = rearrange(q, "b s (n d) -> b s n d", d=head_dim)
        k = rearrange(k, "b s (n d) -> b s n d", d=head_dim)
        v = rearrange(v, "b s (n d) -> b s n d", d=head_dim)
        x = flash_attn_interface.flash_attn_func(q, k, v)
        x = rearrange(x, "b s n d -> b s (n d)", d=head_dim)
    elif FLASH_ATTN_2_AVAILABLE and q.is_cuda:
        q = rearrange(q, "b s (n d) -> b s n d", d=head_dim)
        k = rearrange(k, "b s (n d) -> b s n d", d=head_dim)
        v = rearrange(v, "b s (n d) -> b s n d", d=head_dim)
        x = flash_attn.flash_attn_func(q, k, v)
        x = rearrange(x, "b s n d -> b s (n d)", d=head_dim)
    elif SAGE_ATTN_AVAILABLE and q.is_cuda:
        q = rearrange(q, "b s (n d) -> b n s d", d=head_dim)
        k = rearrange(k, "b s (n d) -> b n s d", d=head_dim)
        v = rearrange(v, "b s (n d) -> b n s d", d=head_dim)
        x = sageattn(q, k, v)
        x = rearrange(x, "b n s d -> b s (n d)", d=head_dim)
    else:
        q = rearrange(q, "b s (n d) -> b n s d", d=head_dim)
        k = rearrange(k, "b s (n d) -> b n s d", d=head_dim)
        v = rearrange(v, "b s (n d) -> b n s d", d=head_dim)
        x = F.scaled_dot_product_attention(q, k, v)
        x = rearrange(x, "b n s d -> b s (n d)", d=head_dim)
    return x

