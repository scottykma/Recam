"""Transformer layers for the ReCam DiT."""

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn

from core.infra.parallelism import get_cp_manager
from example.recam.models.flash_backends import (
    FLASH_RMS_NORM_AVAILABLE,
    _use_flash_rms_norm,
    flash_attention,
    flash_rms_norm,
)
from example.recam.models.prope_attention import prope_qkv
from example.recam.models.wan_rope import apply_qk_rope


T5_CONTEXT_TOKEN_NUMBER = 512


def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor):
    return (x * (1 + scale) + shift)


def sinusoidal_embedding_1d(dim, position):
    sinusoid = torch.outer(position.type(torch.float64), torch.pow(
        10000, -torch.arange(dim//2, dtype=torch.float64, device=position.device).div(dim//2)))
    x = torch.cat([torch.cos(sinusoid), torch.sin(sinusoid)], dim=1)
    return x.to(position.dtype)


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
        self.use_flash = _use_flash_rms_norm() and FLASH_RMS_NORM_AVAILABLE

    def norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)

    def forward(self, x):
        if self.use_flash and x.is_cuda:
            return flash_rms_norm(x, self.weight, self.eps)
        dtype = x.dtype
        return self.norm(x.float()).to(dtype) * self.weight


class AttentionModule(nn.Module):
    def __init__(self, head_dim):
        super().__init__()
        self.head_dim = head_dim

    def forward(self, q, k, v):
        x = flash_attention(q=q, k=k, v=v, head_dim=self.head_dim)
        return x


class SelfAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int, eps: float = 1e-6):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.o = nn.Linear(dim, dim)
        self.norm_q = RMSNorm(dim, eps=eps)
        self.norm_k = RMSNorm(dim, eps=eps)

        self.attn = AttentionModule(self.head_dim)

    def forward(
        self,
        x,
        freqs,
        prope: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
    ):
        q = self.norm_q(self.q(x))
        k = self.norm_k(self.k(x))
        v = self.v(x)
        q, k = apply_qk_rope(q, k, freqs, self.num_heads)
        if prope is not None:
            window = prope[0].shape[1]
            if window > q.shape[1]:
                raise ValueError(
                    f"camera window {window} exceeds sequence length {q.shape[1]}"
                )
            q_w, k_w, v_w, apply_o = prope_qkv(
                q[:, -window:], k[:, -window:], v[:, -window:],
                matrices=prope[0], inverse_matrices=prope[1],
            )
            q = torch.cat([q[:, :-window], q_w], dim=1)
            k = torch.cat([k[:, :-window], k_w], dim=1)
            v = torch.cat([v[:, :-window], v_w], dim=1)
        if get_cp_manager():
            q = get_cp_manager().SeqAllToAll(q, 2, 1)
            k = get_cp_manager().SeqAllToAll(k, 2, 1)
            v = get_cp_manager().SeqAllToAll(v, 2, 1)
        x = self.attn(q, k, v)
        if get_cp_manager():
            x = get_cp_manager().SeqAllToAll(x, 1, 2)
        if prope is not None:
            x = torch.cat(
                [x[:, :-window], apply_o(x[:, -window:])], dim=1,
            )
        return self.o(x)


class CrossAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int, eps: float = 1e-6, has_image_input: bool = False):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.o = nn.Linear(dim, dim)
        self.norm_q = RMSNorm(dim, eps=eps)
        self.norm_k = RMSNorm(dim, eps=eps)
        self.has_image_input = has_image_input
        if has_image_input:
            self.k_img = nn.Linear(dim, dim)
            self.v_img = nn.Linear(dim, dim)
            self.norm_k_img = RMSNorm(dim, eps=eps)

        self.attn = AttentionModule(self.head_dim)

    def forward(self, x: torch.Tensor, y: torch.Tensor):
        if self.has_image_input:
            image_context_length = y.shape[1] - T5_CONTEXT_TOKEN_NUMBER
            img = y[:, :image_context_length]
            ctx = y[:, image_context_length:]
        else:
            ctx = y
        q = self.norm_q(self.q(x))
        k = self.norm_k(self.k(ctx))
        v = self.v(ctx)
        x = self.attn(q, k, v)
        if self.has_image_input:
            k_img = self.norm_k_img(self.k_img(img))
            v_img = self.v_img(img)
            y = flash_attention(q, k_img, v_img, head_dim=self.head_dim)
            x = x + y
        return self.o(x)


class GateModule(nn.Module):
    def __init__(self,):
        super().__init__()

    def forward(self, x, gate, residual):
        return x + gate * residual


class DiTBlock(nn.Module):
    def __init__(self, has_image_input: bool, dim: int, num_heads: int, ffn_dim: int, eps: float = 1e-6):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.ffn_dim = ffn_dim

        self.self_attn = SelfAttention(dim, num_heads, eps)
        self.cross_attn = CrossAttention(
            dim, num_heads, eps, has_image_input=has_image_input)
        self.norm1 = nn.LayerNorm(dim, eps=eps, elementwise_affine=False)
        self.norm2 = nn.LayerNorm(dim, eps=eps, elementwise_affine=False)
        self.norm3 = nn.LayerNorm(dim, eps=eps)
        self.ffn = nn.Sequential(nn.Linear(dim, ffn_dim), nn.GELU(
            approximate='tanh'), nn.Linear(ffn_dim, dim))
        self.modulation = nn.Parameter(torch.randn(1, 6, dim) / dim**0.5)
        self.gate = GateModule()

    def forward(
        self,
        x,
        context,
        t_mod,
        freqs,
        cross_attn_query_mask: Optional[torch.Tensor] = None,
        prope: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
    ):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            self.modulation.unsqueeze(0).to(dtype=t_mod.dtype, device=t_mod.device) + t_mod).chunk(6, dim=2)

        input_x = modulate(self.norm1(x), shift_msa.squeeze(2), scale_msa.squeeze(2))
        attn_out = self.self_attn(input_x, freqs, prope=prope)
        x = self.gate(x, gate_msa.squeeze(2), attn_out)

        if cross_attn_query_mask is None:
            x = x + self.cross_attn(self.norm3(x), context)
        else:
            query_indices = cross_attn_query_mask.nonzero(as_tuple=True)[0]
            if query_indices.numel():
                x_query = x.index_select(1, query_indices)
                cross_out = self.cross_attn(self.norm3(x_query), context)
                x = x.index_add(1, query_indices, cross_out)

        input_x = modulate(self.norm2(x), shift_mlp.squeeze(2), scale_mlp.squeeze(2))
        x = self.gate(x, gate_mlp.squeeze(2), self.ffn(input_x))
        return x


class MLP(torch.nn.Module):
    def __init__(self, in_dim, out_dim, has_pos_emb=False):
        super().__init__()
        self.proj = torch.nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, in_dim),
            nn.GELU(),
            nn.Linear(in_dim, out_dim),
            nn.LayerNorm(out_dim)
        )
        self.has_pos_emb = has_pos_emb
        if has_pos_emb:
            self.emb_pos = torch.nn.Parameter(torch.zeros((1, 514, 1280)))

    def forward(self, x):
        if self.has_pos_emb:
            x = x + self.emb_pos.to(dtype=x.dtype, device=x.device)
        return self.proj(x)


class Head(nn.Module):
    def __init__(self, dim: int, out_dim: int, patch_size: Tuple[int, int, int], eps: float):
        super().__init__()
        self.dim = dim
        self.patch_size = patch_size
        self.norm = nn.LayerNorm(dim, eps=eps, elementwise_affine=False)
        self.head = nn.Linear(dim, out_dim * math.prod(patch_size))
        self.modulation = nn.Parameter(torch.randn(1, 2, dim) / dim**0.5)

    def forward(self, x, t_mod):
        shift, scale = (self.modulation.unsqueeze(0).to(dtype=t_mod.dtype, device=t_mod.device) + t_mod.unsqueeze(2)).chunk(2, dim=2)
        x = (self.head(self.norm(x) * (1 + scale.squeeze(2)) + shift.squeeze(2)))
        return x

