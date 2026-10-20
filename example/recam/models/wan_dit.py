"""ReCam DiT built on the Wan backbone."""

from typing import Optional, Tuple

import torch
import torch.nn as nn
from einops import rearrange

from core.infra.parallelism import get_cp_manager
from example.recam.models.history_mlp import HistoryMLP
from example.recam.models.prope_attention import build_prope_matrices
from example.recam.models.wan_layers import (
    DiTBlock,
    Head,
    MLP,
    sinusoidal_embedding_1d,
)
from example.recam.models.wan_rope import freqs_cis_at, precompute_freqs_cis_3d


class _ReCamBackbone(torch.nn.Module):
    def __init__(
        self,
        dim: int,
        in_dim: int,
        ffn_dim: int,
        out_dim: int,
        text_dim: int,
        freq_dim: int,
        eps: float,
        patch_size: Tuple[int, int, int],
        num_heads: int,
        num_layers: int,
        has_image_input: bool,
        has_image_pos_emb: bool,
    ):
        super().__init__()
        self.dim = dim
        self.freq_dim = freq_dim
        self.has_image_input = has_image_input
        self.patch_size = patch_size

        self.patch_embedding = nn.Conv3d(
            in_dim, dim, kernel_size=patch_size, stride=patch_size)
        self.text_embedding = nn.Sequential(
            nn.Linear(text_dim, dim),
            nn.GELU(approximate='tanh'),
            nn.Linear(dim, dim)
        )
        self.time_embedding = nn.Sequential(
            nn.Linear(freq_dim, dim),
            nn.SiLU(),
            nn.Linear(dim, dim)
        )
        self.time_projection = nn.Sequential(
            nn.SiLU(), nn.Linear(dim, dim * 6))
        self.blocks = nn.ModuleList([
            DiTBlock(has_image_input, dim, num_heads, ffn_dim, eps)
            for _ in range(num_layers)
        ])

        self.num_layers = num_layers
        self.ffn_dim = ffn_dim
        self.eps = eps
        self.num_heads = num_heads
        self.head = Head(dim, out_dim, patch_size, eps)
        self.head_dim = dim // num_heads
        self.freqs = None

        if has_image_input:
            self.img_emb = MLP(1280, dim, has_pos_emb=has_image_pos_emb)
        self.has_image_pos_emb = has_image_pos_emb

    def patchify(self, x: torch.Tensor):
        x = self.patch_embedding(x)
        grid_size = x.shape[2:]
        x = rearrange(x, 'b c f h w -> b (f h w) c').contiguous()
        return x, grid_size

    def unpatchify(self, x: torch.Tensor, grid_size: torch.Tensor):
        return rearrange(
            x, 'b (f h w) (x y z c) -> b c (f x) (h y) (w z)',
            f=grid_size[0], h=grid_size[1], w=grid_size[2],
            x=self.patch_size[0], y=self.patch_size[1], z=self.patch_size[2]
        )


    def forward(
        self,
        x: torch.Tensor,
        timestep: torch.Tensor,
        context: torch.Tensor,
        clip_feature: Optional[torch.Tensor] = None,
        y: Optional[torch.Tensor] = None,
        t_index: Optional[torch.Tensor] = None,
        cross_attn_query_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        b, c, f, h, w = x.shape
        input_num_frames = f
        seq_len = f * h//self.patch_size[-2] * w//self.patch_size[-1]

        if cross_attn_query_mask is not None:
            if (
                cross_attn_query_mask.ndim != 1
                or cross_attn_query_mask.numel() != input_num_frames
            ):
                raise ValueError(
                    "cross_attn_query_mask must be one-dimensional with "
                    f"{input_num_frames} entries; got shape "
                    f"{tuple(cross_attn_query_mask.shape)}"
                )
            if cross_attn_query_mask.dtype != torch.bool:
                raise TypeError(
                    "cross_attn_query_mask must have bool dtype; got "
                    f"{cross_attn_query_mask.dtype}"
                )

        timestep = timestep.view(b, f, 1, 1).expand(b, f, h//self.patch_size[-2], w//self.patch_size[-1]).reshape(b, -1)

        t_emb = sinusoidal_embedding_1d(self.freq_dim, timestep.flatten()).unflatten(0, (b, seq_len))
        t = self.time_embedding(t_emb)
        t_mod = self.time_projection(t)
        t_mod = t_mod.unflatten(2, (6, self.dim))
        context_emb = self.text_embedding(context)

        if y is not None:
            x = torch.cat([x, y], dim=1)

        if self.has_image_input:
            if clip_feature is not None:
                clip_embedding = self.img_emb(clip_feature)
                context_emb = torch.cat([clip_embedding, context_emb], dim=1)

        x, (f, h, w) = self.patchify(x)

        if cross_attn_query_mask is not None:
            if f != input_num_frames:
                raise ValueError(
                    "cross_attn_query_mask requires temporal patch size 1; "
                    f"input has {input_num_frames} frames but patchify produced {f}"
                )
            cross_attn_query_mask = cross_attn_query_mask.to(device=x.device)
            cross_attn_query_mask = cross_attn_query_mask.view(f, 1, 1).expand(
                f, h, w,
            ).reshape(f * h * w)

        if self.freqs == None:
            self.freqs = precompute_freqs_cis_3d(self.head_dim)

        if t_index is None:
            freq_f = self.freqs[0][:f]
        else:
            if t_index.ndim != 1 or t_index.numel() != f:
                raise ValueError(
                    f"t_index must be one-dimensional with {f} entries; "
                    f"got shape {tuple(t_index.shape)}"
                )
            if t_index.is_floating_point() or t_index.dtype == torch.bool:
                raise TypeError(f"t_index must use an integer dtype; got {t_index.dtype}")
            t_index = t_index.to(device=self.freqs[0].device, dtype=torch.long)
            if t_index.numel() and (t_index.min() < 0 or t_index.max() >= self.freqs[0].shape[0]):
                raise ValueError(
                    f"t_index values must be in [0, {self.freqs[0].shape[0] - 1}]"
                )
            freq_f = self.freqs[0].index_select(0, t_index)
        freq_f = freq_f.view(f, 1, 1, -1).expand(f, h, w, -1)
        freq_h = self.freqs[1][:h].view(1, h, 1, -1).expand(f, h, w, -1)
        freq_w = self.freqs[2][:w].view(1, 1, w, -1).expand(f, h, w, -1)
        freqs = torch.cat([freq_f, freq_h, freq_w], dim=-1)

        freqs = freqs.reshape(f * h * w, 1, -1).to(x.device)

        if get_cp_manager():
            cp = get_cp_manager()
            assert self.num_heads % cp.cp_size == 0, "num_heads must be divisible by cp_size."
            assert (f * h * w) % cp.cp_size == 0, "seq_len must be divisible by cp_size."
            x = cp.split_input(x, 1, cache_k="wan_dp_seq")
            freqs = cp.split_input(freqs, 0)
            t_mod = cp.split_input(t_mod, 1)
            if cross_attn_query_mask is not None:
                cross_attn_query_mask = cp.split_input(
                    cross_attn_query_mask, 0,
                )

        for block in self.blocks:
            x = block(
                x,
                context_emb,
                t_mod,
                freqs,
                cross_attn_query_mask=cross_attn_query_mask,
            )

        if get_cp_manager():
            x = get_cp_manager().gather_output(x, 1, cache_k="wan_dp_seq")

        x = self.head(x, t)
        x = self.unpatchify(x, (f, h, w))
        return x


class ReCamModel(_ReCamBackbone):
    """Insert compact history after the first frame, keeping pretrained DiT keys."""

    def __init__(
        self,
        *args,
        history_pack_shape: tuple[int, int, int] = (2, 8, 8),
        history_pad_shape: tuple[int, int, int] = (2, 56, 104),
        camera_focal_x: float = 969.6969696969696 / 1920.0,
        camera_focal_y: float = 969.6969696969696 / 1080.0,
        camera_translation_transform: str = "linear",
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.camera_focal_x = float(camera_focal_x)
        self.camera_focal_y = float(camera_focal_y)
        self.camera_translation_transform = str(camera_translation_transform)
        self.history_pack_shape = tuple(history_pack_shape)
        self.history_pad_shape = tuple(history_pad_shape)
        self.history_mlp = HistoryMLP(
            self.dim,
            pack_shape=self.history_pack_shape,
            pad_shape=self.history_pad_shape,
        )
    def reset_parameters(self) -> None:
        """Initialize the newly added history MLP after materialization."""
        self.history_mlp.reset_parameters()

    @property
    def pack_linear(self) -> HistoryMLP:
        """Compatibility view for older external probes."""
        return self.history_mlp

    def _rope(
        self,
        temporal: torch.Tensor,
        spatial_h: torch.Tensor,
        spatial_w: torch.Tensor,
    ) -> torch.Tensor:
        temporal_dim = self.head_dim - 2 * (self.head_dim // 3)
        return torch.cat([
            freqs_cis_at(temporal_dim, temporal),
            freqs_cis_at(self.head_dim // 3, spatial_h),
            freqs_cis_at(self.head_dim // 3, spatial_w),
        ], dim=-1).unsqueeze(1)

    def _remove_memory_tokens(
        self,
        tokens: torch.Tensor,
        memory_count: int,
        patch_tokens_per_frame: int,
    ) -> torch.Tensor:
        if not memory_count:
            return tokens
        return torch.cat([
            tokens[:, :patch_tokens_per_frame],
            tokens[:, patch_tokens_per_frame + memory_count:],
        ], dim=1)

    def forward(
        self,
        x: torch.Tensor,
        timestep: torch.Tensor,
        context: torch.Tensor,
        *,
        temporal_positions: torch.Tensor,
        main_frame_start: int,
        history: Optional[torch.Tensor] = None,
        packed_history: Optional[
            tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
        ] = None,
        y: Optional[torch.Tensor] = None,
        camera: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        b = x.shape[0]
        context_emb = self.text_embedding(context)

        if y is not None:
            if y.shape[0] != b or y.shape[2:] != x.shape[2:]:
                raise ValueError(
                    "y must share batch and spatial shape with x; "
                    f"got {tuple(y.shape)} vs {tuple(x.shape)}"
                )
            x = torch.cat([x, y], dim=1)

        native_tokens, (f, h, w) = self.patchify(x)
        native_seq = f * h * w

        prope = None
        if camera is not None:
            if camera.ndim != 3 or camera.shape[0] != b or camera.shape[2] != 12:
                raise ValueError(
                    f"camera must have shape [{b}, F, 12]; got {tuple(camera.shape)}"
                )
            if camera.shape[1] > f:
                raise ValueError(
                    "camera window must not exceed the native frame count"
                )
            identity_row = camera.new_tensor([1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0])
            if not bool((camera == identity_row).all().item()):
                matrices, inverse_matrices = build_prope_matrices(
                    camera.to(device=x.device),
                    focal_x=self.camera_focal_x,
                    focal_y=self.camera_focal_y,
                    camera_translation_transform=self.camera_translation_transform,
                )
                camera_frames = camera.shape[1]
                prope = (
                    matrices.view(b, camera_frames, 1, 1, 4, 4)
                    .expand(b, camera_frames, h, w, 4, 4)
                    .reshape(b, camera_frames * h * w, 4, 4),
                    inverse_matrices.view(b, camera_frames, 1, 1, 4, 4)
                    .expand(b, camera_frames, h, w, 4, 4)
                    .reshape(b, camera_frames * h * w, 4, 4),
                )

        native_timestep = timestep.view(b, f, 1, 1).expand(b, f, h, w).reshape(b, native_seq)
        native_t = self.time_embedding(
            sinusoidal_embedding_1d(
                self.freq_dim, native_timestep.flatten(),
            ).unflatten(0, (b, native_seq))
        )
        native_t_mod = self.time_projection(native_t).unflatten(2, (6, self.dim))

        native_temporal = temporal_positions.to(x.device).view(f, 1, 1).expand(f, h, w).reshape(-1)
        native_h = torch.arange(h, device=x.device).view(1, h, 1).expand(f, h, w).reshape(-1)
        native_w = torch.arange(w, device=x.device).view(1, 1, w).expand(f, h, w).reshape(-1)
        native_freqs = self._rope(native_temporal, native_h, native_w)
        native_frame_mask = (
            torch.arange(f, device=x.device) >= int(main_frame_start)
        )
        native_output_mask = native_frame_mask.view(
            f, 1, 1,
        ).expand(f, h, w).reshape(-1)

        memory_tokens = None
        memory_t_mod = None
        memory_freqs = None
        if history is not None or packed_history is not None:
            if history is not None and packed_history is not None:
                raise ValueError("pass history or packed_history, not both")
            if packed_history is None:
                history_positions = torch.arange(
                    -history.shape[2], 0,
                    device=history.device,
                    dtype=torch.long,
                )
                memory_tokens, mem_t, mem_h, mem_w = self.history_mlp(
                    history, history_positions,
                )
            else:
                memory_tokens, mem_t, mem_h, mem_w = packed_history
            mem_count = memory_tokens.shape[1]
            mem_ts = timestep.new_zeros((b, mem_count))
            mem_t_embed = self.time_embedding(
                sinusoidal_embedding_1d(
                    self.freq_dim, mem_ts.flatten(),
                ).unflatten(0, (b, mem_count))
            )
            memory_t_mod = self.time_projection(mem_t_embed).unflatten(2, (6, self.dim))
            memory_freqs = self._rope(mem_t, mem_h, mem_w)

        insert_at = h * w
        if memory_tokens is None:
            tokens = native_tokens
            t_mod = native_t_mod
            freqs = native_freqs
            memory_count = 0
        else:
            memory_count = memory_tokens.shape[1]
            tokens = torch.cat([
                native_tokens[:, :insert_at], memory_tokens,
                native_tokens[:, insert_at:],
            ], dim=1)
            t_mod = torch.cat([
                native_t_mod[:, :insert_at], memory_t_mod,
                native_t_mod[:, insert_at:],
            ], dim=1)
            freqs = torch.cat([
                native_freqs[:insert_at], memory_freqs,
                native_freqs[insert_at:],
            ], dim=0)

        if memory_count:
            text_query_mask = torch.cat([
                native_output_mask[:insert_at],
                torch.zeros(
                    memory_count, dtype=torch.bool, device=x.device,
                ),
                native_output_mask[insert_at:],
            ])
        else:
            text_query_mask = native_output_mask

        if get_cp_manager():
            cp = get_cp_manager()
            assert self.num_heads % cp.cp_size == 0, "num_heads must be divisible by cp_size."
            assert (
                tokens.shape[1] % cp.cp_size == 0
            ), "attention sequence must be divisible by cp_size."
            if prope is not None:
                offset, length = cp.local_window(tokens.shape[1], prope[0].shape[1])
                prope = (
                    (
                        prope[0][:, offset:offset + length],
                        prope[1][:, offset:offset + length],
                    )
                    if length
                    else None
                )
            tokens = cp.split_input(tokens, 1, cache_k="recam_seq")
            freqs = cp.split_input(freqs, 0)
            t_mod = cp.split_input(t_mod, 1)
            text_query_mask = cp.split_input(text_query_mask, 0)

        for block in self.blocks:
            tokens = block(
                tokens, context_emb, t_mod, freqs,
                cross_attn_query_mask=text_query_mask,
                prope=prope,
            )

        if get_cp_manager():
            tokens = get_cp_manager().gather_output(tokens, 1, cache_k="recam_seq")

        tokens = self._remove_memory_tokens(tokens, memory_count, insert_at)
        main_frames = f - int(main_frame_start)
        head_mask = native_output_mask
        output = self.head(tokens[:, head_mask], native_t[:, head_mask])
        return self.unpatchify(output, (main_frames, h, w))

