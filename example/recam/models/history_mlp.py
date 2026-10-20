"""Compact history packer for ReCam."""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


class HistoryMLP(nn.Module):
    """Pack clean history blocks and project them into DiT hidden space."""

    channels = 16

    def __init__(
        self,
        dim: int = 5120,
        pack_shape: tuple[int, int, int] = (2, 8, 8),
        pad_shape: tuple[int, int, int] = (2, 56, 104),
    ) -> None:
        super().__init__()
        self.pack_shape = tuple(int(value) for value in pack_shape)
        self.pad_shape = tuple(int(value) for value in pad_shape)
        if len(self.pack_shape) != 3 or math.prod(self.pack_shape) != 128:
            raise ValueError("pack_shape must contain exactly 128 latent values")
        if len(self.pad_shape) != 3:
            raise ValueError("pad_shape must contain [temporal multiple, height, width]")
        pt, ph, pw = self.pack_shape
        temporal_multiple, padded_h, padded_w = self.pad_shape
        if (
            min(*self.pack_shape, *self.pad_shape) <= 0
            or temporal_multiple % pt
            or padded_h % ph
            or padded_w % pw
            or ph % 2
            or pw % 2
        ):
            raise ValueError(
                "pad_shape must be positive, divisible by pack_shape, and use "
                "even spatial pack sizes"
            )

        self.in_features = self.channels * math.prod(self.pack_shape)
        self.hidden_features = self.in_features
        self.fc1 = nn.Linear(self.in_features, self.hidden_features, bias=True)
        self.activation = nn.SiLU()
        self.fc2 = nn.Linear(self.hidden_features, dim, bias=True)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for layer in (self.fc1, self.fc2):
            nn.init.xavier_uniform_(layer.weight)
            nn.init.zeros_(layer.bias)

    @staticmethod
    def _integer_positions(positions: torch.Tensor) -> torch.Tensor:
        if positions.ndim != 1:
            raise ValueError(
                "temporal_positions must be one-dimensional; "
                f"got {tuple(positions.shape)}"
            )
        if positions.is_floating_point():
            rounded = positions.round()
            if not torch.equal(positions, rounded):
                raise ValueError("history temporal positions must be integers")
            return rounded.to(dtype=torch.long)
        if positions.dtype == torch.bool:
            raise TypeError("history temporal positions must use an integer dtype")
        return positions.to(dtype=torch.long)

    def forward(
        self,
        history: torch.Tensor,
        temporal_positions: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if history.ndim != 5 or history.shape[1] != self.channels:
            raise ValueError(
                "history must have shape [B,16,T,H,W]; got "
                f"{tuple(history.shape)}"
            )
        b, _, t, h, w = history.shape
        temporal_multiple, padded_h, padded_w = self.pad_shape
        pt, ph, pw = self.pack_shape
        if t == 0 or h > padded_h or w > padded_w:
            raise ValueError(
                "history T must be positive and H/W must fit the common "
                f"padding grid {(padded_h, padded_w)}; got {(t, h, w)}"
            )
        if temporal_positions.ndim != 1 or temporal_positions.numel() != t:
            raise ValueError(
                f"temporal_positions must contain {t} values; got "
                f"{tuple(temporal_positions.shape)}"
            )
        positions = self._integer_positions(
            temporal_positions.to(device=history.device)
        )

        pad_t = (-t) % temporal_multiple
        pad_h = padded_h - h
        pad_w = padded_w - w
        history = F.pad(
            history,
            (0, pad_w, 0, pad_h, pad_t, 0),
            mode="constant",
            value=0.0,
        )
        packed = rearrange(
            history,
            "b c (nt pt) (nh ph) (nw pw) -> b (nt nh nw) (c pt ph pw)",
            pt=pt, ph=ph, pw=pw,
        ).contiguous()
        tokens = self.fc2(self.activation(self.fc1(packed)))

        if pad_t:
            positions = torch.cat([positions[:1].expand(pad_t), positions])
        grouped_positions = positions.view(-1, pt)
        pack_t = grouped_positions[:, 0]

        nt = grouped_positions.shape[0]
        nh, nw = padded_h // ph, padded_w // pw
        native_step_h, native_step_w = ph // 2, pw // 2
        token_t = pack_t.view(nt, 1, 1).expand(nt, nh, nw).reshape(-1)
        token_h = (
            torch.arange(nh, device=history.device, dtype=torch.long)
            .mul(native_step_h)
            .view(1, nh, 1)
            .expand(nt, nh, nw)
            .reshape(-1)
        )
        token_w = (
            torch.arange(nw, device=history.device, dtype=torch.long)
            .mul(native_step_w)
            .view(1, 1, nw)
            .expand(nt, nh, nw)
            .reshape(-1)
        )
        return tokens, token_t, token_h, token_w


PackLinear = HistoryMLP

