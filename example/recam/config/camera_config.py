"""Fused-PRoPE camera-control config schema for ReCam."""

import math
from dataclasses import dataclass, field
from typing import ClassVar, Literal, Optional

from example.recam.config.config import ExpConfig, GeometryConfig


@dataclass
class CameraConfig:
    """Fused-PRoPE camera transforms on the AR window self-attention."""

    focal_x: float = 969.6969696969696 / 1920.0
    focal_y: Optional[float] = None
    translation_transform: Literal["linear", "logd4"] = "linear"
    camera_dtype: Literal["bfloat16", "float32"] = "bfloat16"
    camera_dropout_prob: float = 0.0
    trajectory_span: Optional[float] = None

    def __post_init__(self) -> None:
        if not math.isfinite(float(self.focal_x)) or self.focal_x <= 0:
            raise ValueError("camera.focal_x must be finite and positive")
        if self.focal_y is not None and (
            not math.isfinite(float(self.focal_y)) or self.focal_y <= 0
        ):
            raise ValueError("camera.focal_y must be finite and positive")
        if not 0.0 <= float(self.camera_dropout_prob) < 1.0:
            raise ValueError("camera.camera_dropout_prob must be in [0, 1)")
        if self.trajectory_span is not None and (
            not math.isfinite(float(self.trajectory_span))
            or self.trajectory_span <= 0
        ):
            raise ValueError("camera.trajectory_span must be finite and positive")

    def resolved_focal_y(self, geometry: GeometryConfig) -> float:
        if self.focal_y is not None:
            return float(self.focal_y)
        return float(self.focal_x) * geometry.width / geometry.height

    def as_runtime_state(self, geometry: GeometryConfig) -> dict[str, object]:
        return {
            "focal_x": float(self.focal_x),
            "focal_y": self.resolved_focal_y(geometry),
            "translation_transform": self.translation_transform,
            "camera_dropout_prob": float(self.camera_dropout_prob),
            "trajectory_span": (
                None if self.trajectory_span is None else float(self.trajectory_span)
            ),
        }


@dataclass
class CameraExpConfig(ExpConfig):
    camera: CameraConfig = field(default_factory=CameraConfig)

    def __post_init__(self) -> None:
        super().__post_init__()
        transformer = self.model.transformer
        if (transformer.dim // transformer.num_heads) % 4 != 0:
            raise ValueError(
                "camera control requires transformer head_dim divisible by 4"
            )
