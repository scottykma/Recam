"""ReCam camera dataset backed by SpatialVID pose annotations.

SpatialVID stores sparse ``[tx, ty, tz, qx, qy, qz, qw]`` world-to-camera rows
addressed by annotation index.  This module owns the model-side geometry (which
latent frames carry camera rows, and where the window anchor sits) and the pose
lookup; the arithmetic turning poses into canonical ``[F, 12]`` rows lives in
:mod:`example.recam.datasets.camera.format`.

The base dataset owns task sampling, AR window construction, decoding, and
retry behavior.  ``ReCamCameraDataset`` only adds the pose lookup and the
``camera`` field; it intentionally does not alter the training pipeline.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch

from ...config.config import VAE_TEMPORAL_FACTOR
from ..dataset import ReCamDataset, metadata_paths
from ..strategy import Task
from .format import interpolate_pose7, validate_camera_rows
from .sources.spatialvid import pose_rows_for_sample, read_pose_columns




class ReCamCameraDataset(ReCamDataset):
    """``ReCamDataset`` with one additional ``camera`` tensor field."""

    def __init__(
        self,
        metadata_path: Sequence[str] | str,
        *,
        camera_dtype: torch.dtype | str = torch.bfloat16,
        **kwargs: Any,
    ) -> None:
        parquet_paths = metadata_paths(metadata_path)
        super().__init__(parquet_paths, **kwargs)
        self.camera_dtype = self._resolve_camera_dtype(camera_dtype)
        self._metadata_directory = Path(parquet_paths[0]).resolve().parent
        self.pose_paths, self.pose_frame_indices = read_pose_columns(
            parquet_paths, self.total_rows
        )

    @staticmethod
    def _resolve_camera_dtype(value: torch.dtype | str) -> torch.dtype:
        """The config names a dtype; tests hand in the real thing."""
        if isinstance(value, str):
            return getattr(torch, value.removeprefix("torch."))
        return value

    def _camera_latent_positions(
        self,
        task: Task,
        n_history_chunks: int,
        source_indices: np.ndarray,
    ) -> np.ndarray:
        """Latent positions the camera covers: a whole short window, or a tail."""
        if source_indices.ndim != 1 or len(source_indices) == 0:
            raise ValueError("source indices must be a non-empty one-dimensional array")
        latent_positions = source_indices[::VAE_TEMPORAL_FACTOR]
        if task in (Task.T2V, Task.I2V):
            return latent_positions
        camera_start = 1 + self.geometry.history_frames(n_history_chunks)
        return latent_positions[
            camera_start:camera_start + self.geometry.main_frames
        ]

    def _pose_path(self, row_index: int) -> str:
        """Resolve one row's pose sidecar, relative to the metadata directory."""
        pose_path = self.pose_paths[row_index]
        if not pose_path:
            raise ValueError(f"camera metadata has no pose path for row {row_index}")
        candidate = Path(pose_path)
        if candidate.is_absolute() or candidate.is_file():
            return pose_path
        relative_to_metadata = self._metadata_directory / candidate
        if relative_to_metadata.is_file():
            return str(relative_to_metadata)
        return pose_path

    def _extra_sample_fields(
        self,
        row_index: int,
        task: Task,
        n_history_chunks: int,
        indices: np.ndarray,
    ) -> dict[str, torch.Tensor]:
        target_indices = self._camera_latent_positions(
            task, n_history_chunks, np.asarray(indices, dtype=np.int64)
        )
        source_frames, poses = pose_rows_for_sample(
            self._pose_path(row_index), self.pose_frame_indices[row_index],
        )
        anchor_index = self.geometry.overlap - 1 if task is Task.CONTINUE else 0
        camera = interpolate_pose7(
            source_frames, poses, target_indices, anchor_index=anchor_index
        )
        validate_camera_rows(
            camera, anchor_index=anchor_index, expected_length=len(target_indices),
        )
        return {"camera": torch.from_numpy(camera).to(dtype=self.camera_dtype)}
