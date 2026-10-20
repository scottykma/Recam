"""SpatialVID-HQ pose source.

The camera parquet carries one row per clip with a ``poses_path`` sidecar plus
the raw-frame indices that sidecar is keyed by.  The
sidecar is a zero-dimensional NumPy object array holding a
``{raw_video_frame: pose7}`` dictionary, so both the keys and the column are
read and cross-checked rather than one silently standing in for the other.

A null ``pose_frame_indices`` cell instead declares a dense sidecar keyed
``0..N-1``: the reader derives the indices from the sidecar keys and verifies
they really are dense, so an omitted cell can never mis-address a sparse
sidecar.  Every source is read through this one contract.
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np
import pyarrow.dataset as pa_dataset

from .annotation import load_pose_frame_map


POSES_PATH_COLUMN = "poses_path"
POSE_FRAME_INDICES_COLUMN = "pose_frame_indices"

POSE_COLUMN_BATCH_ROWS = 512


def _compact_dense_frame_indices(value: Any) -> Any:
    """Collapse the redundant ``range(N)`` a dense source wrote into a null cell."""

    if value is None or not len(value):
        return value
    if all(index == frame for index, frame in enumerate(value)):
        return None
    return value


def _frame_indices(value: Any, *, pose_path: str) -> np.ndarray | None:
    """Return one row's frame indices, or ``None`` when the cell declares denseness."""

    if value is None:
        return None
    frames = np.asarray(value, dtype=np.float64).reshape(-1)
    if frames.size == 0 or not np.all(np.isfinite(frames)):
        raise ValueError(
            f"pose frame indices are empty for {pose_path}; a dense source must "
            "leave the cell null rather than write an empty list"
        )
    rounded = np.rint(frames).astype(np.int64)
    if not np.allclose(frames, rounded):
        raise ValueError(f"pose frame indices must be integral for {pose_path}")
    if np.any(np.diff(rounded) <= 0):
        raise ValueError(f"pose frame indices must be strictly increasing for {pose_path}")
    return rounded


def pose_rows_for_sample(
    pose_path: str,
    frame_indices: Any,
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(raw_frame_indices, pose7)`` for one sampled clip."""

    mapping = load_pose_frame_map(pose_path)
    keys = sorted(mapping)
    frames = _frame_indices(frame_indices, pose_path=pose_path)
    if frames is None:
        if keys != list(range(len(keys))):
            raise ValueError(
                f"{POSE_FRAME_INDICES_COLUMN} is null for {pose_path}, which only "
                "describes a dense sidecar keyed 0..N-1; a sparse source must "
                "write the explicit frame indices"
            )
        frames = np.arange(len(keys), dtype=np.int64)
    if len(frames) != len(mapping):
        raise ValueError(
            f"pose frame index count {len(frames)} does not match pose rows "
            f"{len(mapping)} for {pose_path}"
        )
    if int(frames[0]) != keys[0] or int(frames[-1]) != keys[-1]:
        raise ValueError(
            f"{POSE_FRAME_INDICES_COLUMN} disagrees with the pose sidecar keys "
            f"for {pose_path}"
        )
    poses = np.stack([mapping[int(frame)] for frame in frames])
    return frames, poses


def read_pose_columns(
    parquet_paths: Sequence[str],
    total_rows: int,
) -> tuple[list[str], list[Any]]:
    """Return ``(pose_paths, frame_indices)`` in parquet row order."""

    source = pa_dataset.dataset(parquet_paths, format="parquet")
    required = (POSES_PATH_COLUMN, POSE_FRAME_INDICES_COLUMN)
    missing = sorted(set(required) - set(source.schema.names))
    if missing:
        raise ValueError(
            f"camera parquet is missing {missing}; rebuild the camera metadata"
        )
    pose_paths: list[str] = []
    frame_indices: list[Any] = []
    for batch in source.to_batches(
        columns=list(required), batch_size=POSE_COLUMN_BATCH_ROWS,
    ):
        columns = batch.to_pydict()
        pose_paths.extend(
            "" if value is None else str(value)
            for value in columns[POSES_PATH_COLUMN]
        )
        frame_indices.extend(
            _compact_dense_frame_indices(value)
            for value in columns[POSE_FRAME_INDICES_COLUMN]
        )
    if len(pose_paths) != total_rows:
        raise ValueError(
            "camera metadata row order/count differs from the base parquet"
        )
    return pose_paths, frame_indices
