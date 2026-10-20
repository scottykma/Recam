"""SpatialVID-HQ annotation primitives for ReCam camera metadata.

Published annotations store camera rows as ``[tx, ty, tz, qx, qy, qz, qw]``
describing a world-to-camera transform, while pose rows are addressed by
annotation index rather than by raw video frame.  The helpers below read the
CSV/annotation contract and materialize a frame-keyed pose sidecar so the
training dataset can look poses up by raw frame number.

Note the convention split: ``pose7_to_w2c`` here returns **w2c**, whereas
``format.pose7_to_c2w`` returns camera-to-world.
"""

from __future__ import annotations

import json
import os
import tempfile
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np

from ..format import EPS, finite_array, quaternion_xyzw_to_matrix


POSE_MAP_FILENAME = "poses_by_frame.npy"


def _clean(value: Any) -> str:
    return "" if value is None else str(value).strip()


def _row_value(row: dict[str, Any], *names: str, default: Any = None) -> Any:
    """Read a metadata field across published and regenerated CSV schemas."""

    for name in names:
        if name in row and _clean(row[name]):
            return row[name]
    return default


def _group_id(value: Any) -> int | None:
    """Parse numeric group ids and labels such as ``group_0001``."""

    text = _clean(value)
    if not text:
        return None
    try:
        return int(float(text))
    except (TypeError, ValueError, OverflowError):
        tail = text.rsplit("_", 1)[-1]
        try:
            return int(tail)
        except (TypeError, ValueError, OverflowError):
            return None


def _int_field(row: dict[str, Any], *names: str, default: int | None = None) -> int | None:
    value = _row_value(row, *names)
    if value is None:
        return default
    try:
        parsed = int(float(value))
    except (TypeError, ValueError, OverflowError):
        return default
    return parsed if parsed > 0 else default


def _load_caption(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8-sig") as handle:
        return json.load(handle)


def resolve_path(
    data_root: str | os.PathLike[str],
    value: str | os.PathLike[str],
    *,
    must_exist: bool = False,
) -> Path:
    """Resolve CSV paths, including an extracted ``root/SpatialVID`` variant."""

    root = Path(data_root)
    raw = Path(str(value))
    candidates: list[Path]
    if raw.is_absolute():
        candidates = [raw]
    else:
        candidates = [root / raw, root / "SpatialVID" / raw]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    if must_exist:
        raise FileNotFoundError(f"could not resolve {value!s}; tried: {candidates}")
    return candidates[0] if candidates else raw


def path_for_record(path: Path) -> str:
    """Store resource paths as absolute strings in every metadata record."""

    return str(path.resolve())


def pose7_to_w2c(pose7: Any, homogeneous: bool = True) -> np.ndarray:
    """Convert SpatialVID pose7 rows to w2c homogeneous or 3x4 matrices."""

    pose = finite_array(pose7, "pose7")
    if pose.ndim == 0 or pose.shape[-1] != 7:
        raise ValueError(f"expected pose shape (..., 7), got {pose.shape}")
    rotation = quaternion_xyzw_to_matrix(pose[..., 3:])
    value34 = np.concatenate([rotation, pose[..., :3, None]], axis=-1)
    if not homogeneous:
        return value34
    result = np.broadcast_to(np.eye(4, dtype=np.float64), (*pose.shape[:-1], 4, 4)).copy()
    result[..., :3, :4] = value34
    return result


def validate_pose_sequence(
    poses: Any, *, rotation_tolerance: float = 1e-3, quaternion_tolerance: float = 1e-3
) -> dict[str, float | int]:
    """Return quality statistics and raise on malformed pose rows."""

    value = finite_array(poses, "poses")
    if value.ndim != 2 or value.shape[1] != 7:
        raise ValueError(f"expected poses with shape (N, 7), got {value.shape}")
    q_norm = np.linalg.norm(value[:, 3:], axis=1)
    matrices = pose7_to_w2c(value, homogeneous=False)
    rotations = matrices[:, :3, :3]
    orthogonality = np.max(
        np.abs(np.matmul(np.swapaxes(rotations, -1, -2), rotations) - np.eye(3)),
        axis=(1, 2),
    )
    determinant = np.linalg.det(rotations)
    if (
        np.any(q_norm < EPS)
        or np.any(np.abs(q_norm - 1.0) > quaternion_tolerance)
        or np.any(np.abs(orthogonality) > rotation_tolerance)
        or np.any(np.abs(determinant - 1.0) > rotation_tolerance)
    ):
        raise ValueError("pose sequence contains an invalid rotation")
    return {
        "count": int(len(value)),
        "quaternion_norm_min": float(q_norm.min()),
        "quaternion_norm_max": float(q_norm.max()),
        "quaternion_norm_max_abs_error": float(np.max(np.abs(q_norm - 1.0))),
        "rotation_orthogonality_max_error": float(orthogonality.max()),
        "rotation_det_min": float(determinant.min()),
        "rotation_det_max": float(determinant.max()),
    }


def read_indexes(path: str | os.PathLike[str]) -> tuple[list[int], list[int]]:
    """Read ``annotation_index raw_video_frame_index`` pairs.

    The first column identifies an annotation row; the second column is the
    only column that may be used to address frames in the source video. This
    helper is used only while creating the frame-keyed pose sidecar.
    """

    annotation: list[int] = []
    raw: list[int] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            parts = line.lstrip("\ufeff \t").split()
            if not parts or parts[0].startswith("#"):
                continue
            if len(parts) < 2:
                raise ValueError(f"{path}:{line_number}: expected two integer columns")
            try:
                ann_index, raw_index = int(parts[0]), int(parts[1])
            except ValueError as exc:
                raise ValueError(f"{path}:{line_number}: non-integer index") from exc
            annotation.append(ann_index)
            raw.append(raw_index)
    if not raw:
        raise ValueError(f"{path} contains no index rows")
    if any(index < 0 for index in annotation):
        raise ValueError(f"{path} contains a negative annotation index")
    if any(index < 0 for index in raw):
        raise ValueError(f"{path} contains a negative raw frame index")
    if any(right <= left for left, right in zip(raw, raw[1:])):
        raise ValueError(f"{path} raw frame indexes are not strictly increasing")
    if len(set(annotation)) != len(annotation):
        raise ValueError(f"{path} annotation indexes are not unique")
    return annotation, raw


def _normalize_pose_map(mapping: Any, path: str | os.PathLike[str]) -> dict[int, np.ndarray]:
    if not isinstance(mapping, dict) or not mapping:
        raise ValueError(f"{path} must contain a non-empty dictionary")
    normalized: dict[int, np.ndarray] = {}
    for raw_frame, pose in mapping.items():
        if isinstance(raw_frame, (bool, np.bool_)) or not isinstance(raw_frame, (int, np.integer)):
            raise ValueError(f"{path} has a non-integer raw-frame key: {raw_frame!r}")
        key = int(raw_frame)
        if key < 0:
            raise ValueError(f"{path} has a negative raw-frame key: {key}")
        value = np.asarray(pose)
        if value.shape != (7,):
            raise ValueError(f"{path}[{key}] must have shape (7,), got {value.shape}")
        try:
            finite = np.isfinite(value)
        except TypeError:
            try:
                value = np.asarray(pose, dtype=np.float64)
                finite = np.isfinite(value)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"{path}[{key}] is not numeric") from exc
        if not np.all(finite):
            raise ValueError(f"{path}[{key}] contains non-finite values")
        value = np.array(value, copy=True)
        value.setflags(write=False)
        normalized[key] = value
    if len(normalized) != len(mapping):
        raise ValueError(f"{path} contains duplicate raw-frame keys")
    return normalized


@lru_cache(maxsize=64)
def _load_pose_frame_map_cached(path: str) -> dict[int, np.ndarray]:
    """Load a generated NumPy object-array dictionary once per worker."""

    payload = np.load(path, allow_pickle=True)
    try:
        if not isinstance(payload, np.ndarray) or payload.shape != ():
            raise ValueError(
                f"{path} must be a zero-dimensional NumPy object array containing "
                "{raw_video_frame: pose7}"
            )
        return _normalize_pose_map(payload.item(), path)
    finally:
        close = getattr(payload, "close", None)
        if close is not None:
            close()


def load_pose_frame_map(path: str | os.PathLike[str]) -> dict[int, np.ndarray]:
    return _load_pose_frame_map_cached(str(Path(path).resolve()))


def _save_numpy_object(path: Path, payload: Any) -> None:
    """Atomically write a trusted NumPy object payload."""

    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{path.stem}.",
        suffix=".tmp",
        dir=str(path.parent),
    )
    os.close(fd)
    temporary = Path(temporary_name)
    try:
        with temporary.open("wb") as handle:
            np.save(handle, payload, allow_pickle=True)
        os.replace(temporary, path)
        path.chmod(0o644)
    finally:
        if temporary.exists():
            temporary.unlink()


def _save_pose_frame_map(path: Path, mapping: dict[int, np.ndarray]) -> None:
    _save_numpy_object(path, mapping)


def prepare_pose_frame_map(
    annotation_dir: str | os.PathLike[str],
    output_path: str | os.PathLike[str] | None = None,
) -> Path:
    """Create or reuse ``poses_by_frame.npy`` for one annotation directory.

    Published SpatialVID annotations store pose rows separately from raw-frame
    numbers. This function performs that conversion once while building meta.
    If ``poses.npy`` is already a keyed dictionary, it is used directly.
    """

    directory = Path(annotation_dir)
    explicit_target = output_path is not None
    target = Path(output_path) if explicit_target else directory / POSE_MAP_FILENAME
    source = directory / "poses.npy"
    indexes = directory / "indexes.txt"

    if target.is_file():
        try:
            load_pose_frame_map(target)
            if not source.is_file() or not indexes.is_file():
                return target
            source_mtime = max(
                (candidate.stat().st_mtime for candidate in (source, indexes) if candidate.exists()),
                default=0.0,
            )
            if target.stat().st_mtime >= source_mtime:
                return target
        except (OSError, ValueError, EOFError, TypeError):
            pass

    if not source.is_file():
        raise FileNotFoundError(source)

    source_payload = np.load(source, allow_pickle=True)
    if (
        not explicit_target
        and isinstance(source_payload, np.ndarray)
        and source_payload.shape == ()
        and isinstance(source_payload.item(), dict)
    ):
        _load_pose_frame_map_cached.cache_clear()
        return source

    if isinstance(source_payload, np.ndarray) and source_payload.shape == () and isinstance(source_payload.item(), dict):
        normalized_source = _normalize_pose_map(source_payload.item(), source)
        validate_pose_sequence(np.stack([normalized_source[key] for key in sorted(normalized_source)]))
        _save_pose_frame_map(target, normalized_source)
        _load_pose_frame_map_cached.cache_clear()
        return target

    if not indexes.is_file():
        raise FileNotFoundError(indexes)
    annotation_indices, raw_frames = read_indexes(indexes)
    poses = np.asarray(source_payload)
    if poses.ndim != 2 or poses.shape[1] != 7:
        raise ValueError(f"{source} must have shape (N, 7), got {poses.shape}")
    if len(poses) != len(raw_frames):
        raise ValueError(f"pose/index length mismatch: poses={len(poses)}, indexes={len(raw_frames)}")
    if sorted(annotation_indices) != list(range(len(raw_frames))):
        raise ValueError(f"{indexes} annotation indexes must be a permutation of pose rows 0..N-1")

    mapping = {
        int(raw_frame): np.array(poses[annotation_row], copy=True)
        for annotation_row, raw_frame in zip(annotation_indices, raw_frames)
    }
    _save_pose_frame_map(target, _normalize_pose_map(mapping, target))
    _load_pose_frame_map_cached.cache_clear()
    return target


def read_npy_shape(path: str | os.PathLike[str]) -> tuple[int, ...]:
    payload = np.load(path, mmap_mode="r", allow_pickle=False)
    try:
        return tuple(int(item) for item in payload.shape)
    finally:
        close = getattr(payload, "close", None)
        if close is not None:
            close()


def validate_annotation_files(
    annotation_dir: Path,
    *,
    source_num_frames: int,
    pose_map_path: str | os.PathLike[str] | None = None,
) -> dict[str, Any]:
    """Validate one annotation and materialize its frame-keyed pose file."""

    annotation_dir = Path(annotation_dir)
    intrinsics_path = annotation_dir / "intrinsics.npy"
    if not annotation_dir.is_dir():
        raise FileNotFoundError(annotation_dir)
    if not intrinsics_path.is_file():
        raise FileNotFoundError(intrinsics_path)
    poses_path = prepare_pose_frame_map(annotation_dir, output_path=pose_map_path)
    pose_map = load_pose_frame_map(poses_path)
    intrinsic_shape = read_npy_shape(intrinsics_path)
    if len(intrinsic_shape) != 2 or intrinsic_shape[1] != 4:
        raise ValueError(f"{intrinsics_path} must have shape (N, 4), got {intrinsic_shape}")
    expected = len(pose_map)
    if intrinsic_shape[0] != expected:
        raise ValueError(
            f"annotation lengths disagree: poses_by_frame={expected}, intrinsics={intrinsic_shape}"
        )
    pose_frame_keys = sorted(pose_map)
    if source_num_frames <= 0 or pose_frame_keys[-1] >= source_num_frames:
        raise ValueError(f"annotated frame {pose_frame_keys[-1]} is outside video length {source_num_frames}")
    validate_pose_sequence(np.stack([pose_map[key] for key in pose_frame_keys]))
    return {
        "annotation_count": expected,
        "pose_shape": [expected, 7],
        "pose_storage": "npy_object_dict",
        "pose_key": "raw_video_frame_index",
        "pose_map_path": path_for_record(poses_path),
        "intrinsics_shape": list(intrinsic_shape),
        "annotation_raw_first": int(pose_frame_keys[0]),
        "annotation_raw_last": int(pose_frame_keys[-1]),
    }
