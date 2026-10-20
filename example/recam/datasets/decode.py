"""Video decode helpers for the ReCam datasets."""

import os

import numpy as np
from decord import VideoReader, cpu
from torch.utils.data import get_worker_info

VIDEO_DECODE_TIMEOUT_SECONDS = 15.0


class OnlineFrameRejected(RuntimeError):
    """Raised when a decoded sample fails the first-frame colorfulness gate."""


def _worker_tag() -> str:
    worker = get_worker_info()
    worker_id = worker.id if worker is not None else "main"
    return f"worker={worker_id} pid={os.getpid()}"


def _cover_crop_geometry(
    source_w: int,
    source_h: int,
    width: int,
    height: int,
) -> tuple[int, int, int, int]:
    """Decode size plus where the canvas sits inside it."""
    scale = max(width / source_w, height / source_h)
    decode_w = max(width, round(source_w * scale))
    decode_h = max(height, round(source_h * scale))
    return decode_w, decode_h, (decode_w - width) // 2, (decode_h - height) // 2


def _read_resized_center_crop(
    video_path: str,
    indices: np.ndarray,
    *,
    width: int,
    height: int,
    num_threads: int = 6,
) -> np.ndarray:
    """Decode at the aspect-preserving scale, then center-crop to the target."""
    indices = np.asarray(indices, dtype=np.int64)
    if indices.ndim != 1 or indices.size == 0:
        raise ValueError("video indices must be a non-empty one-dimensional array")
    probe = VideoReader(
        video_path, width=-1, height=-1,
        num_threads=num_threads, ctx=cpu(0),
    )
    sample = probe.get_batch(indices[:1]).asnumpy()
    if sample.ndim != 4 or sample.shape[-1] != 3:
        raise ValueError(
            f"decoded video must be RGB [T,H,W,3]; got {sample.shape}"
        )
    source_h, source_w = sample.shape[1:3]
    del sample
    decode_w, decode_h, x1, y1 = _cover_crop_geometry(
        source_w, source_h, width, height,
    )
    reader = probe
    del probe
    if (decode_w, decode_h) != (source_w, source_h):
        reader = VideoReader(
            video_path, width=decode_w, height=decode_h,
            num_threads=num_threads, ctx=cpu(0),
        )
    raw = reader.get_batch(indices).asnumpy()
    del reader
    cropped = np.ascontiguousarray(raw[:, y1:y1 + height, x1:x1 + width])
    if cropped.shape[1:3] != (height, width):
        raise RuntimeError(
            f"center crop produced {cropped.shape[1:3]}, expected {(height, width)}"
        )
    return cropped
