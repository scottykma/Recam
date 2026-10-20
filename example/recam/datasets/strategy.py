"""What a ReCam sample trains on: task, context length, and window."""

import math
from enum import IntEnum
from typing import Sequence

import numpy as np

from example.recam.config.config import GeometryConfig, VAE_TEMPORAL_FACTOR


class Task(IntEnum):
    """What one sample trains: a short T2V/I2V window or an AR continuation."""

    T2V = 0
    I2V = 1
    CONTINUE = 2


def get_context_length_dist(
    task_probs: Sequence[float], max_history_chunks: int,
) -> tuple[float, ...]:
    """Spread `[t2v, i2v, ar]` weights over the context lengths they train.

    T2V/I2V context is fixed, so their mass stays as given; the AR mass is split
    evenly over history depths `1..max_history_chunks`.
    """
    if len(task_probs) != 3:
        raise ValueError(
            "task_probs must contain [t2v_prob, i2v_prob, ar_prob]"
        )
    probs = tuple(float(prob) for prob in task_probs)
    if any(not math.isfinite(prob) or prob < 0 for prob in probs):
        raise ValueError("task_probs must be finite and non-negative")
    total = sum(probs)
    if total <= 0:
        raise ValueError("task_probs must have positive mass")
    if max_history_chunks < 1:
        raise ValueError(
            f"max_history_chunks must be positive; got {max_history_chunks}"
        )
    t2v_prob, i2v_prob, ar_prob = tuple(prob / total for prob in probs)
    return (t2v_prob, i2v_prob) + (
        ar_prob / max_history_chunks,
    ) * max_history_chunks


def row_context_capacity(
    src_fps: np.ndarray,
    caption_start: np.ndarray,
    caption_end: np.ndarray,
    max_history_chunks: int,
    geometry: GeometryConfig,
) -> np.ndarray:
    """How many chunks of context every row can supply, 0 when it supplies none.

    The frames a task has to generate must lie inside the caption, and only
    those: a T2V/I2V window generates all of itself, so it needs one chunk of
    caption; a continuation window generates its overlap + new tail and reads
    its history for free, so it needs `1 + n_history_chunks` chunks of video
    head and caption tail together. A row that cannot even host one chunk --
    unknown frame rate, or a caption shorter than the window -- gets 0.
    """
    fps_known = np.isfinite(src_fps) & (src_fps > 0)
    ratio = np.divide(
        src_fps, geometry.target_fps,
        out=np.zeros_like(src_fps), where=fps_known,
    )
    frames_per_chunk = geometry.num_frames - 1
    overlap_frames = geometry.overlap * VAE_TEMPORAL_FACTOR
    short_latest = np.floor(caption_end - 1 - frames_per_chunk * ratio)
    short_window_legal = fps_known & (caption_start <= short_latest)
    tail_latest = np.floor(
        caption_end - 1 - (frames_per_chunk + overlap_frames) * ratio
    )
    continuation_legal = fps_known & (caption_start <= tail_latest)
    context_capacity = np.zeros(len(src_fps), dtype=np.int64)
    context_capacity[short_window_legal] = 1
    context_capacity[continuation_legal] = 1 + np.clip(
        np.floor(
            (tail_latest[continuation_legal] / ratio[continuation_legal]
             + overlap_frames) / frames_per_chunk
        ),
        0,
        max_history_chunks,
    )
    return context_capacity


def sample_frame_indices(
    src_fps: float,
    caption_start: int,
    caption_end: int,
    task: Task,
    n_history_chunks: int,
    geometry: GeometryConfig,
    rng: np.random.Generator,
) -> np.ndarray:
    """Source-frame indices of one window, drawn from the caption range.

    A window is an anchor plus `num_frames` evenly spaced source frames, spaced
    by the source-to-target fps ratio. The anchor is the window's first frame for
    T2V/I2V; for CONTINUE it sits `frames_per_chunk * n_history_chunks -
    overlap_frames` frames into the window, so the anchor opens the frames the
    task has to generate and the history before it may precede the caption.
    """
    ratio = src_fps / geometry.target_fps
    frames_per_chunk = geometry.num_frames - 1
    overlap_frames = geometry.overlap * VAE_TEMPORAL_FACTOR
    if task in (Task.T2V, Task.I2V):
        num_frames = geometry.num_frames
        anchor_offset = 0
        anchor_min = caption_start
        anchor_max = math.floor(caption_end - 1 - frames_per_chunk * ratio)
    else:
        num_frames = frames_per_chunk * n_history_chunks + geometry.num_frames
        anchor_offset = frames_per_chunk * n_history_chunks - overlap_frames
        anchor_min = max(caption_start, math.ceil(anchor_offset * ratio))
        anchor_max = math.floor(
            caption_end - 1 - (frames_per_chunk + overlap_frames) * ratio
        )

    anchor = int(rng.integers(anchor_min, anchor_max + 1))
    positions = (
        anchor - anchor_offset * ratio
        + np.arange(num_frames, dtype=np.float64) * ratio
    )
    indices = np.rint(positions).astype(np.int64)
    if indices[0] < 0:
        raise RuntimeError("sampled history starts before the video")
    if task is Task.CONTINUE:
        if indices[anchor_offset] < caption_start or indices[-1] >= caption_end:
            raise RuntimeError("sampled AR overlap/future escaped caption bounds")
    elif indices[0] < caption_start or indices[-1] >= caption_end:
        raise RuntimeError("sampled short window escaped caption bounds")
    return indices
