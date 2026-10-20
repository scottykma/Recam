"""ReCam windows read directly from parquet rows."""

from __future__ import annotations

import math
import os
from pathlib import Path
from typing import Any, NoReturn, Sequence

import numpy as np
import pyarrow as pa
import pyarrow.dataset as pa_dataset
import pyarrow.parquet as pa_parquet
import torch
from func_timeout import func_set_timeout
from func_timeout.exceptions import FunctionTimedOut
from torch.utils.data import Dataset

from core.infra.logger import get_logger
from example.recam.config.config import MAX_CHUNKS, GeometryConfig
from example.recam.datasets.decode import (
    VIDEO_DECODE_TIMEOUT_SECONDS,
    OnlineFrameRejected,
    _read_resized_center_crop,
    _worker_tag,
)
from example.recam.datasets.strategy import (
    Task,
    get_context_length_dist,
    row_context_capacity,
    sample_frame_indices,
)
from core.data.frame_quality import colorfulness as frame_colorfulness


def metadata_paths(
    metadata_path: Sequence[str] | str | os.PathLike[str],
) -> list[str]:
    """Normalize a parquet file/directory/sequence to Arrow input paths."""

    if isinstance(metadata_path, (str, os.PathLike)):
        values = [metadata_path]
    else:
        values = list(metadata_path)
    paths: list[str] = []
    for value in values:
        path = Path(value)
        if path.is_dir():
            paths.extend(str(item) for item in sorted(path.glob("*.parquet")))
        else:
            paths.append(str(path))
    if not paths:
        raise FileNotFoundError(
            f"no parquet shards found for metadata path: {metadata_path}"
        )
    return paths


def _other_row(
    pool: np.ndarray, row_index: int, rng: np.random.Generator,
) -> int:
    """Draw a row from the pool that is not the one that just failed."""
    replacement = row_index
    while replacement == row_index:
        replacement = int(pool[int(rng.integers(0, len(pool)))])
    return replacement


class _LoadFailures:
    """Books one sample's load failures: logs them, tracks the retry budgets.

    Holds the row table only to name the video in each log line; which row is
    being attempted stays with the caller.
    """

    def __init__(
        self,
        logger,
        video_paths: Sequence[str],
        max_decode_attempts: int,
        max_frame_rejections: int,
    ) -> None:
        self.logger = logger
        self.video_paths = video_paths
        self.max_decode_attempts = max_decode_attempts
        self.max_frame_rejections = max_frame_rejections
        self.decode_errors = 0
        self.frame_rejections = 0
        self.last_error: BaseException | None = None

    def record(
        self,
        error: BaseException,
        row_index: int,
        task: Task,
        n_history_chunks: int,
    ) -> bool:
        """Log one failed load; return whether the sample still has retries."""
        self.last_error = error
        context = (
            f"row={row_index} path={self.video_paths[row_index]} "
            f"task={task.name} history={n_history_chunks}"
        )
        if isinstance(error, OnlineFrameRejected):
            self.frame_rejections += 1
            self.logger.info(
                "ReCam online frame rejection: "
                f"{context} rejection={self.frame_rejections}/"
                f"{self.max_frame_rejections} {_worker_tag()} error={error}",
                rank_0_only=False,
            )
            return self.frame_rejections < self.max_frame_rejections
        self.decode_errors += 1
        timed_out = isinstance(error, FunctionTimedOut)
        detail = (
            f"timeout={VIDEO_DECODE_TIMEOUT_SECONDS:.1f}s "
            f"error_type={type(error).__name__}"
            if timed_out
            else f"error={error!r}"
        )
        self.logger.warning(
            "ReCam video decode "
            f"{'timeout' if timed_out else 'error'}: {context} "
            f"attempt={self.decode_errors}/{self.max_decode_attempts} "
            f"{_worker_tag()} {detail}",
            rank_0_only=False,
        )
        return self.decode_errors < self.max_decode_attempts

    def note_resample(self, from_row: int, to_row: int) -> None:
        self.logger.info(
            "ReCam resampling after load failure: "
            f"from_row={from_row} to_row={to_row} "
            f"decode_errors={self.decode_errors} "
            f"frame_rejections={self.frame_rejections} {_worker_tag()}",
            rank_0_only=False,
        )

    def give_up(
        self, row_index: int, task: Task, n_history_chunks: int,
    ) -> NoReturn:
        """Report the sample that ran out of retries and raise."""
        error_text = str(self.last_error).replace("\n", " ")[:240]
        self.logger.warning(
            "ReCam sample exhausted retries: "
            f"row={row_index} task={task.name} history={n_history_chunks} "
            f"decode_errors={self.decode_errors} "
            f"frame_rejections={self.frame_rejections} {_worker_tag()} "
            f"last_error_type={type(self.last_error).__name__} "
            f"last_error={error_text}",
            rank_0_only=False,
        )
        raise RuntimeError(
            "failed to decode a deterministic sample after "
            f"{self.decode_errors} decode errors and "
            f"{self.frame_rejections} online frame rejections: {error_text}"
        )


def _online_check_budget(
    online_check: dict | None,
) -> tuple[dict | None, int]:
    """Validate the online frame gate; return its threshold and retry budget."""
    if online_check is None:
        return None, 0
    colorfulness_th = float(online_check["colorfulness_th"])
    max_resample = int(online_check["max_resample"])
    if not math.isfinite(colorfulness_th) or colorfulness_th < 0:
        raise ValueError(
            "online colorfulness_th must be finite and non-negative"
        )
    if max_resample < 1:
        raise ValueError("online max_resample must be positive")
    return {"colorfulness_th": colorfulness_th}, max_resample


class ReCamDataset(Dataset):
    """Read geometry-sized windows selected by deterministic sampler buckets."""

    def __init__(
        self,
        metadata_path: Sequence[str] | str | os.PathLike[str],
        *,
        geometry: GeometryConfig,
        n_chunks: int = 5,
        max_decode_attempts: int = 8,
        online_check: dict | None = None,
        task_probs: Sequence[float] = (0.05, 0.35, 0.60),
    ) -> None:
        if not 2 <= n_chunks <= MAX_CHUNKS:
            raise ValueError(
                f"ReCam requires n_chunks in [2, {MAX_CHUNKS}]"
            )
        self.geometry = geometry
        self.logger = get_logger()
        self.max_history_chunks = n_chunks - 1
        self.task_names = ("T2V", "I2V") + tuple(
            f"AR_history_{n_history_chunks}"
            for n_history_chunks in range(1, self.max_history_chunks + 1)
        )
        self.bucket_probs = get_context_length_dist(
            task_probs, self.max_history_chunks,
        )
        self.max_decode_attempts = max(1, int(max_decode_attempts))
        self.online_check, self.max_frame_rejections = _online_check_budget(
            online_check,
        )
        self._read_rows(metadata_path)
        self._index_rows()

    def _read_rows(self, metadata_path) -> None:
        """Load the five columns every sample needs, one entry per row."""
        shard_paths = metadata_paths(metadata_path)
        source_schema = pa.unify_schemas(
            [pa_parquet.read_schema(path) for path in shard_paths],
            promote_options="permissive",
        )
        source = pa_dataset.dataset(
            shard_paths, format="parquet", schema=source_schema,
        )
        required = ["video_path", "src_fps", "caption_start", "caption_end", "text"]
        missing = sorted(set(required) - set(source.schema.names))
        if missing:
            raise ValueError(f"Parquet is missing required columns: {missing}")
        table = source.to_table(columns=required).to_pydict()
        self.video_paths = table["video_path"]
        self.texts = table["text"]
        self.src_fps = np.asarray(table["src_fps"], dtype=np.float64)
        self.caption_start = np.asarray(table["caption_start"], dtype=np.int64)
        self.caption_end = np.asarray(table["caption_end"], dtype=np.int64)
        self.total_rows = len(self.video_paths)

    @staticmethod
    def _chunks_needed(task: Task, n_history_chunks: int) -> int:
        """Chunks one window of this task has to be given: 1, or 1 + its history."""
        return 1 if task in (Task.T2V, Task.I2V) else n_history_chunks + 1

    def _index_rows(self) -> None:
        """Sort the rows into buckets by how many chunks of context they supply.

        The rows sorted by capacity make every bucket one suffix of the same
        array: `depth_starts[c]` is where the rows supplying at least `c`
        chunks begin, and a bucket needs `_chunks_needed` of them.
        """
        self.context_capacity = row_context_capacity(
            self.src_fps,
            self.caption_start,
            self.caption_end,
            self.max_history_chunks,
            self.geometry,
        )
        self.depth_sorted_rows = np.argsort(
            self.context_capacity, kind="stable",
        ).astype(np.int32)
        capacities = self.context_capacity[self.depth_sorted_rows]
        self.depth_starts = np.searchsorted(
            capacities,
            np.arange(self.max_history_chunks + 2),
            side="left",
        ).astype(np.int32)
        self.rows_per_bucket = tuple(
            len(self.depth_sorted_rows)
            - int(self.depth_starts[
                self._chunks_needed(*self.bucket_to_task(bucket_id))
            ])
            for bucket_id in range(len(self.task_names))
        )
        empty = [
            name
            for name, count in zip(self.task_names, self.rows_per_bucket)
            if count == 0
        ]
        if empty:
            raise ValueError(f"ReCam sampling buckets have no rows: {empty}")

    def __len__(self) -> int:
        return self.total_rows

    def bucket_to_task(self, bucket_id: int) -> tuple[Task, int]:
        if bucket_id == 0:
            return Task.T2V, 0
        if bucket_id == 1:
            return Task.I2V, 0
        if 2 <= bucket_id < len(self.task_names):
            return Task.CONTINUE, bucket_id - 1
        raise ValueError(
            f"bucket_id must be in [0, {len(self.task_names)}); got {bucket_id}"
        )

    def legal_rows(self, task: Task, n_history_chunks: int) -> np.ndarray:
        """Row indices whose context capacity covers this task and history depth."""
        if task is Task.CONTINUE and not 1 <= n_history_chunks <= self.max_history_chunks:
            raise ValueError(
                f"continuation depth must be in [1, {self.max_history_chunks}]; "
                f"got {n_history_chunks}"
            )
        return self.depth_sorted_rows[
            self.depth_starts[self._chunks_needed(task, n_history_chunks)]:
        ]

    def sample_bucket(
        self, bucket_id: int, count: int, generator: torch.Generator,
    ) -> list[int]:
        task, n_history_chunks = self.bucket_to_task(bucket_id)
        pool = self.legal_rows(task, n_history_chunks)
        if len(pool) < count:
            raise RuntimeError(
                f"bucket {self.task_names[bucket_id]} has {len(pool)} rows, fewer "
                f"than the global micro-batch size {count}"
            )
        positions: list[int] = []
        seen: set[int] = set()
        while len(positions) < count:
            candidates = torch.randint(
                len(pool),
                (2 * (count - len(positions)),),
                generator=generator,
                dtype=torch.int64,
            ).tolist()
            for position in candidates:
                if position not in seen:
                    seen.add(position)
                    positions.append(position)
                    if len(positions) == count:
                        break
        return [int(pool[position]) for position in positions]

    def make_sample_key(
        self, row_index: int, bucket_id: int, sample_seed: int,
    ) -> tuple[int, int, int, int]:
        task, n_history_chunks = self.bucket_to_task(bucket_id)
        return int(row_index), int(task), n_history_chunks, int(sample_seed)

    def _source_indices(
        self,
        row_index: int,
        task: Task,
        n_history_chunks: int,
        rng: np.random.Generator,
    ) -> np.ndarray:
        return sample_frame_indices(
            float(self.src_fps[row_index]),
            int(self.caption_start[row_index]),
            int(self.caption_end[row_index]),
            task,
            n_history_chunks,
            self.geometry,
            rng,
        )

    @func_set_timeout(VIDEO_DECODE_TIMEOUT_SECONDS)
    def _read_video(self, video_path: str, indices: np.ndarray) -> np.ndarray:
        return _read_resized_center_crop(
            video_path, indices,
            width=self.geometry.width, height=self.geometry.height,
        )

    def _check_online_frame(self, raw: np.ndarray) -> None:
        if self.online_check is None:
            return
        score = frame_colorfulness(raw[0])
        threshold = self.online_check["colorfulness_th"]
        if score < threshold:
            raise OnlineFrameRejected(
                f"first-frame colorfulness {score:.3f} < {threshold:.3f}"
            )

    def _extra_sample_fields(
        self,
        row_index: int,
        task: Task,
        n_history_chunks: int,
        indices: np.ndarray,
    ) -> dict[str, Any]:
        """Return fields supplied by dataset variants after a decode succeeds.

        The hook deliberately receives the sampled source indices so variants
        can derive aligned side information without reimplementing sampling,
        retry, or decode behavior.  The base dataset keeps its original
        return contract by returning no additional fields.
        """

        del row_index, task, n_history_chunks, indices
        return {}

    def __getitem__(self, key) -> dict[str, torch.Tensor | str | int | float]:
        row_index, task_id, n_history_chunks, sample_seed = (
            int(value) for value in key
        )
        task = Task(task_id)
        rng = np.random.default_rng(sample_seed)
        pool = self.legal_rows(task, n_history_chunks)
        failures = _LoadFailures(
            self.logger,
            self.video_paths,
            self.max_decode_attempts,
            self.max_frame_rejections,
        )
        while True:
            indices = self._source_indices(
                row_index, task, n_history_chunks, rng,
            )
            try:
                raw = self._read_video(self.video_paths[row_index], indices)
                self._check_online_frame(raw)
                video = torch.from_numpy(raw).permute(3, 0, 1, 2)
                item: dict[str, Any] = {
                    "video": video,
                    "text": self.texts[row_index],
                    "row_index": row_index,
                    "task_id": int(task),
                    "n_history_chunks": n_history_chunks,
                }
                item.update(
                    self._extra_sample_fields(
                        row_index, task, n_history_chunks, indices,
                    )
                )
                return item
            except (Exception, FunctionTimedOut) as error:
                if not failures.record(
                    error, row_index, task, n_history_chunks,
                ):
                    break
            if len(pool) <= 1:
                break
            failed_row = row_index
            row_index = _other_row(pool, row_index, rng)
            failures.note_resample(failed_row, row_index)
        failures.give_up(row_index, task, n_history_chunks)
