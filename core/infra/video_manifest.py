"""Helpers for manifests emitted next to generated videos."""

from __future__ import annotations

import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


MANIFEST_SCHEMA_VERSION = 1


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return os.fspath(value)
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def load_video_manifest(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {"schema_version": MANIFEST_SCHEMA_VERSION, "videos": []}
    with path.open(encoding="utf-8") as file:
        data = json.load(file)
    if isinstance(data, list):
        data = {"schema_version": MANIFEST_SCHEMA_VERSION, "videos": data}
    if not isinstance(data, dict) or not isinstance(data.get("videos", []), list):
        raise ValueError(f"invalid video manifest: {path}")
    return data


def write_video_manifest(
    output_dir: Path,
    records: Iterable[dict[str, Any]],
    *,
    extra: dict[str, Any] | None = None,
) -> Path:
    """Merge records by relative video path and atomically write manifest.json."""
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "manifest.json"
    existing = load_video_manifest(path)
    merged: dict[str, dict[str, Any]] = {}
    for record in [*existing.get("videos", []), *records]:
        if not isinstance(record, dict) or not record.get("video"):
            continue
        video = Path(os.fspath(record["video"]))
        try:
            key = os.fspath(video.relative_to(output_dir))
        except ValueError:
            key = os.fspath(video)
        normalized = {
            key_name: value
            for key_name, value in record.items()
            if value is not None
        }
        normalized["video"] = key
        merged[key] = _jsonable(normalized)
    payload: dict[str, Any] = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "videos": [merged[key] for key in sorted(merged)],
    }
    if extra:
        payload.update({
            key: _jsonable(value)
            for key, value in extra.items()
            if value is not None
        })

    fd, temp_name = tempfile.mkstemp(prefix=".manifest.", suffix=".json", dir=output_dir)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as file:
            json.dump(payload, file, ensure_ascii=False, indent=2)
            file.write("\n")
            file.flush()
            os.fsync(file.fileno())
        os.replace(temp_name, path)
    finally:
        if os.path.exists(temp_name):
            os.unlink(temp_name)
    return path
