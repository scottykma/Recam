"""ReCam inference entry."""

from __future__ import annotations

import importlib
import math
import os
import re
import sys
from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Optional

import imageio.v2 as imageio
import numpy as np
import torch
import torch.distributed as dist
import tyro
import yaml
from PIL import Image

from core.config.train_schema import _config_from_dict
from core.infra.video_manifest import write_video_manifest
from example.recam.config.camera_config import CameraConfig
from example.recam.config.config import MAX_CHUNKS, GeometryConfig
from example.recam.infer.pipeline import ReCamPipeline, resize_and_crop
from example.recam.infer.trajectory import load_trajectory_file
from example.recam.infer.utils import DEFAULT_NEGATIVE_PROMPT, setup_dist

DEFAULT_INFER_CONFIG = "configs/recam/recam.yaml"
REPO_ROOT = Path(__file__).resolve().parents[3]


@dataclass
class ReCamArgs:
    ckpt_path: str = ""
    model_type: str = "T2V_14B"
    dtype: str = "bfloat16"
    device: str = "cuda"
    vae_path: Optional[str] = None
    text_encoder_path: Optional[str] = None
    tokenizer_path: Optional[str] = None

    config_file: str = DEFAULT_INFER_CONFIG
    geometry: GeometryConfig = field(default_factory=GeometryConfig)
    camera: CameraConfig = field(default_factory=CameraConfig)

    prompt_source: str = ""
    prompt_key: str = "dense_prompt"
    ref_key: str = "ref_images"
    cases: str = ""
    cases_exact: bool = False

    prompt: str = ""
    image: Optional[str] = None
    context_video: Optional[str] = None
    save_file: str = "out.mp4"

    n_chunks: int = 5

    camera_trajectory_file: str = ""

    cfg_scale: float = 3.5
    num_inference_steps: int = 40
    sigma_shift: float = 5.0
    context_sigma_shift: float = 2.0
    negative_prompt: str = DEFAULT_NEGATIVE_PROMPT
    seed: int = 42
    ref_overwrite: bool = True

    output_dir: Optional[Path] = None
    skip_existing: bool = True
    plain_video_filenames: bool = False
    checkpoint_step: Optional[int] = None


def parse_args() -> ReCamArgs:
    """tyro CLI whose defaults come from the inference config YAML."""

    argv = sys.argv[1:]
    config_file = Path(DEFAULT_INFER_CONFIG)
    if "--config-file" in argv:
        index = argv.index("--config-file")
        if index + 1 >= len(argv):
            raise SystemExit("--config-file needs a path")
        config_file = Path(argv[index + 1])
        del sys.argv[index + 1:index + 3]
    path = config_file if config_file.is_absolute() else REPO_ROOT / config_file
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    default = _config_from_dict(ReCamArgs, data)
    default.config_file = str(config_file)
    return tyro.cli(ReCamArgs, default=default)


def _load_prompt_configs(prompt_source: str):
    module_path, separator, attr = prompt_source.partition(":")
    module = importlib.import_module(module_path)
    attr = attr if separator else "PROMPT_CONFIGS"
    if not hasattr(module, attr):
        raise AttributeError(f"prompt module {module_path} has no {attr}")
    return getattr(module, attr)


def _selected(names: list, cases: str, exact: bool) -> list:
    prefixes = tuple(
        value.strip() for value in cases.split(",") if value.strip()
    )
    if not prefixes:
        return names
    if not exact:
        return [name for name in names if str(name).startswith(prefixes)]
    wanted = set(prefixes)
    missing = sorted(wanted - {str(name) for name in names})
    if missing:
        raise ValueError(f"cases not found in the prompt source: {missing}")
    return [name for name in names if str(name) in wanted]


def collect_samples(
    args: ReCamArgs,
) -> list[tuple[str, str, dict[int, str]]]:
    configs = _load_prompt_configs(args.prompt_source)
    names = _selected(list(configs), args.cases, args.cases_exact)
    samples = []
    for name in names:
        config = configs[name]
        raw_images = config.get(args.ref_key, {})
        ref_images = (
            {
                int(frame_id): os.fspath(image_path)
                for frame_id, image_path in raw_images.items()
            }
            if isinstance(raw_images, Mapping)
            else {
                index: os.fspath(image_path)
                for index, image_path in enumerate(raw_images)
            }
        )
        samples.append((str(name), str(config[args.prompt_key]), ref_images))
    return samples


def make_writer(save_path: Path, fps: int):
    save_path.parent.mkdir(parents=True, exist_ok=True)
    return imageio.get_writer(
        str(save_path), fps=fps, codec="libx264", quality=8,
    )


def resolve_output_dir(
    ckpt_path: str, output_dir: Optional[Path], exp_id: str = "recam",
) -> Path:
    if output_dir is not None:
        return output_dir
    path = Path(ckpt_path)
    resolved = path.resolve()
    has_variant = path.name in {"model", "ema"}
    variant = path.name if has_variant else "model"
    checkpoint_dir = resolved.parent if has_variant else resolved
    step_match = re.search(r"step_(\d+)", checkpoint_dir.as_posix())
    step = int(step_match.group(1)) if step_match else None
    state_path = checkpoint_dir / "training_state.pt"
    if step is None and state_path.exists():
        state = torch.load(state_path, map_location="cpu", weights_only=False)
        step = int(state["global_step"])
    if step is None:
        raise ValueError(
            "cannot resolve an immutable checkpoint step; pass --output-dir "
            "or use a step/latest training checkpoint path"
        )
    return Path("outputs") / exp_id / f"step_{step:07d}" / variant


def read_context_video(
    path: str | Path, *, width: int, height: int, fps: int,
) -> list[Image.Image]:
    """Decode and center-crop every context frame to the inference geometry."""
    video_path = Path(path)
    if not video_path.is_file():
        raise FileNotFoundError(f"context video not found: {video_path}")
    reader = imageio.get_reader(os.fspath(video_path))
    try:
        actual_fps = reader.get_meta_data().get("fps")
        if actual_fps is None or not math.isclose(
            float(actual_fps), float(fps), rel_tol=0.0, abs_tol=1e-6,
        ):
            raise ValueError(
                f"context video must be {fps} fps; got {actual_fps!r}"
            )
        frames = []
        for index, frame in enumerate(reader):
            array = np.asarray(frame)
            if array.ndim != 3 or array.shape[2] != 3:
                raise ValueError(
                    f"context frame {index} has shape {array.shape}; "
                    "expected an RGB frame"
                )
            frames.append(
                resize_and_crop(
                    Image.fromarray(array, mode="RGB"), (width, height),
                )
            )
    finally:
        reader.close()
    return frames


def _filename_value(value: object) -> str:
    if isinstance(value, float):
        value = format(value, "g")
    return re.sub(r"[^A-Za-z0-9._-]+", "-", str(value))


def build_output_filename(base_name: str, args: ReCamArgs) -> str:
    """Build a compact filename for the primary sampling controls."""
    if args.plain_video_filenames:
        return _filename_value(base_name) + ".mp4"
    geometry = args.geometry
    parts = [
        _filename_value(base_name),
        f"h{geometry.height}w{geometry.width}",
        "pack" + "x".join(str(value) for value in geometry.history_pack_shape),
    ]
    if args.camera_trajectory_file:
        parts.append(
            "cam" + _filename_value(Path(args.camera_trajectory_file).stem)
        )
    parts += [
        f"cfg{_filename_value(args.cfg_scale)}",
        f"shift{_filename_value(args.sigma_shift)}",
        f"ctxshift{_filename_value(args.context_sigma_shift)}",
    ]
    return "_".join(parts) + ".mp4"


def _run_case(
    pipe: ReCamPipeline,
    args: ReCamArgs,
    prompt: str,
    ref_image: Optional[Image.Image],
    save_path: Path,
    n_chunks: int,
    context_frames: Optional[list[Image.Image]] = None,
    camera_trajectory: Optional[np.ndarray] = None,
) -> None:
    frames = list(context_frames or [])
    temp_path = save_path.with_name(
        f".{save_path.stem}.writing{save_path.suffix}"
    )

    def on_chunk(_index: int, chunk_frames: list[Image.Image]) -> None:
        frames.extend(chunk_frames)
        writer = make_writer(temp_path, int(pipe.geometry.target_fps))
        try:
            for frame in frames:
                writer.append_data(np.asarray(frame))
        finally:
            writer.close()
        os.replace(temp_path, save_path)

    try:
        pipe(
            prompt=prompt,
            negative_prompt=args.negative_prompt,
            ref_image=ref_image,
            context_frames=context_frames,
            camera_trajectory=camera_trajectory,
            seed=args.seed,
            cfg_scale=args.cfg_scale,
            num_inference_steps=args.num_inference_steps,
            sigma_shift=args.sigma_shift,
            context_sigma_shift=args.context_sigma_shift,
            n_chunks=n_chunks,
            ref_overwrite=args.ref_overwrite,
            chunk_callback=on_chunk,
        )
    finally:
        temp_path.unlink(missing_ok=True)


def main(args: ReCamArgs) -> None:
    if args.image and args.context_video:
        raise ValueError("--context-video and --image are mutually exclusive")
    if args.prompt_source and args.context_video:
        raise ValueError(
            "--context-video is a single-case option; a prompt source cannot "
            "carry one"
        )
    geometry = args.geometry
    assert 1 <= args.n_chunks <= MAX_CHUNKS
    rank, world_size, local_rank = setup_dist()
    runtime = replace(
        args,
        device=f"cuda:{local_rank}" if args.device.startswith("cuda") else args.device,
        output_dir=resolve_output_dir(args.ckpt_path, args.output_dir),
    )
    if rank == 0:
        runtime.output_dir.mkdir(parents=True, exist_ok=True)
    dist.barrier()

    if runtime.prompt_source:
        samples = collect_samples(runtime)
    else:
        if not runtime.prompt:
            raise ValueError("either --prompt-source or --prompt must be set")
        ref_images = {0: runtime.image} if runtime.image else {}
        samples = [(None, runtime.prompt, ref_images)]
    camera_trajectory = (
        load_trajectory_file(runtime.camera_trajectory_file)
        if runtime.camera_trajectory_file else None
    )
    pipe = ReCamPipeline(runtime)
    manifest_records = []
    for name, prompt, ref_images in samples[rank::world_size]:
        if name is None:
            requested = Path(runtime.save_file)
            output_parent = runtime.output_dir / requested.parent
            base_name = requested.stem
        else:
            output_parent = runtime.output_dir
            base_name = name
        save_path = output_parent / build_output_filename(base_name, runtime)
        record = {
            "video": save_path,
            "prompt": prompt,
            "reference_image": (
                ref_images[sorted(ref_images)[0]] if ref_images else None
            ),
            "fps": int(geometry.target_fps),
        }
        if runtime.skip_existing and save_path.exists():
            print(f"[Rank {rank}] Skip existing: {save_path}")
            manifest_records.append(record)
            continue
        ref_image: Optional[Image.Image] = None
        if ref_images:
            with Image.open(ref_images[sorted(ref_images)[0]]) as image:
                ref_image = image.convert("RGB")
        context_frames = (
            read_context_video(
                runtime.context_video,
                width=geometry.width,
                height=geometry.height,
                fps=int(geometry.target_fps),
            )
            if runtime.context_video else None
        )
        _run_case(
            pipe, runtime, prompt, ref_image, save_path, runtime.n_chunks,
            context_frames=context_frames, camera_trajectory=camera_trajectory,
        )
        manifest_records.append(record)
        print(f"[Rank {rank}] Saved: {save_path}")

    gathered = [None] * world_size
    dist.all_gather_object(gathered, manifest_records)
    if rank == 0:
        records = [record for batch in gathered for record in (batch or [])]
        write_video_manifest(
            runtime.output_dir,
            records,
            extra={
                "suite": (
                    runtime.prompt_source.rsplit(".", 1)[-1]
                    if runtime.prompt_source else None
                ),
                "prompt_source": runtime.prompt_source or None,
                "checkpoint": runtime.ckpt_path,
                "checkpoint_step": runtime.checkpoint_step,
                "seed": runtime.seed,
                "config_file": runtime.config_file,
                "camera_trajectory_file": runtime.camera_trajectory_file or None,
            },
        )

    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main(parse_args())
