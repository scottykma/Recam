"""Native model inputs, reference conditioning, and loss helpers."""

from dataclasses import dataclass

import torch

from example.recam.config.config import GeometryConfig
from example.recam.datasets.strategy import Task


def identity_camera_like(camera: torch.Tensor) -> torch.Tensor:
    """The all-identity `[F, 12]` row, used to drop camera control per step."""
    identity = torch.zeros_like(camera)
    identity[..., (0, 5, 10)] = 1.0
    return identity


@dataclass
class ReCamInputs:
    """Native model inputs and target for one ReCam training step."""

    x: torch.Tensor
    timestep: torch.Tensor
    temporal_positions: torch.Tensor
    main_frame_start: int
    target: torch.Tensor
    y: torch.Tensor
    history: torch.Tensor | None = None


def loss_metrics(
    diff_sq: torch.Tensor,
    task: Task,
    overlap: int = 2,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Return equal-weight MSE and reference/generation diagnostics."""
    ref_frames = (
        0 if task is Task.T2V
        else 1 if task is Task.I2V
        else overlap
    )
    if diff_sq.ndim != 5:
        raise ValueError(f"diff_sq must be rank 5; got {tuple(diff_sq.shape)}")
    if diff_sq.shape[2] <= ref_frames:
        raise ValueError(
            f"task {task.name} requires more than {ref_frames} main frames; "
            f"got {diff_sq.shape[2]}"
        )
    loss = diff_sq.mean()
    loss_ref = diff_sq[:, :, :ref_frames].mean().item() if ref_frames else 0.0
    loss_gen = diff_sq[:, :, ref_frames:].mean().item()
    return loss, {"loss_ref": loss_ref, "loss_gen": loss_gen}


def _build_ref_y(
    x: torch.Tensor,
    first_frame: torch.Tensor | None,
    overlap: torch.Tensor | None,
    overlap_noise: torch.Tensor | None,
    t_context: torch.Tensor,
    scheduler,
) -> torch.Tensor:
    """y = [indicator(2) | sigma(2) | ref_lat(16)] over the same frames as x:
    frame 0 carries the clean first frame, frames 1..overlap the noised overlap."""
    b, c, t, h, w = x.shape
    y = x.new_zeros(b, 2 + 2 + c, t, h, w)
    if first_frame is not None:
        y[:, :2, 0] = 1.0
        y[:, 2:4, 0] = 1.0
        y[:, 4:, 0] = first_frame.squeeze(2)
    if overlap is not None:
        overlap_frames = overlap.shape[2]
        if 1 + overlap_frames > t:
            raise ValueError(
                f"overlap has {overlap_frames} frames; x holds {t} frames"
            )
        y[:, :2, 1:1 + overlap_frames] = 1.0
        y[:, 2:4, 1:1 + overlap_frames] = (
            (1.0 - t_context / 1000.0).view(b, 1, 1, 1, 1)
        )
        y[:, 4:, 1:1 + overlap_frames] = scheduler.add_noise(
            overlap, overlap_noise, t_context,
        )
    return y


def build_training_inputs(
    latents: torch.Tensor,
    noise: torch.Tensor,
    task: Task,
    n_history_chunks: int,
    t_context: torch.Tensor,
    t_main: torch.Tensor,
    scheduler,
    *,
    geometry: GeometryConfig,
) -> ReCamInputs:
    """Build model inputs and target from a validated dataset sample."""
    b = latents.shape[0]
    overlap = geometry.overlap
    h, w = latents.shape[3:]

    if task is Task.T2V:
        frames = 1 + geometry.new_frames
        noisy_main = scheduler.add_noise(latents, noise, t_main)
        return ReCamInputs(
            x=noisy_main,
            timestep=t_main.view(b, 1).expand(b, frames),
            temporal_positions=torch.arange(
                frames, device=latents.device, dtype=torch.long,
            ),
            main_frame_start=0,
            target=scheduler.training_target(latents, noise),
            y=noisy_main.new_zeros(b, 20, frames, h, w),
        )

    first_frame = latents[:, :, :1]
    if task is Task.I2V:
        history = None
        main_latent = latents
        main_noise = noise
        first_position = 0
        overlap_latent = None
        overlap_noise = None
    else:
        history_end = 1 + geometry.history_frames(n_history_chunks)
        history = latents[:, :, 1:history_end]
        main_latent = latents[:, :, history_end:]
        main_noise = noise[:, :, history_end:]
        first_position = geometry.first_frame_position(n_history_chunks)
        overlap_latent = main_latent[:, :, :overlap]
        overlap_noise = main_noise[:, :, :overlap]

    main_frames = main_latent.shape[2]
    noisy_main = scheduler.add_noise(main_latent, main_noise, t_main)
    x = torch.cat([first_frame, noisy_main], dim=2)
    return ReCamInputs(
        x=x,
        timestep=torch.cat([
            torch.zeros_like(t_context).view(b, 1),
            t_main.view(b, 1).expand(b, main_frames),
        ], dim=1),
        temporal_positions=torch.cat([
            torch.tensor(
                [first_position], device=latents.device, dtype=torch.long,
            ),
            torch.arange(
                main_frames, device=latents.device, dtype=torch.long,
            ),
        ]),
        main_frame_start=geometry.main_frame_start,
        history=history,
        target=scheduler.training_target(main_latent, main_noise),
        y=_build_ref_y(
            x, first_frame, overlap_latent, overlap_noise, t_context, scheduler,
        ),
    )


