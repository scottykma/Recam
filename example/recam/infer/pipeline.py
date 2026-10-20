"""ReCam inference pipeline.

One rollout is a chain of AR windows. Window ``c`` holds

    [first frame | history (memory) | main]

where ``main`` is ``overlap + new_frames`` latents and its head is the overlap.
The overlap appears once in the sequence; its lower-noise ``t_context`` copy
rides in the ``y`` reference channels. The first window of a rollout has no
memory and no overlap (``[first frame | main]``, or ``[main]`` alone for T2V),
so one loop covers T2V, I2V and every continuation depth.

``camera`` rows are the fused-PRoPE conditioning: one ``[F, 12]`` block per
window, sliced out of a single absolute camera path so consecutive windows meet
on the latent they share.
"""

from __future__ import annotations

import math
import random
import sys
from typing import Optional

import numpy as np
import torch
from PIL import Image
from einops import rearrange
from tqdm import tqdm

from example.recam.config.config import (
    MAX_CHUNKS,
    GeometryConfig,
    ModelConfig,
    VAE_TEMPORAL_FACTOR,
)
from example.recam.infer.trajectory import (
    window_latent_indices,
    window_rows,
)
from example.recam.models.wan_dit import ReCamModel
from example.recam.infer.guidance import FlowMatchingGuidance
from example.recam.infer.utils import load_weights_for_model
from example.recam.backbone.wan_prompter import WanPrompter
from example.recam.backbone.wan_video_text_encoder import WanTextEncoder
from example.recam.backbone.wan_video_vae import WanVideoVAE
from example.recam.backbone.wan_video_vae2_2 import WanVideoVAE_2_2, unpatchify
from example.recam.backbone.flow_matching_scheduler import FlowMatchScheduler

PackedHistory = tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]


def resize_and_crop(
    image: Image.Image, target_size: tuple[int, int],
) -> Image.Image:
    target_w, target_h = target_size
    scale = max(target_w / image.width, target_h / image.height)
    new_w, new_h = round(image.width * scale), round(image.height * scale)
    image = image.resize((new_w, new_h), Image.Resampling.LANCZOS)
    x1, y1 = (new_w - target_w) // 2, (new_h - target_h) // 2
    return image.crop((x1, y1, x1 + target_w, y1 + target_h))


def context_sigmas(num_steps: int, shift: float, device, dtype) -> torch.Tensor:
    """Shifted-linear reference-noise schedule from 1 to 0.

    The trailing zero is what makes the last denoising step see a clean overlap
    reference; the Euler loop only ever reads the first ``num_steps`` entries.
    """

    assert num_steps >= 1
    assert math.isfinite(shift) and shift > 0.0
    base = torch.linspace(1.0, 0.0, steps=num_steps, device=device, dtype=dtype)
    base = torch.cat([base, base.new_zeros(1)])
    return shift * base / (1 + (shift - 1) * base)


def build_window_input(
    noisy_main: torch.Tensor,
    main_noise: torch.Tensor,
    sigma_context: torch.Tensor,
    sigma_main: torch.Tensor,
    geometry: GeometryConfig,
    *,
    n_history_chunks: int = 0,
    first_frame: Optional[torch.Tensor] = None,
    overlap: Optional[torch.Tensor] = None,
    history: Optional[torch.Tensor] = None,
    camera: Optional[torch.Tensor] = None,
) -> dict[str, torch.Tensor | int | None]:
    """Assemble one window from its clean context and its noised main block.

    ``overlap`` and ``history`` are both absent on the first window of a
    rollout and both present afterwards, so ``noisy_main`` is ``1 + new_frames``
    latents there and ``main_frames`` at every later depth. ``overlap`` only
    feeds the ``y`` reference channels; the sequence carries it once, as the
    head of ``noisy_main``.
    """

    assert noisy_main.shape == main_noise.shape
    assert (overlap is None) == (history is None)
    if overlap is not None:
        assert first_frame is not None and first_frame.shape[1] == 1
    main_frames = noisy_main.shape[1]
    if history is None:
        assert n_history_chunks == 0
        assert main_frames == 1 + geometry.new_frames
    else:
        assert 1 <= n_history_chunks < MAX_CHUNKS
        assert main_frames == geometry.main_frames
        assert overlap.shape[1] == geometry.overlap
        assert history.shape[1] == geometry.history_frames(n_history_chunks)

    overlap_frames = 0 if overlap is None else overlap.shape[1]
    overlap_ref = (
        None if overlap is None
        else (1.0 - sigma_context) * overlap
        + sigma_context * main_noise[:, :overlap_frames]
    )
    x = noisy_main
    if first_frame is not None:
        x = torch.cat([first_frame, x], dim=1)
    main_frame_start = x.shape[1] - main_frames

    timestep = torch.cat([
        sigma_context.new_zeros(1, 1 if first_frame is not None else 0),
        sigma_main.mul(1000).view(1, 1).expand(1, main_frames),
    ], dim=1)

    first_frame_position = 0 if history is None else (
        geometry.first_frame_position(n_history_chunks)
    )
    positions = [] if first_frame is None else [
        torch.tensor(
            [first_frame_position], device=noisy_main.device, dtype=torch.long,
        ),
    ]
    positions += [
        torch.arange(main_frames, device=noisy_main.device, dtype=torch.long),
    ]
    temporal_positions = torch.cat(positions)

    y = noisy_main.new_zeros(1, 20, x.shape[1], *noisy_main.shape[2:])
    if first_frame is not None:
        y[:, :2, 0] = 1.0
        y[:, 2:4, 0] = 1.0
        y[:, 4:, 0] = first_frame.squeeze(1)
    if overlap_ref is not None:
        y[:, :2, 1:1 + overlap_frames] = 1.0
        y[:, 2:4, 1:1 + overlap_frames] = 1.0 - sigma_context
        y[:, 4:, 1:1 + overlap_frames] = overlap_ref
    return {
        "x": x.unsqueeze(0),
        "timestep": timestep,
        "temporal_positions": temporal_positions,
        "main_frame_start": main_frame_start,
        "history": None if history is None else history.unsqueeze(0),
        "camera": camera,
        "y": y,
    }


def context_chunks(geometry: GeometryConfig, num_frames: int) -> int:
    """How many chunks of context a continuation video of ``num_frames`` holds.

    A chunk covers ``geometry.num_frames`` pixel frames and every chunk after
    the first adds ``num_frames - 1`` more, so a video holds ``n`` chunks when
    its length is ``1 + (num_frames - 1) * n``.
    """

    if num_frames < geometry.num_frames:
        raise ValueError(
            f"a continuation needs at least {geometry.num_frames} context "
            f"frames; got {num_frames}"
        )
    per_chunk = geometry.num_frames - 1
    if (num_frames - 1) % per_chunk:
        raise ValueError(
            f"context frame count must be 1 + {per_chunk}n; got {num_frames}"
        )
    return (num_frames - 1) // per_chunk


def _preprocess_image(image: Image.Image) -> torch.Tensor:
    array = np.array(image, dtype=np.float32) * (2 / 255) - 1
    return torch.Tensor(array).permute(2, 0, 1).unsqueeze(0)


def _cpu_noise(
    shape: tuple[int, ...], generator: torch.Generator,
    device: torch.device | str, dtype: torch.dtype,
) -> torch.Tensor:
    return torch.randn(
        shape, generator=generator, device="cpu", dtype=torch.float32,
    ).to(device=device, dtype=dtype)


class ReCamPipeline:
    """Generate one AR window at a time and stream-decode the frames it adds."""

    def __init__(self, args) -> None:
        self.device = args.device
        self.torch_dtype = (
            getattr(torch, args.dtype)
            if isinstance(args.dtype, str) else args.dtype
        )
        self.geometry: GeometryConfig = args.geometry

        model_cfg = ModelConfig(type=args.model_type)
        transformer = model_cfg.transformer
        encoder_cfg = model_cfg.encoder
        self.out_dim = transformer.out_dim
        print(f"Loading DiT: {args.ckpt_path}")
        with torch.device("meta"):
            self.dit = ReCamModel(
                dim=transformer.dim, in_dim=transformer.in_dim,
                ffn_dim=transformer.ffn_dim, out_dim=transformer.out_dim,
                text_dim=transformer.text_dim, freq_dim=transformer.freq_dim,
                eps=transformer.eps, patch_size=tuple(transformer.patch_size),
                num_heads=transformer.num_heads, num_layers=transformer.num_layers,
                has_image_input=transformer.has_image_input,
                has_image_pos_emb=transformer.has_image_pos_emb,
                history_pack_shape=self.geometry.history_pack_shape,
                history_pad_shape=self.geometry.history_pad_shape,
                camera_focal_x=args.camera.focal_x,
                camera_focal_y=args.camera.resolved_focal_y(self.geometry),
                camera_translation_transform=args.camera.translation_transform,
            )
        self.dit.to_empty(device="cpu")
        self.dit.reset_parameters()
        load_weights_for_model(self.dit, args.ckpt_path, self.device, strict=False)
        self.dit.to(
            device=self.device, dtype=self.torch_dtype,
        ).eval().requires_grad_(False)

        vae_path = args.vae_path or encoder_cfg.vae.path
        self.tiler_kwargs = dict(
            tiled=encoder_cfg.vae.use_tile,
            tile_size=(34, 34),
            tile_stride=(18, 16),
        )
        print(f"Loading VAE: {vae_path}")
        self.vae = (
            WanVideoVAE() if encoder_cfg.vae.type == "WanVideoVAE_2_1"
            else WanVideoVAE_2_2()
        )
        self.vae.to(
            device=self.device, dtype=self.torch_dtype,
        ).eval().requires_grad_(False)
        self.vae.model.load_state_dict(
            torch.load(vae_path, map_location="cpu", weights_only=False),
            strict=True,
        )

        text_encoder_path = args.text_encoder_path or encoder_cfg.text_encoder.path
        tokenizer_path = args.tokenizer_path or encoder_cfg.text_encoder.tokenizer_path
        print(f"Loading text encoder: {text_encoder_path}")
        with torch.device("meta"):
            self.text_encoder = WanTextEncoder()
        self.text_encoder.load_state_dict(
            torch.load(
                text_encoder_path, map_location="cpu", weights_only=False,
            ),
            strict=True,
            assign=True,
        )
        self.text_encoder.to(
            device=self.device, dtype=self.torch_dtype,
        ).eval().requires_grad_(False)
        self.prompter = WanPrompter()
        self.prompter.fetch_models(self.text_encoder)
        self.prompter.fetch_tokenizer(tokenizer_path)

        self.scheduler = FlowMatchScheduler(num_train_timesteps=1000, shift=5)

    def encode_prompt(self, prompt: str) -> torch.Tensor:
        return self.prompter.encode_prompt(prompt, device=self.device)

    def _encode_pixels(
        self, frames: list[Image.Image], height: int, width: int,
    ) -> torch.Tensor:
        tensors = [
            _preprocess_image(
                resize_and_crop(frame.convert("RGB"), (width, height))
            ).squeeze(0)
            for frame in frames
        ]
        video = torch.stack(tensors, dim=1).to(
            device=self.device, dtype=self.torch_dtype,
        )
        latent = self.vae.encode(
            [video], device=self.device, **self.tiler_kwargs,
        )[0].to(dtype=self.torch_dtype)
        return latent

    def encode_image_to_latent(
        self, image: Image.Image, height: int, width: int,
    ) -> torch.Tensor:
        return self._encode_pixels([image], height, width)

    def encode_video_to_latent(
        self, frames: list[Image.Image], height: int, width: int,
    ) -> torch.Tensor:
        return self._encode_pixels(frames, height, width)

    def _pack_history(
        self, history: Optional[torch.Tensor],
    ) -> Optional[PackedHistory]:
        if history is None:
            return None
        return self.dit.history_mlp(
            history.unsqueeze(0),
            torch.arange(
                -history.shape[1], 0, device=history.device, dtype=torch.long,
            ),
        )

    def _camera_rows(
        self, trajectory: Optional[np.ndarray], chunk_index: int, *,
        is_initial: bool,
    ) -> Optional[torch.Tensor]:
        if trajectory is None:
            return None
        rows = window_rows(
            trajectory, chunk_index=chunk_index, geometry=self.geometry,
            is_initial=is_initial,
        )
        return torch.from_numpy(rows).unsqueeze(0).to(device=self.device)

    def _require_camera_coverage(
        self, trajectory: np.ndarray, total_chunks: int,
    ) -> None:
        last = window_latent_indices(
            self.geometry, chunk_index=total_chunks - 1,
            is_initial=total_chunks == 1,
        )
        required = int(last[-1]) * VAE_TEMPORAL_FACTOR + 1
        if len(trajectory) < required:
            raise ValueError(
                f"{total_chunks} chunks need {required} trajectory frames, but "
                f"the path holds {len(trajectory)}"
            )

    def _predict(
        self,
        built: dict[str, torch.Tensor | int | None],
        context: torch.Tensor,
        packed_history: Optional[PackedHistory] = None,
    ) -> torch.Tensor:
        return self.dit(
            x=built["x"],
            timestep=built["timestep"],
            context=context,
            temporal_positions=built["temporal_positions"],
            main_frame_start=built["main_frame_start"],
            history=built["history"] if packed_history is None else None,
            packed_history=packed_history,
            y=built["y"],
            camera=built["camera"],
        )

    @torch.no_grad()
    def _denoise_window(
        self,
        *,
        main_noise: torch.Tensor,
        first_frame: Optional[torch.Tensor],
        overlap: Optional[torch.Tensor],
        history: Optional[torch.Tensor],
        packed_history: Optional[PackedHistory],
        n_history_chunks: int,
        prompt_emb: torch.Tensor,
        negative_emb: Optional[torch.Tensor],
        guidance: Optional[FlowMatchingGuidance],
        timesteps: torch.Tensor,
        context_sigmas: torch.Tensor,
        ref_overwrite: bool,
        camera: Optional[torch.Tensor],
        progress_desc: str,
    ) -> torch.Tensor:
        noisy_main = main_noise.clone()
        pointer = overlap if overlap is not None else first_frame
        with torch.amp.autocast("cuda", dtype=self.torch_dtype):
            for step, t in enumerate(tqdm(
                timesteps[:-1], desc=progress_desc, leave=False,
            )):
                t_curr = t.to(self.device, self.torch_dtype)
                t_next = timesteps[step + 1].to(self.device, self.torch_dtype)
                sigma_main = t_curr / 1000.0
                sigma_context = context_sigmas[step].clamp(max=sigma_main)
                if ref_overwrite and pointer is not None:
                    noisy_main[:, :pointer.shape[1]] = (
                        (1.0 - sigma_main) * pointer
                        + sigma_main * main_noise[:, :pointer.shape[1]]
                    )
                built = build_window_input(
                    noisy_main, main_noise, sigma_context, sigma_main,
                    self.geometry, n_history_chunks=n_history_chunks,
                    first_frame=first_frame, overlap=overlap, history=history,
                    camera=camera,
                )
                pred_cond = self._predict(built, prompt_emb, packed_history)
                if guidance is None:
                    velocity = pred_cond
                else:
                    pred_uncond = self._predict(
                        built, negative_emb, packed_history,
                    )
                    velocity = guidance(
                        pred_cond, pred_uncond, noisy_main.unsqueeze(0), sigma_main,
                    )
                noisy_main = self.scheduler.step(
                    velocity.squeeze(0), t_curr, t_next, noisy_main,
                ).to(dtype=main_noise.dtype)
            if ref_overwrite and pointer is not None:
                noisy_main[:, :pointer.shape[1]] = pointer
        return noisy_main

    @torch.no_grad()
    def _streaming_decode_chunk(
        self, latents: torch.Tensor, *, reset_cache: bool,
    ) -> list[Image.Image]:
        """Decode one latent chunk while preserving temporal VAE state."""
        vae_model = self.vae.model
        scale = self.vae.scale
        if reset_cache:
            vae_model.clear_cache()
        z = latents.unsqueeze(0).to(
            dtype=self.torch_dtype, device=self.device,
        )
        if isinstance(scale[0], torch.Tensor):
            scale_t = [item.to(dtype=z.dtype, device=z.device) for item in scale]
            z = (
                z / scale_t[1].view(1, vae_model.z_dim, 1, 1, 1)
                + scale_t[0].view(1, vae_model.z_dim, 1, 1, 1)
            )
        else:
            scale_t = scale.to(dtype=z.dtype, device=z.device)
            z = z / scale_t[1] + scale_t[0]
        x = vae_model.conv2(z)

        chunk_frames: list[Image.Image] = []
        is_vae_2_2 = isinstance(self.vae, WanVideoVAE_2_2)
        for index in range(x.shape[2]):
            vae_model._conv_idx = [0]
            decoder_kwargs = {}
            if is_vae_2_2:
                decoder_kwargs["first_chunk"] = reset_cache and index == 0
            out = vae_model.decoder(
                x[:, :, index:index + 1],
                feat_cache=vae_model._feat_map,
                feat_idx=vae_model._conv_idx,
                **decoder_kwargs,
            ).clamp_(-1, 1)
            if is_vae_2_2:
                out = unpatchify(out, patch_size=2)
            video = rearrange(out[0], "C T H W -> T H W C")
            array = (
                (video.float() + 1) * 127.5
            ).clip(0, 255).cpu().numpy().astype(np.uint8)
            for raw_frame in array:
                chunk_frames.append(Image.fromarray(raw_frame))
        return chunk_frames

    @torch.no_grad()
    def __call__(
        self,
        prompt: str,
        negative_prompt: str = "",
        ref_image: Optional[Image.Image] = None,
        context_frames: Optional[list[Image.Image]] = None,
        camera_trajectory: Optional[np.ndarray] = None,
        seed: Optional[int] = None,
        cfg_scale: float = 3.5,
        num_inference_steps: int = 40,
        sigma_shift: float = 5.0,
        context_sigma_shift: float = 2.0,
        n_chunks: int = 1,
        ref_overwrite: bool = True,
        chunk_callback=None,
    ) -> list[Image.Image]:
        geometry = self.geometry
        assert not (ref_image is not None and context_frames is not None)
        assert 1 <= n_chunks <= MAX_CHUNKS
        if seed is None or seed < 0:
            seed = random.randint(0, sys.maxsize)

        context_chunk_count = (
            context_chunks(geometry, len(context_frames))
            if context_frames is not None else 0
        )
        total_chunks = context_chunk_count + n_chunks
        assert total_chunks <= MAX_CHUNKS
        if camera_trajectory is not None:
            self._require_camera_coverage(camera_trajectory, total_chunks)

        timesteps = self.scheduler.set_timesteps(
            num_inference_steps, shift=sigma_shift,
        )
        sigmas = context_sigmas(
            num_inference_steps, context_sigma_shift,
            device=self.device, dtype=self.torch_dtype,
        )

        prompt_emb = self.encode_prompt(prompt)
        negative_emb = (
            self.encode_prompt(negative_prompt) if cfg_scale != 1.0 else None
        )
        z_ref = None
        if ref_image is not None:
            z_ref = self.encode_image_to_latent(
                ref_image, geometry.height, geometry.width,
            )
        all_latents = None
        if context_frames is not None:
            all_latents = self.encode_video_to_latent(
                context_frames, geometry.height, geometry.width,
            )
            expected = 1 + geometry.new_frames * context_chunk_count
            assert all_latents.shape[1] == expected, (
                f"{len(context_frames)} context frames encoded to "
                f"{all_latents.shape[1]} latents; expected {expected}"
            )

        generator = torch.Generator(device="cpu").manual_seed(seed)
        guidance = (
            FlowMatchingGuidance(guidance_scale=cfg_scale)
            if cfg_scale != 1.0 else None
        )
        c = self.out_dim
        h_lat, w_lat = geometry.height // 8, geometry.width // 8
        all_frames: list[Image.Image] = []
        for chunk_index in range(total_chunks):
            is_initial = chunk_index == 0
            if is_initial:
                first_frame, overlap, history = z_ref, None, None
                window_start, window_frames = 0, 1 + geometry.new_frames
            else:
                history_end = 1 + geometry.history_frames(chunk_index)
                first_frame = all_latents[:, :1]
                overlap = all_latents[:, history_end:history_end + geometry.overlap]
                history = all_latents[:, 1:history_end]
                window_start, window_frames = history_end, geometry.main_frames

            if chunk_index < context_chunk_count:
                main = all_latents[:, window_start:window_start + window_frames]
            else:
                main_noise = _cpu_noise(
                    (c, window_frames, h_lat, w_lat), generator,
                    self.device, self.torch_dtype,
                )
                main = self._denoise_window(
                    main_noise=main_noise, first_frame=first_frame,
                    overlap=overlap, history=history,
                    packed_history=self._pack_history(history),
                    n_history_chunks=chunk_index,
                    prompt_emb=prompt_emb, negative_emb=negative_emb,
                    guidance=guidance, timesteps=timesteps, context_sigmas=sigmas,
                    ref_overwrite=ref_overwrite,
                    camera=self._camera_rows(
                        camera_trajectory, chunk_index, is_initial=is_initial,
                    ),
                    progress_desc=f"Chunk {chunk_index + 1}/{total_chunks}",
                )
                all_latents = (
                    main if is_initial
                    else torch.cat([all_latents, main[:, geometry.overlap:]], dim=1)
                )

            emitted = main if is_initial else main[:, geometry.overlap:]
            chunk_frames = self._streaming_decode_chunk(
                emitted, reset_cache=is_initial,
            )
            if chunk_index >= context_chunk_count:
                all_frames.extend(chunk_frames)
                if chunk_callback is not None:
                    chunk_callback(chunk_index - context_chunk_count, chunk_frames)
        return all_frames
