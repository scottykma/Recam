"""ReCam trainer."""

from __future__ import annotations

import os
from collections import OrderedDict
from pathlib import Path
from typing import Any, Mapping

import torch
import torch.nn as nn

from core.config.train_schema import build_config_from_cli
from core.runner.trainer import Trainer
from example.recam.config.camera_config import CameraConfig
from example.recam.config.config import (
    COND_IN_CHANNEL,
    GEOMETRY_SCHEMA_VERSION,
    ContextNoiseRegime,
    ExpConfig,
)
from example.recam.datasets.camera.dataset import ReCamCameraDataset
from example.recam.datasets.dataset import ReCamDataset
from example.recam.datasets.strategy import Task
from example.recam.datasets.sampler import ReCamBucketBatchSampler
from example.recam.models.wan_dit import ReCamModel
from example.recam.trainers.inputs import (
    build_training_inputs,
    identity_camera_like,
    loss_metrics,
)
from example.recam.backbone.encoder import WanEncoder
from example.recam.backbone.flow_matching_scheduler import FlowMatchScheduler


def pretrained_adapter(
    state_dict: Mapping[str, torch.Tensor],
) -> OrderedDict[str, torch.Tensor]:
    """Match Wan base weights while retaining any ReCam additions."""
    excluded = (
        "img_emb", "v_img", "norm_k_img", "k_img", "pack_linear",
    )
    return OrderedDict(
        (key, value) for key, value in state_dict.items()
        if not any(name in key for name in excluded)
    )


def is_recam_training_weights(path: str) -> bool:
    checkpoint_path = Path(path)
    if checkpoint_path.name not in {"model", "ema"}:
        return False
    if not (checkpoint_path.parent / "training_state.pt").is_file():
        return False
    return (
        (checkpoint_path / ".metadata").is_file()
        or any(checkpoint_path.glob("*.pt"))
    )


class ReCamTrainer(Trainer):
    def __init__(self, config: ExpConfig):
        super().__init__(config)
        self.encoder = WanEncoder(
            config.model.encoder,
            self.device,
            dtype=getattr(torch, config.model.encoder.dtype),
        )
        self.flow_scheduler = FlowMatchScheduler(
            num_train_timesteps=config.strategy.flow_num_train_timesteps,
            snr_type=config.strategy.flow_snr_type,
            min_timestep=config.strategy.flow_min_timestep,
            max_timestep=config.strategy.flow_max_timestep,
            shift=config.strategy.flow_shift,
            device=self.device,
        )
        self.logger.info("ReCam trainer initialized.")

    @property
    def camera(self) -> CameraConfig | None:
        """Fused-PRoPE camera settings; None selects the base variant."""
        return getattr(self.config, "camera", None)

    def _dataset_common(self) -> dict[str, Any]:
        data = self.config.data
        frame_check = data.online_frame_check
        return dict(
            n_chunks=data.n_chunks,
            max_decode_attempts=data.max_decode_attempts,
            geometry=self.config.geometry,
            online_check=(
                {
                    "colorfulness_th": frame_check.colorfulness_th,
                    "max_resample": frame_check.max_resample,
                }
                if frame_check.enabled else None
            ),
            task_probs=data.task_probs,
        )

    def build_dataset_and_sampler(self):
        paths = self.config.data.parquet_path_list
        if self.is_master:
            missing = [path for path in paths if not os.path.exists(path)]
            if missing:
                raise FileNotFoundError(f"missing parquet files: {missing[:5]}")
        torch.distributed.barrier()
        camera = self.camera
        dataset = (ReCamCameraDataset if camera else ReCamDataset)(
            paths,
            **self._dataset_common(),
            **({"camera_dtype": camera.camera_dtype} if camera else {}),
        )
        self.logger.info(
            f"Chunk dataset: {len(dataset)} rows; "
            f"AR availability history1={dataset.rows_per_bucket[2]}, "
            f"history{dataset.max_history_chunks}={dataset.rows_per_bucket[-1]}"
        )
        sampler = ReCamBucketBatchSampler(
            dataset=dataset,
            batch_size=self.config.dataloader.batch_size,
            num_replicas=self.world_size,
            rank=self.rank,
            seed=self.config.dataloader.seed,
            drop_last=self.config.dataloader.drop_last,
            shuffle=self.config.dataloader.shuffle,
            infinite=self.config.dataloader.infinite,
        )
        return dataset, sampler

    def build_model(self) -> nn.Module:
        cfg = self.config.model.transformer
        camera = self.camera
        return ReCamModel(
            dim=cfg.dim, in_dim=cfg.in_dim, ffn_dim=cfg.ffn_dim,
            out_dim=cfg.out_dim, text_dim=cfg.text_dim, freq_dim=cfg.freq_dim,
            eps=cfg.eps, patch_size=cfg.patch_size,
            num_heads=cfg.num_heads, num_layers=cfg.num_layers,
            has_image_input=cfg.has_image_input,
            has_image_pos_emb=cfg.has_image_pos_emb,
            history_pack_shape=self.config.geometry.history_pack_shape,
            history_pad_shape=self.config.geometry.history_pad_shape,
            **(
                {
                    "camera_focal_x": camera.focal_x,
                    "camera_focal_y": camera.resolved_focal_y(self.config.geometry),
                    "camera_translation_transform": camera.translation_transform,
                }
                if camera else {}
            ),
        )

    def get_pretrained_state_dict_adapter(self):
        if is_recam_training_weights(
            self.config.checkpoint.pretrained_path
        ):
            return None
        return pretrained_adapter

    def strict_pretrained_load(self) -> bool:
        return False

    def checkpoint_runtime_state(self) -> Mapping[str, Any]:
        """Metadata inference reads back out of the checkpoint."""
        state: dict[str, Any] = {
            "recam_geometry_schema": GEOMETRY_SCHEMA_VERSION,
            "recam_geometry": self.config.geometry.as_runtime_state(),
            "recam_cond_mode": COND_IN_CHANNEL,
        }
        camera = self.camera
        if camera:
            state["recam_camera"] = camera.as_runtime_state(
                self.config.geometry,
            )
        return state

    def encode_batch(self, batch: dict[str, Any]) -> dict[str, Any]:
        task_id = int(batch["task_id"].view(-1)[0])
        n_history_chunks = int(batch["n_history_chunks"].view(-1)[0])
        texts = batch["text"]
        video = batch["video"].to(self.device, non_blocking=True)
        with torch.no_grad():
            if video.dtype == torch.uint8:
                video = video.to(dtype=torch.float32)
                video.mul_(1.0 / 127.5).sub_(1.0)
            latents = self.encoder.encode_video(
                video,
            )
            if torch.rand(
                (), device=self.device, generator=self.rng,
            ).item() < self.config.strategy.dropout_cond_prob:
                texts = ["" for _ in texts]
            context = self.encoder.encode_prompt(texts)
        encoded: dict[str, Any] = {
            "latents": latents,
            "context": context,
            "task_id": task_id,
            "n_history_chunks": n_history_chunks,
        }
        if self.camera:
            encoded["camera"] = batch["camera"].to(self.device, non_blocking=True)
        return encoded

    def _sample_timesteps(
        self, batch_size: int, dtype: torch.dtype, device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Draw (t_context, t_main) for one micro-step from the configured mixture.

        The normal regime couples the two: they are the min and max of two draws
        from the main distribution, so context is never noisier than main.
        """
        strategy = self.config.strategy
        weights = torch.as_tensor(
            strategy.mixture_weights, device=device, dtype=torch.float32,
        )
        regime = ContextNoiseRegime(int(torch.multinomial(
            weights, 1, generator=self.rng,
        ).item()))

        def draw(mu: float) -> torch.Tensor:
            return self.flow_scheduler.sample_train_timesteps(
                batch_size, generator=self.rng, lognorm_mu=mu,
            ).to(device=device, dtype=dtype)

        if regime is ContextNoiseRegime.HIGH_HIGH:
            low, high = strategy.high_noise_timestep_range
            sigma = torch.rand(
                batch_size, device=device, dtype=torch.float32,
                generator=self.rng,
            )
            sigma = sigma.mul(high - low).add(low)
            timestep = sigma.mul(
                strategy.flow_num_train_timesteps
            ).to(dtype=dtype)
            return timestep, timestep

        if regime is ContextNoiseRegime.SAME_NOISE:
            timestep = draw(strategy.main_lognorm_mu)
            return timestep, timestep

        first = draw(strategy.main_lognorm_mu)
        second = draw(strategy.main_lognorm_mu)
        return torch.minimum(first, second), torch.maximum(first, second)

    def forward_step(self, encoded_data: dict[str, Any]):
        latents = encoded_data["latents"]
        context = encoded_data["context"]
        task = Task(int(encoded_data["task_id"]))
        n_history_chunks = int(encoded_data["n_history_chunks"])
        b = latents.shape[0]
        noise = torch.randn(
            latents.shape, dtype=latents.dtype, device=latents.device,
            generator=self.rng,
        )
        t_context, t_main = self._sample_timesteps(
            b, latents.dtype, latents.device,
        )
        packed = build_training_inputs(
            latents, noise, task, n_history_chunks, t_context, t_main,
            self.flow_scheduler,
            geometry=self.config.geometry,
        )

        model_kwargs = dict(
            x=packed.x,
            timestep=packed.timestep,
            context=context,
            temporal_positions=packed.temporal_positions,
            main_frame_start=packed.main_frame_start,
            history=packed.history,
            y=packed.y,
        )
        camera = self.camera
        if camera:
            camera_batch = encoded_data["camera"]
            if camera.camera_dropout_prob > 0 and torch.rand(
                (), device=camera_batch.device, generator=self.rng,
            ).item() < camera.camera_dropout_prob:
                camera_batch = identity_camera_like(camera_batch)
            model_kwargs["camera"] = camera_batch
        main_output = self.model(**model_kwargs)
        target = packed.target
        if main_output.shape != target.shape:
            raise RuntimeError(
                f"main prediction {tuple(main_output.shape)} does not match "
                f"target {tuple(target.shape)}"
            )
        diff_sq = (main_output.float() - target.float()) ** 2
        return loss_metrics(
            diff_sq,
            task,
            self.config.geometry.overlap,
        )

    @torch.no_grad()
    def eval(self) -> dict[str, float]:
        return {}


def main(config_class: type[ExpConfig] = ExpConfig) -> None:
    config = build_config_from_cli(config_class)
    trainer = ReCamTrainer(config)
    try:
        trainer.train()
    except Exception as error:
        trainer.logger.error(f"Training failed: {error}", rank_0_only=False)
        raise
    finally:
        trainer.close()


if __name__ == "__main__":
    main()
