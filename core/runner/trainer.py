import gc
import json
import math
import os
from typing import Any, Callable, Mapping

import torch
import torch.distributed as dist
import torch.nn as nn
from peft import LoraConfig, get_peft_model
from peft.tuners.lora import LoraLayer
from torch.distributed.fsdp import FSDPModule
from torch.distributed.tensor import Replicate

from ..config.train_schema import BaseExpConfig
from ..data.data_manager import DataManager
from ..data.dataloader import build_dataloader
from ..infra.checkpointer import Checkpointer
from ..infra.ema import EMA
from ..infra.gradient_checkpointing import apply_gradient_checkpointing
from ..infra.logger import ExpLogger
from ..infra.lr_scheduler import build_lr_scheduler
from ..infra.parallelism import apply_fsdp, init_parallel
from ..infra.utils import get_generator, set_seed


class Trainer:
    def __init__(self, config: BaseExpConfig):
        self.config = config

        self.model_mesh, self.data_mesh = init_parallel(self.config.parallel)
        self.rank = dist.get_rank()
        self.is_master = self.rank == 0
        self.world_size = dist.get_world_size()

        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        if torch.cuda.is_available():
            self.device = torch.device(f"cuda:{local_rank}")
            torch.cuda.set_device(self.device)
        else:
            self.device = torch.device("cpu")

        set_seed(self.config.misc.seed)

        self.logger = ExpLogger(
            log_dir=self.config.logging.log_dir,
            log_file=self.config.logging.log_file,
            use_tensorboard=self.config.logging.use_tensorboard,
            log_interval=self.config.logging.log_interval,
            device=self.device,
            rank=self.rank,
        )
        with self.logger.section("init logger"):
            self.logger.info(
                f"Runtime: rank={self.rank}, world_size={self.world_size}, device={self.device}, seed={self.config.misc.seed}"
            )
            self.logger.info("Experiment Configuration:")
            self.logger.log_config(self.config)

        with self.logger.section("init data"):
            self.logger.info("Building dataset and dataloader...")
            self.dataset, self.sampler = self.build_dataset_and_sampler()
            self.dataloader = build_dataloader(
                self.dataset,
                sampler=self.sampler,
                rank=self.rank,
                seed=self.config.dataloader.seed,
                num_workers=self.config.dataloader.num_workers,
                pin_memory=self.config.dataloader.pin_memory,
                prefetch_factor=self.config.dataloader.prefetch_factor,
            )

            self.logger.log_data_info(self.dataset, self.dataloader)
            self.grad_accum_steps = max(1, int(self.config.optimizer.gradient_accumulation_steps))
            self.global_batch_size = (
                self.config.dataloader.batch_size
                * self.world_size
                * self.grad_accum_steps
                // self.config.parallel.cp_size
            )
            self.global_data_batch_size = self.config.dataloader.batch_size * self.world_size
            self.logger.info(
                f"Global training batch size: {self.global_batch_size}, Global pre-encode batch size: {self.global_data_batch_size}"
            )
            if self.grad_accum_steps > 1:
                self.logger.info(
                    f"Using gradient accumulation steps: {self.grad_accum_steps} (effective global batch: {self.global_batch_size})"
                )
            if (
                self.global_batch_size * self.config.checkpoint.save_latest_interval % self.global_data_batch_size != 0
                or self.global_batch_size * self.config.checkpoint.save_checkpoint_interval % self.global_data_batch_size != 0
            ):
                self.logger.warning(
                    "data consumed of save interval is not divisible by global data batch size. "
                    "This may cause minor mismatches during resume."
                )

        with self.logger.section("init checkpointer"):
            dist.barrier()
            self.logger.info("Initializing checkpointer...")
            self.checkpointer = Checkpointer(
                checkpoint_dir=self.config.checkpoint.dir,
                rank=self.rank,
                use_lora=self.config.lora.enabled,
                ignore_frozen_params=self.config.checkpoint.ignore_frozen_params,
                consolidated=self.config.checkpoint.consolidated,
            )

        with self.logger.section("init model"):
            self.logger.info("Building model...")
            if self.config.checkpoint.init_on_meta_device:
                with torch.device("meta"):
                    self.model = self.build_model()
                    if self.config.lora.enabled:
                        self.model = self.apply_lora(self.model)
            else:
                self.model = self.build_model()
                if self.config.lora.enabled:
                    self.model = self.apply_lora(self.model)

            self.model.train()
            self.logger.log_model_info(self.model)

            if self.config.optimizer.gradient_checkpoint_modules:
                apply_gradient_checkpointing(
                    self.model,
                    checkpoint_module_type=self.config.optimizer.gradient_checkpoint_modules,
                )
            apply_fsdp(
                self.model,
                model_mesh=self.model_mesh,
                reshard_after_forward=self.config.parallel.reshard_after_forward,
                use_amp=self.config.parallel.use_amp,
                param_dtype=self.config.parallel.param_dtype,
                reduce_dtype=self.config.parallel.reduce_dtype,
                wrap_submodules=self.config.parallel.wrap_submodules,
            )
            if self.config.misc.torch_compile:
                self.model = torch.compile(self.model)

            if self.config.checkpoint.init_on_meta_device:
                self.model.to_empty(device=self.device)
                if hasattr(self.model, "reset_parameters"):
                    self.model.reset_parameters()
                if self.config.lora.enabled:
                    for _, module in self.model.named_modules():
                        if isinstance(module, LoraLayer):
                            for adapter_name in module.lora_A.keys():
                                module.reset_lora_parameters(adapter_name, init_lora_weights=True)
            else:
                self.model.to(self.device)

        resume_ckpt = self.config.checkpoint.resume_from
        if not resume_ckpt and self.checkpointer.has_latest_checkpoint():
            resume_ckpt = "latest"

        with self.logger.section("resolve pretrained weights"):
            should_load_pretrained = self.config.checkpoint.pretrained_path and (
                (not self.checkpointer.has_latest_checkpoint()) or self.config.lora.enabled
            )
            uncovered: set[str] = set()
            if should_load_pretrained:
                self.logger.info(f"Loading pretrained model from {self.config.checkpoint.pretrained_path}...")
                uncovered = self.checkpointer.load_pretrained_model(
                    model=self.model,
                    pretrained_path=self.config.checkpoint.pretrained_path,
                    state_dict_adapter=self.get_pretrained_state_dict_adapter(),
                    strict=self.strict_pretrained_load(),
                )
            if resume_ckpt:
                recorded = self.checkpointer.load_runtime_state(resume_ckpt)
                self.fresh_param_names = set(
                    recorded.get("fresh_parameter_names", ())
                )
                fresh_source = f"recorded by {resume_ckpt}"
            else:
                self.fresh_param_names = uncovered
                fresh_source = "uncovered by the pretrained weights"
            names = sorted(self.fresh_param_names)
            self.logger.info(
                f"From-scratch parameters ({fresh_source}): {len(names)} "
                f"[{', '.join(names[:8])}{' ...' if len(names) > 8 else ''}]"
            )

        with self.logger.section("init optimizer"):
            self.logger.info("Initializing optimizer...")
            param_groups = self._build_optimizer_param_groups()
            if param_groups is None:
                trainable_params = [p for p in self.model.parameters() if p.requires_grad]
                self.optimizer = torch.optim.AdamW(
                    trainable_params,
                    lr=self.config.optimizer.learning_rate,
                    weight_decay=self.config.optimizer.weight_decay,
                    betas=(self.config.optimizer.beta1, self.config.optimizer.beta2),
                )
            else:
                self.optimizer = torch.optim.AdamW(
                    param_groups,
                    lr=self.config.optimizer.learning_rate,
                    weight_decay=self.config.optimizer.weight_decay,
                    betas=(self.config.optimizer.beta1, self.config.optimizer.beta2),
                )
            self.logger.log_optimizer_info(self.optimizer)

        with self.logger.section("init lr scheduler"):
            self.logger.info("Initializing lr scheduler...")
            self.lr_scheduler = build_lr_scheduler(
                self.optimizer,
                max_steps=self.config.lr_scheduler.max_steps,
                warmup_steps=self.config.lr_scheduler.warmup_steps,
                decay_steps=self.config.lr_scheduler.decay_steps,
                min_lr_factor=self.config.lr_scheduler.min_lr_factor,
                lr_decay_type=self.config.lr_scheduler.decay_type,
            )

        with self.logger.section("restore training state"):
            self.start_step = 0
            self._gn_ema_mean: float | None = None
            self._gn_ema_var: float = 0.0
            self._consecutive_skips: int = 0

            if (
                self.config.lora.enabled
                and self.config.lora.pretrained_lora_path
                and not resume_ckpt
            ):
                self.checkpointer.load_pretrained_lora_model(
                    model=self.model,
                    pretrained_lora_path=self.config.lora.pretrained_lora_path,
                    logger=self.logger,
                )

            if self.config.ema.enabled:
                self.logger.info("Initializing EMA...")
                self.ema = EMA(
                    model=self.model,
                    decay=self.config.ema.decay,
                )
            else:
                self.ema = None

            if resume_ckpt:
                self.start_step = self.checkpointer.load_checkpoint(
                    model=self.model,
                    optimizer=self.optimizer,
                    lr_scheduler=self.lr_scheduler,
                    ema=self.ema,
                    ckpt_id=resume_ckpt,
                    logger=self.logger,
                )
                if self.config.checkpoint.force_overwrite_optimizer:
                    self._overwrite_optimizer_hparams_from_config(
                        lr_scheduler_state_loaded=self.checkpointer.lr_scheduler_state_loaded,
                    )
                grad_outlier_ema = self.checkpointer.load_grad_outlier_ema(ckpt_id=resume_ckpt)
                if grad_outlier_ema is not None:
                    self._gn_ema_mean = grad_outlier_ema.get("grad_ema_mean")
                    self._gn_ema_var = grad_outlier_ema.get("grad_ema_var", 0.0)
                    self._consecutive_skips = grad_outlier_ema.get("consecutive_skips", 0)
                self._release_resume_memory()
                self.logger.purge_tensorboard(self.start_step)
                runtime_state = self.checkpointer.loaded_training_state.get(
                    "runtime_state", {}
                )
                self.validate_checkpoint_runtime_state(runtime_state)
                if not runtime_state:
                    self.logger.warning(
                        "Checkpoint predates resume runtime metadata; using the "
                        "legacy sampler position and unable to compare the saved "
                        "world size or global batch size."
                    )
                consumed_steps = int(runtime_state.get(
                    "data_step",
                    self.start_step
                    * self.global_batch_size
                    // self.global_data_batch_size,
                ))
                self.dataloader.batch_sampler.set_step(consumed_steps)
                saved_world_size = runtime_state.get("world_size")
                saved_global_batch_size = runtime_state.get("global_batch_size")
                if (
                    saved_world_size is not None
                    and int(saved_world_size) != self.world_size
                ):
                    self.logger.warning(
                        "Resuming with a different world size: "
                        f"{saved_world_size} -> {self.world_size}. DCP restored the "
                        "model, optimizer, and EMA into the current sharding; the "
                        "per-step sample membership changes with the DP topology."
                    )
                if (
                    saved_global_batch_size is not None
                    and int(saved_global_batch_size) != self.global_batch_size
                ):
                    self.logger.warning(
                        "Global training batch size changed across resume: "
                        f"{saved_global_batch_size} -> {self.global_batch_size}."
                    )
                self.logger.info(f"Resumed from {resume_ckpt} at step {self.start_step}.")

        with self.logger.section("init data manager"):
            self.logger.info("Initializing data manager...")
            self.data_manager = DataManager(
                data_iter=iter(self.dataloader),
                encode_fn=self.encode_batch,
                enter_encode_fn=self.enter_encode,
                enter_train_fn=self.enter_train,
                data_mesh=self.data_mesh,
                target_device=self.device,
                pre_encode_batches=self.config.data_manager.pre_encode_batches,
                pre_encode_log_every=self.config.data_manager.pre_encode_log_every,
            )
        self.step = self.start_step
        dist.barrier()

    def apply_lora(self, model: nn.Module) -> nn.Module:
        self.logger.info("Applying LoRA adapters...")
        lora_cfg = LoraConfig(
            r=self.config.lora.lora_r,
            lora_alpha=self.config.lora.lora_alpha,
            lora_dropout=self.config.lora.lora_dropout,
            target_modules=self.config.lora.lora_target_modules,
            modules_to_save=self.config.lora.modules_to_save,
            bias=self.config.lora.lora_bias,
        )
        return get_peft_model(model, lora_cfg)

    def _overwrite_optimizer_hparams_from_config(self, lr_scheduler_state_loaded: bool) -> None:
        """Keep loaded optimizer state, but restore config-controlled hyperparams.

        This also restores the scheduler shape (warmup/stable/decay) from the
        current config while preserving the resumed scheduler step.

        lr_scheduler_state_loaded=False means lr_scheduler.last_epoch did not come from
        the checkpoint (e.g. param-group count mismatch) and is stuck at -1. Recomputing
        _last_lr from that would spuriously factor to 0 (warmup-step formula at epoch -1),
        zeroing every group's lr with no error. Skip the lr write in that case and let the
        next lr_scheduler.step() derive lr from base_lrs/last_epoch consistently.
        """
        opt_cfg = self.config.optimizer
        target_lr = float(opt_cfg.learning_rate)
        self._overwrite_lr_scheduler_hparams_from_config()
        group_hparams = self._optimizer_group_hparams()
        if len(group_hparams) != len(self.optimizer.param_groups):
            raise ValueError(
                "Optimizer group hyperparameter count does not match the current "
                f"optimizer: {len(group_hparams)} vs {len(self.optimizer.param_groups)}"
            )
        target_lrs = [float(lr) for lr, _ in group_hparams]
        for group, group_lr, (_, weight_decay) in zip(
            self.optimizer.param_groups, target_lrs, group_hparams, strict=True
        ):
            group["initial_lr"] = group_lr
            group["weight_decay"] = weight_decay
            group["betas"] = (opt_cfg.beta1, opt_cfg.beta2)

        if hasattr(self.lr_scheduler, "base_lrs"):
            self.lr_scheduler.base_lrs = target_lrs

        if lr_scheduler_state_loaded:
            if hasattr(self.lr_scheduler, "_last_lr"):
                factor = 1.0
                if hasattr(self.lr_scheduler, "_get_lr_factor"):
                    factor = self.lr_scheduler._get_lr_factor(self.lr_scheduler.last_epoch)
                last_lrs = [group_lr * factor for group_lr in target_lrs]
                self.lr_scheduler._last_lr = last_lrs
                for group, lr in zip(self.optimizer.param_groups, last_lrs, strict=True):
                    group["lr"] = lr
        else:
            self.logger.warning(
                "force_overwrite_optimizer: lr_scheduler state was not restored from the "
                "checkpoint, so lr_scheduler.last_epoch does not reflect the resumed step. "
                "Skipping the lr write to avoid zeroing it; base_lrs/initial_lr are updated "
                "and the next lr_scheduler.step() will compute lr from the new base.",
                rank_0_only=True,
            )

        self.logger.info(
            "Overwrote optimizer hyperparameters from config after checkpoint load: "
            f"lr={target_lr}, weight_decay={opt_cfg.weight_decay}, "
            f"betas=({opt_cfg.beta1}, {opt_cfg.beta2})."
        )

    def _overwrite_lr_scheduler_hparams_from_config(self) -> None:
        """Restore scheduler shape from the current config after checkpoint load."""
        scheduler_cfg = self.config.lr_scheduler
        warmup_steps = int(scheduler_cfg.warmup_steps)
        decay_steps = int(scheduler_cfg.decay_steps)
        max_steps = int(scheduler_cfg.max_steps)
        stable_steps = max_steps - warmup_steps - decay_steps
        if stable_steps < 0:
            raise ValueError(
                f"Invalid schedule: warmup_steps ({warmup_steps}) + decay_steps "
                f"({decay_steps}) must be less than or equal to max_steps ({max_steps})."
            )
        for name, value in (
            ("warmup_steps", warmup_steps),
            ("stable_steps", stable_steps),
            ("decay_steps", decay_steps),
            ("lr_decay_type", scheduler_cfg.decay_type),
            ("min_lr_factor", float(scheduler_cfg.min_lr_factor)),
        ):
            if hasattr(self.lr_scheduler, name):
                setattr(self.lr_scheduler, name, value)

    def _optimizer_group_hparams(self) -> list[tuple[float, float]]:
        """Return absolute ``(lr, weight_decay)`` for each optimizer param group.

        Derived from `_build_optimizer_param_groups` so the groups and their
        hyperparameters cannot disagree about which group is which.
        """
        opt_cfg = self.config.optimizer
        groups = self._build_optimizer_param_groups()
        if groups is None:
            return [
                (float(opt_cfg.learning_rate), float(opt_cfg.weight_decay))
                for _ in self.optimizer.param_groups
            ]
        return [
            (
                float(group.get("lr", opt_cfg.learning_rate)),
                float(group.get("weight_decay", opt_cfg.weight_decay)),
            )
            for group in groups
        ]

    def _build_optimizer_param_groups(self) -> list[dict] | None:
        """Split trainable params into a backbone group and, when
        `optimizer.fresh_params` is set, a group for the from-scratch params
        (`self.fresh_param_names`).

        Returns None -> single-group optimizer with config.optimizer.{lr, weight_decay}.
        """
        fresh_cfg = self.config.optimizer.fresh_params
        if fresh_cfg is None or not self.fresh_param_names:
            return None
        backbone: list[torch.nn.Parameter] = []
        fresh: list[torch.nn.Parameter] = []
        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            (
                fresh if name in self.fresh_param_names else backbone
            ).append(param)
        if not fresh or not backbone:
            self.logger.info(
                "Optimizer param groups: single group "
                f"({len(fresh)} from-scratch and {len(backbone)} backbone "
                "trainable params)",
                rank_0_only=True,
            )
            return None
        self.logger.info(
            f"Optimizer param groups: backbone {len(backbone)} params "
            f"@ lr={self.config.optimizer.learning_rate:.2e}; from-scratch "
            f"{len(fresh)} params @ lr={float(fresh_cfg.lr):.2e} "
            f"(weight_decay={float(fresh_cfg.weight_decay):g})",
            rank_0_only=True,
        )
        return [
            {"params": backbone},
            {
                "params": fresh,
                "lr": float(fresh_cfg.lr),
                "weight_decay": float(fresh_cfg.weight_decay),
            },
        ]

    def _release_resume_memory(self) -> None:
        """Release checkpoint-load temporaries before the first training step."""
        gc.collect()
        if self.device.type != "cuda":
            return

        gib = 1024 ** 3
        reserved_before = torch.cuda.memory_reserved(self.device)
        torch.cuda.empty_cache()
        reserved_after = torch.cuda.memory_reserved(self.device)
        allocated = torch.cuda.memory_allocated(self.device)
        self.logger.info(
            "Resume memory cleanup: "
            f"reserved {reserved_before / gib:.2f} -> {reserved_after / gib:.2f} GB; "
            f"allocated {allocated / gib:.2f} GB."
        )

    def get_pretrained_state_dict_adapter(
        self,
    ) -> Callable[[Mapping[str, torch.Tensor]], Mapping[str, torch.Tensor]] | None:
        return None

    def strict_pretrained_load(self) -> bool:
        """Whether fresh-start pretrained loading requires every model parameter."""
        return True

    def checkpoint_runtime_state(self) -> Mapping[str, Any]:
        """Subclass metadata persisted alongside generic resume state."""
        return {}

    def validate_checkpoint_runtime_state(
        self, runtime_state: Mapping[str, Any],
    ) -> None:
        """Validate subclass metadata after checkpoint restoration."""
        return None

    def build_dataset_and_sampler(self) -> tuple[torch.utils.data.Dataset, torch.utils.data.Sampler]:
        raise NotImplementedError

    def build_model(self) -> FSDPModule | nn.Module:
        raise NotImplementedError

    def enter_encode(self) -> None:
        pass

    def enter_train(self) -> None:
        pass

    def encode_batch(self, batch: dict[str, torch.Tensor | Any]) -> dict[str, torch.Tensor | Any]:
        return batch

    def forward_step(self, encoded_data: dict[str, torch.Tensor | Any]) -> tuple[torch.Tensor, dict[str, float]]:
        raise NotImplementedError

    def eval(self) -> dict[str, float]:
        raise NotImplementedError

    def _train_one_micro_step(self, encoded_data: dict[str, torch.Tensor | Any]) -> dict[str, float]:
        loss, metrics = self.forward_step(encoded_data)
        if not torch.isfinite(loss).all().item():
            self.optimizer.zero_grad()
            raise FloatingPointError(
                f"non-finite loss at step {self.step}: {loss.item()}"
            )
        metrics["loss"] = loss.item()
        scaled_loss = loss / self.grad_accum_steps
        scaled_loss.backward()
        return metrics

    def _get_data_parallel_rank(self) -> int:
        if self.data_mesh is not None:
            return self.data_mesh.get_local_rank(0)
        return self.rank

    def _set_train_rng(self) -> None:
        dp_rank = self._get_data_parallel_rank()
        base_seed = self.config.misc.seed * self.config.num_steps * dp_rank + self.step
        self.rng = get_generator(base_seed, self.device)
        sync_seed = self.config.misc.seed * self.config.num_steps + self.step
        self._sync_rng = get_generator(sync_seed, self.device)

    def _run_optimizer_step(self) -> None:
        with self.logger.timer("optim"):
            optim_metrics: dict[str, float] = {}
            opt_cfg = self.config.optimizer

            gn_value: float | None = None
            if opt_cfg.clip_grad and opt_cfg.max_grad_norm > 0:
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    opt_cfg.max_grad_norm,
                )
                grad_norm = grad_norm.redistribute(device_mesh=self.model_mesh, placements=[Replicate(), Replicate()])
                gn_value = grad_norm.item()
                optim_metrics["optim/grad_norm"] = gn_value
                if not math.isfinite(gn_value):
                    self.optimizer.zero_grad()
                    raise FloatingPointError(
                        f"non-finite gradient norm at step {self.step}: {gn_value}"
                    )

            warmup_steps = self.config.lr_scheduler.warmup_steps
            observe_steps = opt_cfg.skip_outlier_grad_observe_steps
            active_start = warmup_steps + observe_steps
            in_lr_warmup = self.step < warmup_steps
            in_active = self.step >= active_start

            skip = False
            threshold: float | None = None
            if (
                opt_cfg.skip_outlier_grad
                and gn_value is not None
                and in_active
                and self._gn_ema_mean is not None
            ):
                effective_factor = opt_cfg.skip_outlier_grad_factor * (self.world_size ** 0.5)
                threshold = self._gn_ema_mean + effective_factor * (self._gn_ema_var ** 0.5)
                if gn_value > threshold:
                    skip = True
                    self.logger.warning(
                        f"[skip-outlier] step {self.step}: grad_norm={gn_value:.3f} > "
                        f"ema_mean({self._gn_ema_mean:.3f}) + {effective_factor:.2f}"
                        f"·sqrt(var)({self._gn_ema_var ** 0.5:.3f}) = {threshold:.3f}; skipping step",
                        rank_0_only=True,
                    )

            if skip:
                self.optimizer.zero_grad()
                optim_metrics["optim/grad_skipped"] = 1.0
                self._consecutive_skips += 1
                reset_n = opt_cfg.skip_outlier_grad_reset_steps
                if reset_n > 0 and self._consecutive_skips >= reset_n:
                    self.logger.warning(
                        f"[skip-outlier] {self._consecutive_skips} consecutive skips — "
                        f"threshold likely corrupted "
                        f"(mean={self._gn_ema_mean:.4f}, threshold={threshold:.4f}). "
                        f"Resetting EMA.",
                        rank_0_only=True,
                    )
                    self._gn_ema_mean = None
                    self._gn_ema_var = 0.0
                    self._consecutive_skips = 0
            else:
                self.optimizer.step()
                self.lr_scheduler.step()
                self.optimizer.zero_grad()
                optim_metrics["optim/grad_skipped"] = 0.0
                self._consecutive_skips = 0

            should_update_ema = (
                opt_cfg.skip_outlier_grad
                and not skip
                and gn_value is not None
                and not in_lr_warmup
            )
            if should_update_ema:
                alpha = opt_cfg.skip_outlier_grad_ema_alpha
                if self._gn_ema_mean is None:
                    self._gn_ema_mean = gn_value
                    self._gn_ema_var = 0.0
                else:
                    delta = gn_value - self._gn_ema_mean
                    self._gn_ema_mean = alpha * self._gn_ema_mean + (1.0 - alpha) * gn_value
                    self._gn_ema_var = alpha * self._gn_ema_var + (1.0 - alpha) * (delta ** 2)

        optim_metrics["optim/lr"] = self.lr_scheduler.get_last_lr()[0]
        if opt_cfg.skip_outlier_grad:
            optim_metrics["optim/gn_ema_mean"] = self._gn_ema_mean if self._gn_ema_mean is not None else 0.0
            optim_metrics["optim/gn_ema_threshold"] = threshold if threshold is not None else 0.0
        self.logger.log_metrics(optim_metrics)

    def train_one_step(self) -> None:
        self._set_train_rng()

        has_all_reduce_toggle = self.grad_accum_steps > 1 and hasattr(self.model, "set_requires_all_reduce")
        if has_all_reduce_toggle:
            self.model.set_requires_all_reduce(False)

        for micro_step in range(self.grad_accum_steps):
            if has_all_reduce_toggle and micro_step == self.grad_accum_steps - 1:
                self.model.set_requires_all_reduce(True)

            encoded_data = self.data_manager.get_batch()
            with self.logger.timer("micro_step"):
                micro_metrics = self._train_one_micro_step(encoded_data)
            self.logger.log_metrics(micro_metrics)

        self._run_optimizer_step()

    def _prepare_training(self) -> None:
        self.logger.info("Starting training...")
        self.model.train()
        self.optimizer.zero_grad()
        self._verify_sync_rng()

    def _verify_sync_rng(self) -> None:
        """Assert _sync_rng draws the same values on every rank.

        The dual-pointer mixture (and any future cross-rank-synchronized task
        decision) relies on _sync_rng being identical across ranks. If it
        diverges, logger.meters' key set diverges and log_step's per-key
        all_reduce hangs without an error message. Checking once at startup
        is much friendlier than waiting for the NCCL watchdog to fire.
        """
        self._set_train_rng()
        if not (dist.is_initialized() and dist.get_world_size() > 1):
            return
        probe = torch.rand(4, device=self.device, generator=self._sync_rng)
        probe_max = probe.clone()
        probe_min = probe.clone()
        dist.all_reduce(probe_max, op=dist.ReduceOp.MAX)
        dist.all_reduce(probe_min, op=dist.ReduceOp.MIN)
        if not torch.allclose(probe_max, probe_min):
            raise RuntimeError(
                f"_sync_rng diverged across ranks: "
                f"max={probe_max.tolist()}, min={probe_min.tolist()}"
            )
        self._set_train_rng()
        self.logger.info("[sanity] _sync_rng synchronized across all ranks")

    def _maybe_update_ema(self) -> None:
        if self.ema is None:
            return
        if self.step <= self.config.ema.start_step:
            self.ema.copy_from(self.model)
        elif self.step % self.config.ema.update_freq == 0:
            self.ema.step(self.model)

    def _save_checkpoint(self, *, is_latest: bool) -> None:
        self.checkpointer.save_checkpoint(
            step=self.step,
            model=self.model,
            optimizer=self.optimizer,
            lr_scheduler=self.lr_scheduler,
            ema=self.ema,
            logger=self.logger,
            is_latest=is_latest,
            grad_outlier_ema={
                "grad_ema_mean": self._gn_ema_mean,
                "grad_ema_var": self._gn_ema_var,
                "consecutive_skips": self._consecutive_skips,
            },
            runtime_state={
                "world_size": self.world_size,
                "global_batch_size": self.global_batch_size,
                "global_data_batch_size": self.global_data_batch_size,
                "gradient_accumulation_steps": self.grad_accum_steps,
                "data_step": (
                    self.step
                    * self.global_batch_size
                    // self.global_data_batch_size
                ),
                "fresh_parameter_names": sorted(self.fresh_param_names),
                **self.checkpoint_runtime_state(),
            },
        )

    def _maybe_save_checkpoint(self) -> None:
        save_regular = self.step % self.config.checkpoint.save_checkpoint_interval == 0
        save_latest = self.step % self.config.checkpoint.save_latest_interval == 0
        if save_regular or save_latest:
            self._save_checkpoint(is_latest=save_latest and not save_regular)

    def _evaluate_current_model(self) -> None:
        self.model.eval()
        if self.ema is not None:
            self.ema.store(self.model)
            self.ema.copy_to(self.model)

        try:
            self.rng = get_generator(self.config.misc.seed * self._get_data_parallel_rank(), self.device)
            self._sync_rng = get_generator(self.config.misc.seed, self.device)
            eval_metrics = self.eval()
            self.logger.log_eval(eval_metrics, self.step)
            if self.is_master and eval_metrics and self.config.eval.eval_dir:
                os.makedirs(self.config.eval.eval_dir, exist_ok=True)
                fname = f"{self.config.eval.eval_dir}/metrics.json"
                history = []
                if os.path.exists(fname):
                    try:
                        with open(fname) as f:
                            history = json.load(f)
                    except (json.JSONDecodeError, OSError):
                        history = []
                history.append({"step": self.step, **eval_metrics})
                with open(fname, "w") as f:
                    json.dump(history, f, indent=2)
        finally:
            if self.ema is not None:
                self.ema.restore(self.model)
            self.model.train()

    def _maybe_run_online_eval(self) -> None:
        if self.config.eval.online_eval and self.step % self.config.eval.online_eval_interval == 0:
            with self.logger.section("online evaluation"):
                self._evaluate_current_model()

    def _should_run_online_eval_on_resume(self) -> bool:
        """Allow task trainers to recover an interrupted checkpoint-step eval."""
        return False

    def _maybe_run_resume_online_eval(self) -> None:
        if self._should_run_online_eval_on_resume():
            with self.logger.section("resume online evaluation"):
                self._evaluate_current_model()

    def _finalize_training(self) -> None:
        with self.logger.section("finalize training"):
            self.logger.info("Training finished.")
            if (
                self.step % self.config.checkpoint.save_latest_interval != 0
                and self.step % self.config.checkpoint.save_checkpoint_interval != 0
            ):
                self._save_checkpoint(is_latest=False)

            if (
                self.config.eval.online_eval
                and self.step % self.config.eval.online_eval_interval != 0
            ):
                with self.logger.section("final evaluation"):
                    self._evaluate_current_model()

    def train(self) -> None:
        with self.logger.section("prepare training"):
            self._prepare_training()

        if self.config.misc.eval_only:
            with self.logger.section("eval-only"):
                self._evaluate_current_model()
            return

        self._maybe_run_resume_online_eval()

        while self.step < self.config.num_steps:
            self.train_one_step()

            self.step += 1
            self._maybe_update_ema()
            self.logger.log_step(self.step)
            self._maybe_save_checkpoint()
            self._maybe_run_online_eval()

        self._finalize_training()

    def close(self) -> None:
        self.logger.close()
        dist.destroy_process_group()
