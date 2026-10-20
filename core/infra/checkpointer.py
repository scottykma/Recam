import os
import json
import shutil
from pathlib import Path
from collections import OrderedDict
from typing import Any, Callable, Mapping

import torch
import torch.nn as nn
from torch.distributed.fsdp import FSDPModule
from torch.distributed.checkpoint.state_dict import (
    get_model_state_dict,
    get_optimizer_state_dict,
    set_model_state_dict,
    set_optimizer_state_dict,
    StateDictOptions,
)
import torch.distributed.checkpoint as dcp
from torch.distributed.tensor import distribute_tensor, DTensor

import safetensors.torch

from .ema import EMA
from .logger import ExpLogger
from .utils import broadcast_object

def get_normal_state_dict(model_path: str):
    if model_path.endswith(".safetensors"):
        return safetensors.torch.load(open(model_path, "rb").read())
    elif os.path.isdir(model_path):
        if "diffusion_pytorch_model.safetensors.index.json" in os.listdir(model_path):
            safetensors_map = json.load(open(os.path.join(model_path, "diffusion_pytorch_model.safetensors.index.json"), "r"))
            diffusion_pytorch_model_files = set(safetensors_map['weight_map'].values())
            state_dict = OrderedDict()
            for fname in diffusion_pytorch_model_files:
                fpath = os.path.join(model_path, fname)
                state_dict.update(get_normal_state_dict(fpath))
            return state_dict
        pt_files = sorted(Path(model_path).glob("*.pt"))
        if pt_files:
            state_dict = OrderedDict()
            for fpath in pt_files:
                state_dict.update(torch.load(fpath, map_location="cpu", weights_only=False))
            return state_dict
        raise ValueError(f"No loadable weights (HF index or *.pt) found in {model_path}")
    else:
        return torch.load(model_path, map_location="cpu", weights_only=False)


def get_lora_state_dict(model_path: str):
    """Load LoRA adapter weights from a PEFT model file or a training checkpoint dir."""
    path = Path(model_path)
    if not path.exists():
        raise ValueError(f"LoRA path {model_path} does not exist.")

    if path.is_file():
        state_dict = get_normal_state_dict(str(path))
    else:
        candidates = [
            path / "model" / "base_model.pt",
            path / "model.pt",
            path / "base_model.pt",
        ]
        for candidate in candidates:
            if candidate.exists():
                state_dict = get_normal_state_dict(str(candidate))
                break
        else:
            model_dir = path / "model"
            pt_files = sorted(model_dir.glob("*.pt")) if model_dir.is_dir() else []
            if pt_files:
                state_dict = OrderedDict()
                for fpath in pt_files:
                    state_dict.update(torch.load(fpath, map_location="cpu", weights_only=False))
            else:
                state_dict = get_normal_state_dict(str(path))

    return OrderedDict(
        (k, v) for k, v in state_dict.items()
        if "lora_" in k or "modules_to_save" in k
    )
    
def _split_by_submodule(state_dict: Mapping[str, torch.Tensor]) -> "OrderedDict[str, OrderedDict]":
    """Group a flat state_dict by its top-level module name (the key prefix before the
    first dot). Keys with no dot fall under '_root'. Used by consolidated checkpoints so
    each top-level submodule (e.g. dit_online / dit_ema / memory) lands in its own .pt,
    letting a loader read only the subset it needs."""
    groups: "OrderedDict[str, OrderedDict]" = OrderedDict()
    for k, v in state_dict.items():
        sub = k.split(".", 1)[0] if "." in k else "_root"
        groups.setdefault(sub, OrderedDict())[k] = v
    return groups


def _strip_peft_prefixes(k: str) -> str:
    """Map a peft-wrapped key back to the plain model key it was derived from.
    Handles both LoRA-adapted Linear layers (base_model.model./base_layer.) and
    modules_to_save-wrapped submodules (original_module./modules_to_save.default.),
    which collapse to the same plain key since both mirror the same pretrained
    submodule."""
    return (
        k.replace("base_model.model.", "")
        .replace("base_layer.", "")
        .replace("original_module.", "")
        .replace("modules_to_save.default.", "")
    )


class Checkpointer:
    """
    A checkpoint manager that uses Distributed Checkpoint (DCP) API to save and load
    training states including model, optimizer, lr_scheduler, ema, and other states.
    """

    def __init__(self, checkpoint_dir: str, rank: int = 0, use_lora: bool = False,
                 ignore_frozen_params: bool = True, consolidated: bool = False) -> None:
        """
        Initializes the Checkpointer.

        Args:
            checkpoint_dir: Root directory for storing checkpoints.
            rank: Process rank for distributed training.
            use_lora: Whether LoRA adapters are used in the model.
            ignore_frozen_params: Whether to exclude frozen params from save/load.
                Set False to include frozen params (larger checkpoints, no pretrained reload on resume).
            consolidated: Save model/optimizer/ema as a single rank0 .pt (full_state_dict
                + cpu_offload) instead of sharded DCP dirs. Load mirrors it via torch.load.
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.is_master = (rank == 0)
        if self.is_master:
            os.makedirs(self.checkpoint_dir, exist_ok=True)
        self.use_lora = use_lora
        self.ignore_frozen_params = ignore_frozen_params
        self.consolidated = consolidated
        torch.distributed.barrier()

    def has_latest_checkpoint(self) -> bool:
        """
        Checks if a 'latest' checkpoint exists.

        Returns:
            True if 'latest' checkpoint exists, False otherwise.
        """
        latest_path = self.checkpoint_dir / "latest"
        return latest_path.exists()

    def _resolve_checkpoint_dir(self, ckpt_id: str | os.PathLike) -> Path:
        """Resolve a checkpoint id under checkpoint_dir, or an explicit path."""
        ckpt_path = Path(ckpt_id)
        if ckpt_path.is_absolute() or ckpt_path.exists():
            return ckpt_path
        return self.checkpoint_dir / ckpt_path

    def save_checkpoint(
        self,
        step: int,
        model: nn.Module | None = None,
        optimizer: torch.optim.Optimizer | None = None,
        lr_scheduler: torch.optim.lr_scheduler._LRScheduler | None = None,
        ema: EMA | None = None,
        logger: ExpLogger | None = None,
        is_latest: bool = True,
        grad_outlier_ema: dict | None = None,
        runtime_state: Mapping[str, Any] | None = None,
    ) -> None:
        """
        Saves a complete training checkpoint.

        Args:
            step: Current training step.
            model: The model to save.
            optimizer: The optimizer to save.
            lr_scheduler: The learning rate scheduler to save.
            ema: The EMA model to save.
            logger: Logger for logging checkpointing events.
            is_latest: If True, saves as the latest checkpoint; otherwise, saves as a regular checkpoint.
            grad_outlier_ema: Optional EMA state for the skip-outlier-grad detector
                (e.g. {"grad_ema_mean": float|None, "grad_ema_var": float}).
                Persisted into training_state.pt so resume continues with a warm threshold.
            runtime_state: Optional data/distributed runtime metadata used to make
                resume behavior explicit across different world sizes.
        """
        torch.cuda.empty_cache()
        if is_latest:
            checkpoint_path = self.checkpoint_dir / "latest_temp"
        else:
            checkpoint_path = self.checkpoint_dir / f"step_{step:07d}"
        
        if self.is_master:
            os.makedirs(checkpoint_path, exist_ok=True)
            if self.consolidated:
                if model is not None:
                    os.makedirs(checkpoint_path / "model", exist_ok=True)
            elif not self.use_lora:
                for subdir in (["model"] if model is not None else []) + \
                               (["optimizer"] if optimizer is not None else []) + \
                               (["ema"] if ema is not None else []):
                    os.makedirs(checkpoint_path / subdir, exist_ok=True)
        torch.distributed.barrier()

        if logger is not None:
            logger.info(f"Saving checkpoint at step {step} to {checkpoint_path}...")
        
        if self.consolidated:
            if model is not None:
                model_state_dict = get_model_state_dict(model, options=StateDictOptions(full_state_dict=True, cpu_offload=True, ignore_frozen_params=self.ignore_frozen_params))
                if self.is_master:
                    for sub, sub_sd in _split_by_submodule(model_state_dict).items():
                        torch.save(sub_sd, checkpoint_path / "model" / f"{sub}.pt")
            if optimizer is not None:
                optim_state_dict = get_optimizer_state_dict(model, optimizer, options=StateDictOptions(full_state_dict=True, cpu_offload=True, ignore_frozen_params=self.ignore_frozen_params))
                if self.is_master:
                    torch.save(optim_state_dict, checkpoint_path / "optimizer.pt")
            if ema is not None:
                static_dict = ema.state_dict()
                for key in static_dict:
                    static_dict[key] = static_dict[key].full_tensor()
                if self.is_master:
                    torch.save(static_dict, checkpoint_path / "ema.pt")
        elif self.use_lora:
            if model is not None:
                model_state_dict = get_model_state_dict(model, options=StateDictOptions(full_state_dict=True, cpu_offload=True, ignore_frozen_params=self.ignore_frozen_params))
                if self.is_master:
                    torch.save(model_state_dict, checkpoint_path / "model.pt")
            if optimizer is not None:
                optim_state_dict = get_optimizer_state_dict(model, optimizer, options=StateDictOptions(full_state_dict=True, cpu_offload=True, ignore_frozen_params=self.ignore_frozen_params))
                if self.is_master:
                    torch.save(optim_state_dict, checkpoint_path / "optimizer.pt")
            if ema is not None:
                static_dict = ema.state_dict()
                for key in static_dict:
                    static_dict[key] = static_dict[key].full_tensor()
                if self.is_master:
                    torch.save(static_dict, checkpoint_path / "ema.pt")
        else: 
            if model is not None:
                model_state_dict = get_model_state_dict(model, options=StateDictOptions(ignore_frozen_params=self.ignore_frozen_params))
                dcp.save(state_dict=model_state_dict, checkpoint_id=checkpoint_path / "model")

            if optimizer is not None:
                optim_state_dict = get_optimizer_state_dict(model, optimizer, options=StateDictOptions(ignore_frozen_params=self.ignore_frozen_params))
                dcp.save(state_dict=optim_state_dict, checkpoint_id=checkpoint_path / "optimizer")

            if ema is not None:
                dcp.save(state_dict=ema.state_dict(), checkpoint_id=checkpoint_path / "ema")

        if self.is_master:
            training_state = {
                "global_step": step,
            }
            if lr_scheduler is not None:
                training_state["lr_scheduler"] = lr_scheduler.state_dict()
            if grad_outlier_ema is not None:
                training_state["grad_outlier_ema"] = grad_outlier_ema
            if runtime_state is not None:
                training_state["runtime_state"] = dict(runtime_state)
            torch.save(training_state, checkpoint_path / "training_state.pt")
        
        if self.is_master:
            latest_path = self.checkpoint_dir / "latest"
            if latest_path.exists() or latest_path.is_symlink():
                if latest_path.is_symlink():
                    latest_path.unlink()
                else:
                    shutil.rmtree(latest_path)
            if is_latest:
                os.rename(checkpoint_path, latest_path)
            else:
                os.symlink(checkpoint_path.name, latest_path)
        
        torch.distributed.barrier()
        torch.cuda.empty_cache()

    def load_checkpoint(
        self,
        model: nn.Module | None = None,
        optimizer: torch.optim.Optimizer | None = None,
        lr_scheduler: torch.optim.lr_scheduler._LRScheduler | None = None,
        ema: EMA | None = None,
        ckpt_id: str | None = None,
        logger: ExpLogger | None = None,
    ) -> int:
        """
        Loads a complete training checkpoint including model, optimizer, lr_scheduler, ema, and training state.

        Args:
            model: The model to load weights into.
            optimizer: The optimizer to load state into.
            lr_scheduler: The learning rate scheduler to load state into.
            ema: The EMA model to load weights into.
            ckpt_id: The checkpoint identifier for load.
            logger: Logger for logging checkpointing events.

        Returns:
            The training step at which the checkpoint was saved.
        """
        if ckpt_id is None:
            ckpt_id = "latest"
        ckpt_dir = self._resolve_checkpoint_dir(ckpt_id)
        
        if not os.path.exists(ckpt_dir):
            raise ValueError(f"Checkpoint path {ckpt_dir} does not exist.")

        if self.use_lora or self.consolidated:
            if model is not None:
                if self.consolidated:
                    model_dir = ckpt_dir / "model"
                    parts = sorted(model_dir.glob("*.pt")) if model_dir.is_dir() else []
                    if not parts:
                        children = sorted(
                            p.name for p in ckpt_dir.iterdir()
                            if p.is_dir() and (p / "training_state.pt").exists()
                        ) if ckpt_dir.is_dir() else []
                        hint = (
                            f"; {ckpt_dir} looks like a checkpoint root. "
                            f"Pass one checkpoint directory/id instead, e.g. {children[:5]}"
                            if children else ""
                        )
                        raise ValueError(f"Model checkpoint not found in {model_dir}{hint}")
                    own = {n for n, _ in model.named_children()}
                    filtered = [p for p in parts if p.stem in own]
                    if not filtered:
                        raise ValueError(
                            f"None of the checkpoint parts {[p.stem for p in parts]} match "
                            f"model submodules {sorted(own)}"
                        )
                    parts = filtered
                    model_state_dict = OrderedDict()
                    for p in parts:
                        model_state_dict.update(torch.load(p, map_location="cpu"))
                else:
                    model_path = ckpt_dir / "model.pt"
                    if not model_path.exists():
                        raise ValueError(f"Model checkpoint not found in {model_path}")
                    model_state_dict = torch.load(model_path, map_location="cpu")
                set_model_state_dict(model, model_state_dict, options=StateDictOptions(full_state_dict=True, ignore_frozen_params=self.ignore_frozen_params))
            if optimizer is not None:
                optim_path = ckpt_dir / "optimizer.pt"
                if not optim_path.exists():
                    raise ValueError(f"Optimizer checkpoint not found in {optim_path}")
                optim_state_dict = torch.load(optim_path, map_location="cpu")
                set_optimizer_state_dict(model, optimizer, optim_state_dict, options=StateDictOptions(full_state_dict=True, ignore_frozen_params=self.ignore_frozen_params))
            if ema is not None:
                ema_path = ckpt_dir / "ema.pt"
                if not ema_path.exists():
                    raise ValueError(f"EMA checkpoint not found in {ema_path}")
                ema_state_dict = torch.load(ema_path, map_location="cpu")
                
                static_dict = ema.state_dict()
                for key in static_dict:
                    ema_state_dict[key] = distribute_tensor(ema_state_dict[key], device_mesh=static_dict[key].device_mesh, placements=static_dict[key].placements)
                ema.load_state_dict(ema_state_dict)
        else:
            if model is not None:
                if not (ckpt_dir / "model").exists():
                    raise ValueError(f"Model checkpoint not found in {ckpt_dir}")
                model_state_dict = get_model_state_dict(model, options=StateDictOptions(ignore_frozen_params=self.ignore_frozen_params))
                dcp.load(state_dict=model_state_dict, checkpoint_id=ckpt_dir / "model")

            if optimizer is not None:
                if not (ckpt_dir / "optimizer").exists():
                    raise ValueError(f"Optimizer checkpoint not found in {ckpt_dir}")
                optim_state_dict = get_optimizer_state_dict(model, optimizer, options=StateDictOptions(ignore_frozen_params=self.ignore_frozen_params))
                dcp.load(state_dict=optim_state_dict, checkpoint_id=ckpt_dir / "optimizer")
            
            if ema is not None:
                if not (ckpt_dir / "ema").exists():
                    raise ValueError(f"EMA checkpoint not found in {ckpt_dir}")
                dcp.load(state_dict=ema.state_dict(), checkpoint_id=ckpt_dir / "ema")

        self.lr_scheduler_state_loaded = lr_scheduler is None

        if not (ckpt_dir / "training_state.pt").exists():
            training_state = {}
            if logger is not None:
                logger.warning(f"Training state checkpoint not found in {ckpt_dir}, returning step 0.")
            step = 0
        else:
            training_state = torch.load(ckpt_dir / "training_state.pt")
            if lr_scheduler is not None:
                try:
                    lr_scheduler.load_state_dict(training_state["lr_scheduler"])
                    self.lr_scheduler_state_loaded = True
                except (IndexError, KeyError):
                    if logger is not None:
                        logger.warning(
                            "LR scheduler state has different param-group count than current "
                            "scheduler (e.g. EMA-trainer ckpt loaded into single-group eval "
                            "trainer). Skipping LR scheduler restoration."
                        )
            step = training_state["global_step"]

        self.loaded_training_state = training_state

        torch.distributed.barrier()
        return step

    def load_grad_outlier_ema(self, ckpt_id: str) -> dict | None:
        """Read the skip-outlier-grad EMA state from training_state.pt.

        Returns None if the checkpoint pre-dates this field (older runs); the
        trainer then cold-starts the EMA. Independent from load_checkpoint to
        keep its signature stable.
        """
        ckpt_dir = self._resolve_checkpoint_dir(ckpt_id)
        ts_path = ckpt_dir / "training_state.pt"
        if not ts_path.exists():
            return None
        training_state = torch.load(ts_path)
        return training_state.get("grad_outlier_ema")

    def load_runtime_state(self, ckpt_id: str) -> dict:
        """Read the trainer-written runtime state from training_state.pt.

        Independent from load_checkpoint (same rationale as
        load_grad_outlier_ema): callers need this *before* the optimizer is
        built, which is earlier than the checkpoint gets restored.
        """
        ckpt_dir = self._resolve_checkpoint_dir(ckpt_id)
        ts_path = ckpt_dir / "training_state.pt"
        if not ts_path.exists():
            return {}
        return torch.load(ts_path).get("runtime_state", {})

    def load_pretrained_model(
        self,
        model: FSDPModule,
        pretrained_path: str,
        *,
        state_dict_adapter: Callable[[Mapping[str, torch.Tensor]], Mapping[str, torch.Tensor]] | None = None,
        strict: bool = True,
    ) -> set[str]:
        """
        Loads model weights from a pretrained checkpoint into a non-distributed model.

        Args:
            model: The non-distributed model to load weights into.
            pretrained_path: Path to the pretrained checkpoint, which can be a .pt, .safetensors file or a directory.

        Returns:
            Names of the model parameters the pretrained source left untouched --
            i.e. the ones this model initializes from zero. Callers use it to give
            those parameters their own optimizer group.
        """
        if not os.path.exists(pretrained_path):
            raise ValueError(f"Pretrained path {pretrained_path} does not exist.")

        if os.path.isdir(pretrained_path) and ".metadata" in os.listdir(pretrained_path):
            model_state_dict = get_model_state_dict(model)
            model_names = set(model_state_dict.keys())
            if self.use_lora:
                without_lora_model_state_dict = {}
                for k, v in model_state_dict.items():
                    if "lora_" in k:
                        continue
                    new_key = _strip_peft_prefixes(k)
                    without_lora_model_state_dict[new_key] = v
                model_state_dict = without_lora_model_state_dict
            if state_dict_adapter is not None:
                model_state_dict = state_dict_adapter(model_state_dict)
            available = set(
                dcp.FileSystemReader(pretrained_path)
                .read_metadata()
                .state_dict_metadata.keys()
            )
            dcp.load(state_dict=model_state_dict, checkpoint_id=pretrained_path)
            return model_names - (set(model_state_dict.keys()) & available)

        if self.is_master:
            model_names = set(model.state_dict().keys())
            pretrained_state_dict = get_normal_state_dict(pretrained_path)
            if state_dict_adapter is not None:
                pretrained_state_dict = state_dict_adapter(pretrained_state_dict)
            if self.use_lora:
                without_lora_state_dict = {}
                for k in model_names:
                    if "lora_" in k:
                        continue
                    src_key = _strip_peft_prefixes(k)
                    if src_key not in pretrained_state_dict:
                        raise ValueError(f"Key {k} not found in pretrained state dict.")
                    without_lora_state_dict[k] = pretrained_state_dict[src_key]
                pretrained_state_dict = without_lora_state_dict
            fresh = model_names - set(pretrained_state_dict.keys())
        else:
            pretrained_state_dict = None
            fresh = None

        torch.distributed.barrier()
        fresh = broadcast_object(fresh, src=0)
        set_model_state_dict(
            model,
            pretrained_state_dict,
            options=StateDictOptions(
                full_state_dict=True,
                broadcast_from_rank0=True,
                strict=bool(strict and not self.use_lora),
            ),
        )
        return fresh

    def load_pretrained_lora_model(
        self,
        model: FSDPModule,
        pretrained_lora_path: str,
        *,
        logger: ExpLogger | None = None,
    ) -> None:
        """Load only LoRA adapter and modules_to_save weights without optimizer state."""
        if self.is_master:
            lora_state_dict = get_lora_state_dict(pretrained_lora_path)
            if not lora_state_dict:
                raise ValueError(f"No LoRA weights found in {pretrained_lora_path}")
            if logger is not None:
                logger.info(
                    f"Loading {len(lora_state_dict)} LoRA/module-to-save tensors from "
                    f"{pretrained_lora_path}..."
                )
        else:
            lora_state_dict = None

        torch.distributed.barrier()
        set_model_state_dict(
            model,
            lora_state_dict,
            options=StateDictOptions(
                full_state_dict=True,
                broadcast_from_rank0=True,
                ignore_frozen_params=self.ignore_frozen_params,
                strict=False,
            ),
        )
