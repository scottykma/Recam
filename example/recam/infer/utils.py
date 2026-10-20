import importlib
import os
from datetime import timedelta
from pathlib import Path

import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
from torch.distributed.checkpoint.api import CheckpointException

from example.recam.backbone.state_dict_adapter import filter_state_dict
from core.infra.checkpointer import get_normal_state_dict

DEFAULT_NEGATIVE_PROMPT = (
    "完全黑暗，完全静止，剧烈抖动，扭曲，不真实，色调艳丽，过曝，静态，细节模糊不清，字幕，静止，"
    "整体发灰，最差质量，低质量，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，"
    "毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿"
)


def _parse_prompt_source(prompt_source: str) -> tuple[str, str]:
    if ":" in prompt_source:
        return tuple(prompt_source.split(":", 1))
    return prompt_source, "PROMPT_CONFIGS"


def load_prompt_configs(prompt_source: str):
    module_path, attr = _parse_prompt_source(prompt_source)
    module = importlib.import_module(module_path)
    if not hasattr(module, attr):
        raise AttributeError(f"Prompt module {module_path} has no attribute {attr}")
    return getattr(module, attr)


def _is_dcp_checkpoint_path(path: str | os.PathLike) -> bool:
    return (Path(path) / ".metadata").exists()


def load_weights_for_model(
    model: torch.nn.Module,
    checkpoint_path: str,
    device: str,
    use_lora: bool = False,
    optional_missing_keys: tuple[str, ...] = (),
    ignored_key_substrings: tuple[str, ...] = (),
    strict: bool = True,
) -> None:
    if _is_dcp_checkpoint_path(checkpoint_path):
        model.to(device=device)
        model_state_dict = model.state_dict()
        if use_lora:
            without_lora_model_state_dict = {}
            for key, value in model_state_dict.items():
                if "lora_" in key:
                    continue
                new_key = key.replace("base_model.model.", "").replace("base_layer.", "")
                without_lora_model_state_dict[new_key] = value
            model_state_dict = without_lora_model_state_dict
        try:
            dcp.load(
                state_dict=model_state_dict, checkpoint_id=checkpoint_path,
            )
        except CheckpointException:
            optional = set(optional_missing_keys)
            fallback_state_dict = {
                key: value for key, value in model_state_dict.items()
                if key not in optional
            }
            if len(fallback_state_dict) == len(model_state_dict):
                raise
            dcp.load(
                state_dict=fallback_state_dict,
                checkpoint_id=checkpoint_path,
            )
            model_state_dict = fallback_state_dict
        model.load_state_dict(model_state_dict, strict=strict and not use_lora)
        return

    if dist.get_rank() == 0:
        pretrained_state_dict = get_normal_state_dict(checkpoint_path)
        pretrained_state_dict = filter_state_dict(pretrained_state_dict)
        if ignored_key_substrings:
            pretrained_state_dict = {
                key: value for key, value in pretrained_state_dict.items()
                if not any(part in key for part in ignored_key_substrings)
            }
        if use_lora:
            without_lora_state_dict = {}
            for key in model.state_dict().keys():
                if "lora_" in key:
                    continue
                source_key = key.replace("base_model.model.", "").replace("base_layer.", "")
                if source_key not in pretrained_state_dict:
                    raise ValueError(f"Key {key} not found in pretrained state dict.")
                without_lora_state_dict[key] = pretrained_state_dict[source_key]
            pretrained_state_dict = without_lora_state_dict
        model.load_state_dict(pretrained_state_dict, strict=strict and not use_lora)

    dist.barrier()
    model.to(device=device)
    for param in model.parameters():
        dist.broadcast(param.data, src=0)


def setup_dist() -> tuple[int, int, int]:
    if not dist.is_initialized():
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        timeout = timedelta(minutes=int(os.environ.get("PG_TIMEOUT_MIN", "120")))
        dist.init_process_group(backend=backend, timeout=timeout)

    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", rank))
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
    return rank, world_size, local_rank
