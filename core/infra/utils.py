from typing import Any

import numpy as np
import random
import torch
import torch.distributed as dist

def set_seed(seed: int = 42):
    """Set global seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = False
        
def get_generator(seed: int, device: torch.device):
    """Get a torch generator with specific seed."""
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    return generator

def broadcast_object(obj: Any, src: int, group: dist.ProcessGroup | None = None) -> Any:
    """Broadcast a Python object across ranks in a process group."""
    world_size = dist.get_world_size(group=group)
    if world_size == 1:
        return obj
    payload = [obj]
    dist.broadcast_object_list(payload, src=src, group=group)
    return payload[0]

def broadcast_tensor(
    tensor: torch.Tensor | None,
    src: int,
    group: dist.ProcessGroup | None = None,
    device: torch.device | None = None,
) -> torch.Tensor:
    """Broadcast a tensor, allocating storage on non-src ranks as needed."""
    world_size = dist.get_world_size(group=group)
    if world_size == 1:
        if tensor is None:
            raise ValueError("tensor must not be None when world_size == 1")
        return tensor

    rank = dist.get_rank()
    is_src = (rank == src)

    meta = None
    if is_src:
        assert tensor is not None
        meta = {
            "shape": tuple(tensor.shape),
            "dtype": tensor.dtype
        }
    meta = broadcast_object(meta, src=src, group=group)

    if not is_src:
        target_device = device if device is not None else torch.device("cpu")
        tensor = torch.empty(meta["shape"], dtype=meta["dtype"], device=target_device)

    dist.broadcast(tensor, src=src, group=group)
    return tensor

def broadcast_dict(
    data: dict[str, Any] | None,
    src: int,
    group: dist.ProcessGroup | None = None,
    device: torch.device | None = None,
) -> dict[str, Any]:
    """Broadcast a dictionary that may contain tensors."""
    world_size = dist.get_world_size(group=group)
    if world_size == 1:
        return data

    rank = dist.get_rank()
    is_src = (rank == src)

    schema = None
    if is_src:
        assert data is not None
        schema = {
            "keys": list(data.keys()),
            "is_tensor": {k: isinstance(v, torch.Tensor) for k, v in data.items()},
        }
    schema = broadcast_object(schema, src=src, group=group)

    if not is_src or data is None:
        data = {}

    for key in schema["keys"]:
        if schema["is_tensor"][key]:
            tensor = data.get(key)
            data[key] = broadcast_tensor(tensor, src=src, group=group, device=device)
        else:
            value = data.get(key)
            data[key] = broadcast_object(value, src=src, group=group)
    return data
