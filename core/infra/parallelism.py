import os

import torch
from torch.distributed.fsdp import fully_shard, MixedPrecisionPolicy
from torch.distributed.device_mesh import DeviceMesh, init_device_mesh
from torch.distributed.tensor import DTensor, Shard, Replicate

from core.config.train_schema import ParallelConfig

class CPManager:
    def __init__(self, cp_mesh: DeviceMesh = None):
        self.cp_mesh = cp_mesh
        self.cp_size = self.cp_mesh.size(0)
        self.cp_rank = self.cp_mesh.get_local_rank()
        self.full_shape: dict[str, torch.Size] = {}
        self.full_stride: dict[str, tuple[int, ...]] = {}

    def local_range(self, total: int) -> tuple[int, int]:
        """Contiguous [start, end) that this rank owns when `total` is sharded."""
        if total % self.cp_size:
            raise ValueError(
                f"cannot shard {total} evenly across cp_size={self.cp_size}"
            )
        width = total // self.cp_size
        start = self.cp_rank * width
        return start, start + width

    def local_window(self, total: int, window: int) -> tuple[int, int]:
        """(offset, length) of a trailing `window` inside this rank's shard.

        `offset` indexes the window itself, so a caller holding per-window data of
        length `window` narrows it with ``data[:, offset:offset + length]``.
        """
        if not 0 <= window <= total:
            raise ValueError(f"window {window} must lie in [0, {total}]")
        start, end = self.local_range(total)
        first = max(start, total - window)
        length = end - first
        if length <= 0:
            return 0, 0
        return first - (total - window), length

    def split_input(self, x: torch.Tensor, dim: int, cache_k: str | None = None) -> torch.Tensor:
        x = DTensor.from_local(x, device_mesh=self.cp_mesh, placements=[Replicate()])
        if cache_k is not None:
            self.full_shape[cache_k] = x.shape
            self.full_stride[cache_k] = x.stride()
        x = x.redistribute(device_mesh=self.cp_mesh, placements=[Shard(dim)])
        x = x.to_local()
        if x.requires_grad:
            x.register_hook(lambda grad: grad / self.cp_size)
        return x
    
    def gather_output(
        self,
        output: torch.Tensor,
        dim: int,
        cache_k: str | None = None,
    ) -> torch.Tensor:
        kwargs = {}
        if cache_k is not None:
            kwargs["shape"] = self.full_shape[cache_k]
            kwargs["stride"] = self.full_stride[cache_k]
        output = DTensor.from_local(output, device_mesh=self.cp_mesh, placements=[Shard(dim)], **kwargs)
        output = output.redistribute(device_mesh=self.cp_mesh, placements=[Replicate()])
        output = output.to_local()
        if output.requires_grad:
            output.register_hook(lambda grad: grad * self.cp_size)
        return output

    def SeqAllToAll(
        self,
        x: torch.Tensor,
        scatter_dim: int,
        gather_dim: int,
        cache_k: str | None = None,
    ) -> torch.Tensor:
        kwargs = {}
        if cache_k is not None:
            kwargs["shape"] = self.full_shape[cache_k]
            kwargs["stride"] = self.full_stride[cache_k]
        x = DTensor.from_local(x, device_mesh=self.cp_mesh, placements=[Shard(gather_dim)], **kwargs)
        x = x.redistribute(self.cp_mesh, placements=[Shard(scatter_dim)])
        return x.to_local()
    
CP_MANAGER: CPManager = None

def get_cp_manager():
    return CP_MANAGER

def init_parallel(config: ParallelConfig) -> tuple[DeviceMesh, DeviceMesh | None]:
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    if torch.cuda.is_available():
        device_type = "cuda"    
    else:
        device_type = "cpu"

    if config.hsdp:
        assert world_size % config.shard_size == 0, "World size must be divisible by shard size for HSDP."
        model_mesh_shape = (world_size // config.shard_size, config.shard_size)
    else:
        model_mesh_shape = (1, world_size)

    model_mesh_names = ("replicate", "shard")
    model_mesh = init_device_mesh(
        device_type=device_type,
        mesh_shape=model_mesh_shape,
        mesh_dim_names=model_mesh_names
    )

    if config.cp_size > 1:
        assert world_size % config.cp_size == 0, "World size must be divisible by context parallelism size."
        data_mesh_shape = (world_size // config.cp_size, config.cp_size)
        data_mesh_names = ("dp", "cp")
        data_mesh = init_device_mesh(
            device_type=device_type,
            mesh_shape=data_mesh_shape,
            mesh_dim_names=data_mesh_names
        )
        global CP_MANAGER
        CP_MANAGER = CPManager(cp_mesh=data_mesh["cp"])
    else:
        data_mesh = None

    return model_mesh, data_mesh


def apply_fsdp(
    model: torch.nn.Module,
    model_mesh: DeviceMesh,
    reshard_after_forward: bool = False,
    use_amp: bool = True,
    param_dtype: str = "bfloat16",
    reduce_dtype: str = "float32",
    wrap_submodules: list[str] | None = None,
):
    """
    Apply distributed training strategy to a model using FSDP.
    
    Args:
        model: The model to be distributed.
        model_mesh: The device mesh for model parallelism.
        reshard_after_forward: Whether to reshard after forward pass.
        use_amp: Whether to use automatic mixed precision.
        param_dtype: Data type for gather model parameters.
        reduce_dtype: Data type for reduction operations.
        wrap_submodules: Names of the submodules to wrap (if any).
    """
    config = {
        "mesh": model_mesh,
        "reshard_after_forward": reshard_after_forward
    }

    if use_amp:
        config["mp_policy"] = MixedPrecisionPolicy(
            param_dtype=getattr(torch, param_dtype),
            reduce_dtype=getattr(torch, reduce_dtype),
        )

    if wrap_submodules:
        for submodule in wrap_submodules:
            layer = model.get_submodule(submodule)
            if isinstance(layer, torch.nn.ModuleList):
                for i, layer in enumerate(layer):
                    fully_shard(layer, **config)

    fully_shard(model, **config)
