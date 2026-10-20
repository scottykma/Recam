from collections import deque
from typing import Any, Callable, Deque, Iterable

import torch
import torch.distributed as dist
from torch.distributed.device_mesh import DeviceMesh

from core.infra.utils import broadcast_dict
from core.infra.logger import get_logger

class DataManager:
    """Coordinates encoder/DiT execution order and caching of encoded batches."""

    def __init__(
        self,
        data_iter: Iterable[Any],
        encode_fn: Callable[[Any], dict[str, torch.Tensor | Any]],
        enter_encode_fn: Callable[[], None],
        enter_train_fn: Callable[[], None],
        target_device: torch.device | None = None,
        data_mesh: DeviceMesh = None,
        pre_encode_batches: int = 1,
        pre_encode_log_every: int = 0,
    ) -> None:
        self.data_iter = iter(data_iter)
        self.encode_fn = encode_fn
        self.enter_encode_fn = enter_encode_fn
        self.enter_train_fn = enter_train_fn
        self.data_mesh = data_mesh
        self.target_device = target_device
        self.pre_encode_batches = max(1, int(pre_encode_batches))
        self.pre_encode_log_every = int(pre_encode_log_every)
        if self.pre_encode_log_every < 0:
            raise ValueError("pre_encode_log_every must be >= 0")
        self._cache: Deque[dict[str, torch.Tensor | Any]] = deque()
        self._reads = 0

        self._cp_mesh = None
        self._cp_group = None
        self._cp_rank = 0
        self._cp_world_size = 1
        if self.data_mesh is not None and "cp" in self.data_mesh.mesh_dim_names:
            self._dp_rank = self.data_mesh["dp"].get_local_rank()
            self._cp_mesh = self.data_mesh["cp"]
            self._cp_group = self._cp_mesh.get_group()
            self._cp_rank = self._cp_mesh.get_local_rank()
            self._cp_world_size = self._cp_mesh.size(0)
        self._has_cp = self._cp_world_size > 1

    def _encode_with_offload(self) -> dict[str, torch.Tensor | Any]:
        batch = next(self.data_iter)
        encoded = self.encode_fn(batch)
        return encoded

    def _fill_cache(self) -> None:
        logger = get_logger()
        log_every = self.pre_encode_log_every
        if log_every > 0:
            logger.info(f"Pre-encoding {self.pre_encode_batches} batch(es)...")
        dist.barrier()
        self.enter_encode_fn()
        while len(self._cache) < self.pre_encode_batches:
            encoded = self._encode_with_offload()
            if len(self._cache) < self.pre_encode_batches - 1:
                encoded = {
                    k: v.to("cpu", non_blocking=True) if isinstance(v, torch.Tensor) else v
                    for k, v in encoded.items()
                }
            self._cache.append(encoded)
            filled = len(self._cache)
            if log_every > 0 and (filled % log_every == 0 or filled == self.pre_encode_batches):
                logger.info(f"Pre-encode progress: {filled}/{self.pre_encode_batches}")
        self.enter_train_fn()
        dist.barrier()

    def get_batch(self) -> dict[str, torch.Tensor | Any]:
        if self._reads % (self._cp_world_size * self.pre_encode_batches) == 0:
            self._fill_cache()
        self._reads += 1
        if self._has_cp:
            return self._get_cp_batch()
        else:
            return self._pop_local_batch()

    def _pop_local_batch(self) -> dict[str, torch.Tensor | Any]:
        encoded = self._cache.popleft()
        encoded = self._move_to_target_device(encoded)
        if self._cache:
            next_batch = self._cache[0]
            self._cache[0] = self._move_to_target_device(next_batch)
        return encoded

    def _get_cp_batch(self) -> dict[str, torch.Tensor | Any]:
        owner = self._reads % self._cp_world_size
        local_batch = None
        if self._cp_rank == owner:
            local_batch = self._pop_local_batch()
        batch = broadcast_dict(
            local_batch,
            src=self.data_mesh.mesh[self._dp_rank, owner].item(),
            group=self._cp_group,
            device=self.target_device,
        )
        return batch

    def _move_to_target_device(self, batch: dict[str, torch.Tensor | Any]) -> dict[str, torch.Tensor | Any]:
        if self.target_device is None:
            return batch
        moved = {}
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                moved[key] = value.to(self.target_device, non_blocking=True)
            else:
                moved[key] = value
        return moved
