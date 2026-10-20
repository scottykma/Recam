import torch
import numpy as np
import random
from .sampler import BatchDistributeSampler


def error_avoidance_collector(batch: list[tuple]) -> tuple:
    """
    A simple collate function that avoids errors by filtering out None samples.
    
    Args:
        batch: List of samples, some of which may be None

    Returns:
        Filtered batch with None samples removed
    """

    batch = [sample for sample in batch if sample is not None]
    return torch.utils.data.default_collate(batch)

def _worker_init_fn(worker_id, rank, seed):
    """
    Docstring for _worker_init_fn
    
    :param worker_id: Description
    :param rank: Description
    :param seed: Description
    """
    worker_seed = seed + rank * 1000 + worker_id
    np.random.seed(worker_seed)
    random.seed(worker_seed)
    torch.manual_seed(worker_seed)

def build_dataloader(
    dataset,
    sampler: torch.utils.data.Sampler,
    rank: int,
    seed: int = 0,
    num_workers: int = 2,
    pin_memory: bool = True,
    prefetch_factor: int = 2,
):
    """
    Build a DataLoader with a distributed batch sampler.
    Args:
        dataset: Dataset object
        sampler: Sampler object
        rank: Rank of the current process
        seed: Random seed for worker initialization
        num_workers: Number of worker processes for data loading
        pin_memory: Whether to use pinned memory for data loading
        prefetch_factor: Number of batches to prefetch per worker
    Returns:
        DataLoader object
    """
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_sampler=sampler,
        collate_fn=error_avoidance_collector,
        num_workers=num_workers,
        pin_memory=pin_memory,
        prefetch_factor=prefetch_factor if num_workers > 0 else None,
        worker_init_fn=lambda worker_id: _worker_init_fn(worker_id, rank, seed),
        persistent_workers=True if num_workers > 1 else False
    )

    return dataloader