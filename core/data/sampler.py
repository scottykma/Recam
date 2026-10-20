import math
import random
from typing import Any, Iterator, List

import torch

class BatchDistributeSampler:
    """
    A distributed batch sampler that supports resume from step, infinite iteration,
    shuffling, and dropping incomplete batches.
    """
    
    def __init__(
        self,
        dataset_size: int,
        batch_size: int,
        num_replicas: int,
        rank: int,
        seed: int = 0,
        drop_last: bool = False,
        shuffle: bool = True,
        infinite: bool = False,
    ):
        """
        Initialize the sampler.
        
        Args:
            dataset_size: Total number of samples in the dataset
            batch_size: Batch size per replica
            num_replicas: Number of processes in distributed training
            rank: Rank of the current process
            seed: Random seed for shuffling
            drop_last: Whether to drop incomplete batches
            shuffle: Whether to shuffle the data
            infinite: Whether to iterate infinitely
        """ 
        if dataset_size <= 0:
            raise ValueError("dataset_size must be a positive integer.")
        self.dataset_size = dataset_size
        self.batch_size = batch_size
        self.num_replicas = num_replicas
        self.rank = rank
        self.seed = seed
        self.drop_last = drop_last
        if self.drop_last and dataset_size < (batch_size * num_replicas):
            self.drop_last = False
        self.shuffle = shuffle
        self.infinite = infinite

        if self.drop_last:
            self.num_batches = math.floor(dataset_size / (batch_size * num_replicas))
        else:
            self.num_batches = math.ceil(dataset_size / (batch_size * num_replicas))
            
        self.num_samples = self.num_batches * batch_size
        self.total_size = self.num_samples * num_replicas

        self.epoch = 0
        self.step = 0
        
    def set_step(self, step: int) -> None:
        """
        Set current step for resuming training.
        
        Args:
            step: Current training step to resume from
        """
        self.step = step
        self.epoch = step // self.num_batches
        
    def _get_epoch_indices(self) -> List[int]:
        """Generate indices for current epoch."""

        indices = list(range(self.dataset_size))

        if self.drop_last:
            indices = indices[:self.total_size]
        else:
            if len(indices) < self.total_size:
                repeat = math.ceil(self.total_size / len(indices))
                indices = (indices * repeat)[:self.total_size]

        if self.shuffle:
            rng = random.Random(self.seed + self.epoch)
            rng.shuffle(indices)
        
        replica_indices = indices[self.rank:self.total_size:self.num_replicas]
        return replica_indices
    
    def __iter__(self) -> Iterator[List[int]]:
        """
        Generate batch indices for current replica.
        
        Returns:
            Iterator yielding lists of sample indices
        """
        while True:
            indices = self._get_epoch_indices()
            batches = [indices[i:i + self.batch_size] for i in range(0, len(indices), self.batch_size)]
            batch_index = self.step % self.num_batches if self.step > 0 else 0
            for b in range(batch_index, self.num_batches):
                yield batches[b]
                self.step += 1
            self.epoch += 1
            if not self.infinite:
                break
    
    def __len__(self) -> int:
        """
        Get number of batches per replica.
        
        Returns:
            Number of batches
        """
        return self.num_batches

class BucketBatchDistributeSampler:
    """
    A distributed bucket batch sampler that supports resume from step, 
    infinite iteration, and weighted bucket sampling.
    """
    
    def __init__(
        self,
        dataset,
        batch_size: int,
        num_replicas: int,
        rank: int,
        seed: int = 0,
        drop_last: bool = False,
        shuffle: bool = True,
        infinite: bool = False
    ):
        self.dataset = dataset
        self.batch_size = batch_size
        self.num_replicas = num_replicas
        self.rank = rank
        self.seed = seed
        self.drop_last = drop_last
        self.shuffle = shuffle
        self.infinite = infinite

        assert shuffle, "BucketBatchDistributeSampler currently only supports shuffling."
        assert infinite, "BucketBatchDistributeSampler currently only supports infinite iteration."
        assert hasattr(dataset, 'bucket_probs'), "Dataset must have 'bucket_probs' attribute."
        has_legacy_buckets = hasattr(dataset, 'bucket_to_indices')
        has_sampling_hook = hasattr(dataset, 'sample_bucket')
        assert has_legacy_buckets or has_sampling_hook, (
            "Dataset must have 'bucket_to_indices' or a 'sample_bucket' hook."
        )

        self.bucket_probs_tensor = torch.as_tensor(dataset.bucket_probs, dtype=torch.float32)
        self.bucket_to_indices = getattr(dataset, 'bucket_to_indices', None)
        self.num_buckets = len(self.bucket_probs_tensor)
        if self.bucket_to_indices is not None:
            assert len(self.bucket_to_indices) == self.num_buckets, (
                "bucket_probs and bucket_to_indices must have the same length"
            )
        self.step_total_samples = self.batch_size * self.num_replicas

        self.step = 0

    def set_step(self, step: int) -> None:
        """
        Set current step for resuming training.
        """
        self.step = step
    
    def __iter__(self) -> Iterator[List[Any]]:
        """
        Generate batch indices. 
        Note: We use a global seed + epoch for bucket selection to ensure all 
        ranks select the same bucket at the same step.
        """
        g = torch.Generator()
        while True:
            g.manual_seed(self.seed + self.step)
            
            bucket_id = torch.multinomial(self.bucket_probs_tensor, 1, generator=g).item()

            if hasattr(self.dataset, 'sample_bucket'):
                global_batch_indices = self.dataset.sample_bucket(
                    bucket_id, self.step_total_samples, g,
                )
            else:
                possible_indices = self.bucket_to_indices[bucket_id]
                num_in_bucket = len(possible_indices)
                if num_in_bucket == 0:
                    raise RuntimeError(f"bucket {bucket_id} has no available samples")
                rand_indices_in_bucket = torch.randint(
                    high=num_in_bucket,
                    size=(self.step_total_samples,),
                    generator=g,
                    dtype=torch.int64,
                ).tolist()
                global_batch_indices = [possible_indices[i] for i in rand_indices_in_bucket]

            if len(global_batch_indices) != self.step_total_samples:
                raise RuntimeError(
                    f"bucket {bucket_id} returned {len(global_batch_indices)} samples; "
                    f"expected {self.step_total_samples}"
                )

            if hasattr(self.dataset, 'make_sample_key'):
                sample_seeds = torch.randint(
                    high=2**31,
                    size=(self.step_total_samples,),
                    generator=g,
                    dtype=torch.int64,
                ).tolist()
                global_batch_indices = [
                    self.dataset.make_sample_key(row_index, bucket_id, sample_seed)
                    for row_index, sample_seed in zip(
                        global_batch_indices, sample_seeds, strict=True,
                    )
                ]
            
            start_idx = self.rank * self.batch_size
            end_idx = start_idx + self.batch_size

            yield global_batch_indices[start_idx:end_idx]
            
            self.step += 1
            
    def __len__(self) -> int:
        return 2**31
