"""Distributed task and history-length sampling for ReCam."""

from __future__ import annotations

from typing import Any, Iterator

import torch

from example.recam.datasets.strategy import Task


class ReCamBucketBatchSampler:
    """Synchronize task across ranks while sampling AR length per rank."""

    def __init__(
        self,
        dataset,
        batch_size: int,
        num_replicas: int,
        rank: int,
        seed: int = 0,
        drop_last: bool = False,
        shuffle: bool = True,
        infinite: bool = False,
    ) -> None:
        if batch_size != 1:
            raise ValueError("ReCam sampler requires batch_size == 1")
        if num_replicas <= 0:
            raise ValueError("num_replicas must be positive")
        if not 0 <= rank < num_replicas:
            raise ValueError(
                f"rank must be in [0, {num_replicas}); got {rank}"
            )
        if not shuffle:
            raise ValueError("ReCam sampler requires shuffle=True")
        if not infinite:
            raise ValueError("ReCam sampler requires infinite=True")

        bucket_probs = torch.as_tensor(
            dataset.bucket_probs, dtype=torch.float32,
        )
        task_probs = torch.stack([
            bucket_probs[0], bucket_probs[1], bucket_probs[2:].sum(),
        ])

        self.dataset = dataset
        self.batch_size = batch_size
        self.num_replicas = num_replicas
        self.rank = rank
        self.seed = seed
        self.drop_last = drop_last
        self.shuffle = shuffle
        self.infinite = infinite
        self.task_probs = task_probs
        self.ar_bucket_probs = bucket_probs[2:]
        self.step = 0

    def set_step(self, step: int) -> None:
        if step < 0:
            raise ValueError("step must be non-negative")
        self.step = step

    def __iter__(self) -> Iterator[list[Any]]:
        generator = torch.Generator()
        while True:
            generator.manual_seed(self.seed + self.step)
            task = Task(int(torch.multinomial(
                self.task_probs, 1, generator=generator,
            ).item()))
            if task is not Task.CONTINUE:
                bucket_ids = [int(task)] * self.num_replicas
            else:
                bucket_ids = (
                    torch.multinomial(
                        self.ar_bucket_probs,
                        self.num_replicas,
                        replacement=True,
                        generator=generator,
                    )
                    .add(2)
                    .tolist()
                )

            global_rows: list[Any] = [None] * self.num_replicas
            for bucket_id in dict.fromkeys(bucket_ids):
                positions = [
                    position
                    for position, selected_bucket in enumerate(bucket_ids)
                    if selected_bucket == bucket_id
                ]
                rows = self.dataset.sample_bucket(
                    bucket_id, len(positions), generator,
                )
                if len(rows) != len(positions):
                    raise RuntimeError(
                        f"bucket {bucket_id} returned {len(rows)} samples; "
                        f"expected {len(positions)}"
                    )
                for position, row in zip(positions, rows, strict=True):
                    global_rows[position] = row

            sample_seeds = torch.randint(
                high=2**31,
                size=(self.num_replicas,),
                generator=generator,
                dtype=torch.int64,
            ).tolist()
            global_keys = [
                self.dataset.make_sample_key(row, bucket_id, sample_seed)
                for row, bucket_id, sample_seed in zip(
                    global_rows, bucket_ids, sample_seeds, strict=True,
                )
            ]
            yield [global_keys[self.rank]]
            self.step += 1

    def __len__(self) -> int:
        return 2**31
