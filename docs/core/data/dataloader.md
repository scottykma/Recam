# build_dataloader

A thin wrapper around `torch.utils.data.DataLoader`.

## Behavior

- The caller supplies `batch_sampler`; the framework does not split batches
  itself
- `collate_fn` filters out `None` samples before the default `collate`
- The worker seed is fixed to `seed + rank * 1000 + worker_id`
- `persistent_workers` is enabled automatically when `num_workers > 1`

## Compatible samplers

- `BatchDistributeSampler`
- `BucketBatchDistributeSampler`
- Any sampler that returns a list of sample indices for one batch

## Common arguments

- `rank`: determines the worker seed
- `num_workers`: data-loading concurrency
- `pin_memory`: usually recommended on CUDA
