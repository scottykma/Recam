# Sampler

## BatchDistributeSampler

The default distributed batch sampler.

- Splits samples by `dataset_size / batch_size / num_replicas / rank`
- Shuffles by `seed + epoch` when `shuffle=True`
- Pads by repeating samples when `drop_last=False`
- Iterates forever when `infinite=True`
- `set_step(step)` resumes at a given batch position

Suitable for most ordinary training tasks.

## BucketBatchDistributeSampler

Samples buckets according to their probabilities.

- Requires the dataset to provide:
  - `bucket_probs`
  - either `bucket_to_indices` or a `sample_bucket` hook
- Currently supports only:
  - `shuffle=True`
  - `infinite=True`

Suitable for datasets bucketed by length or resolution.
