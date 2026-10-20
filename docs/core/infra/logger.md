# ExpLogger

The single entry point for training logs.

## Where output goes

- Colored console logs
- `log_dir/log_file`
- TensorBoard (rank 0 only, and only when `use_tensorboard=True`)

## Main interfaces

- `log_metrics(metrics)`
  - Accumulates first; does not print immediately
- `log_step(step)`
  - Aggregates and prints only when `log_interval` is reached
- `log_eval(metrics, step)`
  - Records evaluation metrics directly
- `section(title)`
  - Prints a titled log block
- `timer(tag)`
  - Records a `timer/<tag>` metric

## How the framework uses it

- Micro-step / optimizer-step / training-step timings can all be attached to
  meters
- `log_step(step)` performs multi-GPU reduction
- Rank 0 additionally prints peak CUDA memory

## Reproducibility

- If the current directory is inside a Git repository, a snapshot of the
  unignored small files is taken at startup
- Snapshot location: `log_dir/code_snapshots/`

## A common mistake

The call is not `log_step(metrics, step)`.

The correct order is:

1. `log_metrics({...})`
2. `log_step(step)`
