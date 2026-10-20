# Checkpointer

Saves, restores, and loads pretrained weights.

## What it handles

- Model
- Optimizer
- LR scheduler
- EMA
- `global_step`

## Supported pretrained formats

- Plain `*.pt`
- `*.safetensors`
- HuggingFace sharded directories
- DCP directories (containing `.metadata`)

If weight keys do not match, use a `state_dict_adapter` to rename them before
loading.

## Save behavior

### Regular training

- `is_latest=False`
  - Write `step_000xxxx/`
  - Then point `latest` at it
- `is_latest=True`
  - Write `latest_temp/` first
  - Atomically replace `latest/` when finished

### LoRA training

- Does not use DCP sharding
- Rank 0 saves directly:
  - `model.pt`
  - `optimizer.pt`
  - `ema.pt`

## Directory layout

Single-model training:

```text
runs/<exp>/checkpoints/
  latest/
  step_0000500/
```

## Resume rules

- `Trainer` automatically resumes when `latest` exists
- Resume restores not only weights but also the optimizer, LR scheduler, EMA,
  and sampler step

## Common points

- `checkpoint.pretrained_path` only initializes weights; it is not a resume
- Regular checkpoints leave `step_xxx` directories behind; latest-only saves do
  not

## Cleaning old checkpoints

The Checkpointer has no retention limit, so `step_xxx` directories accumulate.
Clean them up outside the training loop as needed, and keep the step that
`latest` points to.
