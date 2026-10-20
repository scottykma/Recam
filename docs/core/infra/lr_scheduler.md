# LR Scheduler

The learning-rate curve always has three phases:

1. warmup
2. stable
3. decay

## Entry point

`build_lr_scheduler(optimizer, max_steps, warmup_steps, decay_steps, min_lr_factor, lr_decay_type)`

## Rules

- `stable_steps = max_steps - warmup_steps - decay_steps`
- A negative value raises an error
- The decay phase supports:
  - `linear`
  - `sqrt`
  - `cosine`

## Common configurations

- `warmup_steps=0`
  - The peak learning rate applies from the first step
- `decay_steps=0`
  - No decay phase; the rate stays constant after warmup
- `min_lr_factor=0.1`
  - Decays to at most 10% of the peak rate

## Note

Some configs write `decay_type="constant"`, but what actually takes effect is
`decay_steps=0`; it is not a separate decay algorithm.
