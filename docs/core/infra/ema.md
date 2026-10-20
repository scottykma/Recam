# EMA

Tracks an exponential moving average of the trainable parameters.

## Behavior

- Copies every parameter with `requires_grad=True` at initialization
- `copy_from(model)`: overwrite the EMA with the current model parameters
- `step(model)`: update the EMA by `decay`
- `copy_to(model)`: write the EMA weights back into the model
- `store()` / `restore()`: temporarily swap weights for evaluation

## How it fits into the framework

- `Trainer` updates it automatically during training according to the config
- Before online evaluation, it calls `store()` on the current weights, switches
  to the EMA, and restores the weights after evaluation
- Checkpoints save the EMA alongside the model

## One detail

`state_dict()` strips the prefixes introduced by FSDP / compile to simplify
saving and restoring.
