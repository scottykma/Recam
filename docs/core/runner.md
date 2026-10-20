# Trainer

The single-model training entry point. The contract is simple: you write the
task logic, and the framework owns the training infrastructure.

## Initialization order

1. Initialize distributed runtime and device context
2. Build the logger
3. Call `build_dataset_and_sampler()`, then build the dataloader
4. Build the `Checkpointer`
5. Call `build_model()`, then automatically apply LoRA / gradient checkpointing /
   FSDP / `torch.compile`
6. Build the optimizer, LR scheduler, and EMA
7. Load pretrained weights or `latest`
8. Build the `DataManager`

## Required methods

- `build_dataset_and_sampler() -> (dataset, sampler)`
- `build_model() -> nn.Module`
- `forward_step(encoded_data) -> (loss, metrics)`

## Methods you will usually implement

- `encode_batch(batch) -> dict`
  - Convert a raw batch into the tensor dict used for training
  - Return the batch unchanged when no pre-encoding is needed
- `enter_encode()` / `enter_train()`
  - Move the encoder on and off the GPU and clear caches
- `eval() -> dict[str, float]`
  - Return `{}` when evaluation is not implemented, or set
    `eval.online_eval=False`
- `get_pretrained_state_dict_adapter()`
  - Adapt pretrained weight keys when they do not match the model

## What the main loop does

- Refreshes `self.rng` every step
- Pulls an already-encoded batch from the `DataManager`
- Runs one or more micro-steps according to `gradient_accumulation_steps`
- Clips gradients, steps the optimizer and LR scheduler, and zeroes gradients
- Optionally updates EMA
- Logs, saves checkpoints, and runs online evaluation at the configured intervals

## Key constraints

- `build_model()` returns a model that has not been moved to a device; the
  Trainer handles `.to(device)` and FSDP
- The argument to `forward_step()` is exactly the output of `encode_batch()`
- When `eval.online_eval=True`, `eval()` must be implemented or evaluation fails

## Minimal skeleton

```python
class MyTrainer(Trainer):
    def build_dataset_and_sampler(self):
        return dataset, sampler

    def build_model(self):
        return model

    def encode_batch(self, batch):
        return {"x": batch["x"], "y": batch["y"]}

    def forward_step(self, encoded_data):
        pred = self.model(encoded_data["x"])
        loss = ((pred - encoded_data["y"]) ** 2).mean()
        return loss, {"mse": float(loss.detach().item())}

    @torch.no_grad()
    def eval(self):
        return {}
```

## Commonly used configuration

- `core/config/train_schema.py`
- `dataloader.*`: batch size, workers, shuffling
- `data_manager.*`: pre-encoding cache depth
- `optimizer.*`: learning rate, gradient accumulation, gradient clipping,
  gradient checkpointing
- `parallel.*`: FSDP / HSDP / CP
- `checkpoint.*`: pretrained weights, save intervals, meta-device initialization
