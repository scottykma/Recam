# Documentation

## Where to start

- Integrating a new task with the framework: [`core/runner.md`](core/runner.md)
- Example entry point: [`example/README.md`](example/README.md)

## Core modules

- Training entry point: [`core/runner.md`](core/runner.md)
- Data pipeline: [`core/data/data_manager.md`](core/data/data_manager.md),
  [`core/data/dataloader.md`](core/data/dataloader.md),
  [`core/data/sampler.md`](core/data/sampler.md)
- Infrastructure:
  - Parallelism: [`core/infra/parallelism.md`](core/infra/parallelism.md)
  - Checkpointing: [`core/infra/checkpointer.md`](core/infra/checkpointer.md)
  - Logging: [`core/infra/logger.md`](core/infra/logger.md)
  - EMA: [`core/infra/ema.md`](core/infra/ema.md)
  - LR schedule: [`core/infra/lr_scheduler.md`](core/infra/lr_scheduler.md)
  - Gradient checkpointing: [`core/infra/gradient_checkpointing.md`](core/infra/gradient_checkpointing.md)

## Code map

- `core/config/train_schema.py`: shared configuration for single-model training
- `example/<task>/config/*.py`: task-specific data, model, and strategy configs
- `scripts/*/*.sh`: training and inference launchers
