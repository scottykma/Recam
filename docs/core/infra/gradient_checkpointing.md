# Gradient checkpointing

Entry point: `apply_gradient_checkpointing(model, checkpoint_module_type)`

## Behavior

- Wraps submodules with the `NO_REENTRANT` checkpoint wrapper
- Matches by class name, not by module path

## Configuration

Provide a list of module class names in the config, for example:

```python
optimizer.gradient_checkpoint_modules = ["DiTBlock", "Head"]
```

Only modules whose class name matches are wrapped.
