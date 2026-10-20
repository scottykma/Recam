# Experiment configurations

`configs/recam/recam.yaml` is the single ReCam recipe. It drives all three
entry points:

- training reads the experiment sections (`exp_id`, optimizer, strategy, data,
  geometry, ...);
- camera-control training reads its `camera:` section;
- inference reads `geometry:` and `camera:` from the same file, so the layout
  cannot drift from the weights it evaluates.

Python dataclasses under `example/recam/config/` define the schema and
model-specific defaults. Launchers under `scripts/` point at this file and
accept `CONFIG_FILE=/path/to/override.yaml` when a different recipe is needed.
