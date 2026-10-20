# Parallelism

The framework uses two meshes:

- `model_mesh`
  - Used for FSDP / HSDP
- `data_mesh`
  - Created only when `cp_size > 1`; used for CP and data broadcasting

## init_parallel

- Default: `model_mesh = (1, world_size)`
- With HSDP: `model_mesh = (world_size // shard_size, shard_size)`
- With CP: `data_mesh = (world_size // cp_size, cp_size)`
- Enabling CP also initializes the global `CP_MANAGER`

## apply_fsdp

- First finds submodules listed in `wrap_submodules` and calls `fully_shard` on
  them
- If the target submodule is a `ModuleList`, each layer in the list is wrapped
  separately
- Finally calls `fully_shard` on the whole model

## CPManager

Three main interfaces:

- `split_input(x, dim, cache_k=None)`
  - Splits a tensor into CP shards along one dimension
- `gather_output(y, dim, cache_k=None)`
  - Gathers shards back into the full tensor
- `SeqAllToAll(x, scatter_dim, gather_dim, cache_k=None)`
  - Performs a cross-dimension all-to-all reshuffle

`cache_k` records the original shape / stride for the later gather or all-to-all.

## Constraints

- `world_size % cp_size == 0`
- If the model shards CP by head, it usually also requires
  `num_heads % cp_size == 0`
