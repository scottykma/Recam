# ReCam

ReCam trains on the same chunk boundaries used by streaming generation.
Resolution, fps and chunk geometry come from the config's `geometry` section
(defaults `448x832 @ 16fps`). It is an independent algorithm package with its
own trainer, model, config, inference pipeline, and scripts. The WAN VAE, T5
text encoder, flow-matching scheduler, and weight loader it needs live under
`example/recam/backbone/`.

## Training Mix

- T2V/I2V/AR weights come from `data.task_probs`; the AR task is `Task.CONTINUE`
  in code, since it continues the accumulated sequence.
- One chunk is `num_frames` pixel frames, i.e. `new_frames = (num_frames - 1) / 4`
  VAE latents. T2V and I2V train the first chunk alone, `1 + new_frames` latents.
- AR mass is split uniformly over history depths `1..n_chunks-1`. One history
  chunk adds `num_frames - 1` pixel frames, so the maximum history is `n_chunks-1`
  chunks.
- Every global micro-step uses one synchronized task and history depth across all
  DP ranks. Rows are selected without replacement inside that global
  micro-batch.

## Window Layout

An AR window is four consecutive latent segments (first frame | history |
overlap | new). The dataset window holds `1 + new_frames*(n+1)` latents, but the
model input is `[first frame | main]`, where `main = overlap + new_frames` and
the overlap is its head:

| segment | latent frames | role |
|---|---|---|
| first frame | `1` | clean reference in `x` |
| history | `new_frames*n - overlap` | compressed memory only |
| overlap | `overlap` | head of `main` in `x`; its `t_context` copy is `y.ref_lat` |
| new | `new_frames` | target |

Overlap and new frames must remain inside the caption range; history may precede
it in the same video. `overlap=2` is the shipped recipe: it keeps
`history_frames` even at every depth, so `HistoryMLP` no longer spends a zero
frame on the left pad.

## HistoryMLP

Far history is aligned from the overlap side into groups of two latent frames, so
the newest frames keep their pairing at any depth and the oldest group takes the
left pad. Each `16x2x8x8=2048` pack is projected to one 5120-dimensional DiT token
by `Linear(2048,2048) -> SiLU -> Linear(2048,5120)`. Both linear layers use bias,
Xavier initialization, and zero bias initialization. The common padded latent
canvas is `56x104`, so each temporal group produces a `7x13=91` memory grid and
memory costs `91 * ceil(history_frames / 2)` tokens - at 448x832, 455 tokens per
history chunk, a 32x compression over the 14560 tokens the same frames would take
inline.

The immutable `geometry.history_pack_shape` and `geometry.history_pad_shape`
are shared by training, inference, and checkpoint validation. Native integer
RoPE coordinates use temporal block starts and spatial starts
`0,4,...,24` / `0,4,...,48`.

## Sequence Contract

```text
T2V: [main(F, N)]
I2V: [context(F) | main(F, N)]
AR:  [context(F) | memory(H) | main(O, N)]
```

`F` is the single first frame, `N` the `new_frames` generated latents, `O` the
`overlap` latents, and `H` the `history_frames` packed memory latents. The
overlap appears once in `x`, as the head of `main` at `t_main`; the same clean
latent and noise draw also back `y`'s `1..O` reference slots at `t_context`.

All tokens participate in self-attention. Only main tokens use text
cross-attention and the output head. Loss is velocity MSE over all main latents.

RoPE uses the same integer coordinates in training and inference, on one
consecutive axis anchored at the overlap: memory ends at `-1`, the main sequence
is `0..overlap+new_frames-1` (its first `overlap` slots are the overlap head),
and the first frame sits at `-(history_frames+1)`, one slot before the oldest
memory frame. Compact spatial coordinates are native patch starts in the common
padded `28x52` patch canvas.

## Channel Conditioning

ReCam trains at `in_dim: 36`: a 20-channel reference condition rides
in an extra input concatenated to `x` before patchification,
`y = [indicator(2) | sigma(2) | ref_lat(16)]`, with the same channel order and
sigma semantics used by the ReCam trainer.

- Frame 0 (the clean first frame): `indicator=1`, `sigma=1` (clean), `ref_lat` =
  the clean first frame.
- Frames `1..overlap` (AR only): `indicator=1`, `sigma=1-t_context/1000`,
  `ref_lat` = the overlap latent noised at `t_context`. These slots align with
  the main head in `x`, so the same frame reaches the model at `t_main`
  (sequence) and `t_context` (channels) from one clean latent and one noise draw.
- All other frames are zero. T2V uses an all-zero `y`.

Inference always builds the same 20-channel `y`; there is no 16-channel path.

## Context Parallelism

`parallel.cp_size` may be greater than 1. CP shards the packed attention
sequence across a `(dp, cp)` mesh and, inside each self-attention, swaps that
sequence split for a head split, so every rank still attends over the whole
sequence. There is no ring pass, no online-softmax rescaling and no dropped
token: the forward is bit-exact against `cp_size=1`, and the backward differs
only by floating-point re-association of the split reductions.

**Hard constraint: the packed sequence length must be divisible by `cp_size` at
every sampled history depth.** The all-to-all deadlocks on an uneven split, so
`ExpConfig` rejects such a geometry up front rather than hanging at step 0. The
packed history contributes `(padded_h/ph)*(padded_w/pw)` tokens per `pt` latent
frames, so the parity of the total is set by the geometry:

| geometry | `S(K)` | even? | cp |
|---|---|---|---|
| `448x832`, canvas `(2,56,104)` | `17472 + 455K` | odd when K is odd | 2 blocked |
| `512x896`, canvas `(2,64,112)` | `21504 + 560K` | always | 2, 4, 8 |

At `448x832`, `cp_size=1` carries the same 30 s window without the parity
constraint. Two consequences:

- `cp_size` divides `global_batch_size` (`core/runner/trainer.py`), so a CP run
  needs `optimizer.gradient_accumulation_steps` adjusted to keep the effective
  batch.
- A CP run is not resume-compatible with a `cp_size=1` run of the same
  experiment: the sample stream differs (`DataManager` draws one batch per
  `cp_size * pre_encode_batches` reads). Use a fresh `exp_id`.

The camera window is the only CP-sensitive part of the model. `SelfAttention`
slices `q[:, -window:]`, which assumes the window is the sequence tail, so
`ReCamModel.forward` pre-slices the per-frame PRoPE matrices down to each
rank's own tail before sharding.

## Geometry and Inference

Inference reads `geometry` and `camera` from
`configs/recam/recam.yaml`, the same file that supplied the training
recipe. Nothing about the layout is taken from the checkpoint: the recorded
`recam_geometry` runtime state is training provenance only.

Because both sides share one file, the `geometry:` section cannot drift from
the weights it evaluates. If you split it into a separate inference file later,
keep `history_pack_shape` identical: it changes HistoryMLP parameter shapes and
fails at load, while `history_pad_shape`, `num_frames`, `overlap` and the
spatial size only move RoPE coordinates and silently change the picture.

## Run

The repo ships two 121-frame clips, a demo parquet and two first-frame stills
(one person, one cat) under `assets/demo/`. See
[`assets/demo/README.md`](../../assets/demo/README.md) for the one-GPU smoke
commands; override `--data.n-chunks 3` because the demo clips are shorter than
the 13-chunk recipe.

Training:

```bash
MANIFEST=/path/to/parquets_list.txt \
  CONFIG_FILE=configs/recam/recam.yaml \
  bash scripts/recam/run_train.sh
```

`checkpoint.pretrained_path` starts a new optimizer, scheduler, and global step
from the selected weights; `checkpoint.resume_from` restores a full training
checkpoint. ReCam `model/` and `ema/` DCP directories are loaded
completely, including HistoryMLP. `data.task_probs` stores the T2V/I2V/AR
weights. For a later curriculum stage, stop at a complete checkpoint and raise
`data.n_chunks` up to the supported limit.

Inference from a released checkpoint (`CKPT_PATH`) or a training run:

```bash
CKPT_PATH=/path/to/recam-14b \
N_CHUNKS=5 STEPS=40 \
  bash scripts/recam/run_infer.sh
```

`CKPT_PATH` accepts the checkpoint root, its `model/` or `ema/` directory, or a
single `.pt`/`.safetensors` file. For a checkpoint under `runs/`, replace it with
`EXP_ID=recam CKPT_ID=step_0002000`.

`CONFIG_FILE` defaults to `configs/recam/recam.yaml`; that file, not the
checkpoint, decides the resolution and window layout. There is no sampler to
choose (Euler is the only solver) and no pipeline to switch: camera control is
an optional `--camera-trajectory-file` on the one pipeline.

The default prompt source is `example.recam.infer.prompts.demo`, which ships
two cases backed by `assets/demo/person_lake.png` and `assets/demo/cat.png`.
Replace their prompts or pass your own module with `PROMPT_SOURCE`. To continue
an existing video instead of starting from a still, clear the prompt source and
pass `--context-video`:

```bash
CKPT_PATH=/path/to/recam-14b N_CHUNKS=4 PROMPT_SOURCE= \
  bash scripts/recam/run_infer.sh \
  --prompt "..." --context-video <16fps clip>.mp4
```

The clip must be 16 fps and hold `1 + 40n` frames (41 / 81 / 121 / ...); the
first `n` chunks are encoded as context and are part of the written video, while
`N_CHUNKS` more are generated on top. Outputs use
`outputs/<EXP_ID>/step_<NNNNNNN>/<model|ema>/` for run checkpoints, or
`outputs/<EXP_ID>/<checkpoint-name>/[<model|ema>/]` for released weights.

Camera-control training uses the camera launcher:

```bash
CAMERA_PARQUET_DIR=/path/to/camera_parquets \
  CONFIG_FILE=configs/recam/recam.yaml \
  bash scripts/recam/camera/run_train.sh
```

The same YAML carries the `camera:` section read by camera-control training
and by camera-trajectory inference. The camera warm start must already be a
36-channel ReCam checkpoint at the same geometry; set
`checkpoint.pretrained_path` before launching.
