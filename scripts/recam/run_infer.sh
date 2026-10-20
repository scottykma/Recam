#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$REPO_ROOT"

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  cat <<'EOF'
ReCam inference launcher

Checkpoint (either form):
  CKPT_PATH       path to a released or downloaded checkpoint: a directory
                  holding model/ and/or ema/, one of those weight directories,
                  or a single .pt / .safetensors file
  WEIGHTS         model or ema when both live under CKPT_PATH (default: ema)

  Without CKPT_PATH, the launcher reads a training run:
  EXP_ID          run name (default: recam)
  CKPT_ID         immutable checkpoint under runs/, e.g. step_0002000
  CKPT_ROOT       full checkpoint root; overrides CKPT_ID
  RUNS_DIR        checkpoint root (default: runs)

Other environment:
  CONFIG_FILE     inference recipe with geometry and camera intrinsics
  PROMPT_SOURCE   prompt module; set it empty to run one case from
                  --prompt/--image/--context-video
  CASES, N_CHUNKS, STEPS, CFG_SCALE, SIGMA_SHIFT, CONTEXT_SIGMA_SHIFT, SEED
  GPUS, NPROC_PER_NODE, PYTHON_BIN, TORCHRUN_BIN, MASTER_PORT
  OUT_DIR, OUT_SUBDIR

Additional options are forwarded to example.recam.infer.run.

Examples:
  CKPT_PATH=weights/recam-14b N_CHUNKS=4 bash scripts/recam/run_infer.sh
  EXP_ID=recam CKPT_ID=step_0002000 bash scripts/recam/run_infer.sh
EOF
  exit 0
fi

PYTHON_BIN=${PYTHON_BIN:-python}
TORCHRUN_BIN=${TORCHRUN_BIN:-torchrun}
RUNS_DIR=${RUNS_DIR:-runs}
CONFIG_FILE=${CONFIG_FILE:-configs/recam/recam.yaml}
EXP_ID=${EXP_ID:-recam}

[[ -f "$CONFIG_FILE" ]] || { echo "Inference config not found: $CONFIG_FILE" >&2; exit 1; }

# A released checkpoint can be used directly; the runs/ layout stays as a
# fallback for checkpoints produced by the training launcher.
if [[ -n "${CKPT_PATH:-}" ]]; then
  CHECKPOINT_SOURCE=$CKPT_PATH
elif [[ -n "${CKPT_ROOT:-}" ]]; then
  CHECKPOINT_SOURCE=$CKPT_ROOT
else
  : "${CKPT_ID:?Set CKPT_PATH (a released checkpoint) or CKPT_ID (a checkpoint under runs/)}"
  CHECKPOINT_SOURCE=$RUNS_DIR/$EXP_ID/checkpoints/$CKPT_ID
fi

has_weights() {
  local dir=$1
  if [[ -f "$dir/.metadata" ]]; then return 0; fi
  if [[ -f "$dir/diffusion_pytorch_model.safetensors.index.json" ]]; then return 0; fi
  if compgen -G "$dir/*.pt" >/dev/null; then return 0; fi
  if compgen -G "$dir/*.pth" >/dev/null; then return 0; fi
  if compgen -G "$dir/*.safetensors" >/dev/null; then return 0; fi
  return 1
}

WEIGHTS_LABEL=""
CKPT_NAME=""
if [[ -f "$CHECKPOINT_SOURCE" ]]; then
  CKPT=$CHECKPOINT_SOURCE
  CKPT_ROOT=$(dirname "$CHECKPOINT_SOURCE")
  CKPT_NAME=$(basename "${CHECKPOINT_SOURCE%.*}")
  if [[ -n "${WEIGHTS:-}" ]]; then
    echo "CKPT_PATH is a single weight file; ignoring WEIGHTS=$WEIGHTS" >&2
  fi
  WEIGHTS=""
elif [[ -d "$CHECKPOINT_SOURCE" ]]; then
  CKPT_ROOT=${CHECKPOINT_SOURCE%/}
  SOURCE_NAME=$(basename "$CKPT_ROOT")
  if [[ "$SOURCE_NAME" == "model" || "$SOURCE_NAME" == "ema" ]]; then
    if [[ -n "${WEIGHTS:-}" && "$WEIGHTS" != "$SOURCE_NAME" ]]; then
      echo "CKPT_PATH already points at $SOURCE_NAME/; ignoring WEIGHTS=$WEIGHTS" >&2
    fi
    WEIGHTS=$SOURCE_NAME
    CKPT=$CKPT_ROOT
    CKPT_ROOT=$(dirname "$CKPT_ROOT")
    WEIGHTS_LABEL=$WEIGHTS
    CKPT_NAME=$(basename "$CKPT_ROOT")
  elif [[ -n "${WEIGHTS:-}" ]]; then
    [[ -d "$CKPT_ROOT/$WEIGHTS" ]] || {
      echo "WEIGHTS=$WEIGHTS not found under $CKPT_ROOT" >&2; exit 1; }
    CKPT=$CKPT_ROOT/$WEIGHTS
    WEIGHTS_LABEL=$WEIGHTS
    CKPT_NAME=$SOURCE_NAME
  elif [[ -d "$CKPT_ROOT/ema" ]]; then
    WEIGHTS=ema
    CKPT=$CKPT_ROOT/ema
    WEIGHTS_LABEL=$WEIGHTS
    CKPT_NAME=$SOURCE_NAME
  elif [[ -d "$CKPT_ROOT/model" ]]; then
    WEIGHTS=model
    CKPT=$CKPT_ROOT/model
    WEIGHTS_LABEL=$WEIGHTS
    CKPT_NAME=$SOURCE_NAME
  else
    # A flat weights directory: DCP shards, *.pt, or a safetensors index.
    CKPT=$CKPT_ROOT
    WEIGHTS=""
    CKPT_NAME=$SOURCE_NAME
  fi
else
  echo "Checkpoint not found: $CHECKPOINT_SOURCE" >&2
  exit 1
fi

if [[ -d "$CKPT" ]] && ! has_weights "$CKPT"; then
  echo "No ReCam weights found at: $CKPT" >&2
  echo "Expected a DCP directory (.metadata), .pt/.safetensors files, or a safetensors index." >&2
  exit 1
fi

# The step is metadata for the output layout and the manifest. Released weights
# may not ship training_state.pt, so fall back to a step_<N> directory name and
# otherwise leave the step unset.
STEP=""
STATE_FILE=${CKPT_ROOT}/training_state.pt
if [[ -f "$STATE_FILE" ]]; then
  STEP=$(STATE_FILE="$STATE_FILE" "$PYTHON_BIN" -c \
    'import os, torch; print(int(torch.load(os.environ["STATE_FILE"], map_location="cpu", weights_only=False)["global_step"]))')
fi
if [[ -z "$STEP" && "$(basename "$CKPT_ROOT")" =~ ^step_([0-9]+)$ ]]; then
  STEP=$((10#${BASH_REMATCH[1]}))
fi

if [[ -n "$STEP" ]]; then
  OUT_TAG=$(printf 'step_%07d' "$STEP")
else
  OUT_TAG=$(printf '%s' "${CKPT_NAME:-checkpoint}" | tr -cs 'A-Za-z0-9._-' '-')
  OUT_TAG=${OUT_TAG#-}
  OUT_TAG=${OUT_TAG%-}
  if [[ -z "$OUT_TAG" ]]; then OUT_TAG=checkpoint; fi
fi

if [[ -z "${OUT_DIR:-}" ]]; then
  OUT_DIR=outputs/$EXP_ID/$OUT_TAG
  if [[ -n "$WEIGHTS_LABEL" ]]; then
    OUT_DIR+=/$WEIGHTS_LABEL
  fi
  if [[ -n "${OUT_SUBDIR:-}" ]]; then
    OUT_DIR+=/$OUT_SUBDIR
  fi
fi

export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export TOKENIZERS_PARALLELISM=false

MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
MASTER_PORT=${MASTER_PORT:-12631}
GPUS=${GPUS:-0,1,2,3,4,5,6,7}
NPROC_PER_NODE=${NPROC_PER_NODE:-8}
PROMPT_SOURCE=${PROMPT_SOURCE-example.recam.infer.prompts.demo}
N_CHUNKS=${N_CHUNKS:-4}
STEPS=${STEPS:-40}
CFG_SCALE=${CFG_SCALE:-3.5}
SIGMA_SHIFT=${SIGMA_SHIFT:-5.0}
CONTEXT_SIGMA_SHIFT=${CONTEXT_SIGMA_SHIFT:-2.0}
SEED=${SEED:-42}
CASES=${CASES:-}

echo "Checkpoint: $CKPT"
echo "Output:     $OUT_DIR"

ARGS=(
  --config-file "$CONFIG_FILE"
  --n-chunks "$N_CHUNKS"
  --num-inference-steps "$STEPS"
  --cfg-scale "$CFG_SCALE"
  --sigma-shift "$SIGMA_SHIFT"
  --context-sigma-shift "$CONTEXT_SIGMA_SHIFT"
  --seed "$SEED"
  --ckpt-path "$CKPT"
  --output-dir "$OUT_DIR"
)
if [[ -n "$STEP" ]]; then
  ARGS+=(--checkpoint-step "$STEP")
fi
if [[ -n "$PROMPT_SOURCE" ]]; then
  ARGS+=(--prompt-source "$PROMPT_SOURCE")
fi
if [[ -n "$CASES" ]]; then
  ARGS+=(--cases "$CASES")
fi

CUDA_VISIBLE_DEVICES="$GPUS" "$TORCHRUN_BIN" \
  --nnodes=1 \
  --nproc_per_node="$NPROC_PER_NODE" \
  --node_rank=0 \
  --master_addr="$MASTER_ADDR" \
  --master_port="$MASTER_PORT" \
  -m example.recam.infer.run \
  "${ARGS[@]}" \
  "$@"
