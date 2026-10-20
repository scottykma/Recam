#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$REPO_ROOT"

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  cat <<'EOF'
ReCam training launcher

Environment:
  CONFIG_FILE      training YAML (default: the recam recipe)
  TRAINER_MODULE   trainer entry module
  MANIFEST         text file holding one parquet path per line
  CACHE_DIR        directory holding parquets_list.txt, used when MANIFEST is unset
  RUNS_DIR         checkpoint/log root (default: runs)
  TORCHRUN_BIN, NNODES, NODE_RANK, MASTER_ADDR, MASTER_PORT,
  GPUS_PER_NODE    torchrun settings

Extra arguments are forwarded to the trainer.
EOF
  exit 0
fi

TORCHRUN_BIN=${TORCHRUN_BIN:-torchrun}
CONFIG_FILE=${CONFIG_FILE:-configs/recam/recam.yaml}
TRAINER_MODULE=${TRAINER_MODULE:-example.recam.trainers.trainer}
RUNS_DIR=${RUNS_DIR:-runs}

if [[ -z "${MANIFEST:-}" ]]; then
  if [[ -n "${CACHE_DIR:-}" ]]; then
    MANIFEST="${CACHE_DIR}/parquets_list.txt"
  else
    echo "Set MANIFEST (a parquet list) or CACHE_DIR (a directory containing parquets_list.txt)." >&2
    exit 1
  fi
fi

[[ -f "$CONFIG_FILE" ]] || { echo "Config not found: $CONFIG_FILE" >&2; exit 1; }
[[ -f "$MANIFEST" ]] || { echo "Manifest not found: $MANIFEST" >&2; exit 1; }

mapfile -t PARQUETS < <(sed '/^[[:space:]]*$/d' "$MANIFEST")
[[ ${#PARQUETS[@]} -gt 0 ]] || { echo "Manifest is empty: $MANIFEST" >&2; exit 1; }
for parquet in "${PARQUETS[@]}"; do
  [[ -f "$parquet" ]] || { echo "Parquet not found: $parquet" >&2; exit 1; }
done

export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export TOKENIZERS_PARALLELISM=false

NNODES=${NNODES:-1}
MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
MASTER_PORT=${MASTER_PORT:-12449}
NODE_RANK=${NODE_RANK:-0}
GPUS_PER_NODE=${GPUS_PER_NODE:-8}

echo "[node $NODE_RANK] ${#PARQUETS[@]} parquets from $MANIFEST"
"$TORCHRUN_BIN" \
  --nnodes="$NNODES" \
  --nproc_per_node="$GPUS_PER_NODE" \
  --node_rank="$NODE_RANK" \
  --master_addr="$MASTER_ADDR" \
  --master_port="$MASTER_PORT" \
  -m "$TRAINER_MODULE" \
  --config-file "$CONFIG_FILE" \
  --misc.runs-dir "$RUNS_DIR" \
  "$@" \
  --data.parquet-path-list "${PARQUETS[@]}"
