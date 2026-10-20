#!/usr/bin/env bash
set -euo pipefail

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  cat <<'EOF'
ReCam camera training

Environment:
  CONFIG_FILE          training YAML
  TRAINER_MODULE       trainer entry module
  CAMERA_PARQUET_DIR   directory of camera parquet shards
  CAMERA_PARQUET_DIRS  colon-separated pool directories (overrides the above)
  MANIFEST             explicit parquet list, skipping directory expansion
  RUNS_DIR             checkpoint/log root

Extra arguments are forwarded to the common launcher.
EOF
  REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)
  exec bash "$REPO_ROOT/scripts/recam/run_train.sh" --help
fi

REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)
COMMON_LAUNCHER="$REPO_ROOT/scripts/recam/run_train.sh"
CONFIG_FILE=${CONFIG_FILE:-configs/recam/recam.yaml}
TRAINER_MODULE=${TRAINER_MODULE:-example.recam.trainers.camera_trainer}

if [[ -z "${MANIFEST:-}" ]]; then
  if [[ -z "${CAMERA_PARQUET_DIR:-}" && -z "${CAMERA_PARQUET_DIRS:-}" ]]; then
    echo "Set CAMERA_PARQUET_DIR or CAMERA_PARQUET_DIRS." >&2
    exit 1
  fi
  CAMERA_PARQUET_DIRS=${CAMERA_PARQUET_DIRS:-${CAMERA_PARQUET_DIR:-}}
  IFS=':' read -r -a CAMERA_DIRS <<< "$CAMERA_PARQUET_DIRS"
  PARQUETS=()
  for camera_dir in "${CAMERA_DIRS[@]}"; do
    [[ -n "$camera_dir" ]] || continue
    [[ -d "$camera_dir" ]] || { echo "Camera parquet dir not found: $camera_dir" >&2; exit 1; }
    mapfile -t FOUND < <(ls "$camera_dir"/*.parquet 2>/dev/null | sort)
    [[ ${#FOUND[@]} -gt 0 ]] || { echo "No camera parquets in $camera_dir" >&2; exit 1; }
    PARQUETS+=("${FOUND[@]}")
  done
  [[ ${#PARQUETS[@]} -gt 0 ]] || { echo "No camera parquets in $CAMERA_PARQUET_DIRS" >&2; exit 1; }
  MANIFEST=$(mktemp)
  trap 'rm -f "$MANIFEST"' EXIT
  printf '%s\n' "${PARQUETS[@]}" > "$MANIFEST"
  echo "Camera parquet manifest: ${#PARQUETS[@]} files from $CAMERA_PARQUET_DIRS"
fi

CONFIG_FILE="$CONFIG_FILE" TRAINER_MODULE="$TRAINER_MODULE" MANIFEST="$MANIFEST" \
  bash "$COMMON_LAUNCHER" "$@"
