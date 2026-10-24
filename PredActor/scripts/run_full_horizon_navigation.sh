#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 5 || $# -gt 6 ]]; then
  echo "Usage: $0 CHECKPOINT CONFIG OUTPUT_DIR DEVICE CHECKPOINT_REPLICA [TEXT_EMBEDDING_ARCHIVE]" >&2
  exit 2
fi

checkpoint_path="$1"
config_path="$2"
output_dir="$3"
device_name="$4"
checkpoint_replica="$5"
text_embedding_archive="${6:-}"

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
predactor_dir="$(cd "$script_dir/.." && pwd)"
repository_dir="$(cd "$predactor_dir/.." && pwd)"

mkdir -p "$output_dir/runtime"

run_case() {
  local case_id="$1"
  local prompt="$2"
  local trajectory="$output_dir/${case_id}-seed42.npz"
  local runtime_dir="$output_dir/runtime/${case_id}"
  local embedding_args=()

  if [[ -n "$text_embedding_archive" ]]; then
    embedding_args+=(--text-embedding-archive "$text_embedding_archive")
  fi

  if [[ -f "$trajectory" ]]; then
    echo "[resume] keep $trajectory"
    return
  fi

  python "$predactor_dir/cond_eval.py" \
    --checkpoint "$checkpoint_path" \
    --config "$config_path" \
    --output_dir "$runtime_dir" \
    --device "$device_name" \
    --checkpoint-replica "$checkpoint_replica" \
    --text-command "$prompt" \
    --seed 42 \
    --trajectory-rollout "$trajectory" \
    --rollout-seconds 10 \
    --no-paper-terminal-gates \
    "${embedding_args[@]}" \
    --headless \
    --no-web-ui
}

cd "$repository_dir"
run_case walk-straight "walk straight"
run_case turn-left-walking "turn left while walking"
run_case turn-right-walking "turn right while walking"

python "$predactor_dir/scripts/evaluate_walk_metrics.py" \
  "$output_dir"/*-seed42.npz \
  --output "$output_dir/summary.json"
