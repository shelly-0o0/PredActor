#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${PREDACTOR_REPO_ROOT:-/root/Predactor/PredActor-DAgger}"
DATA_ROOT="${PREDACTOR_DATA_ROOT:-/data/PredActor}"
ROUND_ROOT="${PREDACTOR_ROUND_ROOT:-$DATA_ROOT/dagger/paper_round_01_full_v2}"
COLLECTION_PYTHON="${PREDACTOR_COLLECTION_PYTHON:-/root/miniconda3/envs/predactor-collection/bin/python}"
TRAIN_PYTHON="${PREDACTOR_TRAIN_PYTHON:-/root/Predactor/PredActor/.venv/bin/python}"
PI0="${PREDACTOR_LEARNER_CHECKPOINT:-$DATA_ROOT/outputs/October-23-14-13-57-g1prdp_cond_diffuse-pi0-d0-seed42-final/checkpoints/latest.ckpt}"
TEACHER="$DATA_ROOT/Artifacts/checkpoints/rl_tracker/model_75000.pt"
MOTIONS="$DATA_ROOT/Artifacts/motions/g1_152fk"
D0="$DATA_ROOT/Artifacts/dataset/g1_152fk/motion_dyed_text_t0.01.zarr"
CONDITIONS="$DATA_ROOT/Artifacts/dagger/d0_phase_conditions_v1.npz"
MANIFEST="$ROUND_ROOT/manifest.yaml"
LOG_DIR="$ROUND_ROOT/logs"

CANDIDATES_PER_MOTION="${PREDACTOR_CANDIDATES_PER_MOTION:-3200}"
ORIGINS_PER_MOTION="${PREDACTOR_ORIGINS_PER_MOTION:-2000}"
MIN_D1_WINDOWS="${PREDACTOR_MIN_D1_WINDOWS:-250000}"
MIN_D1_FRACTION="${PREDACTOR_MIN_D1_FRACTION:-0.10}"

# GPU0 resumes the legacy directory containing motions 0-83 and collects
# through motion 91.  The other seven GPUs receive mutually exclusive ranges.
ORIGIN_DIRS=(
  "$ROUND_ROOT/origins"
  "$ROUND_ROOT/origin_shards/shard01"
  "$ROUND_ROOT/origin_shards/shard02"
  "$ROUND_ROOT/origin_shards/shard03"
  "$ROUND_ROOT/origin_shards/shard04"
  "$ROUND_ROOT/origin_shards/shard05"
  "$ROUND_ROOT/origin_shards/shard06"
  "$ROUND_ROOT/origin_shards/shard07"
)
ORIGIN_MOTIONS=(
  "$(seq -s, 0 91)"
  "$(seq -s, 92 100)"
  "$(seq -s, 101 109)"
  "$(seq -s, 110 118)"
  "$(seq -s, 119 127)"
  "$(seq -s, 128 136)"
  "$(seq -s, 137 144)"
  "$(seq -s, 145 151)"
)

cd "$REPO_ROOT"
mkdir -p "$ROUND_ROOT" "$LOG_DIR" "$ROUND_ROOT/origin_shards" "$ROUND_ROOT/branch_shards"
for path in "$COLLECTION_PYTHON" "$TRAIN_PYTHON" "$PI0" "$TEACHER" "$MOTIONS" "$D0" "$CONDITIONS"; do
  test -e "$path" || { echo "missing required input: $path" >&2; exit 2; }
done

export PYTHONPATH="$REPO_ROOT/PredActor:$REPO_ROOT/PredActor/collection/source/rl_tracker:$REPO_ROOT/PredActor/collection/vendor/rsl_rl-modular-normed:${PYTHONPATH:-}"
export PREDACTOR_ROOT="$REPO_ROOT/PredActor"
export PREDACTOR_ASSET_ROOT="$REPO_ROOT/PredActor/assets"

run_origin_shard() {
  local gpu="$1" output="$2" motions="$3" log="$4"
  local runtime="$ROUND_ROOT/runtime/origin$(printf '%02d' "$gpu")"
  # IsaacLab's URDF converter derives its /tmp USD suffix from Python's random
  # stream.  A unique seed prevents simultaneous Kit processes from selecting
  # the same conversion directory; it also gives independent learner rollouts.
  local seed=$((42 + gpu))
  if test -f "$output/origin_receipt.json"; then
    echo "[origin shard $gpu] receipt exists; skipping"
    return 0
  fi
  mkdir -p "$output" "$runtime/tmp" "$runtime/kit-data" "$runtime/cache"
  echo "[origin shard $gpu] motions=$motions output=$output"
  CUDA_VISIBLE_DEVICES="$gpu" \
  TMPDIR="$runtime/tmp" \
  OMNI_KIT_APP_DATA="$runtime/kit-data" \
  XDG_CACHE_HOME="$runtime/cache" \
  "$COLLECTION_PYTHON" \
    PredActor/collection/scripts/data_collection/collect_paper_dagger_origins.py \
    --learner-checkpoint "$PI0" --motion-root "$MOTIONS" \
    --phase-conditions "$CONDITIONS" --output "$output" \
    --motion-ids "$motions" --candidates-per-motion "$CANDIDATES_PER_MOTION" \
    --branch-every 6 --max-episode-steps 240 --max-attempts-per-motion 5000 \
    --device cuda:0 --resume --seed "$seed" --headless >"$log" 2>&1
}

echo "[parallel] launching eight learner-origin shards"
pids=()
for gpu in $(seq 0 7); do
  run_origin_shard "$gpu" "${ORIGIN_DIRS[$gpu]}" "${ORIGIN_MOTIONS[$gpu]}" \
    "$LOG_DIR/origin_shard$(printf '%02d' "$gpu").log" &
  pids+=("$!")
done
failed=0
for pid in "${pids[@]}"; do
  wait "$pid" || failed=1
done
test "$failed" -eq 0 || { echo "one or more origin shards failed" >&2; exit 3; }

total_origins=0
for directory in "${ORIGIN_DIRS[@]}"; do
  test -f "$directory/origin_receipt.json" || {
    echo "missing origin receipt: $directory" >&2; exit 3;
  }
  count=$(find "$directory" -maxdepth 1 -name 'origin_*.npz' | wc -l)
  total_origins=$((total_origins + count))
done
expected_origins=$((152 * CANDIDATES_PER_MOTION))
test "$total_origins" -eq "$expected_origins" || {
  echo "origin count mismatch: expected $expected_origins, got $total_origins" >&2; exit 3;
}
echo "[parallel] learner origins complete: $total_origins"

# Teacher work is repartitioned independently from origin storage.  Each GPU
# gets 19 motion IDs and reads only those IDs from the eight origin directories.
origin_args=()
for directory in "${ORIGIN_DIRS[@]}"; do
  origin_args+=(--origins "$directory")
done

run_teacher_shard() {
  local gpu="$1" output="$2" motions="$3" log="$4"
  local runtime="$ROUND_ROOT/runtime/teacher$(printf '%02d' "$gpu")"
  local seed=$((142 + gpu))
  if test -f "$output/collection_receipt.json"; then
    echo "[teacher shard $gpu] receipt exists; skipping"
    return 0
  fi
  mkdir -p "$output" "$runtime/tmp" "$runtime/kit-data" "$runtime/cache"
  echo "[teacher shard $gpu] motions=$motions output=$output"
  CUDA_VISIBLE_DEVICES="$gpu" \
  TMPDIR="$runtime/tmp" \
  OMNI_KIT_APP_DATA="$runtime/kit-data" \
  XDG_CACHE_HOME="$runtime/cache" \
  "$COLLECTION_PYTHON" \
    PredActor/collection/scripts/data_collection/label_paper_dagger_origins.py \
    "${origin_args[@]}" --motion-ids "$motions" \
    --teacher-checkpoint "$TEACHER" --motion-root "$MOTIONS" \
    --phase-conditions "$CONDITIONS" --output "$output" \
    --origins-per-motion "$ORIGINS_PER_MOTION" \
    --collect-all-valid --minimum-accepted-per-motion 1 --resume \
    --device cuda:0 --seed "$seed" --headless >"$log" 2>&1
}

echo "[parallel] launching eight isolated teacher shards"
pids=()
BRANCH_DIRS=()
for gpu in $(seq 0 7); do
  start=$((gpu * 19))
  end=$((start + 18))
  motions="$(seq -s, "$start" "$end")"
  output="$ROUND_ROOT/branch_shards/shard$(printf '%02d' "$gpu")"
  BRANCH_DIRS+=("$output")
  run_teacher_shard "$gpu" "$output" "$motions" \
    "$LOG_DIR/teacher_shard$(printf '%02d' "$gpu").log" &
  pids+=("$!")
done
failed=0
for pid in "${pids[@]}"; do
  wait "$pid" || failed=1
done
test "$failed" -eq 0 || { echo "one or more teacher shards failed" >&2; exit 4; }

branch_args=()
total_branches=0
for directory in "${BRANCH_DIRS[@]}"; do
  test -f "$directory/collection_receipt.json" || {
    echo "missing teacher receipt: $directory" >&2; exit 4;
  }
  branch_args+=(--branches "$directory")
  count=$(find "$directory" -maxdepth 1 -name 'branch_*.npz' | wc -l)
  total_branches=$((total_branches + count))
done
test "$total_branches" -ge "$MIN_D1_WINDOWS" || {
  echo "D1 count gate failed: $total_branches < $MIN_D1_WINDOWS" >&2; exit 4;
}
echo "[parallel] teacher branches complete: $total_branches"

if ! test -f "$MANIFEST"; then
  "$TRAIN_PYTHON" PredActor/scripts/finalize_paper_dagger_round.py \
    "${branch_args[@]}" --round-dir "$ROUND_ROOT" \
    --d0-zarr "$D0" --normalizer-checkpoint "$PI0" \
    --expected-motion-count 152 --expected-origins-per-motion "$ORIGINS_PER_MOTION" \
    --allow-variable-counts --minimum-origins-per-motion 1 \
    --min-d1-windows "$MIN_D1_WINDOWS" --min-d1-fraction "$MIN_D1_FRACTION"
fi

export PREDACTOR_DAGGER_MANIFEST="$MANIFEST"
export PREDACTOR_PI0_CHECKPOINT="$PI0"
export PREDACTOR_OUTPUT_ROOT="$DATA_ROOT/outputs"
"$TRAIN_PYTHON" PredActor/train.py \
  --cfg g1prdp_dagger_paper.yaml \
  --exp_name paper-dagger-pi1-round01-full-parallel-s42 --gpu 0

echo "PAPER_DAGGER_PARALLEL_ROUND1_COMPLETE round=$ROUND_ROOT"
