#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${PREDACTOR_REPO_ROOT:-/root/Predactor/PredActor-DAgger}"
DATA_ROOT="${PREDACTOR_DATA_ROOT:-/data/PredActor}"
ROUND_ROOT="${PREDACTOR_ROUND_ROOT:-$DATA_ROOT/dagger/paper_round_01_full_v2}"
# Formal round-1 scale.  With a six-step origin stride, 2,000 accepted
# branches per motion are about 50 full 240-step learner rollouts.  The larger
# candidate pool absorbs teacher continuations that terminate before horizon.
ORIGINS_PER_MOTION="${PREDACTOR_ORIGINS_PER_MOTION:-2000}"
CANDIDATES_PER_MOTION="${PREDACTOR_CANDIDATES_PER_MOTION:-3200}"
BRANCH_EVERY="${PREDACTOR_BRANCH_EVERY:-6}"
MAX_EPISODE_STEPS="${PREDACTOR_MAX_EPISODE_STEPS:-240}"
MAX_ATTEMPTS_PER_MOTION="${PREDACTOR_MAX_ATTEMPTS_PER_MOTION:-300}"
# These are launch gates, not sampling weights.  The trainer still performs
# literal D0 union D1 aggregate replay with every window present once.
MIN_D1_WINDOWS="${PREDACTOR_MIN_D1_WINDOWS:-250000}"
MIN_D1_FRACTION="${PREDACTOR_MIN_D1_FRACTION:-0.10}"
COLLECTION_PYTHON="${PREDACTOR_COLLECTION_PYTHON:-/root/miniconda3/envs/predactor-collection/bin/python}"
TRAIN_PYTHON="${PREDACTOR_TRAIN_PYTHON:-/root/Predactor/PredActor/.venv/bin/python}"
PI0="${PREDACTOR_LEARNER_CHECKPOINT:-$DATA_ROOT/outputs/October-23-14-13-57-g1prdp_cond_diffuse-pi0-d0-seed42-final/checkpoints/latest.ckpt}"
TEACHER="$DATA_ROOT/Artifacts/checkpoints/rl_tracker/model_75000.pt"
MOTIONS="$DATA_ROOT/Artifacts/motions/g1_152fk"
D0="$DATA_ROOT/Artifacts/dataset/g1_152fk/motion_dyed_text_t0.01.zarr"
CONDITIONS="$DATA_ROOT/Artifacts/dagger/d0_phase_conditions_v1.npz"
BRANCHES="$ROUND_ROOT/branches"
ORIGINS="$ROUND_ROOT/origins"
MANIFEST="$ROUND_ROOT/manifest.yaml"

cd "$REPO_ROOT"
for path in "$COLLECTION_PYTHON" "$TRAIN_PYTHON" "$PI0" "$TEACHER" "$MOTIONS" "$D0"; do
  test -e "$path" || { echo "missing required input: $path" >&2; exit 2; }
done
if test -e "$ROUND_ROOT"; then
  echo "refusing to overwrite existing round: $ROUND_ROOT" >&2
  exit 2
fi
mkdir -p "$ROUND_ROOT"

EXPECTED_BRANCHES=$((152 * ORIGINS_PER_MOTION))
if test "$EXPECTED_BRANCHES" -lt "$MIN_D1_WINDOWS"; then
  echo "refusing undersized formal round: expected D1=$EXPECTED_BRANCHES, minimum=$MIN_D1_WINDOWS" >&2
  exit 2
fi
printf 'Paper DAgger round 1: motions=152 accepted_per_motion=%s D1=%s candidates_per_motion=%s stride=%s\n' \
  "$ORIGINS_PER_MOTION" "$EXPECTED_BRANCHES" "$CANDIDATES_PER_MOTION" "$BRANCH_EVERY"

if ! test -f "$CONDITIONS"; then
  "$TRAIN_PYTHON" PredActor/scripts/build_d0_phase_conditions.py \
    --d0-zarr "$D0" --motion-root "$MOTIONS" --output "$CONDITIONS"
fi

export PYTHONPATH="$REPO_ROOT/PredActor:$REPO_ROOT/PredActor/collection/source/rl_tracker:$REPO_ROOT/PredActor/collection/vendor/rsl_rl-modular-normed:${PYTHONPATH:-}"
export PREDACTOR_ROOT="$REPO_ROOT/PredActor"
export PREDACTOR_ASSET_ROOT="$REPO_ROOT/PredActor/assets"

"$COLLECTION_PYTHON" \
  PredActor/collection/scripts/data_collection/collect_paper_dagger_origins.py \
  --learner-checkpoint "$PI0" \
  --motion-root "$MOTIONS" \
  --phase-conditions "$CONDITIONS" \
  --output "$ORIGINS" \
  --candidates-per-motion "$CANDIDATES_PER_MOTION" \
  --branch-every "$BRANCH_EVERY" \
  --max-episode-steps "$MAX_EPISODE_STEPS" \
  --max-attempts-per-motion "$MAX_ATTEMPTS_PER_MOTION" \
  --seed 42 --headless

ORIGIN_RECEIPT="$ORIGINS/origin_receipt.json"
test -f "$ORIGIN_RECEIPT" || {
  echo "origin collection did not produce a completion receipt: $ORIGIN_RECEIPT" >&2
  exit 3
}
EXPECTED_ORIGINS=$((152 * CANDIDATES_PER_MOTION))
ACTUAL_ORIGINS=$(find "$ORIGINS" -maxdepth 1 -type f -name 'origin_*.npz' | wc -l)
if test "$ACTUAL_ORIGINS" -ne "$EXPECTED_ORIGINS"; then
  echo "origin collection incomplete: expected $EXPECTED_ORIGINS, got $ACTUAL_ORIGINS" >&2
  exit 3
fi

# The teacher runs in a fresh IsaacLab process.  It never steps or restores the
# live learner simulator, so vector-wide physics/events/resets cannot corrupt
# learner visitation.  Extra candidates absorb rejected teacher continuations.
"$COLLECTION_PYTHON" \
  PredActor/collection/scripts/data_collection/label_paper_dagger_origins.py \
  --origins "$ORIGINS" \
  --teacher-checkpoint "$TEACHER" \
  --motion-root "$MOTIONS" \
  --phase-conditions "$CONDITIONS" \
  --output "$BRANCHES" \
  --origins-per-motion "$ORIGINS_PER_MOTION" \
  --seed 42 --headless

# Isaac Sim may terminate unusually after an exception.  Require a complete
# receipt and exact branch count before finalization.
RECEIPT="$BRANCHES/collection_receipt.json"
test -f "$RECEIPT" || {
  echo "collection did not produce a completion receipt: $RECEIPT" >&2
  exit 3
}
ACTUAL_BRANCHES=$(find "$BRANCHES" -maxdepth 1 -type f -name 'branch_*.npz' | wc -l)
if test "$ACTUAL_BRANCHES" -ne "$EXPECTED_BRANCHES"; then
  echo "collection incomplete: expected $EXPECTED_BRANCHES branches, got $ACTUAL_BRANCHES" >&2
  exit 3
fi

"$TRAIN_PYTHON" PredActor/scripts/finalize_paper_dagger_round.py \
  --branches "$BRANCHES" --round-dir "$ROUND_ROOT" \
  --d0-zarr "$D0" --normalizer-checkpoint "$PI0" \
  --expected-motion-count 152 \
  --expected-origins-per-motion "$ORIGINS_PER_MOTION" \
  --min-d1-windows "$MIN_D1_WINDOWS" \
  --min-d1-fraction "$MIN_D1_FRACTION"

export PREDACTOR_DAGGER_MANIFEST="$MANIFEST"
export PREDACTOR_PI0_CHECKPOINT="$PI0"
export PREDACTOR_OUTPUT_ROOT="$DATA_ROOT/outputs"
"$TRAIN_PYTHON" PredActor/train.py \
  --cfg g1prdp_dagger_paper.yaml \
  --exp_name paper-dagger-pi1-round01-full-s42 \
  --gpu 0

echo "PAPER_DAGGER_ROUND1_COMPLETE round=$ROUND_ROOT"
