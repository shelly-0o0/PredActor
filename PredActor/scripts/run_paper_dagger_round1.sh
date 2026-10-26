#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${PREDACTOR_REPO_ROOT:-/root/Predactor/PredActor-DAgger}"
DATA_ROOT="${PREDACTOR_DATA_ROOT:-/data/PredActor}"
ROUND_ROOT="${PREDACTOR_ROUND_ROOT:-$DATA_ROOT/dagger_strict/paper_round_01}"
ORIGINS_PER_MOTION="${PREDACTOR_ORIGINS_PER_MOTION:-5}"
COLLECTION_PYTHON="${PREDACTOR_COLLECTION_PYTHON:-/root/miniconda3/envs/predactor-collection/bin/python}"
TRAIN_PYTHON="${PREDACTOR_TRAIN_PYTHON:-/root/Predactor/PredActor/.venv/bin/python}"
PI0="${PREDACTOR_LEARNER_CHECKPOINT:-$DATA_ROOT/outputs/October-23-14-13-57-g1prdp_cond_diffuse-pi0-d0-seed42-final/checkpoints/latest.ckpt}"
TEACHER="$DATA_ROOT/Artifacts/checkpoints/rl_tracker/model_75000.pt"
MOTIONS="$DATA_ROOT/Artifacts/motions/g1_152fk"
D0="$DATA_ROOT/Artifacts/dataset/g1_152fk/motion_dyed_text_t0.01.zarr"
CONDITIONS="$DATA_ROOT/Artifacts/dagger/d0_phase_conditions_v1.npz"
BRANCHES="$ROUND_ROOT/branches"
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

if ! test -f "$CONDITIONS"; then
  "$TRAIN_PYTHON" PredActor/scripts/build_d0_phase_conditions.py \
    --d0-zarr "$D0" --motion-root "$MOTIONS" --output "$CONDITIONS"
fi

export PYTHONPATH="$REPO_ROOT/PredActor:$REPO_ROOT/PredActor/collection/source/rl_tracker:$REPO_ROOT/PredActor/collection/vendor/rsl_rl-modular-normed:${PYTHONPATH:-}"
export PREDACTOR_ROOT="$REPO_ROOT/PredActor"
export PREDACTOR_ASSET_ROOT="$REPO_ROOT/PredActor/assets"

"$COLLECTION_PYTHON" \
  PredActor/collection/scripts/data_collection/collect_strict_dagger_round.py \
  --learner-checkpoint "$PI0" \
  --teacher-checkpoint "$TEACHER" \
  --motion-root "$MOTIONS" \
  --phase-conditions "$CONDITIONS" \
  --output "$BRANCHES" \
  --origins-per-motion "$ORIGINS_PER_MOTION" \
  --seed 42 --headless

# Isaac Sim may terminate the Python process unusually after an exception.
# Require a complete receipt and exact branch count before finalization so a
# partial collection can never be mistaken for a finished DAgger round.
RECEIPT="$BRANCHES/collection_receipt.json"
test -f "$RECEIPT" || {
  echo "collection did not produce a completion receipt: $RECEIPT" >&2
  exit 3
}
EXPECTED_BRANCHES=$((152 * ORIGINS_PER_MOTION))
ACTUAL_BRANCHES=$(find "$BRANCHES" -maxdepth 1 -type f -name 'branch_*.npz' | wc -l)
if test "$ACTUAL_BRANCHES" -ne "$EXPECTED_BRANCHES"; then
  echo "collection incomplete: expected $EXPECTED_BRANCHES branches, got $ACTUAL_BRANCHES" >&2
  exit 3
fi

"$TRAIN_PYTHON" PredActor/scripts/finalize_paper_dagger_round.py \
  --branches "$BRANCHES" --round-dir "$ROUND_ROOT" \
  --d0-zarr "$D0" --normalizer-checkpoint "$PI0" \
  --expected-motion-count 152 \
  --expected-origins-per-motion "$ORIGINS_PER_MOTION"

"$TRAIN_PYTHON" PredActor/scripts/validate-dagger-manifest.py "$MANIFEST"

export PREDACTOR_DAGGER_MANIFEST="$MANIFEST"
export PREDACTOR_PI0_CHECKPOINT="$PI0"
export PREDACTOR_OUTPUT_ROOT="$DATA_ROOT/outputs"
"$TRAIN_PYTHON" PredActor/train.py \
  --cfg g1prdp_dagger_paper.yaml \
  --exp_name paper-dagger-pi1-round01-s42 \
  --gpu 0

echo "PAPER_DAGGER_ROUND1_COMPLETE round=$ROUND_ROOT"
