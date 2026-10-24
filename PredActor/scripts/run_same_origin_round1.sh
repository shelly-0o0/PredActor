#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${PREDACTOR_REPO_ROOT:-/root/Predactor/PredActor-DAgger}"
DATA_ROOT="${PREDACTOR_DATA_ROOT:-/data/PredActor}"
ROUND_ROOT="${PREDACTOR_ROUND_ROOT:-$DATA_ROOT/dagger_strict/navigation_canary_v1}"
PAIR_COUNT="${PREDACTOR_ROUND_PAIRS:-12}"
COLLECTION_PYTHON="${PREDACTOR_COLLECTION_PYTHON:-/root/miniconda3/envs/predactor-collection/bin/python}"
TRAIN_PYTHON="${PREDACTOR_TRAIN_PYTHON:-/root/Predactor/PredActor/.venv/bin/python}"
PI0="$DATA_ROOT/outputs/October-23-14-13-57-g1prdp_cond_diffuse-pi0-d0-seed42-final/checkpoints/latest.ckpt"
TEACHER="$DATA_ROOT/Artifacts/checkpoints/rl_tracker/model_75000.pt"
TEXT="$DATA_ROOT/Artifacts/calibration/canonical_text_embeddings.npz"
CANONICAL="$DATA_ROOT/Artifacts/motions/g1_canonical_nav_v1"
D0="$DATA_ROOT/Artifacts/dataset/g1_152fk/motion_dyed_text_t0.01.zarr"
BRANCHES="$ROUND_ROOT/branches"

# This collector intentionally covers only the physically compatible
# straight/left/right navigation family.  It is not a multi-task DAgger round
# and must never auto-launch pi1 training by itself.
if [[ "${PREDACTOR_NAVIGATION_DIAGNOSTIC_ONLY:-0}" != "1" ]]; then
  cat >&2 <<'EOF'
run_same_origin_round1.sh is a navigation-only diagnostic collector.
It does not cover PredActor's 152-motion text-to-motion distribution and no
longer builds a training manifest or starts pi1.  To run a small navigation
canary explicitly, set PREDACTOR_NAVIGATION_DIAGNOSTIC_ONLY=1.
EOF
  exit 2
fi

cd "$REPO_ROOT"
for path in "$COLLECTION_PYTHON" "$TRAIN_PYTHON" "$PI0" "$TEACHER" "$TEXT" \
            "$CANONICAL/canonical_nav_manifest.json" "$D0"; do
  test -e "$path" || { echo "missing required input: $path" >&2; exit 2; }
done
if test -e "$ROUND_ROOT"; then
  echo "refusing to overwrite existing round: $ROUND_ROOT" >&2
  exit 2
fi
mkdir -p "$ROUND_ROOT"

export PYTHONPATH="$REPO_ROOT/PredActor:$REPO_ROOT/PredActor/collection/source/rl_tracker:$REPO_ROOT/PredActor/collection/vendor/rsl_rl-modular-normed:${PYTHONPATH:-}"
export PREDACTOR_ROOT="$REPO_ROOT/PredActor"
export PREDACTOR_ASSET_ROOT="$REPO_ROOT/PredActor/assets"

"$COLLECTION_PYTHON" \
  PredActor/collection/scripts/data_collection/collect_strict_same_origin.py \
  --pi0-checkpoint "$PI0" \
  --text-embeddings "$TEXT" \
  --teacher-checkpoint "$TEACHER" \
  --canonical-root "$CANONICAL" \
  --output "$BRANCHES" \
  --pairs "$PAIR_COUNT" \
  --max-live-steps 20000 \
  --branch-every 12 \
  --episode-steps 220 \
  --seed 42 \
  --headless

"$TRAIN_PYTHON" PredActor/scripts/audit_same_origin_branches.py \
  "$BRANCHES" \
  --output "$ROUND_ROOT/navigation_same_origin_audit.json"

echo "NAVIGATION_DIAGNOSTIC_COMPLETE root=$ROUND_ROOT"
echo "No manifest was built and no training was started."
