#!/bin/bash
# ============================================================================
# MechJEPA: Train ALOE VQA on rollout slots + evaluate
# Prerequisites: Run runpod_train.sh first (produces rollout pkl)
# Usage: bash mechjepa/scripts/runpod_eval_aloe.sh
# ============================================================================
set -e

WORKSPACE="/workspace"
CJEPA="$WORKSPACE/cjepa"
MECHJEPA="$WORKSPACE/mechjepa"
LOGS="$WORKSPACE/logs"
ALOE_OUT="$WORKSPACE/aloe_output"
ROLLOUT="$WORKSPACE/data/rollout_mechjepa.pkl"

cd "$CJEPA"
export PYTHONPATH="$CJEPA"
export WANDB_MODE=disabled
mkdir -p "$LOGS" "$ALOE_OUT"

# ── Verify rollout exists ──
if [ ! -f "$ROLLOUT" ]; then
    echo "ERROR: Rollout not found at $ROLLOUT"
    echo "Run runpod_train.sh first."
    exit 1
fi

ROLLOUT_SIZE=$(python -c "import os; print(f'{os.path.getsize(\"$ROLLOUT\") / 1e9:.1f}')")
echo "========================================"
echo "  ALOE VQA Training on MechJEPA Rollout"
echo "  Started: $(date)"
echo "  Rollout: $ROLLOUT (${ROLLOUT_SIZE} GB)"
echo "========================================"

# ── Train ALOE on MechJEPA rollout ──
echo ""
echo "[1/2] Training ALOE on MechJEPA rollout (400 epochs)..."
echo "  Estimated time: ~4-6 hours on H100"
echo ""

python src/aloe_train.py \
    --task clevrer_vqa \
    --params src/third_party/slotformer/clevrer_vqa/configs/aloe_clevrer_params-rollout.py \
    --exp_name aloe_mechjepa \
    --out_dir "$ALOE_OUT" \
    --slot_root_override "$ROLLOUT" \
    --fp16 --cudnn \
    2>&1 | tee "$LOGS/aloe_mechjepa.log"

echo ""
echo "[1/2] ALOE training DONE at $(date)"

# ── Extract results ──
echo ""
echo "[2/2] Extracting evaluation results..."
echo ""

# Parse ALOE eval metrics from log
echo "──────────────────────────────────────"
echo "  ALOE VQA Results (MechJEPA rollout)"
echo "──────────────────────────────────────"
echo ""

# ALOE logs evaluation metrics every eval_interval epochs
# Format: "Eval epoch X: desc=Y.YY, expl=Y.YY, pred=Y.YY, cf=Y.YY, total=Y.YY"
echo "All evaluation epochs:"
grep -i "eval\|accuracy\|desc\|expl\|pred\|counterfactual\|total" "$LOGS/aloe_mechjepa.log" | tail -20

echo ""
echo "Best results (last eval):"
grep -i "eval" "$LOGS/aloe_mechjepa.log" | tail -1

echo ""
echo "──────────────────────────────────────"
echo "  Baseline comparison reference:"
echo "  C-JEPA:   desc=67.2, expl=58.1, pred=42.3, cf=35.8"
echo "  CTT-JEPA: desc=69.5, expl=61.3, pred=45.1, cf=38.7"
echo "──────────────────────────────────────"

echo ""
echo "========================================"
echo "  ✅ ALOE Evaluation Complete!"
echo "  Finished: $(date)"
echo "========================================"
echo ""
echo "  ALOE model:  $ALOE_OUT/aloe_mechjepa/"
echo "  Full log:    $LOGS/aloe_mechjepa.log"
echo ""
echo "  To run test set evaluation:"
echo "    cd $CJEPA"
echo "    PYTHONPATH=$CJEPA python src/third_party/slotformer/clevrer_vqa/test_clevrer_vqa.py \\"
echo "      --params src/third_party/slotformer/clevrer_vqa/configs/aloe_clevrer_param_for_test.py \\"
echo "      --weight <path_to_best_aloe_weight> \\"
echo "      --slots_root_override $ROLLOUT \\"
echo "      --validate"
