#!/bin/bash
# ============================================================================
# MechJEPA: Train world model + rollout slots on RunPod H100
# Prerequisites: Run runpod_setup.sh first
# Usage: bash mechjepa/scripts/runpod_train.sh
# ============================================================================
set -e

WORKSPACE="/workspace"
MECHJEPA="$WORKSPACE/mechjepa"
DATA="$WORKSPACE/data/clevrer_videosaur_slots.pkl"
CHECKPOINTS="$WORKSPACE/checkpoints"
LOGS="$WORKSPACE/logs"

cd "$MECHJEPA"
export PYTHONPATH="$MECHJEPA"
mkdir -p "$CHECKPOINTS" "$LOGS"

# Verify data exists
if [ ! -f "$DATA" ]; then
    echo "ERROR: Slot data not found at $DATA"
    echo "Run runpod_setup.sh first."
    exit 1
fi

echo "========================================"
echo "  MechJEPA World Model Training"
echo "  Started: $(date)"
echo "  GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "========================================"

# ── Train MechJEPA ──
echo ""
echo "[1/2] Training MechJEPA (30 epochs, batch_size=2048)..."
echo "  Estimated time: ~2-3 hours on H100"
echo ""

python scripts/train_clevrer.py \
    embedding_dir="$DATA" \
    cache_dir="$CHECKPOINTS" \
    output_model_name=mechjepa_clevrer \
    batch_size=2048 \
    num_workers=8 \
    trainer.max_epochs=30 \
    trainer.precision=16-mixed \
    dinowm.history_size=6 \
    dinowm.num_preds=10 \
    frameskip=2 \
    videosaur.NUM_SLOTS=7 \
    videosaur.SLOT_DIM=128 \
    num_masked_slots=4 \
    predictor_lr=5e-4 \
    codebook_lr=1e-3 \
    warmup_epochs=5 \
    max_grad_norm=0.05 \
    predictor.depth=6 \
    predictor.heads=16 \
    predictor.mlp_dim=2048 \
    predictor.dim_head=64 \
    predictor.dropout=0.1 \
    codebook.num_mechanisms=16 \
    codebook.edge_hidden_dim=256 \
    codebook.bottleneck_recon_weight=0.0 \
    system_m.enabled=false \
    system_m.surprise_threshold=1.0 \
    rollout.save_rollout=true \
    rollout.rollout_batch_size=256 \
    wandb.enable=false \
    2>&1 | tee "$LOGS/train_mechjepa.log"

echo ""
echo "[1/2] MechJEPA training DONE at $(date)"

# ── Verify rollout was saved ──
echo ""
echo "[2/2] Verifying rollout..."
ROLLOUT_PATH=$(python -c "
from pathlib import Path
import glob
candidates = glob.glob('$WORKSPACE/data/rollout_mechjepa_*.pkl')
if candidates:
    print(candidates[0])
else:
    print('')
")

if [ -n "$ROLLOUT_PATH" ] && [ -f "$ROLLOUT_PATH" ]; then
    ROLLOUT_SIZE=$(python -c "import os; print(f'{os.path.getsize(\"$ROLLOUT_PATH\") / 1e9:.1f}')")
    echo "  Rollout saved: $ROLLOUT_PATH (${ROLLOUT_SIZE} GB)"
    # Create a stable symlink for the ALOE script
    ln -sf "$ROLLOUT_PATH" "$WORKSPACE/data/rollout_mechjepa.pkl"
    echo "  Symlinked to: $WORKSPACE/data/rollout_mechjepa.pkl"
else
    echo "  WARNING: Rollout not found. You may need to run rollout separately:"
    echo "  python scripts/train_clevrer.py rollout.rollout_only=true \\"
    echo "      rollout.rollout_checkpoint=$CHECKPOINTS/mechjepa_clevrer_best.ckpt \\"
    echo "      embedding_dir=$DATA ..."
fi

echo ""
echo "========================================"
echo "  ✅ MechJEPA Training Complete!"
echo "  Finished: $(date)"
echo "========================================"
echo ""
echo "  Checkpoints: $CHECKPOINTS/"
echo "  Training log: $LOGS/train_mechjepa.log"
echo "  Rollout: $WORKSPACE/data/rollout_mechjepa.pkl"
echo ""
echo "  Next step: bash mechjepa/scripts/runpod_eval_aloe.sh"
