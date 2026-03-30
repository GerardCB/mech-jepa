#!/bin/bash
# ============================================================================
# MechJEPA Push-T: Train Action-Conditioned World Model on RunPod
# Prerequisites: Run runpod_pusht_setup.sh first
# Usage: bash mechjepa/scripts/runpod_pusht_train.sh
# ============================================================================
set -e

WORKSPACE="/workspace"
MECHJEPA="$WORKSPACE/mechjepa"
DATA="$WORKSPACE/data/pusht_slots_actions.pkl"
CHECKPOINTS="$WORKSPACE/checkpoints"
LOGS="$WORKSPACE/logs"

cd "$MECHJEPA"
export PYTHONPATH="$MECHJEPA"
mkdir -p "$CHECKPOINTS" "$LOGS"

# Verify data exists
if [ ! -f "$DATA" ]; then
    echo "ERROR: Merged data not found at $DATA"
    echo "Run runpod_pusht_setup.sh first."
    exit 1
fi

echo "========================================"
echo "  MechJEPA Push-T World Model Training"
echo "  Started: $(date)"
echo "  GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "========================================"

# ── Train MechJEPA Push-T ──
echo ""
echo "[1/1] Training MechJEPA Push-T (200 epochs)..."
echo "  Action conditioning: AdaLN (action_dim=2)"
echo "  Slots: 4 (VideoSAUR), Bottleneck: 8 dims"
echo "  Estimated time: ~2-4 hours on H100"
echo ""

python scripts/train_pusht.py \
    embedding_dir="$DATA" \
    cache_dir="$CHECKPOINTS" \
    output_model_name=mechjepa_pusht \
    batch_size=128 \
    num_workers=8 \
    trainer.max_epochs=200 \
    trainer.precision=16-mixed \
    dinowm.history_size=3 \
    dinowm.num_preds=1 \
    frameskip=5 \
    videosaur.NUM_SLOTS=4 \
    videosaur.SLOT_DIM=128 \
    action_dim=2 \
    num_masked_slots=2 \
    predictor_lr=5e-4 \
    codebook_lr=1e-3 \
    warmup_epochs=10 \
    max_grad_norm=0.05 \
    predictor.depth=6 \
    predictor.heads=16 \
    predictor.mlp_dim=2048 \
    predictor.dim_head=64 \
    predictor.dropout=0.1 \
    codebook.num_mechanisms=8 \
    codebook.edge_hidden_dim=256 \
    codebook.bottleneck_recon_weight=0.0 \
    system_m.enabled=false \
    system_m.surprise_threshold=1.0 \
    wandb.enable=true \
    wandb.project=mechjepa \
    wandb.name=mechjepa_pusht_adaln_v1 \
    2>&1 | tee "$LOGS/train_pusht.log"

echo ""
echo "========================================"
echo "  ✅ Push-T Training Complete!"
echo "  Finished: $(date)"
echo "========================================"
echo ""
echo "  Checkpoints: $CHECKPOINTS/"
echo "  Training log: $LOGS/train_pusht.log"
echo ""
echo "  Next steps:"
echo "    1. Check bottleneck dimension importance in logs"
echo "    2. Implement CEM planner (scripts/plan_pusht.py)"
echo "    3. Run A-B-M adaptation demo (scripts/abm_pusht.py)"
