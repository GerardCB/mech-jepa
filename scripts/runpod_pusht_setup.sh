#!/bin/bash
# ============================================================================
# MechJEPA Push-T: RunPod Environment Setup
# Run this FIRST on a fresh RunPod to install deps and download data.
# Usage: bash mechjepa/scripts/runpod_pusht_setup.sh
# ============================================================================
set -e

WORKSPACE="/workspace"
MECHJEPA="$WORKSPACE/mechjepa"
DATA_DIR="$WORKSPACE/data"

echo "========================================"
echo "  MechJEPA Push-T — Environment Setup"
echo "  Started: $(date)"
echo "  GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "========================================"

# ── 1. Clone / update repo ──
echo ""
echo "[1/5] Setting up repository..."
if [ -d "$MECHJEPA/.git" ]; then
    cd "$MECHJEPA" && git pull origin main
    echo "  Updated existing repo"
else
    git clone https://github.com/GerardCB/mech-jepa.git "$MECHJEPA"
    echo "  Cloned fresh repo"
fi
cd "$MECHJEPA"

# ── 2. Install dependencies ──
echo ""
echo "[2/5] Installing Python dependencies..."
pip install -q torch torchvision --upgrade 2>/dev/null
pip install -q hydra-core omegaconf einops loguru wandb tqdm scikit-learn matplotlib 2>/dev/null

# For data preparation (action extraction)
pip install -q zarr 2>/dev/null
echo "  Done"

# ── 3. Download Push-T slot embeddings ──
echo ""
echo "[3/5] Downloading Push-T VideoSAUR slots..."
mkdir -p "$DATA_DIR"
SLOTS_FILE="$DATA_DIR/pusht_videosaur_slots.pkl"

if [ -f "$SLOTS_FILE" ]; then
    echo "  Slots already downloaded: $SLOTS_FILE"
else
    wget -q --show-progress \
        -O "$SLOTS_FILE" \
        "https://huggingface.co/HazelNam/CJEPA/resolve/main/pusht_videosaur_slots.pkl"
    echo "  Downloaded: $SLOTS_FILE ($(du -h $SLOTS_FILE | cut -f1))"
fi

# ── 4. Download Push-T actions ──
echo ""
echo "[4/5] Downloading Push-T actions from Diffusion Policy..."
PUSHT_ZIP="$DATA_DIR/pusht.zip"
ZARR_DIR="$DATA_DIR/pusht_cchi_v7_replay.zarr"

if [ -d "$ZARR_DIR" ]; then
    echo "  Zarr already exists: $ZARR_DIR"
else
    wget -q --show-progress \
        -O "$PUSHT_ZIP" \
        "https://diffusion-policy.cs.columbia.edu/data/training/pusht.zip"
    echo "  Extracting zarr..."
    cd "$DATA_DIR" && python3 -c "
import zipfile
with zipfile.ZipFile('pusht.zip', 'r') as z:
    z.extractall('.')
print('  Extracted successfully')
"
    rm -f "$PUSHT_ZIP"
    echo "  Zarr ready: $ZARR_DIR"
fi

# ── 5. Merge slots + actions ──
echo ""
echo "[5/5] Merging slots with actions..."
MERGED_FILE="$DATA_DIR/pusht_slots_actions.pkl"

if [ -f "$MERGED_FILE" ]; then
    echo "  Merged data already exists: $MERGED_FILE"
else
    cd "$MECHJEPA"
    export PYTHONPATH="$MECHJEPA"
    python scripts/prepare_pusht_data.py --output "$MERGED_FILE"
fi

# ── Verify ──
echo ""
echo "========================================"
echo "  ✅ Setup Complete!"
echo "  $(date)"
echo "========================================"
echo ""
echo "  Slot data:   $SLOTS_FILE"
echo "  Zarr data:   $ZARR_DIR"
echo "  Merged data: $MERGED_FILE"
echo ""
echo "  Verify data:"
echo "    python -c \"import pickle; d=pickle.load(open('$MERGED_FILE','rb')); print({k:len(v) for k,v in d.items()})\""
echo ""
echo "  Next: bash mechjepa/scripts/runpod_pusht_train.sh"
