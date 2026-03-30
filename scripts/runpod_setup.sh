#!/bin/bash
# ============================================================================
# MechJEPA: RunPod / GPU Server Setup Script
# Run this once after spinning up a pod.
# Tested on: runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04
# ============================================================================
set -e

echo "========================================"
echo "  MechJEPA Environment Setup"
echo "  Started: $(date)"
echo "========================================"

WORKSPACE="/workspace"
cd "$WORKSPACE"

# ── 1. Clone repos ──
echo "[1/7] Cloning repositories..."

if [ ! -d "$WORKSPACE/mechjepa" ]; then
    git clone --quiet https://github.com/GerardCB/mech-jepa.git mechjepa
    echo "  Cloned MechJEPA"
else
    echo "  MechJEPA already present"
    cd mechjepa && git pull --quiet && cd ..
fi

if [ ! -d "$WORKSPACE/cjepa" ]; then
    git clone --quiet -b ctt-jepa https://github.com/GerardCB/cjepa.git cjepa
    echo "  Cloned C-JEPA (ctt-jepa branch, for ALOE framework)"
else
    echo "  C-JEPA already present"
    cd cjepa && git pull --quiet && cd ..
fi

# ── 2. Install MechJEPA dependencies ──
echo "[2/7] Installing MechJEPA dependencies..."
pip install -q hydra-core omegaconf einops loguru wandb tqdm seaborn

# ── 3. Install ALOE dependencies ──
echo "[3/7] Installing ALOE VQA dependencies..."
pip install -q 'torchmetrics<1.0' pycocotools webdataset huggingface_hub

# ── 4. Install third-party libraries (for ALOE) ──
echo "[4/7] Installing third-party libraries..."
cd "$WORKSPACE/cjepa/src/third_party"

if [ ! -d "stable-pretraining" ]; then
    git clone --quiet https://github.com/galilai-group/stable-pretraining.git
    cd stable-pretraining && git checkout 92b5841 && pip install -q -e . && cd ..
else
    echo "  stable-pretraining already installed"
fi

if [ ! -d "stable-worldmodel" ]; then
    git clone --quiet https://github.com/galilai-group/stable-worldmodel.git
    pip install -q -e stable-worldmodel
else
    echo "  stable-worldmodel already installed"
fi

if [ ! -d "nerv" ]; then
    git clone --quiet https://github.com/Wuziyi616/nerv.git
    cd nerv && git checkout v0.1.0 && pip install -q --ignore-installed blinker && pip install -q -e . && cd ..
else
    echo "  nerv already installed"
fi

cd "$WORKSPACE"

# ── 5. Patch torchcodec (crashes on many CUDA versions) ──
echo "[5/7] Patching torchcodec imports..."
pip uninstall torchcodec -y 2>/dev/null || true
cd "$WORKSPACE/cjepa"
PYTHONPATH="$WORKSPACE/cjepa" python scripts/ctt/fix_torchcodec.py
cd "$WORKSPACE"

# ── 6. Download data ──
echo "[6/7] Downloading data..."

# 6a. VideoSAUR slot embeddings (~9 GB)
mkdir -p "$WORKSPACE/data"
if [ ! -f "$WORKSPACE/data/clevrer_videosaur_slots.pkl" ]; then
    echo "  Downloading VideoSAUR slot embeddings from HuggingFace (~9 GB)..."
    python -c "
from huggingface_hub import hf_hub_download
import os
hf_hub_download(repo_id='HazelNam/CJEPA', filename='clevrer_videosaur_slots.pkl',
                local_dir=os.path.join('$WORKSPACE', 'data'))
print('  Slot embeddings download complete.')
"
else
    echo "  Slot embeddings already downloaded"
fi

# 6b. CLEVRER question JSONs (for ALOE VQA) — already in cjepa repo
if [ -d "$WORKSPACE/cjepa/dataset/clevrer/questions" ]; then
    echo "  CLEVRER questions already available in cjepa repo"
else
    echo "  WARNING: CLEVRER questions not found in cjepa repo"
fi

# ── 7. Verify installation ──
echo "[7/7] Verifying installation..."

# Verify MechJEPA
cd "$WORKSPACE/mechjepa"
python -c "
import torch
from mechjepa.model import MechJEPA
from mechjepa.codebook import MechanismCodebook
from mechjepa.dynamics import MechSlotPredictor
from mechjepa.losses import compute_all_losses
from mechjepa.system_m import SystemM
print(f'  PyTorch: {torch.__version__}')
print(f'  CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'  GPU: {torch.cuda.get_device_name(0)}')
    print(f'  GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')
print('  MechJEPA imports OK!')
"

# Verify ALOE
cd "$WORKSPACE/cjepa"
PYTHONPATH="$WORKSPACE/cjepa" python -c "
from src.third_party.slotformer.clevrer_vqa.datasets.clevrer import CLEVRERSlotsVQADataset
from nerv.training import BaseDataModule
print('  ALOE imports OK!')
"

# Verify data
python -c "
import os
data_path = '$WORKSPACE/data/clevrer_videosaur_slots.pkl'
if os.path.exists(data_path):
    size_gb = os.path.getsize(data_path) / 1e9
    print(f'  Slot data: {size_gb:.1f} GB ✓')
else:
    print('  WARNING: Slot data not found!')
"

cd "$WORKSPACE"

echo ""
echo "========================================"
echo "  ✅ Setup complete!"
echo "  Finished: $(date)"
echo ""
echo "  Next steps:"
echo "    1. bash mechjepa/scripts/runpod_train.sh"
echo "    2. bash mechjepa/scripts/runpod_eval_aloe.sh"
echo "========================================"
