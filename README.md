# MechJEPA: World Models with Persistent Mechanism Memory

**MechJEPA** — Mechanism-Memory Joint Embedding Predictive Architecture.

A JEPA-based world model that learns persistent, transferable interaction patterns (collisions, gravity, free-flight) through a mechanism codebook. Built on top of the C-JEPA architecture with three key innovations:

1. **Mechanism Codebook** — `nn.Embedding(16, 128)` storing learned interaction prototypes
2. **Edge Binding** — MLP that matches slot pairs to codebook entries via soft assignment
3. **Mechanism-Biased Attention** — codebook entries bias the slot transformer's dynamics

This is still a work in progress.

## Key Idea

**Why not just add CTT losses to attention?** We tried — CTT-JEPA enforced causal invariance directly on attention weights, and it *hurt* performance. The reason: attention ≠ causality. A slot can attend to another slot without there being a causal interaction (e.g., background attends to foreground for context), and causal mechanisms might not be represented by any slot.

**MechJEPA's fix:** Give physical mechanisms their own representation (the codebook). Invariance operates at the mechanism level — "collision between ball A and ball B should look the same regardless of which other balls are in the scene" — not at the attention level. The codebook entries learn to serve as the canonical vocabulary for physical interactions.

## Codebook Mechanics

The codebook is the persistent memory that survives across episodes:

- **16 entries** start as random vectors
- During training, some learn useful patterns (collisions, gravity) and get high usage
- Others stay "dead" (low EMA usage)
- When a novel interaction appears (high surprise), the most surprised edge's feature replaces a dead entry
- Over time, the codebook fills with the actual mechanism vocabulary of the environment


## Results

| Method | Desc | CF | Expl | Pred |
|--------|------|-----|------|------|
| C-JEPA (published) | 91.0 | 50.3 | 82.5 | 79.6 |
| C-JEPA (reproduced) | 91.4 | 57.5 | 84.9 | 80.1 |
| CTT-JEPA (attention-level) | 89.8 | 49.4 | 81.0 | 77.9 |
| **MechJEPA (codebook)** | **???** | **???** | **???** | **???** |

## Quick Start

### Installation

```bash
git clone https://github.com/GerardCB/mechjepa.git
cd mechjepa
pip install -e ".[dev]"
```

### Run Tests

```bash
python -m pytest tests/ -v
```

### Train on CLEVRER

```bash
# Single GPU
python scripts/train_clevrer.py \
    embedding_dir=/path/to/clevrer_videosaur_slots.pkl \
    wandb.enable=false

# Multi-GPU
torchrun --nproc_per_node=4 scripts/train_clevrer.py \
    embedding_dir=/path/to/clevrer_videosaur_slots.pkl \
    wandb.entity=your-entity
```

### Evaluate (ALOE VQA)

```bash
python scripts/eval_aloe.py \
    rollout_path=/path/to/rollout_mechjepa_clevrer.pkl
```

## Project Structure

```
mechjepa/
├── mechjepa/
│   ├── codebook.py      # Mechanism codebook + binding + maintenance
│   ├── dynamics.py      # Slot transformer + mechanism bias
│   ├── losses.py        # JEPA + commitment + CTT losses
│   ├── system_m.py      # Surprise routing + mode switching
│   ├── model.py         # Full MechJEPA model
│   └── data/
│       └── clevrer_slots.py  # CLEVRER/PushT data loading
├── configs/
│   ├── clevrer.yaml     # CLEVRER training config
│   └── pusht.yaml       # Push-T training config
├── scripts/
│   └── train_clevrer.py # Training with DDP + wandb
├── tests/
│   └── test_model.py    # Unit + integration tests
└── notebooks/
    ├── codebook_viz.ipynb    # Codebook differentiation demo
    ├── surprise_demo.ipynb   # System M in action
    └── transfer_demo.ipynb   # Cross-task mechanism transfer
```

## License

MIT
