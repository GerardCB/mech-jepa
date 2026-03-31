"""
VideoSAUR Slot Encoder for MechJEPA.

Loads the C-JEPA VideoSAUR checkpoint directly:
  - DINOv2 ViT-S/14 backbone (384D, 12 layers, 518×518 input)
  - Output transform: LayerNorm → Linear(384→768) → GELU → Linear(768→128)
  - GRU slot corrector: cross-attention + GRU update, 3 iterations → 4 slots × 128D

Usage:
    encoder = VideoSAUREncoder.from_ckpt("/path/to/pusht_videosaur_model.ckpt")
    frame = env.render()              # (H, W, 3) uint8 RGB
    slots = encoder.encode(frame)     # (4, 128) tensor on device
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image as PILImage

VIT_SIZE = 518
N_SLOTS = 4
SLOT_DIM = 128
N_CORRECTOR_ITERS = 3


class VideoSAUREncoder(nn.Module):
    """
    Minimal VideoSAUR encoder reconstructed from checkpoint weights.

    Architecture (from pusht_videosaur_model.ckpt):
      encoder.module.backbone.*   → DINOv2 ViT-S/14  (384D hidden, 12 layers)
      encoder.module.output_transform.* → LayerNorm + MLP (384→128D)
      processor.module.corrector.* → GRU-based slot corrector
      initializer.*                → Learned slot initialization (1×1×128)
    """

    def __init__(self, ckpt_path: str, device: str = "cuda",
                 n_slots: int = N_SLOTS, n_corrector_iters: int = N_CORRECTOR_ITERS):
        super().__init__()
        self.device = device
        self.n_slots = n_slots
        self.n_iter = n_corrector_iters

        # Import here to keep the dependency optional at module level
        from transformers import Dinov2Config, Dinov2Model

        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        sd = ckpt["state_dict"]

        # ── ViT backbone (DINOv2 architecture) ────────────────────────────────
        vit_sd = {
            k[len("encoder.module.backbone."):]: v
            for k, v in sd.items()
            if k.startswith("encoder.module.backbone.")
        }
        vit_cfg = Dinov2Config(
            hidden_size=384,
            num_hidden_layers=12,
            num_attention_heads=6,
            mlp_ratio=4,
            image_size=VIT_SIZE,
            patch_size=14,
            num_channels=3,
            hidden_act="gelu",
        )
        self.vit = Dinov2Model(vit_cfg)
        self.vit.load_state_dict(vit_sd, strict=True)

        # ── Output transform: LayerNorm → Linear → GELU → Linear (→128D) ─────
        self.ot_ln = nn.LayerNorm(384)
        self.ot_ln.weight = nn.Parameter(sd["encoder.module.output_transform.layers.0.weight"])
        self.ot_ln.bias = nn.Parameter(sd["encoder.module.output_transform.layers.0.bias"])

        self.ot_linear1 = nn.Linear(384, 768)
        self.ot_linear1.weight = nn.Parameter(sd["encoder.module.output_transform.layers.1.weight"])
        self.ot_linear1.bias = nn.Parameter(sd["encoder.module.output_transform.layers.1.bias"])

        self.ot_linear2 = nn.Linear(768, SLOT_DIM)
        self.ot_linear2.weight = nn.Parameter(sd["encoder.module.output_transform.layers.3.weight"])
        self.ot_linear2.bias = nn.Parameter(sd["encoder.module.output_transform.layers.3.bias"])

        # ── Slot corrector (GRU-based cross-attention) ────────────────────────
        self.corr_to_k = nn.Linear(SLOT_DIM, SLOT_DIM, bias=False)
        self.corr_to_k.weight = nn.Parameter(sd["processor.module.corrector.to_k.weight"])

        self.corr_to_v = nn.Linear(SLOT_DIM, SLOT_DIM, bias=False)
        self.corr_to_v.weight = nn.Parameter(sd["processor.module.corrector.to_v.weight"])

        self.corr_to_q = nn.Linear(SLOT_DIM, SLOT_DIM, bias=False)
        self.corr_to_q.weight = nn.Parameter(sd["processor.module.corrector.to_q.weight"])

        self.corr_norm_features = nn.LayerNorm(SLOT_DIM)
        self.corr_norm_features.weight = nn.Parameter(sd["processor.module.corrector.norm_features.weight"])
        self.corr_norm_features.bias = nn.Parameter(sd["processor.module.corrector.norm_features.bias"])

        self.corr_norm_slots = nn.LayerNorm(SLOT_DIM)
        self.corr_norm_slots.weight = nn.Parameter(sd["processor.module.corrector.norm_slots.weight"])
        self.corr_norm_slots.bias = nn.Parameter(sd["processor.module.corrector.norm_slots.bias"])

        self.corr_gru = nn.GRUCell(SLOT_DIM, SLOT_DIM)
        self.corr_gru.weight_ih = nn.Parameter(sd["processor.module.corrector.gru.weight_ih"])
        self.corr_gru.weight_hh = nn.Parameter(sd["processor.module.corrector.gru.weight_hh"])
        self.corr_gru.bias_ih = nn.Parameter(sd["processor.module.corrector.gru.bias_ih"])
        self.corr_gru.bias_hh = nn.Parameter(sd["processor.module.corrector.gru.bias_hh"])

        # ── Slot initialiser ──────────────────────────────────────────────────
        self.slot_init_mean = nn.Parameter(sd["initializer.mean"])  # (1, 1, 128)

        # ── Image preprocessing (ImageNet normalisation) ──────────────────────
        self.register_buffer("img_mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("img_std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

        self.to(device)

    # ── Forward components ────────────────────────────────────────────────────

    def _output_transform(self, x: torch.Tensor) -> torch.Tensor:
        """384D ViT features → 128D slot features."""
        x = self.ot_ln(x)
        x = self.ot_linear1(x)
        x = F.gelu(x)
        x = self.ot_linear2(x)
        return x

    def _slot_corrector_step(self, features: torch.Tensor, slots: torch.Tensor) -> torch.Tensor:
        """One iteration of GRU-based slot attention.

        Args:
            features: (B, N, D) — patch features
            slots:    (B, S, D) — current slot states
        Returns:
            new_slots: (B, S, D)
        """
        B, N, D = features.shape
        S = slots.shape[1]

        feat_n = self.corr_norm_features(features)
        slots_n = self.corr_norm_slots(slots)

        k = self.corr_to_k(feat_n)   # (B, N, D)
        v = self.corr_to_v(feat_n)   # (B, N, D)
        q = self.corr_to_q(slots_n)  # (B, S, D)

        # Slot attention: softmax over slots, then normalise over features
        attn = torch.einsum("bnd,bsd->bsn", k, q) * (D ** -0.5)
        attn = attn.softmax(dim=-2)  # softmax over slots dim
        attn = attn / (attn.sum(dim=-1, keepdim=True) + 1e-8)
        updates = torch.einsum("bsn,bnd->bsd", attn, v)  # (B, S, D)

        # GRU update: input=updates, hidden=slots
        new_slots = self.corr_gru(
            updates.reshape(B * S, D),
            slots.reshape(B * S, D),
        )
        return new_slots.reshape(B, S, D)

    # ── Public API ────────────────────────────────────────────────────────────

    @torch.no_grad()
    def encode(self, frame_rgb: np.ndarray) -> torch.Tensor:
        """Encode a single RGB frame into slot representations.

        Args:
            frame_rgb: (H, W, 3) uint8 numpy RGB image (any resolution)
        Returns:
            slots: (N_SLOTS, SLOT_DIM) float32 tensor on self.device
        """
        # Resize to ViT input size and normalise
        img = PILImage.fromarray(frame_rgb).resize((VIT_SIZE, VIT_SIZE), PILImage.BICUBIC)
        x = torch.from_numpy(np.array(img)).float() / 255.0   # (H, W, 3)
        x = x.permute(2, 0, 1).unsqueeze(0).to(self.device)   # (1, 3, H, W)
        x = (x - self.img_mean) / self.img_std

        # ViT forward → patch features (skip CLS token)
        vit_out = self.vit(pixel_values=x)
        patch_feats = vit_out.last_hidden_state[:, 1:, :]  # (1, 1369, 384)

        # Output transform → 128D
        feats = self._output_transform(patch_feats)  # (1, 1369, 128)

        # Slot corrector: iterative cross-attention refinement
        slots = self.slot_init_mean.expand(1, self.n_slots, SLOT_DIM).clone()
        for _ in range(self.n_iter):
            slots = self._slot_corrector_step(feats, slots)  # (1, 4, 128)

        return slots.squeeze(0)  # (4, 128)

    @classmethod
    def from_ckpt(cls, ckpt_path: str, device: str = "cuda") -> "VideoSAUREncoder":
        """Load encoder from a VideoSAUR checkpoint.

        Args:
            ckpt_path: Path to pusht_videosaur_model.ckpt
            device: Target device ("cuda" or "cpu")
        Returns:
            Encoder in eval mode.
        """
        encoder = cls(ckpt_path, device=device)
        encoder.eval()
        return encoder
