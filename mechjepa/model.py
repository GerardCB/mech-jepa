from __future__ import annotations

"""
MechJEPA — Full model combining all components.

Architecture:
  1. MechanismCodebook: continuous low-rank bottleneck for pairwise edge features
  2. MechSlotPredictor: predicts future states with mechanism-gated attention
  3. System M: monitors surprise and triggers maintenance (optional)
  4. Loss: prediction loss only (structure emerges from the objective)

Usage:
    model = MechJEPA(num_slots=7, slot_dim=128)
    outputs = model(history)
    losses = model.compute_loss(outputs, history, target)
"""

import torch
import torch.nn as nn

from mechjepa.codebook import MechanismCodebook
from mechjepa.dynamics import MechSlotPredictor
from mechjepa.system_m import SystemM, compute_surprise_from_prediction
from mechjepa.losses import compute_all_losses


class MechJEPA(nn.Module):
    """
    MechJEPA: World Models with Persistent Mechanism Memory.

    Combines:
    - MechanismCodebook: continuous low-rank edge bottleneck
    - MechSlotPredictor: slot transformer with mechanism-gated attention
    - SystemM: surprise-driven mode switching (non-parametric)

    Args:
        num_slots: K — number of object slots per frame
        slot_dim: D — dimension of each slot
        num_mechanisms: N — bottleneck dimension
        history_frames: T_hist — number of history frames
        pred_frames: T_pred — number of future frames to predict
        num_masked_slots: M — slots to mask during training
        edge_hidden_dim: hidden dim of edge MLP
        transformer_depth: depth of slot transformer
        transformer_heads: number of attention heads
        transformer_dim_head: dimension per attention head
        transformer_mlp_dim: FFN hidden dimension
        dropout: dropout rate
        seed: random seed for masking
        surprise_threshold: System M surprise threshold
    """

    def __init__(
        self,
        num_slots: int = 7,
        slot_dim: int = 128,
        num_mechanisms: int = 16,
        history_frames: int = 3,
        pred_frames: int = 1,
        num_masked_slots: int = 2,
        edge_hidden_dim: int = 256,
        transformer_depth: int = 6,
        transformer_heads: int = 8,
        transformer_dim_head: int = 64,
        transformer_mlp_dim: int = 2048,
        dropout: float = 0.1,
        seed: int = 42,
        surprise_threshold: float = 1.0,
        # Legacy kwargs from VQ config (accepted, ignored)
        **kwargs,
    ):
        super().__init__()

        self.num_slots = num_slots
        self.slot_dim = slot_dim
        self.num_mechanisms = num_mechanisms
        self.history_frames = history_frames
        self.pred_frames = pred_frames

        # 1. Mechanism Bottleneck (continuous)
        self.codebook = MechanismCodebook(
            num_mechanisms=num_mechanisms,
            slot_dim=slot_dim,
            edge_hidden_dim=edge_hidden_dim,
        )

        # 2. Mechanism-Gated Slot Predictor
        self.predictor = MechSlotPredictor(
            num_slots=num_slots,
            slot_dim=slot_dim,
            history_frames=history_frames,
            pred_frames=pred_frames,
            num_masked_slots=num_masked_slots,
            seed=seed,
            depth=transformer_depth,
            heads=transformer_heads,
            dim_head=transformer_dim_head,
            mlp_dim=transformer_mlp_dim,
            dropout=dropout,
        )

        # 3. System M (non-parametric)
        self.system_m = SystemM(
            surprise_threshold=surprise_threshold,
        )

    def forward(
        self,
        history: torch.Tensor,
        return_attention: bool = False,
        **kwargs,
    ) -> dict[str, torch.Tensor]:
        """
        Full forward pass: bottleneck binding → mechanism-gated dynamics → prediction.

        Args:
            history: (B, T_hist, S, D) — slot history
            return_attention: whether to return attention weights

        Returns:
            dict with pred_embedding, mask_indices, codebook_output, attn_list
        """
        B, T_hist, S, D = history.shape

        # Step 1: Compute mechanism bindings using the last history frame
        z_t = history[:, -1, :, :]  # (B, S, D)
        codebook_output = self.codebook(z_t)
        m_ij = codebook_output["m_ij"]  # (B, S, S, D)

        # Step 2: Run mechanism-gated dynamics
        if return_attention:
            pred_embedding, mask_indices, attn_list = self.predictor(
                history, m_ij=m_ij, return_attention=True,
            )
        else:
            pred_embedding, mask_indices = self.predictor(
                history, m_ij=m_ij,
            )
            attn_list = None

        return {
            "pred_embedding": pred_embedding,
            "mask_indices": mask_indices,
            "codebook_output": codebook_output,
            "attn_list": attn_list,
        }

    def compute_loss(
        self,
        outputs: dict[str, torch.Tensor],
        history: torch.Tensor,
        target: torch.Tensor,
        cfg: dict | None = None,
        **kwargs,
    ) -> dict[str, torch.Tensor]:
        """
        Compute losses from forward outputs.

        Args:
            outputs: dict from forward()
            history: (B, T_hist, S, D)
            target: (B, T_pred, S, D)
            cfg: config dict with loss weights

        Returns:
            dict of all losses
        """
        if cfg is None:
            cfg = {
                "history_size": self.history_frames,
                "num_preds": self.pred_frames,
                "bottleneck_recon_weight": 0.0,
            }

        return compute_all_losses(
            pred_embedding=outputs["pred_embedding"],
            history=history,
            target=target,
            mask_indices=outputs["mask_indices"],
            codebook_output=outputs["codebook_output"],
            cfg=cfg,
        )

    @torch.no_grad()
    def inference(self, history: torch.Tensor) -> torch.Tensor:
        """
        Predict future frames from fully visible history.

        Args:
            history: (B, T_hist, S, D)

        Returns:
            future: (B, T_pred, S, D)
        """
        z_t = history[:, -1, :, :]
        codebook_output = self.codebook(z_t)
        m_ij = codebook_output["m_ij"]
        return self.predictor.inference(history, m_ij=m_ij)

    def get_diagnostics(self) -> dict:
        """Return model diagnostics for logging."""
        diagnostics = {}
        diagnostics.update(self.codebook.get_codebook_stats())
        diagnostics.update(self.system_m.get_stats())
        return diagnostics

    def get_parameter_count(self) -> dict[str, int]:
        """Return parameter counts by component."""
        codebook_params = sum(p.numel() for p in self.codebook.parameters())
        predictor_params = sum(p.numel() for p in self.predictor.parameters())
        total = codebook_params + predictor_params

        return {
            "codebook_params": codebook_params,
            "predictor_params": predictor_params,
            "total_params": total,
            "codebook_overhead": codebook_params,
        }
