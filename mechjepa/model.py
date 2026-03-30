from __future__ import annotations

"""
MechJEPA — Full model combining all components.

Architecture:
  1. MechanismCodebook binds slot pairs to mechanism types
  2. MechSlotPredictor predicts future states with mechanism-biased attention
  3. System M monitors surprise and triggers codebook maintenance
  4. Loss functions combine JEPA prediction + codebook commitment + CTT

Usage:
    model = MechJEPA(num_slots=7, slot_dim=128)
    outputs = model(history_slots)
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
    - MechanismCodebook: learns N mechanism prototypes
    - MechSlotPredictor: slot transformer with mechanism bias
    - SystemM: surprise-driven mode switching (non-parametric)

    Args:
        num_slots: K — number of object slots per frame
        slot_dim: D — dimension of each slot
        num_mechanisms: N — number of mechanism types in codebook
        history_frames: T_hist — number of history frames
        pred_frames: T_pred — number of future frames to predict
        num_masked_slots: M — slots to mask during training
        codebook_temperature: τ_start — Gumbel-softmax initial temperature
        codebook_temperature_min: τ_end — final temperature after annealing
        codebook_anneal_epochs: epochs to anneal over
        commitment_weight: β — codebook commitment loss weight
        sharpness_weight: per-pair entropy minimization weight
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
        codebook_temperature: float = 1.0,
        codebook_temperature_min: float = 0.1,
        codebook_anneal_epochs: int = 30,
        commitment_weight: float = 0.25,
        sharpness_weight: float = 0.1,
        edge_hidden_dim: int = 256,
        transformer_depth: int = 6,
        transformer_heads: int = 8,
        transformer_dim_head: int = 64,
        transformer_mlp_dim: int = 2048,
        dropout: float = 0.1,
        seed: int = 42,
        surprise_threshold: float = 1.0,
    ):
        super().__init__()

        # Store config for loss computation
        self.num_slots = num_slots
        self.slot_dim = slot_dim
        self.num_mechanisms = num_mechanisms
        self.history_frames = history_frames
        self.pred_frames = pred_frames
        self.commitment_weight = commitment_weight
        self.sharpness_weight = sharpness_weight

        # 1. Mechanism Codebook
        self.codebook = MechanismCodebook(
            num_mechanisms=num_mechanisms,
            slot_dim=slot_dim,
            temperature=codebook_temperature,
            temperature_min=codebook_temperature_min,
            anneal_epochs=codebook_anneal_epochs,
            commitment_weight=commitment_weight,
            dead_threshold=0.01,
            edge_hidden_dim=edge_hidden_dim,
        )

        # 2. Mechanism-Biased Slot Predictor
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

        # 3. System M (non-parametric, not nn.Module)
        self.system_m = SystemM(
            surprise_threshold=surprise_threshold,
        )

    def forward(
        self,
        history: torch.Tensor,
        return_attention: bool = False,
        epoch: int = 0,
        max_epochs: int = 100,
    ) -> dict[str, torch.Tensor]:
        """
        Full forward pass: codebook binding → mechanism-gated dynamics → prediction.

        Args:
            history: (B, T_hist, S, D) — slot history
            return_attention: whether to return attention weights
            epoch: current training epoch (for Gumbel temperature annealing)
            max_epochs: total training epochs

        Returns:
            dict with:
                pred_embedding: (B, T_total, S, D)
                mask_indices: (M,)
                codebook_output: dict from MechanismCodebook
                attn_list: attention weights (if requested)
        """
        B, T_hist, S, D = history.shape

        # Step 1: Compute mechanism bindings using the last history frame
        z_t = history[:, -1, :, :]  # (B, S, D)
        codebook_output = self.codebook(z_t, epoch=epoch, max_epochs=max_epochs)
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
        epoch: int = 0,
        cfg: dict | None = None,
    ) -> dict[str, torch.Tensor]:
        """
        Compute all losses from forward outputs.

        Args:
            outputs: dict from forward()
            history: (B, T_hist, S, D)
            target: (B, T_pred, S, D)
            epoch: current training epoch
            cfg: config dict with loss weights (uses defaults if None)

        Returns:
            dict of all losses
        """
        if cfg is None:
            cfg = {
                "history_size": self.history_frames,
                "num_preds": self.pred_frames,
                "commitment_weight": self.commitment_weight,
                "sharpness_weight": self.sharpness_weight,
                "ctt_inv_weight": 0.0,
                "ctt_suf_weight": 0.0,
                "ctt_start_epoch": 20,
                "ctt_adj_threshold": 0.3,
            }

        return compute_all_losses(
            pred_embedding=outputs["pred_embedding"],
            history=history,
            target=target,
            mask_indices=outputs["mask_indices"],
            codebook_output=outputs["codebook_output"],
            cfg=cfg,
            predictor=self.predictor,
            codebook_module=self.codebook,
            epoch=epoch,
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

    def maintain_codebook(
        self,
        e_ij: torch.Tensor,
        surprise_ij: torch.Tensor,
    ):
        """
        Run codebook maintenance: reallocate dead entries.

        Args:
            e_ij: (B, K, K, D) — edge features
            surprise_ij: (B, K, K) — per-edge surprise
        """
        self.codebook.reallocate_dead_entries(e_ij, surprise_ij)

    def get_diagnostics(self) -> dict:
        """
        Return model diagnostics for logging.

        Returns:
            dict with codebook stats and System M stats
        """
        diagnostics = {}
        diagnostics.update(self.codebook.get_codebook_stats())
        diagnostics.update(self.system_m.get_stats())
        return diagnostics

    def get_parameter_count(self) -> dict[str, int]:
        """
        Return parameter counts by component.

        Returns:
            dict with counts for codebook, predictor, total
        """
        codebook_params = sum(p.numel() for p in self.codebook.parameters())
        predictor_params = sum(p.numel() for p in self.predictor.parameters())
        total = codebook_params + predictor_params

        return {
            "codebook_params": codebook_params,
            "predictor_params": predictor_params,
            "total_params": total,
            "codebook_overhead": codebook_params,
        }
