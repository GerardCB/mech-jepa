"""
Loss functions for MechJEPA.

Two loss components (aligned with LeWorldModel's simplicity philosophy):
  1. JEPA prediction loss — MSE on masked slots + future slots (same as C-JEPA)
  2. Bottleneck reconstruction loss — optional ||e_ij - m_ij||² to prevent
     the bottleneck from going to zero (only if needed)

All auxiliary VQ losses (commitment, diversity, sharpness) removed — the
continuous bottleneck + multiplicative gate gets direct gradient from the
prediction objective. No auxiliary signal needed.
"""

import torch
import torch.nn.functional as F


def jepa_prediction_loss(
    pred_embedding: torch.Tensor,
    history: torch.Tensor,
    target: torch.Tensor,
    mask_indices: torch.Tensor,
    history_size: int,
    num_preds: int,
) -> dict[str, torch.Tensor]:
    """
    Standard C-JEPA prediction loss: MSE on masked history + future.

    Args:
        pred_embedding: (B, T_total, S, D) — full predicted sequence
        history: (B, T_hist, S, D) — ground truth history
        target: (B, T_pred, S, D) — ground truth future
        mask_indices: (M,) — indices of masked slots
        history_size: T_hist
        num_preds: T_pred

    Returns:
        dict with loss_masked_history, loss_future, and total loss
    """
    device = pred_embedding.device

    pred_history = pred_embedding[:, :history_size, :, :]
    pred_future = pred_embedding[:, history_size : history_size + num_preds, :, :]

    # Masked history loss (exclude t=0 where all slots are visible anchors)
    if len(mask_indices) > 0 and history_size > 1:
        loss_masked_history = F.mse_loss(
            pred_history[:, 1:, mask_indices, :],
            history[:, 1:, mask_indices, :].detach(),
        )
    else:
        loss_masked_history = torch.tensor(0.0, device=device)

    # Future prediction loss
    loss_future = F.mse_loss(pred_future, target.detach())

    return {
        "loss_masked_history": loss_masked_history,
        "loss_future": loss_future,
        "loss_jepa": loss_masked_history + loss_future,
    }


def bottleneck_reconstruction_loss(
    e_ij: torch.Tensor,
    m_ij: torch.Tensor,
    weight: float = 0.1,
) -> torch.Tensor:
    """
    Optional reconstruction loss for the bottleneck.

    Encourages the encode→decode path to faithfully reconstruct edge features.
    Prevents the bottleneck from collapsing to zero.

    L = weight * ||e_ij - m_ij||²

    Args:
        e_ij: (B, K, K, D) — raw edge features (before bottleneck)
        m_ij: (B, K, K, D) — decoded mechanism vectors (after bottleneck)
        weight: loss weight

    Returns:
        Scalar loss
    """
    return weight * F.mse_loss(e_ij, m_ij)


def compute_all_losses(
    pred_embedding: torch.Tensor,
    history: torch.Tensor,
    target: torch.Tensor,
    mask_indices: torch.Tensor,
    codebook_output: dict[str, torch.Tensor],
    cfg: dict,
    **kwargs,
) -> dict[str, torch.Tensor]:
    """
    Compute all MechJEPA losses.

    Simple: prediction loss + optional bottleneck reconstruction.

    Args:
        pred_embedding: (B, T_total, S, D)
        history: (B, T_hist, S, D)
        target: (B, T_pred, S, D)
        mask_indices: (M,)
        codebook_output: dict from MechanismCodebook.forward()
        cfg: config dict
        **kwargs: ignored (backward compat)

    Returns:
        dict of all losses including total
    """
    device = pred_embedding.device
    history_size = cfg.get("history_size", 3)
    num_preds = cfg.get("num_preds", 1)

    losses = {}

    # 1. JEPA prediction loss (the primary objective)
    jepa_losses = jepa_prediction_loss(
        pred_embedding, history, target, mask_indices, history_size, num_preds
    )
    losses.update(jepa_losses)

    total_loss = jepa_losses["loss_jepa"]

    # 2. Bottleneck reconstruction (optional, prevents zero collapse)
    recon_weight = cfg.get("bottleneck_recon_weight", 0.0)
    if recon_weight > 0 and "e_ij" in codebook_output and "m_ij" in codebook_output:
        loss_recon = bottleneck_reconstruction_loss(
            codebook_output["e_ij"],
            codebook_output["m_ij"],
            recon_weight,
        )
        losses["loss_recon"] = loss_recon
        total_loss = total_loss + loss_recon
    else:
        losses["loss_recon"] = torch.tensor(0.0, device=device)

    losses["loss"] = total_loss
    return losses
