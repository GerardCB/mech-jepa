"""
Loss functions for MechJEPA.

Three loss components:
  1. JEPA prediction loss — MSE on masked slots + future slots (same as C-JEPA)
  2. Codebook commitment loss — VQ-VAE style: ||e_ij - sg(m_ij)||²
  3. Mechanism-level CTT losses — invariance/sufficiency on mechanism assignments
"""

import torch
import torch.nn.functional as F
import numpy as np


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


def codebook_commitment_loss(
    e_ij: torch.Tensor,
    m_ij: torch.Tensor,
    commitment_weight: float = 0.25,
) -> torch.Tensor:
    """
    VQ-VAE style commitment loss.

    Encourages edge features to commit to their assigned codebook entry.
    L = β * ||e_ij - sg(m_ij)||²

    Args:
        e_ij: (B, K, K, D) — edge features
        m_ij: (B, K, K, D) — bound mechanism vectors
        commitment_weight: β

    Returns:
        Scalar loss
    """
    return commitment_weight * F.mse_loss(e_ij, m_ij.detach())


def codebook_sharpness_loss(
    alpha_ij: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    Encourage each slot pair to commit to one or two mechanisms
    by minimizing per-pair entropy.

    L = mean over all pairs of H(alpha_ij[b, i, j, :])

    This is the OPPOSITE of the old diversity loss. Instead of forcing
    uniform usage (which made the codebook decorative), we force each
    individual pair to sharpen its assignment. Across the dataset,
    different pairs will naturally sharpen toward different entries,
    creating real mechanism differentiation.

    Args:
        alpha_ij: (B, K, K, N) — soft mechanism assignments

    Returns:
        Scalar loss (mean per-pair entropy — lower = sharper assignments)
    """
    # Per-pair entropy: H(alpha_ij[b, i, j, :]) for each pair
    per_pair_entropy = -(alpha_ij * (alpha_ij + eps).log()).sum(dim=-1)  # (B, K, K)
    return per_pair_entropy.mean()


def mechanism_invariance_loss(
    predictor,
    codebook,
    history: torch.Tensor,
    alpha_ij: torch.Tensor,
    m_ij: torch.Tensor,
    adj_threshold: float = 0.3,
) -> torch.Tensor:
    """
    Mechanism-level invariance loss (CTT Axiom 6).

    Key difference from ctt-jepa: invariance operates on mechanism ASSIGNMENTS,
    not attention weights. Two slots are mechanism-neighbors if they share
    a non-trivial mechanism binding (alpha_ij has significant weight
    on non-trivial codebook entries).

    Principle: masking slot i should not change the mechanism assignment
    for pairs (j, k) where neither j nor k interact with i.

    Args:
        predictor: MechSlotPredictor
        codebook: MechanismCodebook
        history: (B, T_hist, S, D)
        alpha_ij: (B, K, K, N) — current mechanism assignments
        m_ij: (B, K, K, D) — bound mechanisms
        adj_threshold: threshold for mechanism-based adjacency

    Returns:
        Scalar loss
    """
    B, T_hist, S, D = history.shape
    device = history.device

    # Mechanism-based adjacency: how much slot i and j share mechanisms
    # Use max assignment (strongest mechanism) as adjacency measure
    mech_strength = alpha_ij.max(dim=-1).values.mean(dim=0)  # (K, K)

    # Choose a random slot to mask
    target_slot = np.random.randint(S)

    # Find non-neighbors (slots NOT influenced by target_slot)
    influence = mech_strength[target_slot, :]  # (K,)
    non_neighbors = [
        j for j in range(S)
        if j != target_slot and influence[j].item() < adj_threshold
    ]

    if len(non_neighbors) == 0:
        return torch.tensor(0.0, device=device, requires_grad=True)

    # Masked forward: mask target_slot
    is_slot_masked = torch.zeros(S, dtype=torch.bool, device=device)
    is_slot_masked[target_slot] = True
    masked_indices = torch.tensor([target_slot], dtype=torch.long, device=device)

    x_masked = predictor.prepare_input_with_mask(history, is_slot_masked, masked_indices)

    # Run the masked input through the transformer to get hidden states
    # where the masking actually has an effect (unlike t=0 where all are visible)
    from einops import rearrange

    x_flat = rearrange(x_masked, "b t s d -> b (t s) d")
    out_flat, _ = predictor.transformer(x_flat, m_ij=m_ij, num_slots=S)
    out = rearrange(out_flat, "b (t s) d -> b t s d", t=predictor.total_frames, s=S)

    # Compute mechanism assignments from the transformer's hidden states
    # at the last history timestep (where masking is active)
    z_hidden = out[:, T_hist - 1, :, :]  # (B, S, D)
    codebook_out = codebook(z_hidden)
    alpha_masked = codebook_out["alpha_ij"]

    # For non-neighbor pairs, mechanism assignments should be unchanged
    loss = torch.tensor(0.0, device=device, requires_grad=True)
    count = 0
    for j in non_neighbors:
        for k in range(S):
            if k != target_slot:
                loss = loss + F.mse_loss(
                    alpha_masked[:, j, k, :],
                    alpha_ij[:, j, k, :].detach(),
                )
                count += 1

    return loss / max(count, 1)


def mechanism_sufficiency_loss(
    predictor,
    codebook,
    history: torch.Tensor,
    target: torch.Tensor,
    alpha_ij: torch.Tensor,
    m_ij: torch.Tensor,
    adj_threshold: float = 0.3,
    history_size: int = 3,
    num_preds: int = 1,
) -> torch.Tensor:
    """
    Mechanism-level sufficiency loss (CTT Axiom 4).

    The mechanism neighborhood of slot i should suffice to predict it.
    Masking mechanism-non-neighbors should not hurt prediction of i.

    Args:
        predictor: MechSlotPredictor
        codebook: MechanismCodebook
        history: (B, T_hist, S, D)
        target: (B, T_pred, S, D)
        alpha_ij: (B, K, K, N)
        m_ij: (B, K, K, D)
        adj_threshold: mechanism adjacency threshold
        history_size: number of history frames
        num_preds: number of prediction frames

    Returns:
        Scalar loss
    """
    B, T_hist, S, D = history.shape
    device = history.device

    # Mechanism-based adjacency
    mech_strength = alpha_ij.max(dim=-1).values.mean(dim=0)  # (K, K)

    # Choose target slot
    target_slot = np.random.randint(S)

    # Find non-neighbors of target_slot
    neighbors_of_target = mech_strength[target_slot, :]
    non_neighbors = [
        j for j in range(S)
        if j != target_slot and neighbors_of_target[j].item() < adj_threshold
    ]

    if len(non_neighbors) == 0:
        return torch.tensor(0.0, device=device, requires_grad=True)

    # Mask non-neighbors, keep target and its neighbors visible
    is_slot_masked = torch.zeros(S, dtype=torch.bool, device=device)
    for idx in non_neighbors:
        is_slot_masked[idx] = True
    mask_indices = torch.tensor(non_neighbors, dtype=torch.long, device=device)

    x_masked = predictor.prepare_input_with_mask(history, is_slot_masked, mask_indices)

    # Forward with mechanism bias
    from einops import rearrange

    x_flat = rearrange(x_masked, "b t s d -> b (t s) d")
    out_flat, _ = predictor.transformer(x_flat, m_ij=m_ij, num_slots=S)
    out = rearrange(out_flat, "b (t s) d -> b t s d", t=predictor.total_frames, s=S)
    out = predictor.to_out(out)

    pred_future = out[:, history_size : history_size + num_preds, :, :]

    # Target slot prediction should still be accurate
    loss = F.mse_loss(
        pred_future[:, :, target_slot, :],
        target[:, :, target_slot, :].detach(),
    )

    return loss


def compute_all_losses(
    pred_embedding: torch.Tensor,
    history: torch.Tensor,
    target: torch.Tensor,
    mask_indices: torch.Tensor,
    codebook_output: dict[str, torch.Tensor],
    cfg: dict,
    predictor=None,
    codebook_module=None,
    epoch: int = 0,
) -> dict[str, torch.Tensor]:
    """
    Compute all MechJEPA losses.

    Args:
        pred_embedding: (B, T_total, S, D)
        history: (B, T_hist, S, D)
        target: (B, T_pred, S, D)
        mask_indices: (M,)
        codebook_output: dict from MechanismCodebook.forward()
        cfg: config dict with loss weights
        predictor: MechSlotPredictor (needed for CTT losses)
        codebook_module: MechanismCodebook (needed for CTT losses)
        epoch: current epoch

    Returns:
        dict of all losses including total
    """
    device = pred_embedding.device
    history_size = cfg.get("history_size", 3)
    num_preds = cfg.get("num_preds", 1)

    losses = {}

    # 1. JEPA prediction loss
    jepa_losses = jepa_prediction_loss(
        pred_embedding, history, target, mask_indices, history_size, num_preds
    )
    losses.update(jepa_losses)

    total_loss = jepa_losses["loss_jepa"]

    # 2. Codebook commitment loss
    commitment_weight = cfg.get("commitment_weight", 0.25)
    if commitment_weight > 0:
        loss_commitment = codebook_commitment_loss(
            codebook_output["e_ij"],
            codebook_output["m_ij"],
            commitment_weight,
        )
        losses["loss_commitment"] = loss_commitment
        total_loss = total_loss + loss_commitment
    else:
        losses["loss_commitment"] = torch.tensor(0.0, device=device)

    # 3. Sharpness loss (per-pair entropy minimization)
    sharpness_weight = cfg.get("sharpness_weight", 0.1)
    if sharpness_weight > 0:
        loss_sharpness = codebook_sharpness_loss(codebook_output["alpha_ij"])
        losses["loss_sharpness"] = loss_sharpness
        total_loss = total_loss + sharpness_weight * loss_sharpness
    else:
        losses["loss_sharpness"] = torch.tensor(0.0, device=device)

    # 4. Mechanism-level CTT losses (optional, phased)
    ctt_start_epoch = cfg.get("ctt_start_epoch", 20)
    ctt_active = epoch >= ctt_start_epoch

    inv_weight = cfg.get("ctt_inv_weight", 0.0)
    suf_weight = cfg.get("ctt_suf_weight", 0.0)

    if ctt_active and inv_weight > 0 and predictor is not None and codebook_module is not None:
        loss_inv = mechanism_invariance_loss(
            predictor, codebook_module, history,
            codebook_output["alpha_ij"], codebook_output["m_ij"],
            adj_threshold=cfg.get("ctt_adj_threshold", 0.3),
        )
        losses["loss_inv"] = loss_inv
        total_loss = total_loss + inv_weight * loss_inv
    else:
        losses["loss_inv"] = torch.tensor(0.0, device=device)

    if ctt_active and suf_weight > 0 and predictor is not None:
        loss_suf = mechanism_sufficiency_loss(
            predictor, codebook_module, history, target,
            codebook_output["alpha_ij"], codebook_output["m_ij"],
            adj_threshold=cfg.get("ctt_adj_threshold", 0.3),
            history_size=history_size,
            num_preds=num_preds,
        )
        losses["loss_suf"] = loss_suf
        total_loss = total_loss + suf_weight * loss_suf
    else:
        losses["loss_suf"] = torch.tensor(0.0, device=device)

    losses["loss"] = total_loss
    return losses
