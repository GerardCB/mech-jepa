"""
System M — Surprise-driven mode switching and codebook maintenance.

System M is NOT a neural network. It's a conditional branch on the
per-edge prediction error signal, following LeCun's proposal that
System M outputs "meta-actions" like "switch to learning mode."

Components:
  1. Per-edge surprise: decompose prediction error per slot pair
  2. Mode routing: threshold-based INFER vs LEARN mode switching
  3. Codebook maintenance: dead entry reallocation on novel interactions
"""

import torch
import torch.nn.functional as F
from einops import rearrange


def compute_per_slot_surprise(
    z_pred: torch.Tensor,
    z_actual: torch.Tensor,
) -> torch.Tensor:
    """
    Compute per-slot prediction error.

    Args:
        z_pred: (B, K, D) — predicted slot vectors
        z_actual: (B, K, D) — actual slot vectors (from encoder)

    Returns:
        slot_error: (B, K) — L2 error per slot
    """
    return (z_pred - z_actual).pow(2).sum(dim=-1)  # (B, K)


def compute_per_edge_surprise(
    slot_error: torch.Tensor,
    attn_weights: torch.Tensor,
) -> torch.Tensor:
    """
    Decompose per-slot error into per-edge surprise using attention
    weights as attribution.

    surprise_ij = attn[i, j] * error[i]
    "Slot i's prediction was wrong because of its interaction with slot j"

    Args:
        slot_error: (B, K) — per-slot prediction error
        attn_weights: (B, K, K) — attention weights (slot-to-slot)

    Returns:
        surprise_ij: (B, K, K) — per-edge surprise
    """
    # attn[i, j] * error[i] → how much pair (i, j) contributed to slot i's error
    return attn_weights * slot_error.unsqueeze(2)  # (B, K, K)


def compute_surprise_from_prediction(
    predictor,
    codebook,
    z_history: torch.Tensor,
    z_next_actual: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """
    Full surprise computation pipeline.

    1. Run dynamics to get predicted next state + attention weights
    2. Compute per-slot error against actual next state
    3. Decompose into per-edge surprise

    Args:
        predictor: MechSlotPredictor
        codebook: MechanismCodebook
        z_history: (B, T_hist, S, D) — history slots
        z_next_actual: (B, S, D) — actual next-frame slots

    Returns:
        dict with:
            surprise_ij: (B, K, K) — per-edge surprise
            slot_error: (B, K) — per-slot error
            max_surprise: scalar — maximum surprise value
            most_surprising_pair: (i, j) indices
    """
    B, T_hist, S, D = z_history.shape

    # Get mechanism binding
    z_t = z_history[:, -1, :, :]  # last frame for mechanism computation
    codebook_out = codebook(z_t)
    m_ij = codebook_out["m_ij"]

    # Run predictor with attention extraction
    with torch.no_grad():
        pred_out, _, attn_list = predictor(
            z_history, m_ij=m_ij, return_attention=True,
        )

    # Extract predicted next frame
    z_pred = pred_out[:, T_hist, :, :]  # (B, S, D) — first predicted frame

    # Per-slot error
    slot_error = compute_per_slot_surprise(z_pred, z_next_actual)

    # Get slot-to-slot attention from the last layer
    if attn_list and len(attn_list) > 0:
        attn_weights = attn_list[-1]  # (B, T*S, T*S)

        # Extract slot-to-slot attention at the prediction timestep
        T_total = predictor.total_frames
        attn_reshaped = rearrange(
            attn_weights,
            "b (tq sq) (tk sk) -> b tq sq tk sk",
            tq=T_total, sq=S, tk=T_total, sk=S,
        )
        # Future frame attending to last history frame
        slot_attn = attn_reshaped[:, T_hist, :, T_hist - 1, :]  # (B, S, S)
    else:
        # Uniform attention fallback
        slot_attn = torch.ones(B, S, S, device=z_history.device) / S

    # Per-edge surprise
    surprise_ij = compute_per_edge_surprise(slot_error, slot_attn)

    # Find most surprising pair
    max_surprise = surprise_ij.max()
    flat_idx = surprise_ij.reshape(-1).argmax()
    b_idx = flat_idx // (S * S)
    remaining = flat_idx % (S * S)
    i_idx = remaining // S
    j_idx = remaining % S

    return {
        "surprise_ij": surprise_ij,
        "slot_error": slot_error,
        "max_surprise": max_surprise,
        "most_surprising_pair": (i_idx.item(), j_idx.item()),
        "codebook_output": codebook_out,
    }


class SystemM:
    """
    System M mode controller.

    Routes between INFER mode (use current codebook, plan with dynamics)
    and LEARN mode (run additional learning on surprising interactions).

    This is a simple stateful controller, not a neural network.

    Args:
        surprise_threshold: τ — max surprise below which we stay in INFER mode
        maintenance_interval: how many steps between codebook maintenance
    """

    def __init__(
        self,
        surprise_threshold: float = 1.0,
        maintenance_interval: int = 100,
    ):
        self.surprise_threshold = surprise_threshold
        self.maintenance_interval = maintenance_interval
        self.step_count = 0
        self.mode = "INFER"

        # Running statistics
        self.surprise_history = []
        self.mode_switches = 0

    def should_learn(self, max_surprise: float) -> bool:
        """
        Decide whether to switch to LEARN mode.

        Args:
            max_surprise: maximum per-edge surprise value

        Returns:
            True if we should enter learning mode
        """
        self.step_count += 1
        self.surprise_history.append(max_surprise)

        if max_surprise > self.surprise_threshold:
            if self.mode != "LEARN":
                self.mode = "LEARN"
                self.mode_switches += 1
            return True
        else:
            self.mode = "INFER"
            return False

    def should_maintain_codebook(self) -> bool:
        """Check if it's time for codebook maintenance."""
        return self.step_count % self.maintenance_interval == 0

    def update_threshold(self, factor: float = 0.95):
        """
        Adaptively update the surprise threshold based on recent history.

        As the model improves, surprise naturally decreases. We track the
        running mean and set the threshold relative to it.
        """
        if len(self.surprise_history) < 100:
            return

        recent = self.surprise_history[-100:]
        mean_surprise = sum(recent) / len(recent)
        self.surprise_threshold = max(
            mean_surprise * 2.0,  # 2x the recent mean
            0.01,  # absolute minimum
        )

    def get_stats(self) -> dict:
        """Return System M diagnostics."""
        recent = self.surprise_history[-100:] if self.surprise_history else [0]
        return {
            "system_m/mode": 1.0 if self.mode == "LEARN" else 0.0,
            "system_m/threshold": self.surprise_threshold,
            "system_m/mean_surprise": sum(recent) / len(recent),
            "system_m/mode_switches": self.mode_switches,
            "system_m/step_count": self.step_count,
        }
