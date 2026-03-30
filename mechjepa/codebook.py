"""
Mechanism Bottleneck — continuous low-rank edge representation.

Replaces the discrete VQ-style codebook with a linear bottleneck:
    Edge MLP: [z_i; z_j] → e_ij ∈ R^D
    Encode:   e_ij → h_ij ∈ R^N    (N << D, the mechanism representation)
    Decode:   h_ij → m_ij ∈ R^D    (reconstructed mechanism vector)

The N-dimensional bottleneck vector h_ij IS the mechanism representation.
Different interaction types occupy different directions in this low-dimensional
space. No discrete quantization, no auxiliary losses — the prediction loss
flows continuously through the multiplicative gate in the attention layer,
providing the only signal needed for mechanism differentiation.

Design aligned with C-JEPA, LeWorldModel, Cambrian-S, and Blueprint paper:
continuous representations, simple losses, structure from the objective.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class MechanismCodebook(nn.Module):
    """
    Continuous low-rank mechanism bottleneck.

    Given K slots of dimension D, computes pairwise edge features and
    projects them through a low-rank bottleneck (R^D → R^N → R^D).

    The bottleneck weights are the "mechanism vocabulary" — they persist
    across episodes and can be frozen for transfer to new scenes.

    Args:
        num_mechanisms: N — bottleneck dimension (was: number of codebook entries)
        slot_dim: D — dimension of each slot / mechanism vector
        edge_hidden_dim: hidden dimension of the edge MLP
    """

    def __init__(
        self,
        num_mechanisms: int = 16,
        slot_dim: int = 128,
        edge_hidden_dim: int = 256,
        # Legacy kwargs accepted but ignored (for config compatibility)
        **kwargs,
    ):
        super().__init__()
        self.num_mechanisms = num_mechanisms
        self.slot_dim = slot_dim

        # Edge MLP: [z_i; z_j] -> R^D
        self.edge_mlp = nn.Sequential(
            nn.Linear(2 * slot_dim, edge_hidden_dim),
            nn.GELU(),
            nn.Linear(edge_hidden_dim, slot_dim),
        )

        # Low-rank bottleneck: R^D → R^N → R^D
        self.encode = nn.Linear(slot_dim, num_mechanisms)
        self.decode = nn.Linear(num_mechanisms, slot_dim)

    def compute_edges(self, z: torch.Tensor) -> torch.Tensor:
        """
        Compute pairwise edge features for all slot pairs.

        Args:
            z: (B, K, D) — slot vectors for a single frame

        Returns:
            e_ij: (B, K, K, D) — edge features for each pair
        """
        B, K, D = z.shape

        z_i = z.unsqueeze(2).expand(B, K, K, D)
        z_j = z.unsqueeze(1).expand(B, K, K, D)

        pair_features = torch.cat([z_i, z_j], dim=-1)
        e_ij = self.edge_mlp(pair_features)

        return e_ij

    def bind(
        self, e_ij: torch.Tensor, **kwargs
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Project edge features through the low-rank bottleneck.

        Args:
            e_ij: (B, K, K, D) — edge features

        Returns:
            m_ij: (B, K, K, D) — mechanism vectors (decoded)
            h_ij: (B, K, K, N) — bottleneck representations
        """
        h_ij = self.encode(e_ij)   # (B, K, K, N)
        m_ij = self.decode(h_ij)   # (B, K, K, D)

        return m_ij, h_ij

    def forward(
        self, z: torch.Tensor, **kwargs
    ) -> dict[str, torch.Tensor]:
        """
        Full forward: compute edges → bottleneck projection.

        Args:
            z: (B, K, D) — slot vectors

        Returns:
            dict with:
                m_ij: (B, K, K, D) — mechanism vectors
                h_ij: (B, K, K, N) — bottleneck representations
                e_ij: (B, K, K, D) — raw edge features
        """
        e_ij = self.compute_edges(z)
        m_ij, h_ij = self.bind(e_ij)

        return {
            "m_ij": m_ij,
            "h_ij": h_ij,
            "e_ij": e_ij,
        }

    def get_codebook_stats(self) -> dict[str, torch.Tensor]:
        """
        Return diagnostic statistics about the bottleneck.

        Analyzes the bottleneck weight matrices to understand
        which mechanism dimensions are being used.
        """
        # Analyze encode weight usage (which bottleneck dims carry signal)
        encode_norms = self.encode.weight.data.norm(dim=1)  # (N,)
        decode_norms = self.decode.weight.data.norm(dim=0)  # (N,)
        dim_importance = encode_norms * decode_norms  # (N,)

        return {
            "codebook/dim_importance": dim_importance,
            "codebook/dim_max": dim_importance.max(),
            "codebook/dim_min": dim_importance.min(),
            "codebook/dim_ratio": dim_importance.max() / (dim_importance.min() + 1e-8),
            "codebook/num_dead": torch.tensor(0),  # continuous = no dead entries
        }
