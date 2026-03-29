"""
Mechanism Codebook — the persistent memory of MechJEPA.

This module implements:
  - MechanismCodebook: nn.Embedding(N, D) storing N mechanism prototypes
  - EdgeMLP: computes pairwise edge features from slot pairs
  - Soft binding: matches edge features to codebook entries via temperature-scaled dot product
  - EMA usage tracking + dead entry reallocation

The codebook is the key architectural innovation. Each row captures
"one way objects can influence each other." The model discovers these
patterns (collisions, gravity, free-flight) without supervision.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class MechanismCodebook(nn.Module):
    """
    Persistent mechanism memory with edge binding.

    Given K slots of dimension D, computes pairwise edge features and
    soft-assigns each pair to one of N mechanism prototypes.

    Args:
        num_mechanisms: N — number of mechanism types in the codebook
        slot_dim: D — dimension of each slot / mechanism vector
        temperature: τ — softmax temperature for binding sharpness
        commitment_weight: β — weight for commitment loss (VQ-VAE style)
        dead_threshold: minimum EMA usage below which an entry is "dead"
        edge_hidden_dim: hidden dimension of the edge MLP
    """

    def __init__(
        self,
        num_mechanisms: int = 16,
        slot_dim: int = 128,
        temperature: float = 0.1,
        commitment_weight: float = 0.25,
        dead_threshold: float = 0.01,
        edge_hidden_dim: int = 256,
    ):
        super().__init__()
        self.num_mechanisms = num_mechanisms
        self.slot_dim = slot_dim
        self.temperature = temperature
        self.commitment_weight = commitment_weight
        self.dead_threshold = dead_threshold

        # The codebook: N mechanism prototypes in R^D
        self.codebook = nn.Embedding(num_mechanisms, slot_dim)
        nn.init.normal_(self.codebook.weight, std=0.02)

        # Edge MLP: [z_i; z_j] -> R^D
        self.edge_mlp = nn.Sequential(
            nn.Linear(2 * slot_dim, edge_hidden_dim),
            nn.GELU(),
            nn.Linear(edge_hidden_dim, slot_dim),
        )

        # EMA usage tracker (not a parameter, survives across episodes)
        self.register_buffer("codebook_usage", torch.zeros(num_mechanisms))
        self.register_buffer("usage_initialized", torch.tensor(False))

    def compute_edges(self, z: torch.Tensor) -> torch.Tensor:
        """
        Compute pairwise edge features for all slot pairs.

        Args:
            z: (B, K, D) — slot vectors for a single frame

        Returns:
            e_ij: (B, K, K, D) — edge features for each pair
        """
        B, K, D = z.shape

        # Expand slots for pairwise computation
        z_i = z.unsqueeze(2).expand(B, K, K, D)  # (B, K, K, D) — source
        z_j = z.unsqueeze(1).expand(B, K, K, D)  # (B, K, K, D) — target

        # Concatenate pair features and project
        pair_features = torch.cat([z_i, z_j], dim=-1)  # (B, K, K, 2D)
        e_ij = self.edge_mlp(pair_features)  # (B, K, K, D)

        return e_ij

    def bind(
        self, e_ij: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Soft-assign edge features to codebook entries.

        Args:
            e_ij: (B, K, K, D) — edge features

        Returns:
            m_ij: (B, K, K, D) — bound mechanism vectors
            alpha_ij: (B, K, K, N) — soft assignment weights
            logits: (B, K, K, N) — raw logits before softmax
        """
        # Dot-product similarity with codebook: (B, K, K, D) @ (D, N)
        logits = (
            e_ij @ self.codebook.weight.T / self.temperature
        )  # (B, K, K, N)

        # Soft assignment
        alpha_ij = F.softmax(logits, dim=-1)  # (B, K, K, N)

        # Weighted sum of codebook entries
        m_ij = alpha_ij @ self.codebook.weight  # (B, K, K, D)

        return m_ij, alpha_ij, logits

    def forward(
        self, z: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        """
        Full forward: compute edges, bind to codebook.

        Args:
            z: (B, K, D) — slot vectors

        Returns:
            dict with keys:
                m_ij: (B, K, K, D) — bound mechanism vectors
                alpha_ij: (B, K, K, N) — soft assignments (THE graph edges)
                e_ij: (B, K, K, D) — raw edge features
                logits: (B, K, K, N) — binding logits
        """
        e_ij = self.compute_edges(z)
        m_ij, alpha_ij, logits = self.bind(e_ij)

        # Update usage tracker (no grad)
        if self.training:
            self._update_usage(alpha_ij)

        return {
            "m_ij": m_ij,
            "alpha_ij": alpha_ij,
            "e_ij": e_ij,
            "logits": logits,
        }

    def commitment_loss(
        self, e_ij: torch.Tensor, m_ij: torch.Tensor
    ) -> torch.Tensor:
        """
        VQ-VAE style commitment loss: encourages edge features to
        commit to their assigned codebook entry.

        L = β * ||e_ij - sg(m_ij)||²

        Args:
            e_ij: (B, K, K, D) — edge features
            m_ij: (B, K, K, D) — bound mechanism vectors

        Returns:
            Scalar loss
        """
        return self.commitment_weight * F.mse_loss(e_ij, m_ij.detach())

    @torch.no_grad()
    def _update_usage(self, alpha_ij: torch.Tensor):
        """
        Update EMA usage tracker.

        Args:
            alpha_ij: (B, K, K, N) — soft assignments
        """
        # Sum over batch and spatial dims: how much each codebook entry is used
        usage = alpha_ij.sum(dim=(0, 1, 2))  # (N,)

        if not self.usage_initialized:
            self.codebook_usage.copy_(usage)
            self.usage_initialized.fill_(True)
        else:
            self.codebook_usage.mul_(0.99).add_(usage, alpha=0.01)

    @torch.no_grad()
    def get_dead_entries(self) -> torch.Tensor:
        """
        Return boolean mask of dead codebook entries (usage below threshold).

        Returns:
            (N,) bool tensor — True for dead entries
        """
        return self.codebook_usage < self.dead_threshold

    @torch.no_grad()
    def reallocate_dead_entries(
        self, e_ij: torch.Tensor, surprise_ij: torch.Tensor
    ):
        """
        Replace dead codebook entries with the edge features of
        the most surprising (novel) interactions.

        Args:
            e_ij: (B, K, K, D) — edge features
            surprise_ij: (B, K, K) — per-edge surprise values
        """
        dead = self.get_dead_entries()
        if not dead.any():
            return

        # Find the most surprising pair
        # Flatten to find global argmax
        flat_idx = surprise_ij.reshape(-1).argmax()
        b_idx = flat_idx // (surprise_ij.shape[1] * surprise_ij.shape[2])
        remaining = flat_idx % (surprise_ij.shape[1] * surprise_ij.shape[2])
        i_idx = remaining // surprise_ij.shape[2]
        j_idx = remaining % surprise_ij.shape[2]

        # Get the novel interaction feature
        new_mechanism = e_ij[b_idx, i_idx, j_idx]  # (D,)

        # Replace first dead entry
        dead_idx = dead.nonzero(as_tuple=True)[0][0]
        self.codebook.weight.data[dead_idx] = new_mechanism
        self.codebook_usage[dead_idx] = self.codebook_usage.max()  # Reset usage

    def get_codebook_stats(self) -> dict[str, torch.Tensor]:
        """
        Return diagnostic statistics about the codebook.

        Returns:
            dict with usage, num_dead, entropy, etc.
        """
        usage_normalized = self.codebook_usage / (self.codebook_usage.sum() + 1e-8)
        entropy = -(usage_normalized * (usage_normalized + 1e-8).log()).sum()
        dead = self.get_dead_entries()

        return {
            "codebook/usage": self.codebook_usage,
            "codebook/num_dead": dead.sum(),
            "codebook/entropy": entropy,
            "codebook/usage_max": self.codebook_usage.max(),
            "codebook/usage_min": self.codebook_usage.min(),
        }
