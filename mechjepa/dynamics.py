from __future__ import annotations

"""
Slot Transformer Dynamics with Mechanism Bias — MechJEPA's prediction engine.

Adapted from ctt-jepa's MaskedSlotPredictor. The key modifications:
  1. MechanismAttention: multiplicative mechanism gating (bottleneck controls
     which slot pairs interact)
  2. ActionAdaLN: Adaptive Layer Normalization for action conditioning (actions
     control global dynamics intensity) — composing with mechanism gating
  3. MechanismFFN: feed-forward network with mechanism context
  4. Same masking & positional embedding strategy as C-JEPA for compatibility

Composition order per layer:
  LN → AdaLN modulation (action) → Q,K,V → standard attention
  → multiplicative mechanism gate (bottleneck) → output

Actions and mechanisms address orthogonal questions:
  - Actions: global dynamics intensity (hard push vs soft push)
  - Mechanisms: per-pair interaction topology (which objects affect which)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
import numpy as np


# ---------------------------------------------------------------------------
# Adaptive Layer Normalization (Action Conditioning)
# ---------------------------------------------------------------------------


class ActionAdaLN(nn.Module):
    """
    Adaptive Layer Normalization for action conditioning.

    Given an action embedding, produces (scale, shift) parameters that
    modulate LayerNorm output: AdaLN(x) = scale * LN(x) + shift.

    This follows LeWorldModel's approach: actions control HOW STRONGLY
    the dynamics operate at each layer, while the mechanism gate controls
    WHICH pairs interact.

    Args:
        dim: feature dimension (slot_dim)
    """

    def __init__(self, dim: int):
        super().__init__()
        # Produces (scale, shift) from action embedding
        self.modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(dim, 2 * dim),
        )
        # Initialize to identity: scale=1, shift=0
        nn.init.zeros_(self.modulation[-1].weight)
        nn.init.zeros_(self.modulation[-1].bias)

    def forward(
        self, x_normed: torch.Tensor, action_emb: torch.Tensor | None = None
    ) -> torch.Tensor:
        """
        Modulate normalized features with action embedding.

        Args:
            x_normed: (B, N, D) — output of LayerNorm
            action_emb: (B, N, D) — action embedding (None = identity)

        Returns:
            (B, N, D) — modulated features
        """
        if action_emb is None:
            return x_normed

        # (B, N, 2D) → split to (B, N, D) each
        params = self.modulation(action_emb)
        scale, shift = params.chunk(2, dim=-1)
        # scale is centered at 0, so add 1 for identity initialization
        return x_normed * (1 + scale) + shift


# ---------------------------------------------------------------------------
# Mechanism-Aware Attention
# ---------------------------------------------------------------------------


class MechanismAttention(nn.Module):
    """
    Multi-head self-attention with multiplicative mechanism gating.

    Standard attention computes: attn = softmax(Q @ K^T / sqrt(d))
    We GATE the attention with mechanism-derived gates:
        standard_attn = softmax(Q @ K^T / sqrt(d))
        mech_gate = sigmoid(gate_proj(m_ij))  # 0-1 per head per pair
        attn = standard_attn * mech_gate

    When mech_gate ≈ 0 for a pair, no information flows between those
    slots regardless of how high the standard attention is. This makes
    the codebook LOAD-BEARING — the model must learn meaningful mechanism
    assignments to route information correctly.

    Args:
        dim: model dimension
        heads: number of attention heads
        dim_head: dimension per head
        dropout: attention dropout
    """

    def __init__(self, dim: int, heads: int = 8, dim_head: int = 64, dropout: float = 0.0):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.dim_head = dim_head

        self.norm = nn.LayerNorm(dim)
        self.adaln = ActionAdaLN(dim)  # Action modulation after LN
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = nn.Sequential(nn.Linear(inner_dim, dim), nn.Dropout(dropout))
        self.attn_dropout = nn.Dropout(dropout)

        # Mechanism gate projection: (D) -> (heads) per slot pair
        # Initialized with positive bias so gates start open (~sigmoid(2)≈0.88)
        self.gate_proj = nn.Sequential(
            nn.Linear(dim, dim // 2),
            nn.GELU(),
            nn.Linear(dim // 2, heads),
        )
        # Initialize final layer bias to +2.0 so gates start open
        nn.init.constant_(self.gate_proj[-1].bias, 2.0)

    def forward(
        self,
        x: torch.Tensor,
        m_ij: torch.Tensor | None = None,
        action_emb: torch.Tensor | None = None,
        return_attention: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """
        Composition order:
          LN → AdaLN (action intensity) → Q,K,V → attention → mechanism gate (pair topology) → output

        Args:
            x: (B, T*S, D) — flattened slot sequence
            m_ij: (B, S, S, D) — mechanism vectors (None = no gating)
            action_emb: (B, T*S, D) — action embedding (None = no modulation)
            return_attention: whether to return attention weights

        Returns:
            output: (B, T*S, D)
            attn_weights: (B, T*S, T*S) if return_attention else None
        """
        # Step 1: LayerNorm → AdaLN modulation (action controls dynamics intensity)
        x_normed = self.norm(x)
        x_normed = self.adaln(x_normed, action_emb)

        B, N, _ = x_normed.shape

        # Step 2: Q, K, V from action-modulated features
        qkv = self.to_qkv(x_normed).chunk(3, dim=-1)
        q, k, v = (rearrange(t, "b n (h d) -> b h n d", h=self.heads) for t in qkv)

        # Step 3: Standard attention
        attn_logits = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn = F.softmax(attn_logits, dim=-1)

        # Step 4: Multiplicative mechanism gating (bottleneck controls pair topology)
        if m_ij is not None:
            mech_gate = self._compute_mech_gate(m_ij, N)
            attn = attn * mech_gate

        attn = self.attn_dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, "b h n d -> b n (h d)")
        out = self.to_out(out)

        if return_attention:
            return out, attn.mean(dim=1)
        return out, None

    def _compute_mech_gate(self, m_ij: torch.Tensor, seq_len: int) -> torch.Tensor:
        """
        Compute multiplicative mechanism gate.

        m_ij is (B, S, S, D) — slot-to-slot mechanism vectors.
        We produce a (B, H, T*S, T*S) gate in [0, 1] via sigmoid.

        The gate is shared across all time steps: the mechanism between
        slot i and slot j is the same regardless of which timestep they're in.

        Args:
            m_ij: (B, S, S, D)
            seq_len: T*S — total sequence length

        Returns:
            gate: (B, H, T*S, T*S) in [0, 1]
        """
        B, S, _, D = m_ij.shape
        T = seq_len // S

        # Project mechanism vectors to per-head gate logits: (B, S, S, H)
        gate_ss = self.gate_proj(m_ij)  # (B, S, S, H)
        gate_ss = torch.sigmoid(gate_ss)  # [0, 1]
        gate_ss = gate_ss.permute(0, 3, 1, 2)  # (B, H, S, S)

        # Tile across time: (B, H, S, S) -> (B, H, T*S, T*S)
        gate = gate_ss.unsqueeze(2).unsqueeze(4)  # (B, H, 1, S, 1, S)
        gate = gate.expand(B, self.heads, T, S, T, S)  # (B, H, T, S, T, S)
        gate = gate.reshape(B, self.heads, T * S, T * S)  # (B, H, T*S, T*S)

        return gate


# ---------------------------------------------------------------------------
# Mechanism-Aware FFN
# ---------------------------------------------------------------------------


class MechanismFFN(nn.Module):
    """
    Feed-forward network that incorporates mechanism context.

    Standard FFN takes slot features only:
        z_next = FFN(z + attn_out)

    We augment the input with the mean mechanism context per slot:
        z_next = FFN([z + attn_out; mean_j(m_ij)])

    This allows the dynamics to use mechanism information for state updates.

    Args:
        dim: input dimension (slot dim)
        hidden_dim: FFN hidden dimension
        dropout: dropout rate
    """

    def __init__(self, dim: int, hidden_dim: int, dropout: float = 0.0):
        super().__init__()
        self.norm = nn.LayerNorm(2 * dim)
        self.adaln = ActionAdaLN(2 * dim)  # Action modulation on concatenated input
        # Input is dim (slot features) + dim (mean mechanism context)
        self.net = nn.Sequential(
            nn.Linear(2 * dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),  # Output is back to slot dim
            nn.Dropout(dropout),
        )

    def forward(
        self,
        x: torch.Tensor,
        mech_context: torch.Tensor,
        action_emb: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            x: (B, T*S, D) — slot features after attention
            mech_context: (B, T*S, D) — mean mechanism context per slot
            action_emb: (B, T*S, D) — action embedding (None = no modulation)

        Returns:
            (B, T*S, D) — updated slot features
        """
        combined = torch.cat([x, mech_context], dim=-1)  # (B, T*S, 2D)
        combined = self.norm(combined)
        # Expand action_emb to match 2D dimension if provided
        if action_emb is not None:
            action_emb_2d = action_emb.repeat(1, 1, 2)  # (B, T*S, 2D)
            combined = self.adaln(combined, action_emb_2d)
        return self.net(combined)


# ---------------------------------------------------------------------------
# Standard FFN (fallback without mechanism)
# ---------------------------------------------------------------------------


class StandardFFN(nn.Module):
    """Standard feed-forward network (no mechanism context)."""

    def __init__(self, dim: int, hidden_dim: int, dropout: float = 0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ---------------------------------------------------------------------------
# Mechanism Transformer
# ---------------------------------------------------------------------------


class MechanismTransformer(nn.Module):
    """
    Transformer encoder with mechanism-biased attention and mechanism-aware FFN.

    Args:
        dim: model dimension
        depth: number of transformer layers
        heads: number of attention heads
        dim_head: dimension per head
        mlp_dim: FFN hidden dimension
        dropout: dropout rate
    """

    def __init__(
        self,
        dim: int,
        depth: int,
        heads: int,
        dim_head: int,
        mlp_dim: int,
        dropout: float = 0.0,
        action_dim: int | None = None,
    ):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.layers = nn.ModuleList()
        self.action_dim = action_dim

        # Action embedder: raw action → slot_dim embedding
        if action_dim is not None:
            self.action_embedder = nn.Sequential(
                nn.Linear(action_dim, dim // 2),
                nn.GELU(),
                nn.Linear(dim // 2, dim),
            )
        else:
            self.action_embedder = None

        for _ in range(depth):
            self.layers.append(
                nn.ModuleList([
                    MechanismAttention(dim, heads, dim_head, dropout),
                    MechanismFFN(dim, mlp_dim, dropout),
                ])
            )

    def _embed_actions(
        self, actions: torch.Tensor | None, seq_len: int, num_slots: int
    ) -> torch.Tensor | None:
        """
        Embed actions and expand to match slot sequence shape.

        Args:
            actions: (B, T, action_dim) — one action per timestep transition
            seq_len: T*S — total sequence length
            num_slots: S

        Returns:
            action_emb: (B, T*S, D) or None
        """
        if actions is None or self.action_embedder is None:
            return None

        B, T_act, _ = actions.shape
        T = seq_len // num_slots

        # Embed: (B, T_act, action_dim) → (B, T_act, D)
        act_emb = self.action_embedder(actions)  # (B, T_act, D)

        # Pad if T_act < T (e.g. future frames have no action)
        if T_act < T:
            pad = torch.zeros(B, T - T_act, act_emb.shape[-1], device=act_emb.device)
            act_emb = torch.cat([act_emb, pad], dim=1)  # (B, T, D)

        # Expand across slots: (B, T, D) → (B, T, S, D) → (B, T*S, D)
        act_emb = act_emb[:, :T].unsqueeze(2).expand(B, T, num_slots, -1)
        act_emb = act_emb.reshape(B, T * num_slots, -1)

        return act_emb

    def forward(
        self,
        x: torch.Tensor,
        m_ij: torch.Tensor | None = None,
        num_slots: int | None = None,
        actions: torch.Tensor | None = None,
        return_attention: bool = False,
    ) -> tuple[torch.Tensor, list[torch.Tensor] | None]:
        """
        Args:
            x: (B, T*S, D) — flattened slot sequence
            m_ij: (B, S, S, D) — mechanism vectors (None = standard transformer)
            num_slots: S — needed to compute per-slot mechanism context
            actions: (B, T, action_dim) — actions per timestep (None = unconditional)
            return_attention: whether to return attention weights

        Returns:
            output: (B, T*S, D)
            attn_list: list of (B, T*S, T*S) attention weights per layer (or None)
        """
        attn_list = [] if return_attention else None

        # Embed actions once, shared across all layers
        action_emb = self._embed_actions(actions, x.shape[1], num_slots) if num_slots else None

        for attn_layer, ffn_layer in self.layers:
            # Mechanism-biased attention with action modulation
            # Order: LN → AdaLN(action) → QKV → attn → mechanism gate → output
            attn_out, attn_weights = attn_layer(
                x, m_ij=m_ij, action_emb=action_emb, return_attention=return_attention
            )
            x = x + attn_out

            if return_attention and attn_weights is not None:
                attn_list.append(attn_weights)

            # Mechanism-aware FFN with action modulation
            if m_ij is not None and num_slots is not None:
                mech_context = self._compute_mech_context(m_ij, x.shape[1], num_slots)
                x = x + ffn_layer(x, mech_context, action_emb=action_emb)
            else:
                mech_context = torch.zeros_like(x)
                x = x + ffn_layer(x, mech_context, action_emb=action_emb)

        return self.norm(x), attn_list

    def _compute_mech_context(
        self, m_ij: torch.Tensor, seq_len: int, num_slots: int
    ) -> torch.Tensor:
        """
        Compute mean mechanism context per slot, tiled across time.

        m_ij: (B, S, S, D) — mean over axis 2 gives per-slot context
        Returns: (B, T*S, D) — mechanism context for each position in the sequence
        """
        B, S, _, D = m_ij.shape
        T = seq_len // S

        # Mean over interaction partners: (B, S, D)
        mech_per_slot = m_ij.mean(dim=2)  # (B, S, D)

        # Tile across time: (B, S, D) -> (B, T, S, D) -> (B, T*S, D)
        mech_tiled = mech_per_slot.unsqueeze(1).expand(B, T, S, D)
        mech_tiled = mech_tiled.reshape(B, T * S, D)

        return mech_tiled


# ---------------------------------------------------------------------------
# MechSlotPredictor (the full dynamics module)
# ---------------------------------------------------------------------------


class MechSlotPredictor(nn.Module):
    """
    Masked Slot Predictor with Mechanism Bias.

    Adapted from C-JEPA's MaskedSlotPredictor. Same masking and positional
    embedding strategy, but with mechanism-biased attention and mechanism-aware FFN.

    Args:
        num_slots: K — number of slots per frame
        slot_dim: D — dimension of each slot
        history_frames: T_hist — number of input history frames
        pred_frames: T_pred — number of future frames to predict
        num_masked_slots: M — number of slots to mask during training
        seed: random seed for reproducible masking
        depth: transformer depth
        heads: number of attention heads
        dim_head: dimension per head
        mlp_dim: FFN hidden dimension
        dropout: dropout rate
    """

    def __init__(
        self,
        num_slots: int,
        slot_dim: int = 128,
        history_frames: int = 3,
        pred_frames: int = 1,
        num_masked_slots: int = 2,
        seed: int = 42,
        depth: int = 6,
        heads: int = 8,
        dim_head: int = 64,
        mlp_dim: int = 2048,
        dropout: float = 0.1,
        action_dim: int | None = None,
    ):
        super().__init__()
        self.num_slots = num_slots
        self.slot_dim = slot_dim
        self.history_frames = history_frames
        self.pred_frames = pred_frames
        self.total_frames = history_frames + pred_frames
        self.max_masked_slots = num_masked_slots
        self.seed = seed
        self.action_dim = action_dim

        # Persistent RNG for eval reproducibility (only used when not training)
        self._eval_rng = np.random.RandomState(seed)

        # Learnable mask token
        self.mask_token = nn.Parameter(torch.zeros(1, 1, slot_dim))
        nn.init.trunc_normal_(self.mask_token, std=0.02)

        # Time positional embedding
        self.time_pos_embed = nn.Parameter(
            torch.randn(1, self.total_frames, 1, slot_dim)
        )

        # ID projector (anchor mechanism)
        self.id_projector = nn.Linear(slot_dim, slot_dim)

        # Mechanism-aware transformer backbone (with optional action conditioning)
        self.transformer = MechanismTransformer(
            dim=slot_dim,
            depth=depth,
            heads=heads,
            dim_head=dim_head,
            mlp_dim=mlp_dim,
            dropout=dropout,
            action_dim=action_dim,
        )

        # Output head
        self.to_out = nn.Linear(slot_dim, slot_dim)

    def get_mask_indices(self, batch_size: int, device: torch.device):
        """
        Select slots to mask.

        During training: randomly vary both *which* and *how many* slots
        are masked (uniformly from 0 to max_masked_slots), following C-JEPA.
        During eval: use fixed seed for reproducibility.
        """
        if self.training:
            # Randomly sample the NUMBER of slots to mask (0 to max)
            num_masked = np.random.randint(0, self.max_masked_slots + 1)
            if num_masked == 0:
                is_slot_masked = torch.zeros(self.num_slots, dtype=torch.bool, device=device)
                return is_slot_masked, torch.tensor([], dtype=torch.long, device=device)
            masked_indices = np.random.choice(self.num_slots, num_masked, replace=False)
        else:
            masked_indices = self._eval_rng.choice(
                self.num_slots, self.max_masked_slots, replace=False,
            )

        is_slot_masked = torch.zeros(self.num_slots, dtype=torch.bool, device=device)
        is_slot_masked[masked_indices] = True

        return is_slot_masked, torch.from_numpy(masked_indices).to(device)

    def prepare_input(self, x: torch.Tensor):
        """
        Construct input sequence with masking.

        Logic (same as C-JEPA):
        - t=0: ALWAYS visible (identity anchor)
        - Masked slots: visible at t=0, masked at t=1..T_total
        - Unmasked slots: visible at t=0..T_hist, masked at future

        Args:
            x: (B, T_hist, S, D) — ground truth history

        Returns:
            final_input: (B, T_total, S, D)
            mask_indices: indices of masked slots
        """
        B, T_hist, S, D = x.shape
        T_total = self.total_frames
        device = x.device

        # Get mask
        if self.max_masked_slots > 0:
            is_slot_masked, masked_indices = self.get_mask_indices(B, device)
        else:
            masked_indices = torch.tensor([], dtype=torch.long, device=device)

        # Anchors from t=0
        anchors = x[:, 0, :, :]
        anchor_queries = self.id_projector(anchors)

        # Query grid: mask_token + time_pos + anchor_query
        tokens_grid = self.mask_token.expand(B, T_total, S, D)
        pos_grid = self.time_pos_embed.expand(B, T_total, S, D)
        anchor_grid = anchor_queries.unsqueeze(1).expand(B, T_total, S, D)
        query_input = tokens_grid + pos_grid + anchor_grid

        final_input = query_input.clone()

        # t=0: always real data for all slots
        final_input[:, 0, :, :] = x[:, 0, :, :] + self.time_pos_embed[:, 0, :, :]

        # Unmasked slots: real data for t=1..T_hist-1
        if self.max_masked_slots > 0:
            unmasked_indices = torch.where(~is_slot_masked)[0]
        else:
            unmasked_indices = torch.arange(0, S)

        if len(unmasked_indices) > 0 and T_hist > 1:
            real_history = x[:, 1:, unmasked_indices, :]
            history_pos = self.time_pos_embed[:, 1:T_hist, :, :].expand(B, T_hist - 1, S, D)
            history_pos_unmasked = history_pos[:, :, unmasked_indices, :]
            final_input[:, 1:T_hist, unmasked_indices, :] = real_history + history_pos_unmasked

        return final_input, masked_indices

    def prepare_input_with_mask(
        self,
        x: torch.Tensor,
        is_slot_masked: torch.Tensor,
        masked_indices: torch.Tensor,
    ) -> torch.Tensor:
        """
        Prepare input with a custom mask (for CTT losses / System M).

        Same as prepare_input but with arbitrary mask indices.
        """
        B, T_hist, S, D = x.shape
        T_total = self.total_frames
        device = x.device

        anchors = x[:, 0, :, :]
        anchor_queries = self.id_projector(anchors)

        tokens_grid = self.mask_token.expand(B, T_total, S, D)
        pos_grid = self.time_pos_embed.expand(B, T_total, S, D)
        anchor_grid = anchor_queries.unsqueeze(1).expand(B, T_total, S, D)
        query_input = tokens_grid + pos_grid + anchor_grid

        final_input = query_input.clone()
        final_input[:, 0, :, :] = x[:, 0, :, :] + self.time_pos_embed[:, 0, :, :]

        unmasked_indices = torch.where(~is_slot_masked)[0]
        if len(unmasked_indices) > 0 and T_hist > 1:
            real_history = x[:, 1:, unmasked_indices, :]
            history_pos = self.time_pos_embed[:, 1:T_hist, :, :].expand(B, T_hist - 1, S, D)
            history_pos_unmasked = history_pos[:, :, unmasked_indices, :]
            final_input[:, 1:T_hist, unmasked_indices, :] = real_history + history_pos_unmasked

        return final_input

    def forward(
        self,
        x: torch.Tensor,
        m_ij: torch.Tensor | None = None,
        actions: torch.Tensor | None = None,
        return_attention: bool = False,
    ):
        """
        Forward pass with mechanism bias and optional action conditioning.

        Args:
            x: (B, T_hist, S, D) — history slots
            m_ij: (B, S, S, D) — bound mechanism vectors from codebook
            actions: (B, T_hist, action_dim) — per-transition actions (None = unconditional)
            return_attention: whether to return attention weights

        Returns:
            pred: (B, T_total, S, D) — predicted slots
            mask_indices: indices of masked slots
            attn_list: attention weights (if requested)
        """
        B, T_hist, S, D = x.shape

        # Prepare masked input
        x_input, masked_indices = self.prepare_input(x)

        # Flatten for transformer
        x_flat = rearrange(x_input, "b t s d -> b (t s) d")

        # Forward through mechanism transformer with action conditioning
        out_flat, attn_list = self.transformer(
            x_flat,
            m_ij=m_ij,
            num_slots=S,
            actions=actions,
            return_attention=return_attention,
        )

        # Unflatten
        out = rearrange(out_flat, "b (t s) d -> b t s d", t=self.total_frames, s=S)
        out = self.to_out(out)

        if return_attention:
            return out, masked_indices, attn_list
        return out, masked_indices

    @torch.no_grad()
    def inference(
        self,
        x: torch.Tensor,
        m_ij: torch.Tensor | None = None,
        actions: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Inference: predict future frames from fully visible history.

        Args:
            x: (B, T_hist, S, D) — fully visible history
            m_ij: (B, S, S, D) — mechanism vectors
            actions: (B, T_hist, action_dim) — actions (None = unconditional)

        Returns:
            future_prediction: (B, T_pred, S, D)
        """
        B, T_hist, S, D = x.shape
        T_pred = self.pred_frames
        T_total = T_hist + T_pred

        inf_time_pos_embed = self.time_pos_embed[:, -T_total:, :, :]

        # Anchor query from t=0
        anchors = x[:, 0, :, :]
        anchor_queries = self.id_projector(anchors)

        # History: real data + pos embed
        input_history = x + inf_time_pos_embed[:, :T_hist, :, :]

        # Future: mask_token + pos_embed + anchor_query
        tokens_grid = self.mask_token.expand(B, T_pred, S, D)
        pos_grid = inf_time_pos_embed[:, T_hist:T_total, :, :].expand(B, T_pred, S, D)
        anchor_grid = anchor_queries.unsqueeze(1).expand(B, T_pred, S, D)
        input_future = tokens_grid + pos_grid + anchor_grid

        full_input = torch.cat([input_history, input_future], dim=1)

        # Flatten and forward with actions
        x_flat = rearrange(full_input, "b t s d -> b (t s) d")
        out_flat, _ = self.transformer(
            x_flat, m_ij=m_ij, num_slots=S, actions=actions
        )

        # Unflatten and extract future
        out = rearrange(out_flat, "b (t s) d -> b t s d", t=T_total, s=S)
        out = self.to_out(out)
        return out[:, T_hist:, :, :]
