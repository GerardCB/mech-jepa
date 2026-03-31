"""
MechJEPA Cost Model — SWM-compatible interface for planning.

Implements the `Costable` protocol required by SWM's CEMSolver:
    get_cost(info_dict, action_candidates) → costs

SWM's CEM solver expands info_dict so that:
  - info_dict['pixels'] has shape (n_envs, n_samples, H, W, C)
  - action_candidates has shape (n_envs, n_samples, horizon, action_dim)
  - returned costs must be (n_envs, n_samples) — 2D

The pixels are identical across n_samples (same observation replicated).
We only encode pixels[:, 0] and expand in latent space.

Usage with SWM:
    from stable_worldmodel.solver import CEMSolver
    cost_model = MechJEPACostModel(world_model, encoder)
    solver = CEMSolver(model=cost_model, num_samples=300, n_steps=5)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class MechJEPACostModel(nn.Module):
    """
    Wraps MechJEPA + VideoSAUR encoder for use with SWM's CEMSolver.
    """

    def __init__(self, world_model: nn.Module, encoder: nn.Module,
                 history_len: int = 3):
        super().__init__()
        self.model = world_model
        self.encoder = encoder
        self.history_len = history_len

        # Rolling history (updated via auto-commit in get_cost)
        self._slot_history: list[torch.Tensor] = []   # list of (n_envs, S, D)
        self._action_history: list[torch.Tensor] = []  # list of (n_envs, act_dim)

        # Per-step cache
        self._cached_curr_slots: torch.Tensor | None = None
        self._cached_goal_slots: torch.Tensor | None = None
        self._step_counter: int = 0

    def reset(self):
        """Clear history at episode start."""
        self._slot_history.clear()
        self._action_history.clear()
        self._cached_curr_slots = None
        self._cached_goal_slots = None
        self._step_counter = 0

    def _encode_single(self, frame_hwc: np.ndarray) -> torch.Tensor:
        """Encode a single (H, W, C) frame into slots (S, D)."""
        return self.encoder.encode(frame_hwc.astype(np.uint8))

    def _encode_batch(self, pixels: np.ndarray) -> torch.Tensor:
        """Encode (N, H, W, C) batch of frames into (N, S, D) slots."""
        slots_list = []
        for i in range(pixels.shape[0]):
            s = self._encode_single(pixels[i])
            slots_list.append(s.unsqueeze(0))
        return torch.cat(slots_list, dim=0)

    def _to_nhwc(self, pixels) -> np.ndarray:
        """Convert any SWM pixel format to (N, H, W, C) numpy uint8."""
        if isinstance(pixels, torch.Tensor):
            pixels = pixels.cpu().numpy()

        # Squeeze extra leading dims until 4D: (N, H, W, C)
        while pixels.ndim > 4:
            pixels = pixels[:, 0]

        # Handle (N, C, H, W) -> (N, H, W, C)
        if pixels.ndim == 4 and pixels.shape[1] in (1, 3):
            pixels = pixels.transpose(0, 2, 3, 1)

        # Handle single image (H, W, C)
        if pixels.ndim == 3:
            pixels = pixels[np.newaxis]

        return pixels

    @torch.no_grad()
    def get_cost(self, info_dict: dict, action_candidates: torch.Tensor) -> torch.Tensor:
        """
        Evaluate candidate action sequences by rolling out the world model.

        Called by SWM's CEMSolver with:
          - info_dict['pixels']:  (n_envs, n_samples, H, W, C) — replicated
          - info_dict['goal']:    (n_envs, n_samples, H, W, C) — replicated
          - action_candidates:    (n_envs, n_samples, horizon, action_dim)

        Returns:
          costs: (n_envs, n_samples) — 2D tensor, lower is better.
        """
        device = action_candidates.device
        n_envs, n_samples, horizon, action_dim = action_candidates.shape

        # Encode ONCE per step (all n_samples have the same pixels)
        pixels_nhwc = self._to_nhwc(info_dict['pixels'])  # (n_envs, H, W, C)
        goal_nhwc = self._to_nhwc(info_dict['goal'])      # (n_envs, H, W, C)

        # Detect new step via pixel hash (CEM calls get_cost multiple times
        # per step with the same pixels — avoid re-encoding + re-committing)
        pixel_hash = hash(pixels_nhwc.tobytes())

        if pixel_hash != self._step_counter:  # _step_counter stores last hash
            # New observation → commit previous to history, then encode
            if self._cached_curr_slots is not None:
                self._slot_history.append(self._cached_curr_slots.detach())
                if len(self._slot_history) > self.history_len:
                    self._slot_history.pop(0)
                # Zero action placeholder (replaced by update_step if ABM)
                if len(self._action_history) < len(self._slot_history):
                    self._action_history.append(
                        torch.zeros(n_envs, action_dim, device=device)
                    )
                    if len(self._action_history) > self.history_len:
                        self._action_history.pop(0)

            self._cached_curr_slots = self._encode_batch(pixels_nhwc).to(device)
            self._cached_goal_slots = self._encode_batch(goal_nhwc).to(device)
            self._step_counter = pixel_hash

        curr_slots = self._cached_curr_slots  # (n_envs, S, D)
        goal_slots = self._cached_goal_slots  # (n_envs, S, D)
        S, D = curr_slots.shape[1], curr_slots.shape[2]

        # Build history: existing + current, padded to history_len
        temp_history = list(self._slot_history) + [curr_slots]
        while len(temp_history) < self.history_len:
            temp_history.insert(0, temp_history[0])
        temp_history = temp_history[-self.history_len:]
        history = torch.stack(temp_history, dim=1).to(device)  # (n_envs, T, S, D)

        # Action history, padded
        temp_acts = list(self._action_history)
        while len(temp_acts) < self.history_len:
            temp_acts.insert(0, torch.zeros(n_envs, action_dim, device=device))
        temp_acts = [a.to(device) for a in temp_acts[-self.history_len:]]
        action_hist = torch.stack(temp_acts, dim=1)

        # Expand for n_samples: (n_envs * n_samples, T, S, D)
        hist_exp = history.unsqueeze(1).expand(-1, n_samples, -1, -1, -1)
        hist_exp = hist_exp.reshape(n_envs * n_samples, self.history_len, S, D)

        ahist_exp = action_hist.unsqueeze(1).expand(-1, n_samples, -1, -1)
        ahist_exp = ahist_exp.reshape(n_envs * n_samples, self.history_len, action_dim)

        # Flatten candidates: (n_envs * n_samples, horizon, action_dim)
        flat_actions = action_candidates.reshape(n_envs * n_samples, horizon, action_dim)

        # Roll out world model step-by-step
        curr_history = hist_exp
        curr_acts = ahist_exp

        for h in range(horizon):
            step_act = flat_actions[:, h:h+1, :]
            curr_acts = torch.cat([curr_acts[:, 1:, :], step_act], dim=1)
            next_pred = self.model.inference(
                curr_history, actions=curr_acts
            )
            curr_history = torch.cat([curr_history[:, 1:], next_pred], dim=1)

        # Final predicted state: (n_envs * n_samples, S, D)
        final_pred = curr_history[:, -1, :, :]

        # Expand goal: (n_envs * n_samples, S, D)
        goal_exp = goal_slots.unsqueeze(1).expand(-1, n_samples, -1, -1)
        goal_exp = goal_exp.reshape(n_envs * n_samples, S, D).to(device)

        # Cost = Permutation-invariant Chamfer-style distance to goal
        # Since slots are unordered, naive element-wise MSE fails.
        # Compute pairwise distances: (B, S, S)
        dists = torch.cdist(final_pred, goal_exp)
        # For each predicted slot, find the closest goal slot distance, and average them
        costs = dists.min(dim=2).values.mean(dim=1)
        
        # Returns (n_envs, n_samples) 2D
        costs = costs.reshape(n_envs, n_samples)

        return costs

    def update_step(self, action: np.ndarray | torch.Tensor):
        """Update action history after committing an action.

        Called by ABMPolicy after get_action() — for Frozen policy this
        is handled automatically by the auto-commit in get_cost().
        """
        if isinstance(action, np.ndarray):
            action = torch.from_numpy(action).float()

        # Replace the zero-action placeholder with the actual action
        if self._action_history:
            self._action_history[-1] = action
        else:
            self._action_history.append(action)
            if len(self._action_history) > self.history_len:
                self._action_history.pop(0)

    # Legacy alias
    def update_action_history(self, action):
        """Legacy alias for update_step."""
        self.update_step(action)

    @property
    def slot_history(self) -> list[torch.Tensor]:
        """Access to slot history for System M surprise computation."""
        return self._slot_history

    @property
    def action_history(self) -> list[torch.Tensor]:
        """Access to action history for System M surprise computation."""
        return self._action_history
