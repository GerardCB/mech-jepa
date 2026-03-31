"""
MechJEPA Cost Model — SWM-compatible interface for planning.

Implements the `Costable` protocol required by SWM's CEMSolver:
    get_cost(info_dict, action_candidates) → costs

The cost is the MSE between the predicted final slot state and the goal
slot state, computed by rolling out MechJEPA in latent space for each
candidate action sequence.

Architecture note:
    SWM's CEMSolver calls get_cost multiple times per env step (once per
    CEM iteration). We must NOT modify internal state (history) inside
    get_cost. Instead we cache the encoded slots and only update history
    when update_step() is called after the action is committed.

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

    The solver calls `get_cost(info_dict, action_candidates)` with:
      - info_dict['pixels']:  current obs  (n_envs, ...)
      - info_dict['goal']:    goal obs     (n_envs, ...)
      - action_candidates:    (n_envs, n_samples, horizon, action_dim) float32

    Returns costs: (n_envs, n_samples, 1) — lower is better.
    """

    def __init__(self, world_model: nn.Module, encoder: nn.Module,
                 history_len: int = 3):
        super().__init__()
        self.model = world_model
        self.encoder = encoder
        self.history_len = history_len

        # Rolling history (updated externally via update_step, NOT in get_cost)
        self._slot_history: list[torch.Tensor] = []   # list of (n_envs, S, D)
        self._action_history: list[torch.Tensor] = []  # list of (n_envs, act_dim)

        # Cache: encoded once per step, reused across CEM iterations
        self._cached_curr_slots: torch.Tensor | None = None
        self._cached_goal_slots: torch.Tensor | None = None
        self._cached_info_id: int | None = None

    def reset(self):
        """Clear history at episode start."""
        self._slot_history.clear()
        self._action_history.clear()
        self._cached_curr_slots = None
        self._cached_goal_slots = None
        self._cached_info_id = None

    def _encode_pixels(self, pixels: torch.Tensor | np.ndarray) -> torch.Tensor:
        """Encode a batch of pixel observations into slots.

        Args:
            pixels: various shapes from SWM, e.g. (n_envs, 1, H, W, C) or (n_envs, H, W, C)
        Returns:
            slots: (n_envs, S, D)
        """
        if isinstance(pixels, torch.Tensor):
            pixels = pixels.cpu().numpy()

        # Squeeze extra leading dims until (N, H, W, C) or (N, C, H, W)
        while pixels.ndim > 4:
            pixels = pixels.squeeze(1) if pixels.shape[1] == 1 else pixels.reshape(-1, *pixels.shape[-3:])

        # Handle (N, C, H, W) → (N, H, W, C)
        if pixels.ndim == 4 and pixels.shape[1] in (1, 3):
            pixels = pixels.transpose(0, 2, 3, 1)

        # Handle single image (H, W, C) → (1, H, W, C)
        if pixels.ndim == 3:
            pixels = pixels[np.newaxis]

        slots_list = []
        for i in range(pixels.shape[0]):
            s = self.encoder.encode(pixels[i].astype(np.uint8))  # (S, D)
            slots_list.append(s.unsqueeze(0))
        return torch.cat(slots_list, dim=0)  # (n_envs, S, D)

    def _get_history_tensors(self, n_envs: int, action_dim: int, device: torch.device):
        """Build padded history tensors for rollout.

        Returns:
            history: (n_envs, history_len, S, D)
            action_hist: (n_envs, history_len, action_dim)
        """
        # Pad slot history if needed
        slot_hist = list(self._slot_history)
        while len(slot_hist) < self.history_len:
            slot_hist.insert(0, slot_hist[0] if slot_hist else
                             torch.zeros(n_envs, 4, 128, device=device))

        # Pad action history if needed
        act_hist = list(self._action_history)
        while len(act_hist) < self.history_len:
            act_hist.insert(0, torch.zeros(n_envs, action_dim, device=device))

        history = torch.stack(slot_hist[-self.history_len:], dim=1).to(device)
        action_hist = torch.stack(act_hist[-self.history_len:], dim=1).to(device)
        return history, action_hist

    @torch.no_grad()
    def get_cost(self, info_dict: dict, action_candidates: torch.Tensor) -> torch.Tensor:
        """
        Evaluate candidate action sequences by rolling out the world model.

        This is called MULTIPLE times per env step by CEMSolver (once per
        CEM iteration). We cache the encoded slots so encoding happens only once.
        When a NEW observation arrives, we auto-commit the previous cached
        slots to history (so both Frozen and ABM policies maintain history).

        Args:
            info_dict: contains 'pixels' and 'goal' observations
            action_candidates: (n_envs, n_samples, horizon, action_dim)
        Returns:
            costs: (n_envs, n_samples, 1)
        """
        device = action_candidates.device
        n_envs, n_samples, horizon, action_dim = action_candidates.shape

        # Detect new step: if info changed, commit previous step to history
        info_id = id(info_dict.get('pixels', None))
        if self._cached_info_id != info_id:
            # Commit previous cached slots to history (if any)
            if self._cached_curr_slots is not None:
                self._slot_history.append(self._cached_curr_slots.detach())
                if len(self._slot_history) > self.history_len:
                    self._slot_history.pop(0)
                # Also commit a zero action if no action was committed
                if len(self._action_history) < len(self._slot_history):
                    self._action_history.append(
                        torch.zeros(self._cached_curr_slots.shape[0], action_dim, device=device)
                    )
                    if len(self._action_history) > self.history_len:
                        self._action_history.pop(0)

            # Encode new observation
            self._cached_curr_slots = self._encode_pixels(info_dict['pixels']).to(device)
            self._cached_goal_slots = self._encode_pixels(info_dict['goal']).to(device)
            self._cached_info_id = info_id

        curr_slots = self._cached_curr_slots  # (n_envs, S, D)
        goal_slots = self._cached_goal_slots  # (n_envs, S, D)
        S, D = curr_slots.shape[1], curr_slots.shape[2]

        # Build history including current slots (temporary, NOT persisted)
        temp_history = list(self._slot_history) + [curr_slots]
        while len(temp_history) < self.history_len:
            temp_history.insert(0, temp_history[0])
        temp_history = temp_history[-self.history_len:]
        history = torch.stack(temp_history, dim=1).to(device)  # (n_envs, T, S, D)

        # Action history (padded)
        temp_acts = list(self._action_history)
        while len(temp_acts) < self.history_len:
            temp_acts.insert(0, torch.zeros(n_envs, action_dim, device=device))
        action_hist = torch.stack(temp_acts[-self.history_len:], dim=1).to(device)

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
            step_act = flat_actions[:, h:h+1, :]  # (B, 1, act_dim)
            curr_acts = torch.cat([curr_acts[:, 1:, :], step_act], dim=1)
            next_pred = self.model.inference(
                curr_history, actions=curr_acts
            )  # (B, 1, S, D)
            curr_history = torch.cat([curr_history[:, 1:], next_pred], dim=1)

        # Final predicted state
        final_pred = curr_history[:, -1, :, :]  # (n_envs * n_samples, S, D)

        # Expand goal
        goal_exp = goal_slots.unsqueeze(1).expand(-1, n_samples, -1, -1)
        goal_exp = goal_exp.reshape(n_envs * n_samples, S, D).to(device)

        # Cost = MSE to goal
        costs = (final_pred - goal_exp).pow(2).mean(dim=(1, 2))
        costs = costs.reshape(n_envs, n_samples, 1)

        return costs

    def update_step(self, action: np.ndarray | torch.Tensor):
        """Update history after an action is committed to the environment.

        Call this AFTER the action has been executed in the env, to register
        the current observation and action in the rolling history.

        Args:
            action: (n_envs, action_dim) numpy or tensor
        """
        # Add current slots to history (from cache)
        if self._cached_curr_slots is not None:
            self._slot_history.append(self._cached_curr_slots.detach())
            if len(self._slot_history) > self.history_len:
                self._slot_history.pop(0)

        # Add action to history
        if isinstance(action, np.ndarray):
            action = torch.from_numpy(action).float()
        self._action_history.append(action)
        if len(self._action_history) > self.history_len:
            self._action_history.pop(0)

        # Invalidate cache for next step
        self._cached_info_id = None

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
