"""
MechJEPA Cost Model — SWM-compatible interface for planning.

Implements the `Costable` protocol required by SWM's CEMSolver:
    get_cost(info_dict, action_candidates) → costs

The cost is the MSE between the predicted final slot state and the goal
slot state, computed by rolling out MechJEPA in latent space for each
candidate action sequence.

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
      - info_dict['pixels']: current observation (n_envs, C, H, W) uint8
      - info_dict['goal']:   goal observation    (n_envs, C, H, W) uint8
      - action_candidates:   (n_envs, n_samples, horizon, action_dim) float32

    Returns costs: (n_envs, n_samples, 1) — lower is better.
    """

    def __init__(self, world_model: nn.Module, encoder: nn.Module,
                 history_len: int = 3):
        super().__init__()
        self.model = world_model
        self.encoder = encoder
        self.history_len = history_len

        # Rolling history maintained across steps
        self._slot_history: list[torch.Tensor] = []  # list of (1, S, D)
        self._action_history: list[torch.Tensor] = []  # list of (1, act_dim)

    def reset(self):
        """Clear history at episode start."""
        self._slot_history.clear()
        self._action_history.clear()

    def _encode_pixels(self, pixels: torch.Tensor | np.ndarray) -> torch.Tensor:
        """Encode a batch of pixel observations into slots.

        Args:
            pixels: various shapes from SWM, e.g. (n_envs, 1, H, W, C) or (n_envs, H, W, C)
        Returns:
            slots: (n_envs, S, D)
        """
        if isinstance(pixels, torch.Tensor):
            pixels = pixels.cpu().numpy()

        # Squeeze extra leading dims until we have (N, H, W, C) or (N, C, H, W)
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

    @torch.no_grad()
    def get_cost(self, info_dict: dict, action_candidates: torch.Tensor) -> torch.Tensor:
        """
        Evaluate candidate action sequences by rolling out the world model.

        Args:
            info_dict: contains 'pixels' and 'goal' observations
            action_candidates: (n_envs, n_samples, horizon, action_dim)
        Returns:
            costs: (n_envs, n_samples, 1)
        """
        device = action_candidates.device
        n_envs, n_samples, horizon, action_dim = action_candidates.shape

        # Encode current observation
        curr_slots = self._encode_pixels(info_dict['pixels'])  # (n_envs, S, D)
        S, D = curr_slots.shape[1], curr_slots.shape[2]

        # Encode goal
        goal_slots = self._encode_pixels(info_dict['goal'])  # (n_envs, S, D)

        # Update history
        self._slot_history.append(curr_slots)
        if len(self._slot_history) > self.history_len:
            self._slot_history.pop(0)

        # Pad history if not enough frames yet
        while len(self._slot_history) < self.history_len:
            self._slot_history.insert(0, self._slot_history[0])

        # Build history tensor: (n_envs, T_hist, S, D)
        history = torch.stack(self._slot_history, dim=1).to(device)

        # Build action history: (n_envs, T_hist, action_dim)
        while len(self._action_history) < self.history_len:
            self._action_history.insert(
                0, torch.zeros(n_envs, action_dim, device=device)
            )
        action_hist = torch.stack(
            self._action_history[-self.history_len:], dim=1
        ).to(device)

        # Expand for n_samples: (n_envs * n_samples, T_hist, S, D)
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
            # Shift action buffer
            curr_acts = torch.cat([curr_acts[:, 1:, :], step_act], dim=1)
            # Predict next slots
            next_pred = self.model.inference(
                curr_history, actions=curr_acts
            )  # (B, 1, S, D)
            # Shift slot history
            curr_history = torch.cat([curr_history[:, 1:], next_pred], dim=1)

        # Final predicted state
        final_pred = curr_history[:, -1, :, :]  # (n_envs * n_samples, S, D)

        # Expand goal
        goal_exp = goal_slots.unsqueeze(1).expand(-1, n_samples, -1, -1)
        goal_exp = goal_exp.reshape(n_envs * n_samples, S, D).to(device)

        # Cost = MSE to goal
        costs = (final_pred - goal_exp).pow(2).mean(dim=(1, 2))  # (n_envs * n_samples,)
        costs = costs.reshape(n_envs, n_samples, 1)

        return costs

    def update_action_history(self, action: np.ndarray | torch.Tensor):
        """Update the action history after executing an action.

        Args:
            action: (n_envs, action_dim) numpy or tensor
        """
        if isinstance(action, np.ndarray):
            action = torch.from_numpy(action).float()
        self._action_history.append(action)
        if len(self._action_history) > self.history_len:
            self._action_history.pop(0)

    @property
    def slot_history(self) -> list[torch.Tensor]:
        """Access to slot history for System M surprise computation."""
        return self._slot_history

    @property
    def action_history(self) -> list[torch.Tensor]:
        """Access to action history for System M surprise computation."""
        return self._action_history
