import torch
import torch.nn as nn
import numpy as np
from loguru import logger as logging
from tqdm import tqdm

class CEMPlanner:
    """
    CEM (Cross-Entropy Method) Planner for MechJEPA in latent slot space.
    
    Optimizes a sequence of actions by rolling out the world model and
    minimizing the distance to a goal slot configuration.
    """
    def __init__(
        self,
        model: nn.Module,
        horizon: int = 16,
        num_samples: int = 512,
        num_elites: int = 64,
        num_iterations: int = 5,
        action_dim: int = 2,
        device: str = "cuda",
        alpha: float = 0.1,  # momentum for mean/std update
    ):
        self.model = model
        self.horizon = horizon
        self.num_samples = num_samples
        self.num_elites = num_elites
        self.num_iterations = num_iterations
        self.action_dim = action_dim
        self.device = device
        self.alpha = alpha

    @torch.no_grad()
    def plan(self, history: torch.Tensor, goal_slots: torch.Tensor) -> torch.Tensor:
        """
        Args:
            history: (B, T_hist, S, D) — current history of slots
            goal_slots: (S, D) or (B, S, D) — target slot configuration
        Returns:
            best_action_sequence: (B, horizon, action_dim)
        """
        B, T_hist, S, D = history.shape
        if goal_slots.dim() == 2:
            goal_slots = goal_slots.unsqueeze(0).expand(B, -1, -1)
            
        # Initialize mean and std for action sequences
        # Push-T actions are in [-1, 1] range
        mean = torch.zeros(B, self.horizon, self.action_dim, device=self.device)
        std = torch.ones(B, self.horizon, self.action_dim, device=self.device) * 0.5
        
        best_actions = None
        min_cost = torch.full((B,), float('inf'), device=self.device)

        for i in range(self.num_iterations):
            # 1. Sample action sequences: (B, num_samples, horizon, action_dim)
            # Use reparameterization-style sampling for vectorized CEM
            epsilon = torch.randn(B, self.num_samples, self.horizon, self.action_dim, device=self.device)
            candidate_actions = mean.unsqueeze(1) + std.unsqueeze(1) * epsilon
            candidate_actions = torch.clamp(candidate_actions, -1.0, 1.0)
            
            # 2. Vectorized rollout: (B * num_samples, horizon, action_dim)
            B_total = B * self.num_samples
            flat_actions = candidate_actions.reshape(B_total, self.horizon, self.action_dim)
            flat_history = history.unsqueeze(1).expand(-1, self.num_samples, -1, -1, -1).reshape(B_total, T_hist, S, D)
            
            # Roll out step-by-step
            curr_history = flat_history
            for h in range(self.horizon):
                # Predict next frame: (B_total, 1, S, D)
                # Note: We use the transition's action. In our model, action[t] handles transition (t -> t+1).
                # Current frame is at index T_hist-1. We need action[T_hist-1] for the rollout.
                # However, the predictor's forward/inference logic takes actions for the entire history.
                # During rollout, we append the predicted frame to history and continue.
                
                # Taking only the relevant action for the *current* transition
                step_action = flat_actions[:, h:h+1, :] # (B_total, 1, action_dim)
                # Construct history actions (padding zeros for previous steps if needed,
                # but predictor only cares about the last T_hist actions)
                # Actually, predictor.inference takes history (B, T_hist, S, D) and actions (B, T_hist, D_act).
                # The action at history[:, -1] is the one that produces the next frame.
                
                hist_actions = torch.zeros(B_total, T_hist, self.action_dim, device=self.device)
                hist_actions[:, -1, :] = step_action.squeeze(1)
                
                next_pred = self.model.inference(curr_history, actions=hist_actions) # (B_total, 1, S, D)
                
                # Update history: drop oldest, append newest
                curr_history = torch.cat([curr_history[:, 1:], next_pred], dim=1)
            
            # 3. Final predicted state: (B_total, S, D)
            final_pred = curr_history[:, -1, :, :]
            
            # 4. Compute cost: MSE to goal slots: (B_total,)
            target_flat = goal_slots.unsqueeze(1).expand(-1, self.num_samples, -1, -1).reshape(B_total, S, D)
            # Cost is per-sample distance
            costs = (final_pred - target_flat).pow(2).mean(dim=(1, 2))
            costs = costs.reshape(B, self.num_samples)
            
            # 5. Select elites
            values, indices = torch.topk(costs, self.num_elites, largest=False)
            # indices: (B, num_elites)
            
            # Gather elite action sequences: (B, num_elites, horizon, action_dim)
            elite_actions = torch.gather(
                candidate_actions, 1, 
                indices.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, self.horizon, self.action_dim)
            )
            
            # 6. Update distribution
            new_mean = elite_actions.mean(dim=1) # (B, horizon, action_dim)
            new_std = elite_actions.std(dim=1)   # (B, horizon, action_dim)
            
            mean = (1 - self.alpha) * new_mean + self.alpha * mean
            std = (1 - self.alpha) * new_std + self.alpha * std
            
            # Track best ever
            current_min_cost, min_idx = torch.min(costs, dim=1)
            mask = current_min_cost < min_cost
            min_cost[mask] = current_min_cost[mask]
            
            batch_best_actions = torch.gather(
                candidate_actions, 1,
                min_idx.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).expand(-1, -1, self.horizon, self.action_dim)
            ).squeeze(1)
            
            if best_actions is None:
                best_actions = batch_best_actions
            else:
                best_actions[mask] = batch_best_actions[mask]

        return best_actions
