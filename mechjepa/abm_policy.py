"""
ABMPolicy — System M adaptation layered on top of SWM's WorldModelPolicy.

This policy subclasses WorldModelPolicy to add surprise-triggered online
adaptation (System M) before each planning step. The flow is:

  1. Encode current pixels → slots (via VideoSAUR)
  2. Compute surprise: MSE(predicted_slots, actual_slots)
  3. If surprise > threshold → adapt codebook + predictor (few gradient steps)
  4. Delegate to WorldModelPolicy.get_action() → CEM planning

Usage:
    from mechjepa.abm_policy import ABMPolicy
    from stable_worldmodel.solver import CEMSolver
    from stable_worldmodel.policy import PlanConfig

    solver = CEMSolver(model=cost_model, num_samples=300, n_steps=5)
    config = PlanConfig(horizon=10, receding_horizon=1, history_len=3)

    policy = ABMPolicy(
        solver=solver, config=config,
        cost_model=cost_model,
        world_model=mech_jepa_model,
        threshold=0.015,
    )
    world.set_policy(policy)
    results = world.evaluate_from_dataset(dataset, ...)
"""

import numpy as np
import torch
import torch.nn.functional as F
from loguru import logger as logging

from stable_worldmodel.policy import WorldModelPolicy


class ABMPolicy(WorldModelPolicy):
    """
    A-B-M policy: Anticipate–Behave–Modulate.

    Extends SWM's WorldModelPolicy with System M (surprise-triggered
    online adaptation). Compatible with all SWM infrastructure:
    evaluate(), evaluate_from_dataset(), record_video().
    """

    def __init__(
        self,
        solver,
        config,
        cost_model,
        world_model,
        threshold: float = 0.015,
        adapt_steps: int = 3,
        adapt_lr_codebook: float = 5e-4,
        adapt_lr_predictor: float = 1e-4,
        grad_clip: float = 0.1,
        **kwargs,
    ):
        """
        Args:
            solver: SWM CEMSolver (or any Solver) wrapping the cost_model
            config: SWM PlanConfig for MPC loop
            cost_model: MechJEPACostModel (holds slot/action history)
            world_model: raw MechJEPA model (for gradient updates)
            threshold: surprise threshold to trigger adaptation
            adapt_steps: number of gradient steps per adaptation
            adapt_lr_codebook: learning rate for codebook parameters
            adapt_lr_predictor: learning rate for predictor parameters
            grad_clip: max gradient norm
            **kwargs: passed to WorldModelPolicy
        """
        super().__init__(solver=solver, config=config, **kwargs)

        self.cost_model = cost_model
        self.world_model = world_model
        self.threshold = threshold
        self.adapt_steps = adapt_steps
        self.grad_clip = grad_clip

        self.optimizer = torch.optim.AdamW([
            {"params": world_model.codebook.parameters(), "lr": adapt_lr_codebook},
            {"params": world_model.predictor.parameters(), "lr": adapt_lr_predictor},
        ])

        # Logging / metrics
        self.surprise_log: list[float] = []
        self.adapted_log: list[bool] = []
        self._step_count = 0

    def reset_metrics(self):
        """Reset per-episode metrics."""
        self.surprise_log.clear()
        self.adapted_log.clear()
        self._step_count = 0
        self.cost_model.reset()

    def _compute_surprise(self, curr_slots: torch.Tensor) -> float:
        """Compute prediction error between model's prediction and actual observation.

        Args:
            curr_slots: (n_envs, S, D) current encoded slots
        Returns:
            surprise: scalar float (mean MSE across envs)
        """
        hist = self.cost_model.slot_history
        ahist = self.cost_model.action_history

        if len(hist) < 3:
            return 0.0

        device = curr_slots.device
        # Build history tensor: (1, T_hist, S, D)
        history = torch.stack(hist[-3:], dim=1).to(device)
        # Build action tensor: (1, T_hist, act_dim)
        actions = torch.stack(ahist[-3:], dim=1).to(device)

        with torch.no_grad():
            pred = self.world_model.inference(history, actions=actions)
            pred = pred.squeeze(1)  # (n_envs, S, D)

        surprise = F.mse_loss(pred, curr_slots).item()
        return surprise

    def _adapt(self, curr_slots: torch.Tensor):
        """System M: adapt codebook + predictor with a few gradient steps.

        Args:
            curr_slots: (n_envs, S, D) current encoded slots (target)
        """
        hist = self.cost_model.slot_history
        ahist = self.cost_model.action_history
        device = curr_slots.device

        history = torch.stack(hist[-3:], dim=1).to(device)
        actions = torch.stack(ahist[-3:], dim=1).to(device)

        self.world_model.train()
        for _ in range(self.adapt_steps):
            self.optimizer.zero_grad()
            pred = self.world_model.differentiable_inference(
                history, actions=actions
            ).squeeze(1)
            loss = F.mse_loss(pred, curr_slots)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                self.world_model.parameters(), self.grad_clip
            )
            self.optimizer.step()
        self.world_model.eval()

    def get_action(self, info_dict: dict, **kwargs) -> np.ndarray:
        """Get action with System M adaptation before planning.

        Overrides WorldModelPolicy.get_action():
        1. Encode current pixels → slots
        2. Compute surprise
        3. Adapt if surprised
        4. Delegate to parent for CEM planning

        Args:
            info_dict: SWM info dict with 'pixels', 'goal', etc.
        Returns:
            actions: (n_envs, action_dim) numpy array
        """
        # Encode current observation for surprise computation
        device = next(self.world_model.parameters()).device
        pixels_nhwc = self.cost_model._to_nhwc(info_dict['pixels'])
        curr_slots = self.cost_model._encode_batch(pixels_nhwc).to(device)

        # System M: check surprise and adapt if needed
        surprise = self._compute_surprise(curr_slots)
        adapted = False

        if surprise > self.threshold and len(self.cost_model.slot_history) >= 3:
            adapted = True
            self._adapt(curr_slots)
            logging.debug(
                f"Step {self._step_count}: System M triggered "
                f"(surprise={surprise:.4f} > {self.threshold})"
            )

        self.surprise_log.append(surprise)
        self.adapted_log.append(adapted)
        self._step_count += 1

        # Delegate to SWM's WorldModelPolicy for CEM planning
        actions = super().get_action(info_dict, **kwargs)

        # Update action history in cost model
        self.cost_model.update_action_history(actions)

        return actions

    def get_metrics(self) -> dict:
        """Return System M metrics for this episode."""
        return {
            "mean_surprise": float(np.mean(self.surprise_log)) if self.surprise_log else 0.0,
            "max_surprise": float(np.max(self.surprise_log)) if self.surprise_log else 0.0,
            "total_adaptations": int(sum(self.adapted_log)),
            "steps": self._step_count,
        }
