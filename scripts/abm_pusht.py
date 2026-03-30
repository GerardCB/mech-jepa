"""
A-B-M Agent Demo for Push-T (latent-space version).

Demonstrates the full System A / System B / System M loop:
  - System A: MechJEPA world model predicts the next slot state
  - System B: CEM planner finds an action sequence toward a goal
  - System M: Monitors per-slot prediction surprise; triggers online
              adaptation when surprise exceeds a threshold

Distribution shift: we perturb the slot observations with a scaling
factor alpha > 1 to simulate a heavier block (larger, slower dynamics).
This is the cleanest way to test the loop without a pixel encoder.

Conditions compared:
  1. Frozen model  — no System M, no gradient updates at test time
  2. A-B-M agent   — System M triggers adaptation on surprising steps

Usage:
    python scripts/abm_pusht.py \\
        --ckpt /workspace/checkpoints/mechjepa_pusht_act_best.ckpt \\
        --data /workspace/data/pusht_slots_actions.pkl
"""

import argparse
import torch
import torch.nn.functional as F
import numpy as np
import pickle as pkl
from copy import deepcopy
from loguru import logger as logging

from mechjepa.model import MechJEPA
from mechjepa.planner import CEMPlanner


# ── Model config must match the training run exactly ──────────────────────────
MODEL_CFG = dict(
    num_slots=4, slot_dim=128, num_mechanisms=8,
    history_frames=3, pred_frames=1, action_dim=2,
    transformer_depth=6, transformer_heads=16,
    transformer_dim_head=64, transformer_mlp_dim=2048,
    edge_hidden_dim=256,
)


def load_model(ckpt_path: str, device: str) -> MechJEPA:
    model = MechJEPA(**MODEL_CFG)
    sd = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(sd)
    return model.to(device)


def compute_surprise(model: MechJEPA, history: torch.Tensor,
                     hist_actions: torch.Tensor,
                     actual_next: torch.Tensor) -> float:
    """
    Simple per-slot prediction error (MSE) as the surprise metric.

    Args:
        history:      (1, T_hist, S, D)
        hist_actions: (1, T_hist, action_dim)
        actual_next:  (1, S, D)  — the REAL observed next frame

    Returns:
        scalar surprise (mean MSE over all slots)
    """
    with torch.no_grad():
        pred_next = model.inference(history, actions=hist_actions)  # (1, 1, S, D)
    pred_next = pred_next.squeeze(1)  # (1, S, D)
    return F.mse_loss(pred_next, actual_next).item()


def differentiable_predict(model: MechJEPA, history: torch.Tensor,
                           hist_actions: torch.Tensor) -> torch.Tensor:
    """
    Differentiable version of model.inference() — same logic but no
    @torch.no_grad() wrapper, so gradients can flow for adaptation.

    Args:
        history:      (B, T_hist, S, D)
        hist_actions: (B, T_hist, action_dim)

    Returns:
        pred_next: (B, S, D) — predicted next frame
    """
    z_t = history[:, -1, :, :]
    codebook_output = model.codebook(z_t)
    m_ij = codebook_output["m_ij"]
    pred = model.predictor.inference(history, m_ij=m_ij, actions=hist_actions)
    return pred.squeeze(1)  # (B, S, D)


def adaptation_step(model: MechJEPA, optimizer: torch.optim.Optimizer,
                    history: torch.Tensor, hist_actions: torch.Tensor,
                    actual_next: torch.Tensor, n_steps: int = 3) -> float:
    """
    Take n_steps gradient steps to minimise prediction error on the
    current surprising transition.  Only the codebook and predictor
    are updated; this is intentionally narrow to avoid forgetting.

    Returns: loss after the last step
    """
    model.train()
    last_loss = float("nan")
    for _ in range(n_steps):
        optimizer.zero_grad()
        pred_next = differentiable_predict(model, history, hist_actions)
        loss = F.mse_loss(pred_next, actual_next)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
        optimizer.step()
        last_loss = loss.item()
    model.eval()
    return last_loss



def run_episode(
    model: MechJEPA,
    planner: CEMPlanner,
    slots_seq: torch.Tensor,       # (T, S, D) — full episode slots
    actions_seq: torch.Tensor,     # (T, action_dim)
    goal_slots: torch.Tensor,      # (S, D)
    shift_alpha: float = 1.0,      # OOD slot-scale factor (1 = in-distribution)
    system_m: bool = False,
    optimizer=None,
    surprise_threshold: float = 0.02,
    adaptation_steps: int = 3,
    device: str = "cuda",
) -> dict:
    """
    Run one 'episode' stepping through the supplied slot sequence.

    The OOD distribution shift is applied by scaling observed slots
    by shift_alpha (>1 makes objects appear 'heavier'/larger, <1 lighter).
    The world model was trained on alpha=1.0 data.

    Returns: dict with per-step metrics
    """
    T, S, D = slots_seq.shape
    history_size = 3
    horizon = planner.horizon

    # Circular history buffers
    hist_slots = []   # list of (1, S, D) tensors
    hist_acts  = []   # list of (1, action_dim) tensors

    adapt_count = 0
    surprise_log = []
    planning_error_log = []

    for t in range(T - 1):
        # ── Observe (with optional OOD shift) ──────────────────────────────
        obs_slots = slots_seq[t].unsqueeze(0) * shift_alpha  # (1, S, D)

        hist_slots.append(obs_slots)
        if len(hist_slots) > history_size:
            hist_slots.pop(0)

        if len(hist_acts) == 0 or len(hist_acts) < history_size:
            # Pad with the actual expert action (or zero for very first step)
            hist_acts.insert(0, actions_seq[max(0, t - 1)].unsqueeze(0))
        if len(hist_acts) > history_size:
            hist_acts.pop(0)

        if len(hist_slots) < history_size:
            continue  # need full history before monitoring or planning

        # Stack buffers
        hist_t  = torch.cat(hist_slots, dim=0).unsqueeze(0)   # (1, T_h, S, D)
        hact_t  = torch.cat(hist_acts,  dim=0).unsqueeze(0)   # (1, T_h, 2)

        # ── System M: compute surprise vs actual next frame ─────────────────
        next_obs = slots_seq[t + 1].unsqueeze(0) * shift_alpha  # (1, S, D)
        surprise = compute_surprise(model, hist_t, hact_t, next_obs)
        surprise_log.append(surprise)

        if system_m and optimizer is not None and surprise > surprise_threshold:
            adapt_count += 1
            adaptation_step(model, optimizer, hist_t, hact_t, next_obs,
                            n_steps=adaptation_steps)

        # ── System B: CEM plan toward goal ─────────────────────────────────
        planned = planner.plan(hist_t, hact_t, goal_slots)  # (1, H, 2)

        # One-step planning error: compare predicted next vs actual next (in distribution)
        with torch.no_grad():
            step_act = planned[:, 0:1, :]  # (1, 1, 2)
            curr_acts = torch.cat([hact_t[:, 1:, :], step_act], dim=1)
            pred_next  = model.inference(hist_t, actions=curr_acts).squeeze(1)
        planning_err = F.mse_loss(pred_next, next_obs).item()
        planning_error_log.append(planning_err)

        # Advance expert action buffer
        hist_acts.append(actions_seq[t].unsqueeze(0))
        if len(hist_acts) > history_size:
            hist_acts.pop(0)

    return {
        "surprise":      np.array(surprise_log),
        "planning_err":  np.array(planning_error_log),
        "adapt_count":   adapt_count,
        "mean_surprise": float(np.mean(surprise_log)) if surprise_log else 0.0,
        "mean_plan_err": float(np.mean(planning_error_log)) if planning_error_log else 0.0,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt",      default="/workspace/checkpoints/mechjepa_pusht_act_best.ckpt")
    parser.add_argument("--data",      default="/workspace/data/pusht_slots_actions.pkl")
    parser.add_argument("--shift",     type=float, default=1.4,
                        help="OOD slot scale factor (1.0 = in-distribution, 1.4 = heavy-block shift)")
    parser.add_argument("--threshold", type=float, default=0.015,
                        help="Surprise threshold for System M adaptation trigger")
    parser.add_argument("--adapt_steps", type=int, default=3)
    parser.add_argument("--horizon",   type=int, default=10)
    parser.add_argument("--episodes",  type=int, default=5,
                        help="Number of validation episodes to evaluate")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ── Load data ─────────────────────────────────────────────────────────
    logging.info(f"Loading data from {args.data}")
    with open(args.data, "rb") as f:
        data = pkl.load(f)
    val_data = data["val"]
    ep_keys  = sorted(list(val_data.keys()))[:args.episodes]
    logging.info(f"Using {len(ep_keys)} val episodes, OOD shift alpha={args.shift}")

    # Hardcode a fixed goal: last frame of the first validation episode
    goal_slots = torch.from_numpy(
        val_data[ep_keys[0]]["slots"][-1]
    ).float().to(device)

    # ── Condition 1: Frozen model ─────────────────────────────────────────
    logging.info("▶ Condition 1: Frozen model (no System M)")
    frozen_model = load_model(args.ckpt, device).eval()
    frozen_planner = CEMPlanner(frozen_model, horizon=args.horizon,
                                num_samples=256, num_iterations=5, device=device)

    frozen_results = []
    for key in ep_keys:
        ep = val_data[key]
        slots   = torch.from_numpy(ep["slots"]).float().to(device)
        actions = torch.from_numpy(ep["actions"]).float().to(device)
        r = run_episode(frozen_model, frozen_planner, slots, actions, goal_slots,
                        shift_alpha=args.shift, system_m=False, device=device)
        frozen_results.append(r)
        logging.info(f"  {key}: surprise={r['mean_surprise']:.4f}  plan_err={r['mean_plan_err']:.4f}")

    # ── Condition 2: A-B-M Agent ──────────────────────────────────────────
    logging.info("▶ Condition 2: A-B-M Agent (System M active)")
    abm_model = load_model(args.ckpt, device)
    abm_model.eval()
    optimizer = torch.optim.AdamW([
        {"params": abm_model.codebook.parameters(),  "lr": 5e-4},
        {"params": abm_model.predictor.parameters(), "lr": 1e-4},
    ])
    abm_planner = CEMPlanner(abm_model, horizon=args.horizon,
                             num_samples=256, num_iterations=5, device=device)

    abm_results = []
    for key in ep_keys:
        ep = val_data[key]
        slots   = torch.from_numpy(ep["slots"]).float().to(device)
        actions = torch.from_numpy(ep["actions"]).float().to(device)
        r = run_episode(abm_model, abm_planner, slots, actions, goal_slots,
                        shift_alpha=args.shift, system_m=True, optimizer=optimizer,
                        surprise_threshold=args.threshold,
                        adaptation_steps=args.adapt_steps, device=device)
        abm_results.append(r)
        logging.info(f"  {key}: surprise={r['mean_surprise']:.4f}  plan_err={r['mean_plan_err']:.4f}  adaptations={r['adapt_count']}")

    # ── Summary ───────────────────────────────────────────────────────────
    frozen_pe = np.mean([r["mean_plan_err"]  for r in frozen_results])
    abm_pe    = np.mean([r["mean_plan_err"]  for r in abm_results])
    frozen_sr = np.mean([r["mean_surprise"]  for r in frozen_results])
    abm_sr    = np.mean([r["mean_surprise"]  for r in abm_results])
    total_ada = sum(r["adapt_count"] for r in abm_results)

    logging.info("")
    logging.info("=" * 55)
    logging.info(f"  OOD shift alpha = {args.shift}")
    logging.info(f"  {'':20s}  {'Frozen':>10s}  {'A-B-M':>10s}")
    logging.info(f"  {'Mean surprise':20s}  {frozen_sr:>10.4f}  {abm_sr:>10.4f}")
    logging.info(f"  {'Mean plan error':20s}  {frozen_pe:>10.4f}  {abm_pe:>10.4f}")
    logging.info(f"  {'Improvement':20s}  {'—':>10s}  {frozen_pe/abm_pe:>9.2f}x")
    logging.info(f"  Total System M adaptations: {total_ada}")
    logging.info("=" * 55)


if __name__ == "__main__":
    main()
