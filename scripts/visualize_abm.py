"""
visualize_abm.py — Generate publication-quality figures and comparison videos
for the MechJEPA A-B-M demo.

Outputs (all saved to /workspace/results/):
  - surprise_comparison.png   — per-step surprise: Frozen vs A-B-M
  - plan_err_comparison.png   — per-step planning error: Frozen vs A-B-M
  - summary_bar.png           — summary bar chart (both metrics)
  - abm_demo.mp4              — side-by-side latent trajectory animation

Usage:
    python scripts/visualize_abm.py \\
        --ckpt /workspace/checkpoints/mechjepa_pusht_act_best.ckpt \\
        --data /workspace/data/pusht_slots_actions.pkl
"""

import argparse
import os
import sys
import numpy as np
import torch
import torch.nn.functional as F
import pickle as pkl
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
from copy import deepcopy
from loguru import logger as logging

sys.path.insert(0, "/workspace/mechjepa")
from mechjepa.model import MechJEPA
from mechjepa.planner import CEMPlanner

# ── Colours ────────────────────────────────────────────────────────────────
FROZEN_COLOR = "#E63946"   # warm red
ABM_COLOR    = "#2EC4B6"   # teal
ADAPT_COLOR  = "#FFB703"   # amber — marks adaptation events

MODEL_CFG = dict(
    num_slots=4, slot_dim=128, num_mechanisms=8,
    history_frames=3, pred_frames=1, action_dim=2,
    transformer_depth=6, transformer_heads=16,
    transformer_dim_head=64, transformer_mlp_dim=2048,
    edge_hidden_dim=256,
)

SHIFT = 1.4
THRESHOLD = 0.015
ADAPT_STEPS = 3
HORIZON = 10
N_EPISODES = 5
N_SAMPLES = 256


def load_model(ckpt, device):
    model = MechJEPA(**MODEL_CFG)
    model.load_state_dict(torch.load(ckpt, map_location=device, weights_only=False))
    return model.to(device)


def run_episode_detailed(model, planner, slots_seq, actions_seq, goal_slots,
                         shift_alpha, system_m=False, optimizer=None,
                         surprise_threshold=THRESHOLD, adaptation_steps=ADAPT_STEPS,
                         device="cuda"):
    """
    Like run_episode() in abm_pusht.py but also returns per-step logs and
    the raw predicted slot sequences for video rendering.
    """
    T, S, D = slots_seq.shape
    history_size = 3
    hist_s, hist_a = [], []

    surprise_log, plan_err_log, adapt_mask = [], [], []
    pred_traj, real_traj = [], []

    for t in range(T - 1):
        obs_slots = slots_seq[t].unsqueeze(0) * shift_alpha

        hist_s.append(obs_slots)
        if len(hist_s) > history_size: hist_s.pop(0)
        if len(hist_a) < history_size:
            hist_a.insert(0, actions_seq[max(0, t - 1)].unsqueeze(0))
        if len(hist_a) > history_size: hist_a.pop(0)
        if len(hist_s) < history_size: continue

        hist_t  = torch.cat(hist_s, dim=0).unsqueeze(0)
        hact_t  = torch.cat(hist_a, dim=0).unsqueeze(0)
        next_obs = slots_seq[t + 1].unsqueeze(0) * shift_alpha

        # Surprise
        with torch.no_grad():
            pred_next = model.inference(hist_t, actions=hact_t).squeeze(1)
        surprise = F.mse_loss(pred_next, next_obs).item()
        surprise_log.append(surprise)

        adapted = False
        if system_m and optimizer and surprise > surprise_threshold:
            # Adaptation
            model.train()
            for _ in range(adaptation_steps):
                optimizer.zero_grad()
                p = model.differentiable_inference(hist_t, actions=hact_t).squeeze(1)
                loss = F.mse_loss(p, next_obs)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
                optimizer.step()
            model.eval()
            adapted = True
            with torch.no_grad():
                pred_next = model.inference(hist_t, actions=hact_t).squeeze(1)

        adapt_mask.append(adapted)

        # Planning error
        planned = planner.plan(hist_t, hact_t, goal_slots)
        with torch.no_grad():
            step_act  = planned[:, 0:1, :]
            curr_acts = torch.cat([hact_t[:, 1:, :], step_act], dim=1)
            p2 = model.inference(hist_t, actions=curr_acts).squeeze(1)
        plan_err = F.mse_loss(p2, next_obs).item()
        plan_err_log.append(plan_err)

        # Record trajectories (first slot, first two PCA dims for plotting)
        pred_traj.append(pred_next[0, 0, :4].cpu().numpy())
        real_traj.append(next_obs[0, 0, :4].cpu().numpy())

        hist_a.append(actions_seq[t].unsqueeze(0))
        if len(hist_a) > history_size: hist_a.pop(0)

    return {
        "surprise": np.array(surprise_log),
        "plan_err": np.array(plan_err_log),
        "adapt_mask": np.array(adapt_mask),
        "pred_traj": np.array(pred_traj),
        "real_traj": np.array(real_traj),
    }


def make_line_fig(frozen_logs, abm_logs, key, ylabel, title, out_path):
    """Two-panel: ep0 step-by-step, + all-episode mean ribbon."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 4), dpi=150)
    fig.patch.set_facecolor("#0f1117")

    # ── Panel 1: Single episode step-by-step ──────────────────────────────
    ax = axes[0]
    ax.set_facecolor("#1a1d27")
    f = frozen_logs[0][key]
    a = abm_logs[0][key]
    adapt = abm_logs[0]["adapt_mask"]
    xs = np.arange(len(f))
    ax.plot(xs, f, color=FROZEN_COLOR, lw=1.8, label="Frozen Model")
    ax.plot(xs, a, color=ABM_COLOR,    lw=1.8, label="A-B-M Agent")
    # Mark adaptation events
    adapt_xs = np.where(adapt)[0]
    for ax_time in adapt_xs:
        ax.axvline(ax_time, color=ADAPT_COLOR, alpha=0.35, lw=0.8)
    ax.set_xlabel("Step", color="white"); ax.set_ylabel(ylabel, color="white")
    ax.set_title("Episode 1 — Step-by-step", color="white", fontsize=11)
    ax.tick_params(colors="white"); ax.spines[:].set_color("#444")
    adapt_patch = mpatches.Patch(color=ADAPT_COLOR, alpha=0.6, label=f"Adaptation ({len(adapt_xs)} steps)")
    ax.legend(handles=[
        mpatches.Patch(color=FROZEN_COLOR, label="Frozen"),
        mpatches.Patch(color=ABM_COLOR,    label="A-B-M"),
        adapt_patch,
    ], facecolor="#1a1d27", labelcolor="white", fontsize=8)

    # ── Panel 2: Mean ± std across episodes ───────────────────────────────
    ax2 = axes[1]
    ax2.set_facecolor("#1a1d27")
    min_len = min(min(len(l[key]) for l in frozen_logs), min(len(l[key]) for l in abm_logs))
    f_arr = np.stack([l[key][:min_len] for l in frozen_logs])
    a_arr = np.stack([l[key][:min_len] for l in abm_logs])
    xs2   = np.arange(min_len)
    ax2.plot(xs2, f_arr.mean(0), color=FROZEN_COLOR, lw=1.8)
    ax2.fill_between(xs2, f_arr.mean(0)-f_arr.std(0), f_arr.mean(0)+f_arr.std(0),
                     color=FROZEN_COLOR, alpha=0.2)
    ax2.plot(xs2, a_arr.mean(0), color=ABM_COLOR, lw=1.8)
    ax2.fill_between(xs2, a_arr.mean(0)-a_arr.std(0), a_arr.mean(0)+a_arr.std(0),
                     color=ABM_COLOR, alpha=0.2)
    ax2.set_xlabel("Step", color="white"); ax2.set_ylabel(ylabel, color="white")
    ax2.set_title(f"All {len(frozen_logs)} Episodes — Mean ± Std", color="white", fontsize=11)
    ax2.tick_params(colors="white"); ax2.spines[:].set_color("#444")
    ax2.legend(handles=[
        mpatches.Patch(color=FROZEN_COLOR, label="Frozen"),
        mpatches.Patch(color=ABM_COLOR,    label="A-B-M"),
    ], facecolor="#1a1d27", labelcolor="white", fontsize=8)

    fig.suptitle(title, color="white", fontsize=13, y=1.01)
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    logging.info(f"Saved {out_path}")


def make_bar_fig(frozen_logs, abm_logs, out_path):
    """Summary grouped bar chart ."""
    metrics = ["Mean Surprise", "Mean Plan Error"]
    f_vals = [np.mean([l["surprise"].mean() for l in frozen_logs]),
              np.mean([l["plan_err"].mean()  for l in frozen_logs])]
    a_vals = [np.mean([l["surprise"].mean() for l in abm_logs]),
              np.mean([l["plan_err"].mean()  for l in abm_logs])]

    fig, ax = plt.subplots(figsize=(7, 4), dpi=150)
    fig.patch.set_facecolor("#0f1117"); ax.set_facecolor("#1a1d27")
    x = np.arange(len(metrics)); w = 0.32
    b1 = ax.bar(x - w/2, f_vals, w, label="Frozen Model", color=FROZEN_COLOR, alpha=0.9)
    b2 = ax.bar(x + w/2, a_vals, w, label="A-B-M Agent",  color=ABM_COLOR,    alpha=0.9)
    for bar, val in zip(b1, f_vals):
        ax.text(bar.get_x() + bar.get_width()/2, val + 0.001, f"{val:.3f}",
                ha="center", va="bottom", color="white", fontsize=8)
    for bar, val in zip(b2, a_vals):
        impr = f_vals[list(b2).index(bar)] / val
        ax.text(bar.get_x() + bar.get_width()/2, val + 0.001, f"{val:.3f}\n({impr:.1f}×↓)",
                ha="center", va="bottom", color=ABM_COLOR, fontsize=8, fontweight="bold")
    ax.set_xticks(x); ax.set_xticklabels(metrics, color="white")
    ax.tick_params(colors="white"); ax.spines[:].set_color("#444")
    ax.set_ylabel("Error (MSE)", color="white")
    ax.set_title(f"MechJEPA A-B-M vs Frozen  |  OOD shift α={SHIFT}", color="white", fontsize=11)
    ax.legend(facecolor="#1a1d27", labelcolor="white", fontsize=9)
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    logging.info(f"Saved {out_path}")


def make_video(frozen_log, abm_log, goal_slots, out_path):
    """
    Side-by-side animation: slot position traces for Frozen (left) vs A-B-M (right).
    Uses first 2 dimensions of slot-0 as 2D proxy for position.
    """
    try:
        import cv2
    except ImportError:
        logging.warning("opencv not available, skipping video")
        return

    f_traj = frozen_log["pred_traj"]   # (T, 4)
    a_traj = abm_log["pred_traj"]
    r_traj = frozen_log["real_traj"]   # ground truth (same for both)
    adapt  = abm_log["adapt_mask"]
    T = min(len(f_traj), len(a_traj), len(r_traj))

    W, H = 540, 300
    fps = 10
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(out_path, fourcc, fps, (W * 2, H))

    # Normalise trajectories to [0,1] for rendering
    all_xy = np.concatenate([f_traj[:T, :2], a_traj[:T, :2], r_traj[:T, :2]], axis=0)
    lo, hi = all_xy.min(0) - 0.05, all_xy.max(0) + 0.05

    def to_px(pt, w, h):
        x = int(np.clip((pt[0] - lo[0]) / (hi[0] - lo[0]), 0, 1) * (w - 30) + 15)
        y = int(np.clip(1 - (pt[1] - lo[1]) / (hi[1] - lo[1]), 0, 1) * (h - 30) + 15)
        return (x, y)

    BG      = (17, 17, 25)
    COL_F   = (70,  57, 230)   # blue-ish for frozen pred
    COL_A   = (46, 196, 182)   # teal for ABM pred
    COL_GT  = (200, 200, 200)  # white for GT
    COL_AD  = (3,  183, 255)   # amber adapt flash

    trail_f, trail_a, trail_r = [], [], []

    for t in range(T):
        frame_f = np.full((H, W, 3), BG, dtype=np.uint8)
        frame_a = np.full((H, W, 3), BG, dtype=np.uint8)

        pf = to_px(f_traj[t, :2], W, H)
        pa = to_px(a_traj[t, :2], W, H)
        pr = to_px(r_traj[t, :2], W, H)
        trail_f.append(pf); trail_a.append(pa); trail_r.append(pr)

        is_adapt = t < len(adapt) and adapt[t]

        for i in range(1, len(trail_f)):
            alpha = i / len(trail_f)
            cv2.line(frame_f, trail_r[i-1], trail_r[i], tuple(int(c*alpha) for c in COL_GT), 1)
            cv2.line(frame_f, trail_f[i-1], trail_f[i], tuple(int(c*alpha) for c in COL_F),  1)
            cv2.line(frame_a, trail_r[i-1], trail_r[i], tuple(int(c*alpha) for c in COL_GT), 1)
            cv2.line(frame_a, trail_a[i-1], trail_a[i], tuple(int(c*alpha) for c in COL_A),  1)

        cv2.circle(frame_f, pr, 5, COL_GT, -1)
        cv2.circle(frame_f, pf, 6, COL_F,  -1)
        cv2.circle(frame_a, pr, 5, COL_GT, -1)
        cv2.circle(frame_a, pa, 6, COL_A,  -1)

        # Adaptation flash
        if is_adapt:
            cv2.rectangle(frame_a, (0, 0), (W-1, H-1), COL_AD, 3)
            cv2.putText(frame_a, "ADAPTING", (10, 22), cv2.FONT_HERSHEY_SIMPLEX,
                        0.55, COL_AD, 1, cv2.LINE_AA)

        err_f = frozen_log["surprise"][t] if t < len(frozen_log["surprise"]) else 0
        err_a = abm_log["surprise"][t]    if t < len(abm_log["surprise"])    else 0

        cv2.putText(frame_f, f"Frozen   |  Surprise: {err_f:.4f}", (10, H-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, COL_GT, 1, cv2.LINE_AA)
        cv2.putText(frame_a, f"A-B-M    |  Surprise: {err_a:.4f}", (10, H-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, COL_GT, 1, cv2.LINE_AA)
        cv2.putText(frame_f, f"t={t}", (W-45, 18), cv2.FONT_HERSHEY_SIMPLEX,
                    0.4, (180, 180, 180), 1, cv2.LINE_AA)

        frame = np.concatenate([frame_f, frame_a], axis=1)
        # White divider
        frame[:, W-1:W+1] = 255
        out.write(frame)

    out.release()
    logging.info(f"Saved {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt",     default="/workspace/checkpoints/mechjepa_pusht_act_best.ckpt")
    parser.add_argument("--data",     default="/workspace/data/pusht_slots_actions.pkl")
    parser.add_argument("--out_dir",  default="/workspace/results")
    parser.add_argument("--episodes", type=int, default=N_EPISODES)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    logging.info("Loading data...")
    with open(args.data, "rb") as f:
        data = pkl.load(f)
    val  = data["val"]
    keys = sorted(list(val.keys()))[:args.episodes]
    goal_slots = torch.from_numpy(val[keys[0]]["slots"][-1]).float().to(device)

    frozen_model = load_model(args.ckpt, device).eval()
    frozen_plan  = CEMPlanner(frozen_model, horizon=HORIZON, num_samples=N_SAMPLES,
                              num_iterations=5, device=device)

    abm_model = load_model(args.ckpt, device).eval()
    optimizer = torch.optim.AdamW([
        {"params": abm_model.codebook.parameters(),  "lr": 5e-4},
        {"params": abm_model.predictor.parameters(), "lr": 1e-4},
    ])
    abm_plan = CEMPlanner(abm_model, horizon=HORIZON, num_samples=N_SAMPLES,
                          num_iterations=5, device=device)

    logging.info("Running Frozen condition...")
    frozen_logs = []
    for key in keys:
        ep = val[key]
        slots   = torch.from_numpy(ep["slots"]).float().to(device)
        actions = torch.from_numpy(ep["actions"]).float().to(device)
        r = run_episode_detailed(frozen_model, frozen_plan, slots, actions,
                                 goal_slots, shift_alpha=SHIFT, device=device)
        frozen_logs.append(r)
        logging.info(f"  Frozen {key}: surprise={r['surprise'].mean():.4f}  plan_err={r['plan_err'].mean():.4f}")

    logging.info("Running A-B-M condition...")
    abm_logs = []
    for key in keys:
        ep = val[key]
        slots   = torch.from_numpy(ep["slots"]).float().to(device)
        actions = torch.from_numpy(ep["actions"]).float().to(device)
        r = run_episode_detailed(abm_model, abm_plan, slots, actions,
                                 goal_slots, shift_alpha=SHIFT, system_m=True,
                                 optimizer=optimizer, device=device)
        abm_logs.append(r)
        n_adapt = r["adapt_mask"].sum()
        logging.info(f"  A-B-M  {key}: surprise={r['surprise'].mean():.4f}  plan_err={r['plan_err'].mean():.4f}  adapts={n_adapt}")

    # ── Figures ────────────────────────────────────────────────────────────
    logging.info("Generating figures...")

    make_line_fig(frozen_logs, abm_logs, "surprise", "Surprise (MSE)",
                  "Per-Step Prediction Surprise — Frozen vs A-B-M (OOD α=1.4)",
                  os.path.join(args.out_dir, "surprise_comparison.png"))

    make_line_fig(frozen_logs, abm_logs, "plan_err", "Planning Error (MSE)",
                  "Per-Step Latent Planning Error — Frozen vs A-B-M (OOD α=1.4)",
                  os.path.join(args.out_dir, "plan_err_comparison.png"))

    make_bar_fig(frozen_logs, abm_logs,
                 os.path.join(args.out_dir, "summary_bar.png"))

    # ── Video (episode 0 only) ─────────────────────────────────────────────
    logging.info("Generating video...")
    make_video(frozen_logs[0], abm_logs[0], goal_slots,
               os.path.join(args.out_dir, "abm_demo.mp4"))

    # ── Print final table ──────────────────────────────────────────────────
    f_s = np.mean([l["surprise"].mean() for l in frozen_logs])
    a_s = np.mean([l["surprise"].mean() for l in abm_logs])
    f_p = np.mean([l["plan_err"].mean()  for l in frozen_logs])
    a_p = np.mean([l["plan_err"].mean()  for l in abm_logs])
    tot_adapt = sum(l["adapt_mask"].sum() for l in abm_logs)

    logging.info("")
    logging.info("=" * 60)
    logging.info(f"  OOD shift alpha = {SHIFT}")
    logging.info(f"  {'':22s}  {'Frozen':>10s}  {'A-B-M':>10s}")
    logging.info(f"  {'Mean Surprise':22s}  {f_s:>10.4f}  {a_s:>10.4f}  ({f_s/a_s:.1f}x)")
    logging.info(f"  {'Mean Plan Error':22s}  {f_p:>10.4f}  {a_p:>10.4f}  ({f_p/a_p:.1f}x)")
    logging.info(f"  Total System M adaptations: {tot_adapt}")
    logging.info("=" * 60)
    logging.info(f"Results saved to {args.out_dir}/")


if __name__ == "__main__":
    main()
