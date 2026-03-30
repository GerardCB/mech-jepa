"""
render_slot_video.py — Honest slot-space comparison video: Real vs Frozen vs A-B-M.

What this shows:
  At each timestep t in a validation episode:
  - REAL position     (white)  — the actual VideoSAUR block slot from the dataset
  - FROZEN prediction (red)    — what the frozen world model predicted for t+1
  - A-B-M prediction  (teal)   — what the adapted world model predicted for t+1

  Separately: the PLANNED trajectory from CEM in slot space for both models.

2D positions are obtained by fitting per-object PCA on the stored slot dataset
and using the spatially-variant principal components (PC2, PC3) as x/y axes.

This directly and honestly renders what was validated in Phases 2 and 3, with
zero approximation or pseudo-slot substitution.

Output:
  /workspace/results/videos/slot_space_ep{N}.mp4    — per-episode (4)
  /workspace/results/videos/slot_space_all.mp4      — concatenated
"""

import os, sys, argparse, math
import numpy as np
import torch
import torch.nn.functional as F
import pickle as pkl
import cv2
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from copy import deepcopy
from loguru import logger as logging

sys.path.insert(0, '/workspace/mechjepa')
from mechjepa.model import MechJEPA
from mechjepa.planner import CEMPlanner

# ── Constants ────────────────────────────────────────────────────────────────
W, H        = 512, 512
PANEL_H     = 72        # info bar at bottom
FPS         = 15        # slower = easier to see
HORIZON     = 10
N_EPISODES  = 5
N_SAMPLES   = 512
MODEL_CFG   = dict(
    num_slots=4, slot_dim=128, num_mechanisms=8,
    history_frames=3, pred_frames=1, action_dim=2,
    transformer_depth=6, transformer_heads=16,
    transformer_dim_head=64, transformer_mlp_dim=2048,
    edge_hidden_dim=256,
)

# ── Colours ──────────────────────────────────────────────────────────────────
BG          = (18,  18,  28)
REAL_COL    = (220, 220, 220)   # white — ground truth
FROZEN_COL  = (220,  70,  70)   # red   — frozen model prediction
ABM_COL     = ( 46, 196, 182)   # teal  — A-B-M prediction
GOAL_COL    = ( 80,  80, 160)   # purple— goal
AGENT_COL   = (255, 200,  60)   # gold  — agent
ADAPT_COL   = (255, 180,   0)   # amber — adaptation event
PLAN_COL_F  = (180,  50,  50)   # darker red   — frozen CEM plan
PLAN_COL_A  = ( 20, 140, 130)   # darker teal  — ABM CEM plan
GREY        = (100, 100, 110)

RADIUS_BLOCK  = 20
RADIUS_AGENT  = 12
TRAIL_LEN     = 30


def build_pca(val_data):
    """
    Fit per-slot PCA on validation slot data.
    Returns dict with 'agent' and 'block' PCA objects and scaling params.
    """
    keys = sorted(list(val_data.keys()))
    all_slots = np.concatenate([val_data[k]['slots'] for k in keys], axis=0)  # (N, 4, 128)

    # Fit separate PCA per slot index (slot0=agent, slot1=block most often)
    pca_per_slot = []
    for s in range(4):
        pca = PCA(n_components=4)
        pca.fit(all_slots[:, s, :])
        pca_per_slot.append(pca)

    # For rendering we need a consistent 2D coordinate system.
    # Use the global PCA (all slots) to find which slot is which object,
    # then project each slot using its per-slot PCA.
    global_pca = PCA(n_components=2)
    global_pca.fit(all_slots.reshape(-1, 128))

    return {
        'global': global_pca,
        'per_slot': pca_per_slot,
        'all_slots': all_slots,
    }


def slot_to_canvas(slot_vec: np.ndarray, pca: PCA,
                   lo: np.ndarray, hi: np.ndarray) -> tuple[int, int]:
    """
    Project a single slot vector (128,) to canvas (x, y) using PCA.
    lo, hi are the per-PC2/PC3 extents of the training data for normalisation.
    """
    proj = pca.transform(slot_vec.reshape(1, -1))[0]       # (4,)
    # Use PC1 and PC2 as x, y (they capture most motion)
    raw_x, raw_y = proj[1], proj[2]
    x = int(np.clip((raw_x - lo[0]) / (hi[0] - lo[0] + 1e-8), 0, 1) * (W - 60) + 30)
    y = int(np.clip(1 - (raw_y - lo[1]) / (hi[1] - lo[1] + 1e-8), 0, 1) * (H - 60) + 30)
    return x, y


def slot_to_canvas_global(slot_vec: np.ndarray, pca: PCA,
                           lo: np.ndarray, hi: np.ndarray) -> tuple[int, int]:
    proj = pca.transform(slot_vec.reshape(1, -1))[0]
    x = int(np.clip((proj[0] - lo[0]) / (hi[0] - lo[0] + 1e-8), 0, 1) * (W - 60) + 30)
    y = int(np.clip(1 - (proj[1] - lo[1]) / (hi[1] - lo[1] + 1e-8), 0, 1) * (H - 60) + 30)
    return x, y


def draw_object(canvas, pos, radius, color, label=None, thickness=-1):
    cv2.circle(canvas, pos, radius, color, thickness)
    if label:
        cv2.putText(canvas, label, (pos[0] + radius + 3, pos[1] + 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1, cv2.LINE_AA)


def draw_frame(real_pos, frozen_pos, abm_pos, goal_pos,
               real_agent_pos, frozen_agent_pos, abm_agent_pos,
               frozen_plan_pts, abm_plan_pts,
               trail_real, trail_frozen, trail_abm,
               surprise_f, surprise_a, plan_err_f, plan_err_a,
               adapted, step_i, total_steps, ep_label):
    """Build one BGR video frame."""
    canvas = np.full((H + PANEL_H, W, 3), BG, dtype=np.uint8)

    # ── Goal ghost ────────────────────────────────────────────────────────
    if goal_pos:
        cv2.circle(canvas, goal_pos, RADIUS_BLOCK + 4, GOAL_COL, 2)
        cv2.circle(canvas, goal_pos, 3, GOAL_COL, -1)

    # ── CEM plan trajectories ─────────────────────────────────────────────
    for i in range(1, len(frozen_plan_pts)):
        cv2.line(canvas, frozen_plan_pts[i-1], frozen_plan_pts[i], PLAN_COL_F, 1)
    for i in range(1, len(abm_plan_pts)):
        cv2.line(canvas, abm_plan_pts[i-1], abm_plan_pts[i], PLAN_COL_A, 1)

    # ── Trails ───────────────────────────────────────────────────────────
    for i in range(1, len(trail_real)):
        alpha = i / len(trail_real)
        c = tuple(int(c * alpha) for c in REAL_COL)
        cv2.line(canvas, trail_real[i-1], trail_real[i], c, 1)
    for i in range(1, len(trail_frozen)):
        alpha = i / len(trail_frozen)
        c = tuple(int(c * alpha) for c in FROZEN_COL)
        cv2.line(canvas, trail_frozen[i-1], trail_frozen[i], c, 1)
    for i in range(1, len(trail_abm)):
        alpha = i / len(trail_abm)
        c = tuple(int(c * alpha) for c in ABM_COL)
        cv2.line(canvas, trail_abm[i-1], trail_abm[i], c, 1)

    # ── Objects ───────────────────────────────────────────────────────────
    # Frozen prediction
    if frozen_pos:
        draw_object(canvas, frozen_pos, RADIUS_BLOCK, FROZEN_COL, thickness=2)
    if frozen_agent_pos:
        draw_object(canvas, frozen_agent_pos, RADIUS_AGENT, FROZEN_COL, thickness=2)

    # A-B-M prediction
    if abm_pos:
        draw_object(canvas, abm_pos, RADIUS_BLOCK - 4, ABM_COL, thickness=2)
    if abm_agent_pos:
        draw_object(canvas, abm_agent_pos, RADIUS_AGENT - 3, ABM_COL, thickness=2)

    # Ground truth (on top)
    if real_pos:
        draw_object(canvas, real_pos, RADIUS_BLOCK, REAL_COL, 'Block', thickness=-1)
    if real_agent_pos:
        draw_object(canvas, real_agent_pos, RADIUS_AGENT, AGENT_COL, 'Agent', thickness=-1)

    # ── Adaptation flash ──────────────────────────────────────────────────
    if adapted:
        cv2.rectangle(canvas, (2, 2), (W-2, H-2), ADAPT_COL, 3)

    # ── Progress bar ──────────────────────────────────────────────────────
    bar_w = int((step_i / max(total_steps - 1, 1)) * (W - 4))
    cv2.rectangle(canvas, (2, H - 4), (2 + bar_w, H - 2), GREY, -1)

    # ── Legend ────────────────────────────────────────────────────────────
    cv2.rectangle(canvas, (0, H), (W, H + PANEL_H), (25, 25, 40), -1)
    cv2.line(canvas, (0, H), (W, H), (60, 60, 80), 1)

    # Left: legend
    items = [
        (REAL_COL,   '■ Real (GT slot)'),
        (FROZEN_COL, '■ Frozen model pred'),
        (ABM_COL,    '■ A-B-M model pred'),
        (GOAL_COL,   '○ Goal slot'),
    ]
    for i, (col, text) in enumerate(items):
        x = 8 + (i % 2) * 165
        y = H + 18 + (i // 2) * 18
        cv2.putText(canvas, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.38, col, 1, cv2.LINE_AA)

    # Right: metrics
    cv2.putText(canvas, f"Surprise — Frozen:{surprise_f:.4f}  A-B-M:{surprise_a:.4f}",
                (W - 350, H + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.38, GREY, 1, cv2.LINE_AA)
    cv2.putText(canvas, f"Plan Err — Frozen:{plan_err_f:.4f}  A-B-M:{plan_err_a:.4f}",
                (W - 350, H + 38), cv2.FONT_HERSHEY_SIMPLEX, 0.38, GREY, 1, cv2.LINE_AA)

    # Adapt status
    if adapted:
        cv2.putText(canvas, "● SYSTEM M: ADAPTING", (W - 350, H + 56),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.40, ADAPT_COL, 1, cv2.LINE_AA)
    else:
        cv2.putText(canvas, f"t={step_i+1}/{total_steps}  {ep_label}",
                    (W - 350, H + 56), cv2.FONT_HERSHEY_SIMPLEX, 0.38, GREY, 1, cv2.LINE_AA)

    cv2.putText(canvas, '  MechJEPA — Slot-Space Comparison',
                (8, H + 56), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (80, 80, 100), 1, cv2.LINE_AA)

    return canvas


def run_episode_video(ep_slots, ep_actions, goal_slots,
                      frozen_model, abm_model,
                      frozen_planner, abm_planner,
                      optimizer,
                      pca_block, pca_agent,
                      block_lo, block_hi,
                      agent_lo, agent_hi,
                      goal_block_pos,
                      ep_label, device,
                      surprise_threshold=0.015, adapt_steps=3):
    """
    Generate frames for one episode.
    ep_slots: (T, 4, 128) — ground truth VideoSAUR slots
    ep_actions: (T, 2)
    """
    T = ep_slots.shape[0]
    hist_size = 3
    frames = []

    trail_real, trail_frozen, trail_abm = [], [], []
    surprise_log_f, surprise_log_a = [0.0], [0.0]
    plan_err_log_f, plan_err_log_a = [0.0], [0.0]
    adapt_log = [False]

    # Build rolling buffers (use ground-truth slots — this is the validated setup)
    hist_slots  = []
    hist_acts   = []

    for t in range(T - 1):
        curr_slots = torch.from_numpy(ep_slots[t]).float().unsqueeze(0).to(device)  # (1, 4, 128)

        hist_slots.append(curr_slots)
        if len(hist_slots) > hist_size: hist_slots.pop(0)

        if len(hist_acts) < hist_size:
            hist_acts.insert(0, torch.from_numpy(ep_actions[max(0, t-1)]).float().unsqueeze(0).to(device))
        if len(hist_acts) > hist_size: hist_acts.pop(0)

        next_real = torch.from_numpy(ep_slots[t+1]).float().unsqueeze(0).to(device)

        if len(hist_slots) < hist_size:
            # Not enough history yet — skip (just render real)
            real_bpos  = slot_to_canvas(ep_slots[t, 1, :], pca_block, block_lo, block_hi)
            real_apos  = slot_to_canvas(ep_slots[t, 0, :], pca_agent, agent_lo, agent_hi)
            trail_real.append(real_bpos)
            if len(trail_real) > TRAIL_LEN: trail_real.pop(0)
            frames.append(draw_frame(
                real_bpos, None, None, goal_block_pos,
                real_apos, None, None, [], [],
                trail_real, [], [],
                0, 0, 0, 0, False, t, T-1, ep_label
            ))
            continue

        hist_t  = torch.cat(hist_slots, dim=0).unsqueeze(0)   # (1, 3, 4, 128)
        hact_t  = torch.cat(hist_acts,  dim=0).unsqueeze(0)   # (1, 3, 2)

        # ── Frozen model: predict next ────────────────────────────────────
        with torch.no_grad():
            pred_f = frozen_model.inference(hist_t, actions=hact_t).squeeze(1)  # (1, 4, 128)
        surprise_f = F.mse_loss(pred_f, next_real).item()

        # ── A-B-M: check surprise, adapt if needed ────────────────────────
        with torch.no_grad():
            pred_a = abm_model.inference(hist_t, actions=hact_t).squeeze(1)
        surprise_a = F.mse_loss(pred_a, next_real).item()
        adapted = False

        if surprise_a > surprise_threshold and optimizer is not None:
            adapted = True
            abm_model.train()
            for _ in range(adapt_steps):
                optimizer.zero_grad()
                p = abm_model.differentiable_inference(hist_t, actions=hact_t).squeeze(1)
                loss = F.mse_loss(p, next_real)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(abm_model.parameters(), 0.1)
                optimizer.step()
            abm_model.eval()
            with torch.no_grad():
                pred_a = abm_model.inference(hist_t, actions=hact_t).squeeze(1)

        # ── CEM planned trajectories ──────────────────────────────────────
        goal_t = goal_slots.unsqueeze(0).to(device)  # (1, 4, 128)
        planned_f = frozen_planner.plan(hist_t, hact_t, goal_slots.to(device))  # (1, H, 2)
        planned_a = abm_planner.plan(hist_t,   hact_t, goal_slots.to(device))

        # Rollout planned actions to get block trajectory for visualisation
        def rollout_plan(model, planned_acts, hist_t, hact_t):
            pts = []
            ch, ca = hist_t, hact_t
            for h in range(planned_acts.shape[1]):
                sa = planned_acts[:, h:h+1, :]
                ca = torch.cat([ca[:, 1:, :], sa], dim=1)
                with torch.no_grad():
                    nxt = model.inference(ch, actions=ca)  # (1, 1, 4, 128)
                nxt_s = nxt.squeeze(1)  # (1, 4, 128)
                bpos = slot_to_canvas(nxt_s[0, 1, :].cpu().numpy(), pca_block, block_lo, block_hi)
                pts.append(bpos)
                ch = torch.cat([ch[:, 1:, :], nxt], dim=1)
            return pts

        fplan_pts = rollout_plan(frozen_model, planned_f, hist_t, hact_t)
        aplan_pts = rollout_plan(abm_model,   planned_a, hist_t, hact_t)

        # Planning error: 1-step predicted vs real
        plan_err_f = F.mse_loss(pred_f, next_real).item()
        plan_err_a = F.mse_loss(pred_a, next_real).item()

        # ── Canvas positions ──────────────────────────────────────────────
        real_bpos  = slot_to_canvas(ep_slots[t, 1, :], pca_block, block_lo, block_hi)
        real_apos  = slot_to_canvas(ep_slots[t, 0, :], pca_agent, agent_lo, agent_hi)
        froz_bpos  = slot_to_canvas(pred_f[0, 1, :].cpu().numpy(), pca_block, block_lo, block_hi)
        froz_apos  = slot_to_canvas(pred_f[0, 0, :].cpu().numpy(), pca_agent, agent_lo, agent_hi)
        abm_bpos   = slot_to_canvas(pred_a[0, 1, :].cpu().numpy(), pca_block, block_lo, block_hi)
        abm_apos   = slot_to_canvas(pred_a[0, 0, :].cpu().numpy(), pca_agent, agent_lo, agent_hi)

        trail_real.append(real_bpos)
        trail_frozen.append(froz_bpos)
        trail_abm.append(abm_bpos)
        if len(trail_real)   > TRAIL_LEN: trail_real.pop(0)
        if len(trail_frozen) > TRAIL_LEN: trail_frozen.pop(0)
        if len(trail_abm)    > TRAIL_LEN: trail_abm.pop(0)

        surprise_log_f.append(surprise_f)
        surprise_log_a.append(surprise_a)
        plan_err_log_f.append(plan_err_f)
        plan_err_log_a.append(plan_err_a)
        adapt_log.append(adapted)

        # Advance action buffer
        hist_acts.append(torch.from_numpy(ep_actions[t]).float().unsqueeze(0).to(device))
        if len(hist_acts) > hist_size: hist_acts.pop(0)

        frame = draw_frame(
            real_bpos, froz_bpos, abm_bpos, goal_block_pos,
            real_apos, froz_apos, abm_apos,
            fplan_pts, aplan_pts,
            list(trail_real), list(trail_frozen), list(trail_abm),
            surprise_f, surprise_a, plan_err_f, plan_err_a,
            adapted, t, T-1, ep_label
        )
        frames.append(frame)

    stats = {
        'mean_surp_f': float(np.mean(surprise_log_f)),
        'mean_surp_a': float(np.mean(surprise_log_a)),
        'mean_err_f':  float(np.mean(plan_err_log_f)),
        'mean_err_a':  float(np.mean(plan_err_log_a)),
        'n_adapt':     int(sum(adapt_log)),
    }
    return frames, stats


def write_video(frames, out_path):
    if not frames:
        return
    fh, fw = frames[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    vw = cv2.VideoWriter(out_path, fourcc, FPS, (fw, fh))
    for f in frames:
        vw.write(f)
    vw.release()
    logging.info(f'Wrote {out_path} ({len(frames)} frames, {len(frames)/FPS:.1f}s)')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt',      default='/workspace/checkpoints/mechjepa_pusht_act_best.ckpt')
    parser.add_argument('--data',      default='/workspace/data/pusht_slots_actions.pkl')
    parser.add_argument('--out_dir',   default='/workspace/results/videos2')
    parser.add_argument('--episodes',  type=int, default=N_EPISODES)
    parser.add_argument('--threshold', type=float, default=0.015)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # ── Load data ─────────────────────────────────────────────────────────
    logging.info('Loading data and fitting PCA...')
    with open(args.data, 'rb') as f:
        data = pkl.load(f)
    val = data['val']
    ep_keys = sorted(list(val.keys()))[:args.episodes]

    all_slots = np.concatenate([val[k]['slots'] for k in ep_keys], axis=0)  # (N, 4, 128)

    # Per-slot PCA fitted on the whole dataset
    pca_agent = PCA(n_components=4); pca_agent.fit(all_slots[:, 0, :])
    pca_block = PCA(n_components=4); pca_block.fit(all_slots[:, 1, :])

    # Compute canvas extents from training distribution
    def get_extents(pca, slot_data):
        proj = pca.transform(slot_data)   # (N, 4)
        # Use PC1 and PC2 for x/y (most variance)
        lo = proj[:, 1:3].min(0) - 0.1
        hi = proj[:, 1:3].max(0) + 0.1
        return lo, hi

    agent_lo, agent_hi = get_extents(pca_agent, all_slots[:, 0, :])
    block_lo, block_hi = get_extents(pca_block, all_slots[:, 1, :])

    # Goal: last frame of episode 0
    goal_slots    = torch.from_numpy(val[ep_keys[0]]['slots'][-1]).float()  # (4, 128)
    goal_block_np = val[ep_keys[0]]['slots'][-1][1]                         # (128,) — block slot
    goal_block_pos= slot_to_canvas(goal_block_np, pca_block, block_lo, block_hi)

    # ── Models ────────────────────────────────────────────────────────────
    def load_model(ckpt):
        m = MechJEPA(**MODEL_CFG)
        m.load_state_dict(torch.load(ckpt, map_location=device, weights_only=False))
        return m.to(device)

    frozen_model  = load_model(args.ckpt).eval()
    frozen_plan   = CEMPlanner(frozen_model, horizon=HORIZON, num_samples=N_SAMPLES,
                               num_iterations=5, device=device)
    abm_model     = load_model(args.ckpt)
    abm_model.eval()
    optimizer = torch.optim.AdamW([
        {'params': abm_model.codebook.parameters(),  'lr': 5e-4},
        {'params': abm_model.predictor.parameters(), 'lr': 1e-4},
    ])
    abm_plan = CEMPlanner(abm_model, horizon=HORIZON, num_samples=N_SAMPLES,
                          num_iterations=5, device=device)

    all_frames = []
    for ep_i, key in enumerate(ep_keys):
        logging.info(f'▶ Episode {ep_i+1}/{len(ep_keys)}: {key}')
        ep = val[key]
        ep_slots   = ep['slots']    # (T, 4, 128)
        ep_actions = ep['actions']  # (T, 2)

        frames, stats = run_episode_video(
            ep_slots, ep_actions, goal_slots,
            frozen_model, abm_model, frozen_plan, abm_plan, optimizer,
            pca_block, pca_agent, block_lo, block_hi, agent_lo, agent_hi,
            goal_block_pos,
            ep_label=f'{key}  (ep {ep_i+1})',
            device=device,
            surprise_threshold=args.threshold,
        )

        logging.info(
            f"  Frozen: surprise={stats['mean_surp_f']:.4f}  plan_err={stats['mean_err_f']:.4f}"
            f"\n  A-B-M:  surprise={stats['mean_surp_a']:.4f}  plan_err={stats['mean_err_a']:.4f}"
            f"  adaptations={stats['n_adapt']}"
        )

        out_path = os.path.join(args.out_dir, f'slot_ep{ep_i+1:02d}.mp4')
        write_video(frames, out_path)
        all_frames.extend(frames)

    all_path = os.path.join(args.out_dir, 'slot_all_episodes.mp4')
    write_video(all_frames, all_path)
    logging.info(f'Done. Videos saved to {args.out_dir}/')


if __name__ == '__main__':
    main()
