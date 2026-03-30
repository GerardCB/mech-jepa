"""
render_pusht_video.py — Record side-by-side Push-T videos: Frozen model vs A-B-M agent.

Strategy:
  The stable_worldmodel reports exact env state:
      state[0..1] = agent x/y  (pixels, 0-512)
      state[2]    = block_x
      state[3]    = block_y
      state[4]    = block_angle (radians)

  We:
  1. Map state → pseudo-slots that approximate the VideoSAUR training distribution
     using PCA statistics extracted from the stored slot dataset.
  2. Register a custom Policy that runs CEM in latent space to select actions.
  3. Render each frame using cv2 (2D top-down view of the T-piece).
  4. Record both agents for N episodes and stitch into a side-by-side mp4.

Output: /workspace/results/pusht_frozen_vs_abm_{ep}.mp4  (one per episode)
        /workspace/results/pusht_comparison_all.mp4       (all episodes concatenated)
"""

import os, sys, argparse, math
import os; os.environ['SDL_VIDEODRIVER'] = 'offscreen'
import numpy as np
import torch
import torch.nn.functional as F
import pickle as pkl
import cv2
from copy import deepcopy
from loguru import logger as logging

sys.path.insert(0, '/workspace/mechjepa')
from mechjepa.model import MechJEPA
from mechjepa.planner import CEMPlanner

import stable_worldmodel as swm
from stable_worldmodel.policy import Policy  # base class


# ── Resolution ─────────────────────────────────────────────────────────────
W, H       = 512, 512   # main canvas (matches env coordinate space)
PANEL_PAD  = 60         # top bar height
FPS        = 20
MAX_STEPS  = 150
N_EPISODES = 4

MODEL_CFG = dict(
    num_slots=4, slot_dim=128, num_mechanisms=8,
    history_frames=3, pred_frames=1, action_dim=2,
    transformer_depth=6, transformer_heads=16,
    transformer_dim_head=64, transformer_mlp_dim=2048,
    edge_hidden_dim=256,
)

# ── Colours ─────────────────────────────────────────────────────────────────
BG         = (18, 18, 28)
AGENT_COL  = (255, 220, 100)
BLOCK_COL  = (80, 190, 160)
GOAL_COL   = (80,  80, 140)
FROZEN_COL = (230,  80,  80)     # panel header
ABM_COL    = (46,  196, 182)
ADAPT_COL  = (255, 180,   0)
WHITE      = (255, 255, 255)
GREY       = (130, 130, 130)


# ── T-piece geometry (in local frame, scale=30) ──────────────────────────────
def t_piece_polygon(cx, cy, angle, scale=30):
    """
    Returns list of (x, y) int tuples defining the T-piece polygon,
    centred at (cx, cy) rotated by angle (radians).
    Coordinates are in env pixel space (0-512).
    """
    s = scale
    # T-piece: rectangle top + rectangle stem (local coords)
    pts = np.array([
        [-2*s, -s], [ 2*s, -s], [ 2*s,  0], [ s,  0],
        [ s,   2*s],[-s,  2*s], [-s,   0], [-2*s,  0],
    ], dtype=float)
    cos_a, sin_a = math.cos(angle), math.sin(angle)
    R = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
    pts = (R @ pts.T).T
    pts[:, 0] += cx
    pts[:, 1] += cy
    return pts.astype(np.int32)


def draw_frame(state, goal_state, label, adapt_event=False, surprise=0.0, plan_err=0.0):
    """
    Draw one env frame using the given state vector.
    state: (7,) — [agent_x, agent_y, block_x, block_y, block_angle, ...]
    goal_state: same format, but for the goal
    """
    canvas = np.full((H + PANEL_PAD, W, 3), BG, dtype=np.uint8)

    # ── Goal (ghost) ────────────────────────────────────────────────────────
    if goal_state is not None:
        gx, gy, gang = int(goal_state[2]), int(goal_state[3]), goal_state[4]
        gpts = t_piece_polygon(gx, gy, gang)
        cv2.fillPoly(canvas, [gpts], GOAL_COL)
        cv2.polylines(canvas, [gpts], True, GREY, 1)

    # ── T-piece (block) ──────────────────────────────────────────────────────
    bx, by, bang = int(state[2]), int(state[3]), state[4]
    bpts = t_piece_polygon(bx, by, bang)
    cv2.fillPoly(canvas, [bpts], BLOCK_COL)
    cv2.polylines(canvas, [bpts], True, WHITE, 2)

    # ── Agent (circle) ───────────────────────────────────────────────────────
    ax, ay = int(state[0]), int(state[1])
    cv2.circle(canvas, (ax, ay), 14, AGENT_COL, -1)
    cv2.circle(canvas, (ax, ay), 14, WHITE,      2)

    # ── Adaptation flash border ──────────────────────────────────────────────
    if adapt_event:
        cv2.rectangle(canvas, (3, 3), (W - 3, H + PANEL_PAD - 3), ADAPT_COL, 4)

    # ── Header bar ───────────────────────────────────────────────────────────
    col    = FROZEN_COL if 'Frozen' in label else ABM_COL
    header = canvas[H:, :]
    header[:] = (30, 30, 45)
    cv2.putText(canvas, label, (10, H + 38),
                cv2.FONT_HERSHEY_DUPLEX, 0.7, col, 1, cv2.LINE_AA)
    cv2.putText(canvas,
                f"surprise={surprise:.4f}  plan_err={plan_err:.4f}",
                (W - 310, H + 38),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, GREY, 1, cv2.LINE_AA)
    if adapt_event:
        cv2.putText(canvas, "● ADAPTING", (10, H + 56),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, ADAPT_COL, 1, cv2.LINE_AA)

    return canvas


# ── State → pseudo-slots ─────────────────────────────────────────────────────
def state_to_slots(state_np: np.ndarray,
                   slot_mean: np.ndarray,
                   slot_std:  np.ndarray,
                   device: str = 'cuda') -> torch.Tensor:
    """
    Map env state vector (7,) → slot tensor (1, 4, 128) that lives in the
    same distribution as the VideoSAUR training slots.

    We embed object positions into the first few dims of each slot,
    then add Gaussian noise at the training scale (slot_std) to fill
    the remaining dims.  The mean is set to match the training centroid.

    slot 0 → agent
    slot 1 → block
    slot 2,3 → noise (background)
    """
    # Normalize coords to [0,1]
    norm = state_np / 512.0
    S, D = 4, 128

    slots = np.tile(slot_mean.copy(), (S, 1))        # (4, 128) at training mean
    noise = np.random.randn(S, D) * slot_std * 0.5   # small noise

    # Embed agent (slot 0)
    slots[0, 0] = (norm[0] - 0.5) * 4 * slot_std[0]
    slots[0, 1] = (norm[1] - 0.5) * 4 * slot_std[1]

    # Embed block (slot 1): pos + angle
    slots[1, 0] = (norm[2] - 0.5) * 4 * slot_std[0]
    slots[1, 1] = (norm[3] - 0.5) * 4 * slot_std[1]
    slots[1, 2] = math.sin(norm[4] * 2 * math.pi) * slot_std[2]
    slots[1, 3] = math.cos(norm[4] * 2 * math.pi) * slot_std[3]

    # Background slots
    slots[2:] = slots[2:] + noise[2:]

    return torch.from_numpy(slots).float().unsqueeze(0).to(device)   # (1, 4, 128)


# ── Policy wrapper ────────────────────────────────────────────────────────────
class MechJEPAPolicy(Policy):
    """
    Wraps MechJEPA + CEMPlanner as a stable_worldmodel Policy.
    Reads state from env's `states` dict, converts to slots, runs CEM.
    """
    def __init__(self, model, planner, goal_slots_np, slot_mean, slot_std,
                 device='cuda', system_m=False, optimizer=None,
                 surprise_threshold=0.015, adapt_steps=3):
        self.model    = model
        self.planner  = planner
        self.goal_slots = torch.from_numpy(goal_slots_np).float().to(device)
        self.slot_mean  = slot_mean
        self.slot_std   = slot_std
        self.device     = device
        self.system_m   = system_m
        self.optimizer  = optimizer
        self.threshold  = surprise_threshold
        self.adapt_steps = adapt_steps

        self.hist_slots  = []   # list of (1, 4, 128)
        self.hist_acts   = []   # list of (1, 2)
        self.last_action = np.zeros((1, 2), dtype=np.float32)

        # Logging
        self.surprise_log  = []
        self.plan_err_log  = []
        self.adapt_events  = []

    def reset(self):
        self.hist_slots = []
        self.hist_acts  = []
        self.last_action = np.zeros((1, 2), dtype=np.float32)
        self.surprise_log  = []
        self.plan_err_log  = []
        self.adapt_events  = []

    def get_action(self, infos) -> np.ndarray:
        """Called by World.step(). infos not used — we read world.states."""
        return self.last_action

    def compute_step(self, state_np: np.ndarray) -> tuple[np.ndarray, dict]:
        """
        Main per-step logic. Call this before world.step().
        Returns: (action (1,2), metrics dict)
        """
        curr_slots = state_to_slots(state_np, self.slot_mean, self.slot_std, self.device)

        self.hist_slots.append(curr_slots)
        if len(self.hist_slots) > 3: self.hist_slots.pop(0)

        if len(self.hist_acts) < 3:
            pad = torch.zeros(1, 2, device=self.device)
            self.hist_acts.insert(0, pad)
        if len(self.hist_acts) > 3: self.hist_acts.pop(0)

        if len(self.hist_slots) < 3:
            self.last_action = np.zeros((1, 2), dtype=np.float32)
            return self.last_action, {"surprise": 0.0, "plan_err": 0.0, "adapted": False}

        hist_t  = torch.cat(self.hist_slots, dim=0).unsqueeze(0)   # (1, 3, 4, 128)
        hact_t  = torch.cat(self.hist_acts,  dim=0).unsqueeze(0)   # (1, 3, 2)

        # Surprise (vs previous predicted)
        surprise_val = 0.0
        adapted = False
        if len(self.surprise_log) > 0:
            with torch.no_grad():
                pred_next = self.model.inference(hist_t, actions=hact_t).squeeze(1)
            surprise_val = F.mse_loss(pred_next, curr_slots).item()

            if self.system_m and self.optimizer and surprise_val > self.threshold:
                adapted = True
                self.model.train()
                for _ in range(self.adapt_steps):
                    self.optimizer.zero_grad()
                    p = self.model.differentiable_inference(hist_t, actions=hact_t).squeeze(1)
                    loss = F.mse_loss(p, curr_slots)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.1)
                    self.optimizer.step()
                self.model.eval()

        self.surprise_log.append(surprise_val)

        # Plan
        planned = self.planner.plan(hist_t, hact_t, self.goal_slots)   # (1, H, 2)
        action_t = planned[:, 0, :]   # (1, 2)

        # Planning error
        with torch.no_grad():
            step_act  = action_t.unsqueeze(1)
            curr_acts = torch.cat([hact_t[:, 1:, :], step_act], dim=1)
            p2 = self.model.inference(hist_t, actions=curr_acts).squeeze(1)
        plan_err = F.mse_loss(p2, curr_slots).item()
        self.plan_err_log.append(plan_err)
        self.adapt_events.append(adapted)

        # Advance action buffer
        self.hist_acts.append(action_t.detach())
        if len(self.hist_acts) > 3: self.hist_acts.pop(0)

        action_np = action_t.detach().cpu().numpy()   # (1, 2)
        self.last_action = action_np

        return action_np, {"surprise": surprise_val, "plan_err": plan_err, "adapted": adapted}


def load_model(ckpt, device):
    model = MechJEPA(**MODEL_CFG)
    model.load_state_dict(torch.load(ckpt, map_location=device, weights_only=False))
    return model.to(device)


def run_episode(model, planner, goal_slots_np, goal_state_np,
                slot_mean, slot_std,
                system_m=False, optimizer=None,
                label="Frozen", seed=42, device='cuda'):
    """Run one episode, record frames, return frames + logs."""
    env = swm.World('swm/PushT-v1', num_envs=1, image_shape=(96, 96),
                    max_episode_steps=MAX_STEPS, verbose=0)
    policy = MechJEPAPolicy(
        model, planner, goal_slots_np, slot_mean, slot_std, device,
        system_m=system_m, optimizer=optimizer,
    )
    env.set_policy(policy)
    env.reset(seed=seed)

    frames = []
    for step_i in range(MAX_STEPS):
        state = env.states['state'][0]   # (7,)
        action, metrics = policy.compute_step(state)
        env.step()

        frame = draw_frame(
            state, goal_state_np, label=label,
            adapt_event=metrics['adapted'],
            surprise=metrics['surprise'],
            plan_err=metrics['plan_err'],
        )
        frames.append(frame)

        if env.terminateds is not None and env.terminateds[0]:
            break

    env.close()
    return frames, policy.surprise_log, policy.plan_err_log, policy.adapt_events


def write_side_by_side(frames_a, frames_b, out_path):
    """Write a side-by-side mp4 from two frame lists."""
    n = min(len(frames_a), len(frames_b))
    fh, fw = frames_a[0].shape[:2]
    sep = np.full((fh, 4, 3), 60, dtype=np.uint8)   # narrow white separator

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    vw = cv2.VideoWriter(out_path, fourcc, FPS, (fw * 2 + 4, fh))
    for i in range(n):
        frame = np.concatenate([frames_a[i], sep, frames_b[i]], axis=1)
        vw.write(frame)
    vw.release()
    logging.info(f"Wrote {out_path}  ({n} frames, {n/FPS:.1f}s)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt',      default='/workspace/checkpoints/mechjepa_pusht_act_best.ckpt')
    parser.add_argument('--data',      default='/workspace/data/pusht_slots_actions.pkl')
    parser.add_argument('--out_dir',   default='/workspace/results/videos')
    parser.add_argument('--episodes',  type=int, default=N_EPISODES)
    parser.add_argument('--horizon',   type=int, default=10)
    parser.add_argument('--max_steps', type=int, default=MAX_STEPS)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # ── Load dataset to compute slot statistics ────────────────────────────
    logging.info('Loading dataset for slot statistics...')
    with open(args.data, 'rb') as f:
        data = pkl.load(f)
    val = data['val']
    ep_keys = sorted(list(val.keys()))

    all_slots = np.concatenate([val[k]['slots'] for k in ep_keys], axis=0)  # (N, 4, 128)
    all_slots_flat = all_slots.reshape(-1, 128)
    slot_mean = all_slots_flat.mean(0)
    slot_std  = all_slots_flat.std(0).clip(0.01)    # (128,)

    # ── Use last frame of ep0 as goal ─────────────────────────────────────
    goal_slots_np  = val[ep_keys[0]]['slots'][-1]   # (4, 128)
    # We need a goal state too — use nominal center of Push-T board for visual
    goal_state_np  = np.array([256.0, 256.0, 256.0, 200.0, 0.2, 0.0, 0.0])

    # ── Models ────────────────────────────────────────────────────────────
    frozen_model  = load_model(args.ckpt, device).eval()
    frozen_plan   = CEMPlanner(frozen_model, horizon=args.horizon,
                               num_samples=256, num_iterations=5, device=device)

    abm_model = load_model(args.ckpt, device).eval()
    optimizer = torch.optim.AdamW([
        {'params': abm_model.codebook.parameters(),  'lr': 5e-4},
        {'params': abm_model.predictor.parameters(), 'lr': 1e-4},
    ])
    abm_plan = CEMPlanner(abm_model, horizon=args.horizon,
                          num_samples=256, num_iterations=5, device=device)

    all_frozen_frames, all_abm_frames = [], []

    for ep_i in range(args.episodes):
        seed = 1000 + ep_i * 37
        logging.info(f'▶ Episode {ep_i+1}/{args.episodes}  (seed={seed})')

        f_frames, f_surp, f_err, _ = run_episode(
            frozen_model, frozen_plan, goal_slots_np, goal_state_np,
            slot_mean, slot_std,
            system_m=False, label=f'Frozen  | ep{ep_i+1}',
            seed=seed, device=device,
        )
        logging.info(f'  Frozen:  surprise={np.mean(f_surp):.4f}  plan_err={np.mean(f_err):.4f}  steps={len(f_frames)}')

        a_frames, a_surp, a_err, a_adapt = run_episode(
            abm_model, abm_plan, goal_slots_np, goal_state_np,
            slot_mean, slot_std,
            system_m=True, optimizer=optimizer,
            label=f'A-B-M   | ep{ep_i+1}',
            seed=seed, device=device,
        )
        n_adapt = sum(a_adapt)
        logging.info(f'  A-B-M:   surprise={np.mean(a_surp):.4f}  plan_err={np.mean(a_err):.4f}  adapts={n_adapt}')

        # Per-episode video
        ep_path = os.path.join(args.out_dir, f'pusht_ep{ep_i+1:02d}.mp4')
        write_side_by_side(f_frames, a_frames, ep_path)
        all_frozen_frames.extend(f_frames)
        all_abm_frames.extend(a_frames)

    # ── Concatenated full video ─────────────────────────────────────────────
    all_path = os.path.join(args.out_dir, 'pusht_all_episodes.mp4')
    write_side_by_side(all_frozen_frames, all_abm_frames, all_path)
    logging.info(f'All done. Videos saved to {args.out_dir}/')


if __name__ == '__main__':
    main()
