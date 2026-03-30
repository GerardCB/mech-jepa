"""
render_env_gif.py — Physically correct Push-T GIFs from real environment state.

Strategy:
  1. Replay stored EXPERT actions in the env → get real agent (x,y) + block (x,y,angle)
     This is the "expert / real" track.
  2. Run CEM planning using stored ground-truth VideoSAUR slots → get planned actions
     Execute THOSE actions in a SECOND copy of the env (same seed)
     This gives physically correct T-piece motion for both Frozen and A-B-M models.
  3. Render from env.states['state'] = [agent_x, agent_y, block_x, block_y, block_angle, ...]
     All objects stay in physical contact because we're using real physics.

Three-panel GIF:
  Left:   Expert trajectory (green agent + blue T-piece)
  Middle: Frozen CEM trajectory (red agent + T-piece)
  Right:  A-B-M CEM trajectory (teal agent + T-piece)

Single-panel GIF (for README):
  One canvas showing all three T-piece trajectories overlaid (agent shown for each).

Outputs to:
  /workspace/results/gifs2/pusht_env_compare_ep{N}.gif
  /workspace/results/gifs2/pusht_env_compare_all.gif
  /workspace/results/gifs2/pusht_env_overlay_ep{N}.gif
"""

import os, sys, math, argparse
import os; os.environ['SDL_VIDEODRIVER'] = 'offscreen'
import numpy as np
import pickle as pkl
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont
from copy import deepcopy
from loguru import logger as logging

sys.path.insert(0, '/workspace/mechjepa')
from mechjepa.model import MechJEPA
from mechjepa.planner import CEMPlanner

import stable_worldmodel as swm
from stable_worldmodel.policy import Policy

# ── Canvas constants ─────────────────────────────────────────────────────────
PANEL_W     = 320        # each panel width
PANEL_H     = 320        # each panel height
INFO_H      = 56
ENV_SIZE    = 512.0      # env coordinate range
BLOCK_SCALE = 24         # T-piece polygon scale (px, at 320px canvas)
AGENT_R     = 10
FPS_DELAY   = 80         # ms per frame (~12.5 fps)
TRAIL_LEN   = 25
MAX_STEPS   = 100

# ── Palette ───────────────────────────────────────────────────────────────────
BG          = (248, 246, 242)
GRID_COL    = (228, 226, 222)
INFO_BG     = ( 38,  38,  50)

# Expert (green)
EXPERT_BLOCK = ( 60, 130, 200)   # blue block
EXPERT_AGENT = ( 20, 170,  80)   # green agent

# Frozen model (red)
FROZEN_BLOCK = (210,  65,  50)
FROZEN_AGENT = (210,  65,  50)

# A-B-M model (teal)
ABM_BLOCK    = ( 25, 165, 135)
ABM_AGENT    = ( 25, 165, 135)

GOAL_COL     = (180, 160, 220)   # lavender goal
ADAPT_FLASH  = (255, 160,   0)
TEXT_COL     = (240, 240, 240)
DARK_TEXT    = ( 40,  40,  50)

MODEL_CFG = dict(
    num_slots=4, slot_dim=128, num_mechanisms=8,
    history_frames=3, pred_frames=1, action_dim=2,
    transformer_depth=6, transformer_heads=16,
    transformer_dim_head=64, transformer_mlp_dim=2048,
    edge_hidden_dim=256,
)


# ── Geometry helpers ──────────────────────────────────────────────────────────
def env_to_canvas(x, y, size=PANEL_W):
    """Convert env coord (0-512) to canvas pixel."""
    px = int(np.clip(x / ENV_SIZE, 0, 1) * (size - 20) + 10)
    py = int(np.clip(1 - y / ENV_SIZE, 0, 1) * (size - 20) + 10)
    return px, py


def t_piece_pts(cx, cy, angle, scale=BLOCK_SCALE):
    """T-piece polygon around (cx,cy) rotated by angle (rad)."""
    s = scale
    local = np.array([
        [-2*s, -s], [2*s, -s], [2*s, 0],
        [ s,    0], [s,   2*s], [-s, 2*s],
        [-s,    0], [-2*s, 0],
    ], dtype=float)
    cos_a, sin_a = math.cos(angle), math.sin(angle)
    R = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
    pts = (R @ local.T).T
    pts[:, 0] += cx
    pts[:, 1] += cy
    return [tuple(p.astype(int)) for p in pts]


def draw_scene(state, block_col, agent_col, trail, size=PANEL_W,
               adapted=False, goal_state=None, label='', surp=None):
    """Render a single panel from env state [ax, ay, bx, by, bang, ...]."""
    img = Image.new('RGBA', (size, size + INFO_H), BG + (255,))
    draw = ImageDraw.Draw(img, 'RGBA')

    # Grid
    for i in range(0, size, 40):
        draw.line([(i, 0), (i, size)], fill=GRID_COL + (255,), width=1)
        draw.line([(0, i), (size, i)], fill=GRID_COL + (255,), width=1)

    # Goal ghost
    if goal_state is not None:
        gx, gy = env_to_canvas(goal_state[2], goal_state[3], size)
        bang = goal_state[4]
        gpts = t_piece_pts(gx, gy, bang, scale=BLOCK_SCALE - 5)
        draw.polygon(gpts, outline=GOAL_COL + (180,), width=2)

    # Trail
    for i in range(1, len(trail)):
        alpha = int(40 + 180 * i / len(trail))
        draw.line([trail[i-1], trail[i]], fill=block_col + (alpha,), width=2)

    ax, ay = state[0], state[1]
    bx, by, bang = state[2], state[3], state[4]
    apx, apy = env_to_canvas(ax, ay, size)
    bpx, bpy = env_to_canvas(bx, by, size)

    # Block (T-piece)
    bpts = t_piece_pts(bpx, bpy, bang)
    draw.polygon(bpts, fill=block_col + (200,), outline=block_col + (255,), width=1)

    # Agent (circle)
    r = AGENT_R
    draw.ellipse([apx - r, apy - r, apx + r, apy + r],
                 fill=agent_col + (230,), outline=(20, 20, 20, 200), width=1)

    # Adapt flash
    if adapted:
        for w in range(4):
            draw.rectangle([w, w, size-1-w, size-1-w],
                           outline=ADAPT_FLASH + (200 - w*40,), width=1)

    # Info bar
    draw.rectangle([0, size, size, size + INFO_H], fill=INFO_BG + (255,))
    draw.line([(0, size), (size, size)], fill=(70, 70, 90, 255), width=1)

    # Color dot + label
    draw.ellipse([8, size + 10, 20, size + 22], fill=block_col + (255,))
    draw.text((24, size + 8), label, fill=block_col + (255,))

    if surp is not None:
        draw.text((8, size + 30), f'Prediction error: {surp:.4f}', fill=TEXT_COL + (180,))
    if adapted:
        draw.text((size - 120, size + 30), '⚡ ADAPTING', fill=ADAPT_FLASH + (255,))

    return img


def make_tripanel(expert_img, frozen_img, abm_img, size=PANEL_W):
    gap = 3
    W = size * 3 + gap * 2
    H = size + INFO_H
    out = Image.new('RGBA', (W, H), (200, 200, 200, 255))
    out.paste(expert_img, (0, 0))
    out.paste(frozen_img, (size + gap, 0))
    out.paste(abm_img,    (size * 2 + gap * 2, 0))
    return out


def make_overlay(states_exp, states_froz, states_abm, size=PANEL_W,
                 trails_e=None, trails_f=None, trails_a=None,
                 adapted=False, surp_f=0, surp_a=0, goal_state=None, step_i=0, total=1):
    """Single canvas with all 3 T-pieces overlaid + labels."""
    img = Image.new('RGBA', (size, size + INFO_H + 20), BG + (255,))
    draw = ImageDraw.Draw(img, 'RGBA')

    for i in range(0, size, 40):
        draw.line([(i, 0), (i, size)], fill=GRID_COL + (255,), width=1)
        draw.line([(0, i), (size, i)], fill=GRID_COL + (255,), width=1)

    if goal_state is not None:
        gx, gy = env_to_canvas(goal_state[2], goal_state[3], size)
        gpts = t_piece_pts(gx, gy, goal_state[4], scale=BLOCK_SCALE - 5)
        draw.polygon(gpts, outline=GOAL_COL + (160,), width=2)

    for trail, col in [(trails_e or [], EXPERT_BLOCK),
                       (trails_f or [], FROZEN_BLOCK),
                       (trails_a or [], ABM_BLOCK)]:
        for i in range(1, len(trail)):
            alpha = int(40 + 180 * i / len(trail))
            draw.line([trail[i-1], trail[i]], fill=col + (alpha,), width=2)

    def draw_state(state, bcol, acol, alpha_factor=1.0, outline_only=False):
        ax, ay = env_to_canvas(state[0], state[1], size)
        bx, by = env_to_canvas(state[2], state[3], size)
        bang   = state[4]
        bpts   = t_piece_pts(bx, by, bang)
        al     = int(200 * alpha_factor)
        if outline_only:
            draw.polygon(bpts, outline=bcol + (al,), width=2)
        else:
            draw.polygon(bpts, fill=bcol + (al,), outline=bcol + (255,), width=1)
        r = AGENT_R
        if outline_only:
            draw.ellipse([ax-r, ay-r, ax+r, ay+r], outline=acol + (al,), width=2)
        else:
            draw.ellipse([ax-r, ay-r, ax+r, ay+r], fill=acol + (al,), outline=(20,20,20,180), width=1)

    draw_state(states_froz, FROZEN_BLOCK, FROZEN_AGENT, outline_only=True)
    draw_state(states_abm,  ABM_BLOCK,    ABM_AGENT,    outline_only=True)
    draw_state(states_exp,  EXPERT_BLOCK, EXPERT_AGENT)   # expert on top, filled

    if adapted:
        for w in range(4):
            draw.rectangle([w, w, size-1-w, size-1-w], outline=ADAPT_FLASH + (200-w*40,), width=1)

    # Progress bar
    bar_w = int((step_i / max(total-1, 1)) * (size - 4))
    draw.rectangle([2, size-3, 2+bar_w, size-1], fill=(100, 100, 120, 200))

    # Info bar
    H_tot = size + INFO_H + 20
    draw.rectangle([0, size, size, H_tot], fill=INFO_BG + (255,))

    items = [(EXPERT_BLOCK, '● Expert'), (FROZEN_BLOCK, '● Frozen'), (ABM_BLOCK, '●  A-B-M')]
    for i, (col, lbl) in enumerate(items):
        draw.ellipse([8 + i*100, size+8, 18+i*100, size+18], fill=col+(255,))
        draw.text((22+i*100, size+6), lbl, fill=col+(255,))

    impr = surp_f / (surp_a + 1e-8)
    draw.text((8, size+28),
              f'Pred err — Frozen: {surp_f:.4f}  A-B-M: {surp_a:.4f}  ({impr:.1f}× better)',
              fill=TEXT_COL + (180,))
    if adapted:
        draw.text((size-130, size+28), '⚡ SYSTEM M ADAPT', fill=ADAPT_FLASH+(255,))

    return img


# ── Slot-to-action via CEM ────────────────────────────────────────────────────
class SlotCEMPolicy:
    """
    Given stored VideoSAUR slot sequences, run CEM at each step.
    Maintains a rolling slot history buffer.
    """
    def __init__(self, model, planner, goal_slots, ep_slots, ep_actions, device,
                 system_m=False, optimizer=None, threshold=0.015, adapt_steps=3):
        self.model       = model
        self.planner     = planner
        self.goal_slots  = goal_slots.to(device)
        self.ep_slots    = ep_slots    # (T, 4, 128)
        self.ep_actions  = ep_actions  # (T, 2)
        self.device      = device
        self.system_m    = system_m
        self.optimizer   = optimizer
        self.threshold   = threshold
        self.adapt_steps = adapt_steps

        self.hist_slots = []
        self.hist_acts  = []
        self.t          = 0
        self.surp_log   = []
        self.adapt_log  = []

    def reset(self):
        self.hist_slots = []
        self.hist_acts  = []
        self.t = 0
        self.surp_log   = []
        self.adapt_log  = []

    def step(self):
        """Advance one timestep. Returns (action np (2,), surprise, adapted)."""
        t = self.t
        T = self.ep_slots.shape[0]
        if t >= T - 1:
            return np.zeros(2, dtype=np.float32), 0.0, False

        curr = torch.from_numpy(self.ep_slots[t]).float().unsqueeze(0).to(self.device)
        self.hist_slots.append(curr)
        if len(self.hist_slots) > 3: self.hist_slots.pop(0)

        if len(self.hist_acts) < 3:
            self.hist_acts.insert(0, torch.from_numpy(
                self.ep_actions[max(0, t-1)]).float().unsqueeze(0).to(self.device))
        if len(self.hist_acts) > 3: self.hist_acts.pop(0)

        self.t += 1

        if len(self.hist_slots) < 3:
            return np.zeros(2, dtype=np.float32), 0.0, False

        next_real = torch.from_numpy(self.ep_slots[min(t+1, T-1)]).float().unsqueeze(0).to(self.device)
        hist_t  = torch.cat(self.hist_slots, dim=0).unsqueeze(0)
        hact_t  = torch.cat(self.hist_acts,  dim=0).unsqueeze(0)

        with torch.no_grad():
            pred = self.model.inference(hist_t, actions=hact_t).squeeze(1)
        surp = F.mse_loss(pred, next_real).item()
        adapted = False

        if self.system_m and self.optimizer and surp > self.threshold:
            adapted = True
            self.model.train()
            for _ in range(self.adapt_steps):
                self.optimizer.zero_grad()
                p = self.model.differentiable_inference(hist_t, actions=hact_t).squeeze(1)
                loss = F.mse_loss(p, next_real)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.1)
                self.optimizer.step()
            self.model.eval()

        planned = self.planner.plan(hist_t, hact_t, self.goal_slots)  # (1, H, 2)
        action = planned[0, 0, :].detach().cpu().numpy()  # (2,)

        # Advance action buffer
        self.hist_acts.append(torch.from_numpy(action).float().unsqueeze(0).to(self.device))
        if len(self.hist_acts) > 3: self.hist_acts.pop(0)

        self.surp_log.append(surp)
        self.adapt_log.append(adapted)
        return action, surp, adapted


class ExpertPolicy(Policy):
    def __init__(self, actions):
        self.actions = actions  # (T, 2)
        self.t = 0
    def reset(self):
        self.t = 0
    def get_action(self, infos):
        a = self.actions[self.t] if self.t < len(self.actions) else np.zeros(2)
        self.t += 1
        return a.reshape(1, 2).astype(np.float32)


class CEMEnvPolicy(Policy):
    def __init__(self, slot_cem, actions_fallback):
        self.slot_cem = slot_cem
        self.actions_fallback = actions_fallback
        self._next_action = np.zeros(2, dtype=np.float32)
        self._surp = 0.0
        self._adapted = False

    def get_action(self, infos):
        return self._next_action.reshape(1, 2)

    def compute_next(self):
        a, s, ad = self.slot_cem.step()
        self._next_action = a
        self._surp = s
        self._adapted = ad
        return a, s, ad


def run_episode_gifs(ep_slots, ep_actions, goal_slots,
                     frozen_model, abm_model, frozen_plan, abm_plan, optimizer,
                     goal_state, device, threshold=0.015):
    T = min(len(ep_slots), MAX_STEPS + 3)

    # Policy objects
    expert_pol = ExpertPolicy(ep_actions)

    frozen_slot = SlotCEMPolicy(frozen_model, frozen_plan, goal_slots, ep_slots, ep_actions, device)
    abm_slot    = SlotCEMPolicy(abm_model,   abm_plan,   goal_slots, ep_slots, ep_actions, device,
                                system_m=True, optimizer=optimizer, threshold=threshold)

    frozen_pol = CEMEnvPolicy(frozen_slot, ep_actions)
    abm_pol    = CEMEnvPolicy(abm_slot,    ep_actions)

    # Three env instances (same seed → same starting state)
    seed = 42
    def make_env():
        e = swm.World('swm/PushT-v1', num_envs=1, image_shape=(96,96),
                      max_episode_steps=MAX_STEPS + 10, verbose=0)
        return e

    env_exp   = make_env(); env_exp.set_policy(expert_pol);  env_exp.reset(seed=seed)
    env_froz  = make_env(); env_froz.set_policy(frozen_pol); env_froz.reset(seed=seed)
    env_abm   = make_env(); env_abm.set_policy(abm_pol);     env_abm.reset(seed=seed)

    tri_frames, ov_frames = [], []
    tr_e, tr_f, tr_a = [], [], []

    for step_i in range(MAX_STEPS):
        # Get current states
        s_exp  = env_exp.states['state'][0]   if env_exp.states  else np.zeros(7)
        s_froz = env_froz.states['state'][0]  if env_froz.states else np.zeros(7)
        s_abm  = env_abm.states['state'][0]   if env_abm.states  else np.zeros(7)

        # Block centre for trails
        bp_e = env_to_canvas(s_exp[2],  s_exp[3],  PANEL_W)
        bp_f = env_to_canvas(s_froz[2], s_froz[3], PANEL_W)
        bp_a = env_to_canvas(s_abm[2],  s_abm[3],  PANEL_W)
        tr_e.append(bp_e); tr_f.append(bp_f); tr_a.append(bp_a)
        if len(tr_e) > TRAIL_LEN: tr_e.pop(0)
        if len(tr_f) > TRAIL_LEN: tr_f.pop(0)
        if len(tr_a) > TRAIL_LEN: tr_a.pop(0)

        # Compute CEM actions (sets internal state for next step call)
        _, surp_f, adapted_f = frozen_pol.compute_next()
        _, surp_a, adapted_a = abm_pol.compute_next()

        # Draw panels
        p_exp  = draw_scene(s_exp,  EXPERT_BLOCK, EXPERT_AGENT, list(tr_e),
                            PANEL_W, goal_state=goal_state, label='Expert Actions (GT)')
        p_froz = draw_scene(s_froz, FROZEN_BLOCK, FROZEN_AGENT, list(tr_f),
                            PANEL_W, adapted=False, goal_state=goal_state,
                            label='Frozen Model', surp=surp_f)
        p_abm  = draw_scene(s_abm,  ABM_BLOCK, ABM_AGENT, list(tr_a),
                            PANEL_W, adapted=adapted_a, goal_state=goal_state,
                            label='A-B-M Agent', surp=surp_a)

        tri_frames.append(make_tripanel(p_exp, p_froz, p_abm))
        ov_frames.append(make_overlay(
            s_exp, s_froz, s_abm, PANEL_W,
            list(tr_e), list(tr_f), list(tr_a),
            adapted_a, surp_f, surp_a, goal_state,
            step_i, MAX_STEPS,
        ))

        # Step all envs
        env_exp.step()
        env_froz.step()
        env_abm.step()

        done_e = env_exp.terminateds is not None and env_exp.terminateds[0]
        done_f = env_froz.terminateds is not None and env_froz.terminateds[0]
        done_a = env_abm.terminateds  is not None and env_abm.terminateds[0]
        if done_e and done_f and done_a:
            break

    env_exp.close(); env_froz.close(); env_abm.close()
    return tri_frames, ov_frames


def save_gif(frames, path):
    rgb = [f.convert('RGB') for f in frames]
    rgb[0].save(path, save_all=True, append_images=rgb[1:],
                duration=FPS_DELAY, loop=0, optimize=False)
    logging.info(f'Saved {path} ({len(frames)} frames, {len(frames)*FPS_DELAY/1000:.1f}s)')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt',      default='/workspace/checkpoints/mechjepa_pusht_act_best.ckpt')
    parser.add_argument('--data',      default='/workspace/data/pusht_slots_actions.pkl')
    parser.add_argument('--out_dir',   default='/workspace/results/gifs2')
    parser.add_argument('--episodes',  type=int, default=4)
    parser.add_argument('--threshold', type=float, default=0.015)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    logging.info('Loading data...')
    with open(args.data, 'rb') as f:
        data = pkl.load(f)
    val = data['val']
    ep_keys = sorted(list(val.keys()))[:args.episodes]

    goal_slots = torch.from_numpy(val[ep_keys[0]]['slots'][-1]).float()
    # Goal state: use env center as approximate goal visual
    goal_state = np.array([256.0, 256.0, 256.0, 180.0, 0.3, 0.0, 0.0])

    def load_model(ckpt):
        m = MechJEPA(**MODEL_CFG)
        m.load_state_dict(torch.load(ckpt, map_location=device, weights_only=False))
        return m.to(device)

    frozen_model = load_model(args.ckpt).eval()
    frozen_plan  = CEMPlanner(frozen_model, horizon=10, num_samples=256, num_iterations=5, device=device)
    abm_model    = load_model(args.ckpt)
    abm_model.eval()
    optimizer = torch.optim.AdamW([
        {'params': abm_model.codebook.parameters(),  'lr': 5e-4},
        {'params': abm_model.predictor.parameters(), 'lr': 1e-4},
    ])
    abm_plan = CEMPlanner(abm_model, horizon=10, num_samples=256, num_iterations=5, device=device)

    all_tri, all_ov = [], []

    for ep_i, key in enumerate(ep_keys):
        logging.info(f'▶ Episode {ep_i+1}/{len(ep_keys)}: {key}')
        ep = val[key]
        tri_frames, ov_frames = run_episode_gifs(
            ep['slots'], ep['actions'], goal_slots,
            frozen_model, abm_model, frozen_plan, abm_plan, optimizer,
            goal_state, device=device, threshold=args.threshold,
        )
        save_gif(tri_frames, os.path.join(args.out_dir, f'pusht_env_compare_ep{ep_i+1:02d}.gif'))
        save_gif(ov_frames,  os.path.join(args.out_dir, f'pusht_env_overlay_ep{ep_i+1:02d}.gif'))
        all_tri.extend(tri_frames)
        all_ov.extend(ov_frames)
        logging.info(f'  Episode {ep_i+1} done ({len(tri_frames)} frames)')

    save_gif(all_tri, os.path.join(args.out_dir, 'pusht_env_compare_all.gif'))
    save_gif(all_ov,  os.path.join(args.out_dir, 'pusht_env_overlay_all.gif'))
    logging.info(f'All done. GIFs saved to {args.out_dir}/')


if __name__ == '__main__':
    main()
