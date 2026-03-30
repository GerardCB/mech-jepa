"""
render_env_gif.py — Physically-correct Push-T GIFs. Fixed version.

Root cause of previous bug:
  Expert actions were recorded from specific initial states (agent+block positions).
  Replaying them in a randomly-seeded env caused misalignment.

Fix:
  - `pusht_expert_state_meta.pkl` has per-episode initial states.
  - We pass `options={'state': init_state}` to env.reset(), which sets exact
    agent_x, agent_y, block_x, block_y, block_angle (from env source: PushT.reset).
  - Expert replay now starts from the SAME state as the original recordings.
  - CEM models also start from the same initial state.

Three panels per GIF:
  Left:   Expert actions (green agent, blue T-piece)
  Centre: Frozen CEM planner (red)
  Right:  A-B-M CEM planner with System M (teal)
"""

import os, sys, math, argparse
os.environ['SDL_VIDEODRIVER'] = 'offscreen'
import numpy as np
import pickle as pkl
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw
from copy import deepcopy
from loguru import logger as logging

sys.path.insert(0, '/workspace/mechjepa')
from mechjepa.model import MechJEPA
from mechjepa.planner import CEMPlanner

import stable_worldmodel as swm
from stable_worldmodel.policy import Policy

# ── Canvas constants ──────────────────────────────────────────────────────────
PANEL_W     = 340
PANEL_H     = 340
INFO_H      = 56
ENV_SIZE    = 512.0
BLOCK_SCALE = 24
AGENT_R     = 11
FPS_DELAY   = 80        # ~12.5 fps
TRAIL_LEN   = 30
MAX_STEPS   = 120

# ── Palette ────────────────────────────────────────────────────────────────────
BG           = (248, 246, 240)
GRID_COL     = (228, 226, 220)
INFO_BG      = ( 35,  35,  48)
EXPERT_BLOCK = ( 55, 125, 200)
EXPERT_AGENT = ( 20, 170,  80)
FROZEN_BLOCK = (210,  60,  50)
FROZEN_AGENT = (180,  50,  40)
ABM_BLOCK    = ( 25, 165, 130)
ABM_AGENT    = ( 20, 140, 110)
GOAL_COL     = (180, 155, 220)
ADAPT_FLASH  = (255, 155,   0)
TEXT_COL     = (235, 235, 235)
GREY         = (120, 120, 130)

MODEL_CFG = dict(
    num_slots=4, slot_dim=128, num_mechanisms=8,
    history_frames=3, pred_frames=1, action_dim=2,
    transformer_depth=6, transformer_heads=16,
    transformer_dim_head=64, transformer_mlp_dim=2048,
    edge_hidden_dim=256,
)


def env_to_canvas(x, y, w=PANEL_W, h=PANEL_H, margin=18):
    px = int(np.clip(x / ENV_SIZE, 0, 1) * (w - 2*margin) + margin)
    py = int(np.clip(1 - y / ENV_SIZE, 0, 1) * (h - 2*margin) + margin)
    return px, py


def t_piece_pts(cx, cy, angle, scale=BLOCK_SCALE):
    s = scale
    local = np.array([
        [-2*s, -s], [2*s, -s], [2*s, 0],
        [ s,    0], [s,   2*s], [-s, 2*s],
        [-s,    0], [-2*s, 0],
    ], dtype=float)
    ca, sa = math.cos(angle), math.sin(angle)
    R = np.array([[ca, -sa], [sa, ca]])
    pts = (R @ local.T).T
    pts[:, 0] += cx
    pts[:, 1] += cy
    return [tuple(p.astype(int)) for p in pts]


def draw_panel(state, block_col, agent_col, trail, goal_state,
               label, surp, adapted, step_i, total,
               w=PANEL_W, h=PANEL_H):
    img  = Image.new('RGBA', (w, h + INFO_H), BG + (255,))
    draw = ImageDraw.Draw(img, 'RGBA')

    # Grid
    for i in range(0, w, 40):
        draw.line([(i, 0), (i, h)], fill=GRID_COL + (255,), width=1)
    for i in range(0, h, 40):
        draw.line([(0, i), (w, i)], fill=GRID_COL + (255,), width=1)

    # Goal ghost
    if goal_state is not None:
        gx, gy = env_to_canvas(goal_state[2], goal_state[3], w, h)
        gpts = t_piece_pts(gx, gy, goal_state[4], scale=BLOCK_SCALE - 5)
        draw.polygon(gpts, outline=GOAL_COL + (160,), width=2)

    # Trail
    for i in range(1, len(trail)):
        alpha = int(30 + 180 * i / len(trail))
        draw.line([trail[i-1], trail[i]], fill=block_col + (alpha,), width=2)

    # Objects
    ax, ay = env_to_canvas(state[0], state[1], w, h)
    bx, by = env_to_canvas(state[2], state[3], w, h)
    bang    = state[4]

    bpts = t_piece_pts(bx, by, bang)
    draw.polygon(bpts, fill=block_col + (210,), outline=block_col + (255,), width=1)

    r = AGENT_R
    draw.ellipse([ax-r, ay-r, ax+r, ay+r],
                 fill=agent_col + (240,), outline=(15,15,15,200), width=1)

    # Adapt border
    if adapted:
        for ww in range(5):
            draw.rectangle([ww, ww, w-1-ww, h-1-ww],
                           outline=ADAPT_FLASH + (220 - ww*40,), width=1)

    # Progress bar
    bar_w = int(step_i / max(total-1,1) * (w-4))
    draw.rectangle([2, h-3, 2+bar_w, h-1], fill=(100,100,120,220))

    # Info strip
    draw.rectangle([0, h, w, h+INFO_H], fill=INFO_BG + (255,))
    draw.line([(0, h), (w, h)], fill=(65, 65, 85, 255), width=1)
    dot_r = 5
    draw.ellipse([8, h+10, 8+dot_r*2, h+10+dot_r*2], fill=block_col+(255,))
    draw.text((8+dot_r*2+5, h+7), label, fill=block_col+(255,))
    if surp is not None:
        draw.text((8, h+30), f'Pred err: {surp:.4f}', fill=TEXT_COL+(160,))
    if adapted:
        draw.text((w-130, h+30), '⚡ ADAPTING', fill=ADAPT_FLASH+(255,))

    return img


def stitch(p1, p2, p3, w=PANEL_W):
    gap = 3
    W = w*3 + gap*2
    H = p1.height
    out = Image.new('RGBA', (W, H), (180,180,180,255))
    out.paste(p1, (0, 0))
    out.paste(p2, (w + gap, 0))
    out.paste(p3, (w*2 + gap*2, 0))
    return out


# ── Slot-CEM policy (runs model in slot space, outputs env actions) ───────────
class SlotCEMPolicy:
    def __init__(self, model, planner, goal_slots, ep_slots, ep_actions,
                 device, system_m=False, optimizer=None, threshold=0.015, adapt_steps=3):
        self.model     = model
        self.planner   = planner
        self.goal_s    = goal_slots.to(device)
        self.ep_slots  = ep_slots
        self.ep_acts   = ep_actions
        self.device    = device
        self.system_m  = system_m
        self.optimizer = optimizer
        self.thresh    = threshold
        self.n_adapt   = adapt_steps
        self.hist_s    = []
        self.hist_a    = []
        self.t         = 0
        self.surp_log  = []
        self.adapt_log = []

    def reset(self):
        self.hist_s = []; self.hist_a = []; self.t = 0
        self.surp_log = []; self.adapt_log = []

    def step(self):
        t = self.t; T = len(self.ep_slots)
        if t >= T - 1:
            self.t += 1
            return np.zeros(2, dtype=np.float32), 0.0, False

        cur  = torch.from_numpy(self.ep_slots[t]).float().unsqueeze(0).to(self.device)
        self.hist_s.append(cur)
        if len(self.hist_s) > 3: self.hist_s.pop(0)
        if len(self.hist_a) < 3:
            self.hist_a.insert(0, torch.from_numpy(
                self.ep_acts[max(0, t-1)]).float().unsqueeze(0).to(self.device))
        if len(self.hist_a) > 3: self.hist_a.pop(0)
        self.t += 1

        if len(self.hist_s) < 3:
            return np.zeros(2, dtype=np.float32), 0.0, False

        nxt  = torch.from_numpy(self.ep_slots[min(t+1, T-1)]).float().unsqueeze(0).to(self.device)
        ht   = torch.cat(self.hist_s, dim=0).unsqueeze(0)
        hat  = torch.cat(self.hist_a, dim=0).unsqueeze(0)

        with torch.no_grad():
            pred = self.model.inference(ht, actions=hat).squeeze(1)
        surp = F.mse_loss(pred, nxt).item()
        adapted = False

        if self.system_m and self.optimizer and surp > self.thresh:
            adapted = True
            self.model.train()
            for _ in range(self.n_adapt):
                self.optimizer.zero_grad()
                p = self.model.differentiable_inference(ht, actions=hat).squeeze(1)
                loss = F.mse_loss(p, nxt)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.1)
                self.optimizer.step()
            self.model.eval()

        planned = self.planner.plan(ht, hat, self.goal_s)
        action  = planned[0, 0, :].detach().cpu().numpy()

        self.hist_a.append(torch.from_numpy(action).float().unsqueeze(0).to(self.device))
        if len(self.hist_a) > 3: self.hist_a.pop(0)
        self.surp_log.append(surp)
        self.adapt_log.append(adapted)
        return action, surp, adapted


class ExpertPolicy(Policy):
    def __init__(self, actions):
        self.actions = actions; self.t = 0
    def reset(self): self.t = 0
    def get_action(self, infos):
        a = self.actions[self.t] if self.t < len(self.actions) else np.zeros(2)
        self.t = min(self.t + 1, len(self.actions) - 1)
        return a.reshape(1,2).astype(np.float32)


class WrapCEMPolicy(Policy):
    def __init__(self, slot_cem):
        self.slot_cem = slot_cem
        self._action  = np.zeros(2, dtype=np.float32)
        self.surp     = 0.0
        self.adapted  = False

    def get_action(self, infos):
        return self._action.reshape(1, 2)

    def compute(self):
        a, s, ad = self.slot_cem.step()
        self._action = a; self.surp = s; self.adapted = ad
        return a, s, ad


def run_episode(ep_key, ep_slots, ep_actions, ep_states,
                goal_slots, goal_state,
                frozen_model, abm_model, frozen_plan, abm_plan, optimizer,
                device, threshold=0.015):
    """Run expert + frozen CEM + A-B-M CEM in 3 envs with matching initial state."""

    init_state = ep_states[0]   # [ax, ay, bx, by, bang, vx, vy]
    reset_opts = {'state': init_state, 'goal_state': goal_state}

    expert_pol = ExpertPolicy(ep_actions)

    frz_cem  = SlotCEMPolicy(frozen_model, frozen_plan, goal_slots, ep_slots, ep_actions, device)
    abm_cem  = SlotCEMPolicy(abm_model,   abm_plan,   goal_slots, ep_slots, ep_actions, device,
                             system_m=True, optimizer=optimizer, threshold=threshold)
    frz_pol  = WrapCEMPolicy(frz_cem)
    abm_pol  = WrapCEMPolicy(abm_cem)

    def make_env(pol):
        e = swm.World('swm/PushT-v1', num_envs=1, image_shape=(96,96),
                      max_episode_steps=MAX_STEPS+10, verbose=0)
        e.set_policy(pol)
        e.reset(options=reset_opts)
        return e

    env_e = make_env(expert_pol)
    env_f = make_env(frz_pol)
    env_a = make_env(abm_pol)

    tri_frames = []
    tr_e, tr_f, tr_a = [], [], []

    for step_i in range(MAX_STEPS):
        s_e = env_e.states['state'][0] if env_e.states else np.zeros(7)
        s_f = env_f.states['state'][0] if env_f.states else np.zeros(7)
        s_a = env_a.states['state'][0] if env_a.states else np.zeros(7)

        # Block trail positions
        tr_e.append(env_to_canvas(s_e[2], s_e[3]))
        tr_f.append(env_to_canvas(s_f[2], s_f[3]))
        tr_a.append(env_to_canvas(s_a[2], s_a[3]))
        if len(tr_e) > TRAIL_LEN: tr_e.pop(0)
        if len(tr_f) > TRAIL_LEN: tr_f.pop(0)
        if len(tr_a) > TRAIL_LEN: tr_a.pop(0)

        _,  surp_f, adpt_f = frz_pol.compute()
        _,  surp_a, adpt_a = abm_pol.compute()

        p_e = draw_panel(s_e, EXPERT_BLOCK, EXPERT_AGENT, list(tr_e),
                         goal_state, 'Expert Actions', None, False, step_i, MAX_STEPS)
        p_f = draw_panel(s_f, FROZEN_BLOCK, FROZEN_AGENT, list(tr_f),
                         goal_state, 'Frozen Model', surp_f, False, step_i, MAX_STEPS)
        p_a = draw_panel(s_a, ABM_BLOCK, ABM_AGENT, list(tr_a),
                         goal_state, 'A-B-M Agent', surp_a, adpt_a, step_i, MAX_STEPS)

        tri_frames.append(stitch(p_e, p_f, p_a))

        env_e.step(); env_f.step(); env_a.step()

        done = all([
            env_e.terminateds is not None and env_e.terminateds[0],
            env_f.terminateds is not None and env_f.terminateds[0],
            env_a.terminateds is not None and env_a.terminateds[0],
        ])
        if done:
            break

    logstr = (f'  Frozen surp={np.mean(frz_cem.surp_log):.4f}'
              f'  A-B-M surp={np.mean(abm_cem.surp_log):.4f}'
              f'  adaptations={sum(abm_cem.adapt_log)}')
    env_e.close(); env_f.close(); env_a.close()
    return tri_frames, logstr


def save_gif(frames, path):
    rgb = [f.convert('RGB') for f in frames]
    rgb[0].save(path, save_all=True, append_images=rgb[1:],
                duration=FPS_DELAY, loop=0, optimize=False)
    logging.info(f'Saved {path} ({len(frames)} frames, {len(frames)*FPS_DELAY/1000:.1f}s)')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt',       default='/workspace/checkpoints/mechjepa_pusht_act_best.ckpt')
    parser.add_argument('--data',       default='/workspace/data/pusht_slots_actions.pkl')
    parser.add_argument('--state_meta', default='/workspace/data/pusht_expert_state_meta.pkl')
    parser.add_argument('--out_dir',    default='/workspace/results/gifs_fixed')
    parser.add_argument('--episodes',   type=int, default=4)
    parser.add_argument('--threshold',  type=float, default=0.015)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    logging.info('Loading data...')
    with open(args.data, 'rb') as f:
        data = pkl.load(f)
    with open(args.state_meta, 'rb') as f:
        state_meta = pkl.load(f)

    val      = data['val']
    val_stat = state_meta['val']
    ep_keys  = sorted([k for k in val.keys() if k in val_stat])[:args.episodes]

    # Goal: use last state of ep0 as goal
    goal_state_arr = val_stat[ep_keys[0]][-1]  # [ax, ay, bx, by, bang, vx, vy]
    goal_slots = torch.from_numpy(val[ep_keys[0]]['slots'][-1]).float()

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

    all_frames = []
    for ep_i, key in enumerate(ep_keys):
        logging.info(f'▶ Episode {ep_i+1}/{len(ep_keys)}: {key}')
        ep_slots   = val[key]['slots']
        ep_actions = val[key]['actions']
        ep_states  = val_stat[key]

        init = ep_states[0]
        logging.info(f'  Init: agent=({init[0]:.0f},{init[1]:.0f})  block=({init[2]:.0f},{init[3]:.0f},ang={init[4]:.2f})')

        frames, logstr = run_episode(
            key, ep_slots, ep_actions, ep_states,
            goal_slots, goal_state_arr,
            frozen_model, abm_model, frozen_plan, abm_plan, optimizer,
            device=device, threshold=args.threshold,
        )
        logging.info(logstr)

        save_gif(frames, os.path.join(args.out_dir, f'pusht_ep{ep_i+1:02d}.gif'))
        all_frames.extend(frames)

    save_gif(all_frames, os.path.join(args.out_dir, 'pusht_all.gif'))
    logging.info(f'Done. Saved to {args.out_dir}/')


if __name__ == '__main__':
    main()
