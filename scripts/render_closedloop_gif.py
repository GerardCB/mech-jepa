"""
render_closedloop_gif.py — Fully closed-loop Push-T comparison GIF.

Pipeline per step:
  env.render(518×518 pixels)
    → DINOv2 ViT backbone (loaded from pusht_videosaur_model.ckpt)
    → output_transform (384→128)
    → GRU Slot Corrector (cross-attention, 4 slots × 128D)
    → slots (4, 128)
    → CEM Planner (horizon=10, N=256)
    → action (2,)
    → env.step(action)
    → next frame

Three panels:
  Left:   Expert (replays stored expert actions)
  Centre: Frozen MechJEPA CEM
  Right:  A-B-M MechJEPA CEM (System M active)

All agents start from the SAME initial state from the expert recording.
"""

import os, sys, math, argparse
os.environ['SDL_VIDEODRIVER'] = 'offscreen'
import numpy as np
import pickle as pkl
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageDraw
from loguru import logger as logging
from transformers import Dinov2Config, Dinov2Model

sys.path.insert(0, '/workspace/mechjepa')
from mechjepa.model import MechJEPA
from mechjepa.planner import CEMPlanner

import stable_worldmodel as swm
from stable_worldmodel.policy import Policy

# ── Constants ─────────────────────────────────────────────────────────────────
PANEL_W     = 340
PANEL_H     = 340
INFO_H      = 56
ENV_SIZE    = 512.0
BLOCK_SCALE = 24
AGENT_R     = 11
FPS_DELAY   = 80
TRAIL_LEN   = 30
MAX_STEPS   = 120
VIT_SIZE    = 518
N_SLOTS     = 4
SLOT_DIM    = 128
N_CORRECTOR = 3  # slot corrector iterations per forward pass

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

MODEL_CFG = dict(
    num_slots=4, slot_dim=128, num_mechanisms=8,
    history_frames=3, pred_frames=1, action_dim=2,
    transformer_depth=6, transformer_heads=16,
    transformer_dim_head=64, transformer_mlp_dim=2048,
    edge_hidden_dim=256,
)


# ══════════════════════════════════════════════════════════════════════════════
# VideoSAUR Slot Encoder
# ══════════════════════════════════════════════════════════════════════════════

class VideoSAUREncoder(nn.Module):
    """
    Minimal VideoSAUR encoder: DINOv2 ViT + output_transform + GRU slot corrector.
    Loaded directly from checkpoint weights.
    """
    def __init__(self, ckpt_path: str, device='cuda', n_slots=N_SLOTS,
                 n_corrector_iters=N_CORRECTOR):
        super().__init__()
        self.device = device
        self.n_slots = n_slots
        self.n_iter  = n_corrector_iters

        ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
        sd   = ckpt['state_dict']

        # ── ViT backbone (DINOv2 architecture) ────────────────────────────────
        vit_sd = {k[len('encoder.module.backbone.'):]: v
                  for k, v in sd.items() if k.startswith('encoder.module.backbone.')}
        vit_cfg = Dinov2Config(
            hidden_size=384, num_hidden_layers=12, num_attention_heads=6,
            mlp_ratio=4, image_size=VIT_SIZE, patch_size=14, num_channels=3,
            hidden_act='gelu',
        )
        self.vit = Dinov2Model(vit_cfg)
        self.vit.load_state_dict(vit_sd, strict=True)

        # ── Output transform: LayerNorm → Linear → GELU → Linear (128D) ──────
        self.ot_ln_w = nn.Parameter(sd['encoder.module.output_transform.layers.0.weight'])
        self.ot_ln_b = nn.Parameter(sd['encoder.module.output_transform.layers.0.bias'])
        self.ot_l1_w = nn.Parameter(sd['encoder.module.output_transform.layers.1.weight'])
        self.ot_l1_b = nn.Parameter(sd['encoder.module.output_transform.layers.1.bias'])
        self.ot_l2_w = nn.Parameter(sd['encoder.module.output_transform.layers.3.weight'])
        self.ot_l2_b = nn.Parameter(sd['encoder.module.output_transform.layers.3.bias'])

        # ── Slot corrector (GRU-based cross-attention) ────────────────────────
        self.corr_to_k  = nn.Parameter(sd['processor.module.corrector.to_k.weight'])
        self.corr_to_v  = nn.Parameter(sd['processor.module.corrector.to_v.weight'])
        self.corr_to_q  = nn.Parameter(sd['processor.module.corrector.to_q.weight'])
        self.corr_nf_w  = nn.Parameter(sd['processor.module.corrector.norm_features.weight'])
        self.corr_nf_b  = nn.Parameter(sd['processor.module.corrector.norm_features.bias'])
        self.corr_ns_w  = nn.Parameter(sd['processor.module.corrector.norm_slots.weight'])
        self.corr_ns_b  = nn.Parameter(sd['processor.module.corrector.norm_slots.bias'])
        self.corr_gru_wih = nn.Parameter(sd['processor.module.corrector.gru.weight_ih'])
        self.corr_gru_whh = nn.Parameter(sd['processor.module.corrector.gru.weight_hh'])
        self.corr_gru_bih = nn.Parameter(sd['processor.module.corrector.gru.bias_ih'])
        self.corr_gru_bhh = nn.Parameter(sd['processor.module.corrector.gru.bias_hh'])

        # ── Slot initialiser ──────────────────────────────────────────────────
        self.slot_init_mean = nn.Parameter(sd['initializer.mean'])    # (1, 1, 128)

        # Image preprocessing (ImageNet-like mean/std used in VideoSAUR)
        self.register_buffer('img_mean', torch.tensor([0.485, 0.456, 0.406]).view(1,3,1,1))
        self.register_buffer('img_std',  torch.tensor([0.229, 0.224, 0.225]).view(1,3,1,1))

        self.to(device)

    def _output_transform(self, x):
        x = F.layer_norm(x, [x.shape[-1]], self.ot_ln_w, self.ot_ln_b)
        x = F.linear(x, self.ot_l1_w, self.ot_l1_b)
        x = F.gelu(x)
        x = F.linear(x, self.ot_l2_w, self.ot_l2_b)
        return x

    def _slot_corrector(self, features, slots):
        """features: (B,N,D), slots: (B,S,D) → new_slots: (B,S,D)"""
        B, N, D = features.shape
        S = slots.shape[1]
        feat_n  = F.layer_norm(features, [D], self.corr_nf_w, self.corr_nf_b)
        slots_n = F.layer_norm(slots,    [D], self.corr_ns_w, self.corr_ns_b)
        k = F.linear(feat_n,  self.corr_to_k)   # (B, N, D)
        v = F.linear(feat_n,  self.corr_to_v)
        q = F.linear(slots_n, self.corr_to_q)   # (B, S, D)
        # Slot attention: softmax over N for each slot, then normalise over slots
        attn = torch.einsum('bnd,bsd->bsn', k, q) * (D ** -0.5)
        attn_s = attn.softmax(dim=-2)           # over slots
        attn_n = attn_s / (attn_s.sum(dim=-1, keepdim=True) + 1e-8)  # over features
        updates = torch.einsum('bsn,bnd->bsd', attn_n, v)
        # GRU update
        slots_f  = slots.reshape(B*S, D)
        updates_f = updates.reshape(B*S, D)
        wih   = self.corr_gru_wih; whh = self.corr_gru_whh
        bih   = self.corr_gru_bih; bhh = self.corr_gru_bhh
        z = torch.sigmoid(F.linear(updates_f, wih[:D], bih[:D]) +
                          F.linear(slots_f,   whh[:D], bhh[:D]))
        r = torch.sigmoid(F.linear(updates_f, wih[D:2*D], bih[D:2*D]) +
                          F.linear(slots_f,   whh[D:2*D], bhh[D:2*D]))
        n = torch.tanh(F.linear(updates_f,   wih[2*D:], bih[2*D:]) +
                       F.linear(r * slots_f, whh[2*D:], bhh[2*D:]))
        new_slots = (1 - z) * slots_f + z * n
        return new_slots.reshape(B, S, D)

    @torch.no_grad()
    def encode(self, frame_rgb: np.ndarray) -> torch.Tensor:
        """
        Args:
            frame_rgb: (H, W, 3) uint8 numpy RGB image
        Returns:
            slots: (N_SLOTS, SLOT_DIM) float32 tensor on self.device
        """
        # Resize and normalise
        from PIL import Image as PILImage
        img = PILImage.fromarray(frame_rgb).resize((VIT_SIZE, VIT_SIZE), PILImage.BICUBIC)
        x = torch.from_numpy(np.array(img)).float() / 255.0   # (H,W,3)
        x = x.permute(2, 0, 1).unsqueeze(0).to(self.device)   # (1,3,H,W)
        x = (x - self.img_mean) / self.img_std

        # ViT forward
        vit_out = self.vit(pixel_values=x)
        patch_feats = vit_out.last_hidden_state[:, 1:, :]  # skip CLS: (1,1369,384)

        # Output transform → 128D
        feats = self._output_transform(patch_feats)          # (1, 1369, 128)

        # Slot corrector
        slots = self.slot_init_mean.expand(1, self.n_slots, SLOT_DIM).clone()
        for _ in range(self.n_iter):
            slots = self._slot_corrector(feats, slots)       # (1, 4, 128)

        return slots.squeeze(0)  # (4, 128)


# ══════════════════════════════════════════════════════════════════════════════
# CEM policy using real-time VideoSAUR encoding
# ══════════════════════════════════════════════════════════════════════════════

class ClosedLoopCEMPolicy:
    """
    At each step:
      1. Encode current env frame → slots via VideoSAUR
      2. Roll history buffer
      3. Run CEM to get planned actions
      4. Execute action in env
    """
    def __init__(self, encoder, model, planner, goal_slots, device,
                 system_m=False, optimizer=None, threshold=0.015, adapt_steps=3):
        self.encoder   = encoder
        self.model     = model
        self.planner   = planner
        self.goal_s    = goal_slots.to(device)
        self.device    = device
        self.system_m  = system_m
        self.optimizer = optimizer
        self.thresh    = threshold
        self.n_adapt   = adapt_steps
        self.hist_slots = []
        self.hist_acts  = []
        self.surp_log   = []
        self.adapt_log  = []
        self._last_action = np.zeros(2, dtype=np.float32)
        self._surp     = 0.0
        self._adapted  = False

    def reset(self):
        self.hist_slots = []; self.hist_acts = []
        self.surp_log = []; self.adapt_log = []
        self._last_action = np.zeros(2, dtype=np.float32)

    def step(self, frame_rgb):
        """frame_rgb: (H,W,3) uint8 numpy. Returns (action (2,), surprise, adapted)."""
        slots = self.encoder.encode(frame_rgb)   # (4, 128) on device
        curr  = slots.unsqueeze(0)               # (1, 4, 128)

        self.hist_slots.append(curr)
        if len(self.hist_slots) > 3: self.hist_slots.pop(0)

        action_t = torch.from_numpy(self._last_action).float().unsqueeze(0).to(self.device)
        self.hist_acts.append(action_t)
        if len(self.hist_acts) > 3: self.hist_acts.pop(0)

        if len(self.hist_slots) < 3:
            return np.zeros(2, dtype=np.float32), 0.0, False

        hist_t  = torch.cat(self.hist_slots, dim=0).unsqueeze(0)  # (1, 3, 4, 128)
        hact_t  = torch.cat(self.hist_acts,  dim=0).unsqueeze(0)  # (1, 3, 2)

        # Surprise: compare prediction of previous step to current observation
        with torch.no_grad():
            pred = self.model.inference(hist_t, actions=hact_t).squeeze(1)
        surp = F.mse_loss(pred, curr).item()
        adapted = False

        if self.system_m and self.optimizer and surp > self.thresh:
            adapted = True
            self.model.train()
            for _ in range(self.n_adapt):
                self.optimizer.zero_grad()
                p = self.model.differentiable_inference(hist_t, actions=hact_t).squeeze(1)
                loss = F.mse_loss(p, curr)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.1)
                self.optimizer.step()
            self.model.eval()

        planned = self.planner.plan(hist_t, hact_t, self.goal_s)  # (1, H, 2)
        action  = planned[0, 0, :].detach().cpu().numpy()

        self.hist_acts.pop()  # replace warm-up action with planned one
        self.hist_acts.append(torch.from_numpy(action).float().unsqueeze(0).to(self.device))

        self._last_action = action
        self._surp  = surp
        self._adapted = adapted
        self.surp_log.append(surp)
        self.adapt_log.append(adapted)
        return action, surp, adapted


class ExpertPolicy(Policy):
    def __init__(self, actions):
        self.actions = actions; self.t = 0
    def reset(self): self.t = 0
    def get_action(self, infos):
        a = self.actions[min(self.t, len(self.actions)-1)]
        self.t += 1
        return a.reshape(1,2).astype(np.float32)


# ══════════════════════════════════════════════════════════════════════════════
# Rendering helpers
# ══════════════════════════════════════════════════════════════════════════════

def env_to_canvas(x, y, w=PANEL_W, h=PANEL_H, margin=18):
    px = int(np.clip(x / ENV_SIZE, 0, 1) * (w - 2*margin) + margin)
    py = int(np.clip(1 - y / ENV_SIZE, 0, 1) * (h - 2*margin) + margin)
    return px, py


def t_piece_pts(cx, cy, angle, scale=BLOCK_SCALE):
    s = scale
    local = np.array([
        [-2*s,-s],[2*s,-s],[2*s,0],[s,0],[s,2*s],[-s,2*s],[-s,0],[-2*s,0]
    ], dtype=float)
    ca, sa = math.cos(angle), math.sin(angle)
    R = np.array([[ca,-sa],[sa,ca]])
    pts = (R @ local.T).T + np.array([cx, cy])
    return [tuple(p.astype(int)) for p in pts]


def draw_panel(state, block_col, agent_col, trail, goal_state,
               label, surp, adapted, step_i, total, w=PANEL_W, h=PANEL_H):
    img  = Image.new('RGBA', (w, h+INFO_H), BG+(255,))
    draw = ImageDraw.Draw(img, 'RGBA')
    for i in range(0, w, 40):
        draw.line([(i,0),(i,h)], fill=GRID_COL+(255,), width=1)
    for i in range(0, h, 40):
        draw.line([(0,i),(w,i)], fill=GRID_COL+(255,), width=1)
    if goal_state is not None:
        gx, gy = env_to_canvas(goal_state[2], goal_state[3], w, h)
        gpts = t_piece_pts(gx, gy, goal_state[4], scale=BLOCK_SCALE-5)
        draw.polygon(gpts, outline=GOAL_COL+(160,), width=2)
    for i in range(1, len(trail)):
        alpha = int(30 + 180 * i / len(trail))
        draw.line([trail[i-1], trail[i]], fill=block_col+(alpha,), width=2)
    ax, ay = env_to_canvas(state[0], state[1], w, h)
    bx, by = env_to_canvas(state[2], state[3], w, h)
    draw.polygon(t_piece_pts(bx, by, state[4]),
                 fill=block_col+(210,), outline=block_col+(255,), width=1)
    r = AGENT_R
    draw.ellipse([ax-r,ay-r,ax+r,ay+r], fill=agent_col+(240,), outline=(15,15,15,200), width=1)
    if adapted:
        for ww in range(5):
            draw.rectangle([ww,ww,w-1-ww,h-1-ww], outline=ADAPT_FLASH+(220-ww*40,), width=1)
    bar_w = int(step_i/max(total-1,1)*(w-4))
    draw.rectangle([2,h-3,2+bar_w,h-1], fill=(100,100,120,220))
    draw.rectangle([0,h,w,h+INFO_H], fill=INFO_BG+(255,))
    draw.line([(0,h),(w,h)], fill=(65,65,85,255), width=1)
    draw.ellipse([8,h+10,18,h+20], fill=block_col+(255,))
    draw.text((22,h+7), label, fill=block_col+(255,))
    if surp is not None:
        draw.text((8,h+30), f'Pred err: {surp:.4f}', fill=TEXT_COL+(160,))
    if adapted:
        draw.text((w-130,h+30), '⚡ ADAPTING', fill=ADAPT_FLASH+(255,))
    return img


def stitch(p1, p2, p3, w=PANEL_W):
    gap = 3
    out = Image.new('RGBA', (w*3+gap*2, p1.height), (180,180,180,255))
    out.paste(p1, (0,0)); out.paste(p2, (w+gap,0)); out.paste(p3, (w*2+gap*2,0))
    return out


def save_gif(frames, path):
    rgb = [f.convert('RGB') for f in frames]
    rgb[0].save(path, save_all=True, append_images=rgb[1:],
                duration=FPS_DELAY, loop=0, optimize=False)
    logging.info(f'Saved {path} ({len(frames)} frames, {len(frames)*FPS_DELAY/1000:.1f}s)')


# ══════════════════════════════════════════════════════════════════════════════
# Main episode loop
# ══════════════════════════════════════════════════════════════════════════════

def run_episode(ep_actions, init_state, goal_state,
                goal_slots, encoder, frozen_cem, abm_cem, device):
    """Run all 3 agents in separate envs with matching initial state."""
    reset_opts = {'state': init_state, 'goal_state': goal_state}

    expert_pol = ExpertPolicy(ep_actions)

    def make_env(pol):
        e = swm.World('swm/PushT-v1', num_envs=1, image_shape=(VIT_SIZE, VIT_SIZE),
                      max_episode_steps=MAX_STEPS+10, verbose=0)
        e.set_policy(pol)
        e.reset(options=reset_opts)
        return e

    # Simple wrapper for CEM agents
    class CEMEnvPol(Policy):
        def __init__(self, cem): self.cem = cem; self._a = np.zeros(2,dtype=np.float32)
        def get_action(self, infos): return self._a.reshape(1,2)

    fz_pol = CEMEnvPol(frozen_cem)
    ab_pol = CEMEnvPol(abm_cem)

    env_e = make_env(expert_pol)
    env_f = make_env(fz_pol)
    env_a = make_env(ab_pol)

    frames = []
    tr_e, tr_f, tr_a = [], [], []

    for step_i in range(MAX_STEPS):
        s_e = env_e.states['state'][0] if env_e.states else np.zeros(7)
        s_f = env_f.states['state'][0] if env_f.states else np.zeros(7)
        s_a = env_a.states['state'][0] if env_a.states else np.zeros(7)

        tr_e.append(env_to_canvas(s_e[2], s_e[3]))
        tr_f.append(env_to_canvas(s_f[2], s_f[3]))
        tr_a.append(env_to_canvas(s_a[2], s_a[3]))
        if len(tr_e) > TRAIL_LEN: tr_e.pop(0)
        if len(tr_f) > TRAIL_LEN: tr_f.pop(0)
        if len(tr_a) > TRAIL_LEN: tr_a.pop(0)

        # ── Frozen CEM: encode current env_f frame ────────────────────────────
        frame_f = env_f.envs.envs[0].render()  # (H, W, 3) uint8 RGB
        fz_action, surp_f, _ = frozen_cem.step(frame_f.astype(np.uint8))
        fz_pol._a = fz_action

        # ── A-B-M CEM: encode current env_a frame ────────────────────────────
        frame_a = env_a.envs.envs[0].render()  # (H, W, 3) uint8 RGB
        ab_action, surp_a, adapted = abm_cem.step(frame_a.astype(np.uint8))
        ab_pol._a = ab_action

        # Draw panels
        p_e = draw_panel(s_e, EXPERT_BLOCK, EXPERT_AGENT, list(tr_e),
                         goal_state, 'Expert Actions', None, False, step_i, MAX_STEPS)
        p_f = draw_panel(s_f, FROZEN_BLOCK, FROZEN_AGENT, list(tr_f),
                         goal_state, 'Frozen Model', surp_f, False, step_i, MAX_STEPS)
        p_a = draw_panel(s_a, ABM_BLOCK, ABM_AGENT, list(tr_a),
                         goal_state, 'A-B-M Agent', surp_a, adapted, step_i, MAX_STEPS)
        frames.append(stitch(p_e, p_f, p_a))

        # Step envs
        done_e = env_e.terminateds is not None and bool(env_e.terminateds[0])
        done_f = env_f.terminateds is not None and bool(env_f.terminateds[0])
        done_a = env_a.terminateds is not None and bool(env_a.terminateds[0])

        if not done_e: env_e.step()
        if not done_f: env_f.step()
        if not done_a: env_a.step()

        if done_e and done_f and done_a:
            break

    env_e.close(); env_f.close(); env_a.close()
    stats = {
        'surp_f': float(np.mean(frozen_cem.surp_log)) if frozen_cem.surp_log else 0,
        'surp_a': float(np.mean(abm_cem.surp_log))   if abm_cem.surp_log   else 0,
        'adapts': int(sum(abm_cem.adapt_log)),
    }
    return frames, stats


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt',       default='/workspace/checkpoints/mechjepa_pusht_act_best.ckpt')
    parser.add_argument('--encoder',    default='/workspace/data/pusht_videosaur_model.ckpt')
    parser.add_argument('--data',       default='/workspace/data/pusht_slots_actions.pkl')
    parser.add_argument('--state_meta', default='/workspace/data/pusht_expert_state_meta.pkl')
    parser.add_argument('--out_dir',    default='/workspace/results/gifs_closedloop')
    parser.add_argument('--episodes',   type=int,   default=4)
    parser.add_argument('--threshold',  type=float, default=0.015)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # ── Load data ─────────────────────────────────────────────────────────────
    logging.info('Loading data...')
    with open(args.data, 'rb') as f:
        data = pkl.load(f)
    with open(args.state_meta, 'rb') as f:
        state_meta = pkl.load(f)
    val      = data['val']
    val_stat = state_meta['val']
    ep_keys  = sorted([k for k in val.keys() if k in val_stat])[:args.episodes]

    goal_state = val_stat[ep_keys[0]][-1]
    goal_slots = torch.from_numpy(val[ep_keys[0]]['slots'][-1]).float()

    # ── Load VideoSAUR encoder ────────────────────────────────────────────────
    logging.info('Loading VideoSAUR encoder...')
    encoder = VideoSAUREncoder(args.encoder, device=device)
    encoder.eval()

    # ── Load MechJEPA models ──────────────────────────────────────────────────
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

    # ── Run episodes ──────────────────────────────────────────────────────────
    all_frames = []
    for ep_i, key in enumerate(ep_keys):
        logging.info(f'▶ Episode {ep_i+1}/{len(ep_keys)}: {key}')
        ep_actions = val[key]['actions']
        ep_states  = val_stat[key]
        init_state = ep_states[0]
        logging.info(f'  Init: agent=({init_state[0]:.0f},{init_state[1]:.0f}) '
                     f'block=({init_state[2]:.0f},{init_state[3]:.0f})')

        frozen_cem = ClosedLoopCEMPolicy(encoder, frozen_model, frozen_plan, goal_slots, device)
        abm_cem    = ClosedLoopCEMPolicy(encoder, abm_model,   abm_plan,   goal_slots, device,
                                          system_m=True, optimizer=optimizer, threshold=args.threshold)

        frames, stats = run_episode(
            ep_actions, init_state, goal_state,
            goal_slots, encoder, frozen_cem, abm_cem, device,
        )
        logging.info(f'  Frozen surp={stats["surp_f"]:.4f}  '
                     f'A-B-M surp={stats["surp_a"]:.4f}  '
                     f'adaptations={stats["adapts"]}')

        save_gif(frames, os.path.join(args.out_dir, f'pusht_cl_ep{ep_i+1:02d}.gif'))
        all_frames.extend(frames)

    save_gif(all_frames, os.path.join(args.out_dir, 'pusht_cl_all.gif'))
    logging.info(f'Done. Saved to {args.out_dir}/')


if __name__ == '__main__':
    main()
