"""
render_gif.py — Generate clean Push-T comparison GIFs for the GitHub repo.

Visual style:
  - Light off-white background (#F7F5F2)
  - Solid T-shaped polygon for the block (rotated correctly)
  - Filled circle for the agent
  - THREE overlaid trajectories in one panel:
      ● Real slot trajectory (dark gray, filled shape)
      ● Frozen model prediction (coral red, outline)
      ● A-B-M model prediction (emerald teal, outline + filled when better)
  - Amber flash indicator when System M adapts
  - Info strip at bottom

Also outputs a SIDE-BY-SIDE version: left=Frozen, right=A-B-M (both vs ground truth)

Outputs:
  /workspace/results/gifs/pusht_overview.gif    — single panel, all 3 overlaid (best for README header)
  /workspace/results/gifs/pusht_compare_ep{N}.gif — side-by-side, per episode
  /workspace/results/gifs/pusht_compare_all.gif   — side-by-side, all episodes concatenated
"""

import os, sys, math, argparse
import numpy as np
import pickle as pkl
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont
from sklearn.decomposition import PCA
from copy import deepcopy
from loguru import logger as logging

sys.path.insert(0, '/workspace/mechjepa')
from mechjepa.model import MechJEPA
from mechjepa.planner import CEMPlanner

# ── Display constants ─────────────────────────────────────────────────────────
SIZE        = 400       # canvas W and H
INFO_H      = 52        # bottom info strip
FPS_DELAY   = 80        # ms per GIF frame (~12.5 fps)
TRAIL_LEN   = 20
BLOCK_SCALE = 22        # T-piece half-width in pixels
AGENT_R     = 11

# ── Palette (light theme) ────────────────────────────────────────────────────
BG          = (247, 245, 242)     # off-white
GRID_COL    = (230, 228, 225)     # faint grid
AGENT_REAL  = ( 30,  30,  30)    # near-black agent
BLOCK_REAL  = ( 60, 120, 180)    # blue block (real GT)
BLOCK_FROZ  = (210,  70,  55)    # coral red (frozen pred)
BLOCK_ABM   = ( 32, 160, 130)    # emerald (A-B-M pred)
GOAL_COL    = (180, 160, 220)    # lavender goal
ADAPT_FLASH = (255, 165,   0)    # amber
INFO_BG     = ( 40,  40,  50)    # dark info bar
TEXT_COL    = (240, 240, 240)
LABEL_FROZ  = (210,  70,  55)
LABEL_ABM   = ( 32, 160, 130)

MODEL_CFG = dict(
    num_slots=4, slot_dim=128, num_mechanisms=8,
    history_frames=3, pred_frames=1, action_dim=2,
    transformer_depth=6, transformer_heads=16,
    transformer_dim_head=64, transformer_mlp_dim=2048,
    edge_hidden_dim=256,
)


# ── T-piece polygon ───────────────────────────────────────────────────────────
def t_piece_pts(cx, cy, angle, scale=BLOCK_SCALE):
    """Return list of (x,y) ints for a T-piece polygon centred at (cx,cy)."""
    s = scale
    # Local coords of the T:  wide bar on top, stem below
    local = np.array([
        [-2*s, -s], [2*s, -s], [2*s, 0],
        [ s,    0], [s,   2*s], [-s, 2*s],
        [-s,    0], [-2*s, 0],
    ], dtype=float)
    cos_a, sin_a = math.cos(angle), math.sin(angle)
    R = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
    rotated = (R @ local.T).T
    rotated[:, 0] += cx
    rotated[:, 1] += cy
    return [tuple(p.astype(int)) for p in rotated]


def draw_block(draw, cx, cy, angle, color, outline_only=False, scale=BLOCK_SCALE, width=2):
    pts = t_piece_pts(cx, cy, angle, scale=scale)
    if outline_only:
        draw.polygon(pts, outline=color + (200,), width=width)
    else:
        draw.polygon(pts, fill=color + (220,), outline=color + (255,), width=1)


def draw_agent(draw, cx, cy, color, outline_only=False, r=AGENT_R):
    box = [cx - r, cy - r, cx + r, cy + r]
    if outline_only:
        draw.ellipse(box, outline=color + (200,), width=2)
    else:
        draw.ellipse(box, fill=color + (230,), outline=(0, 0, 0, 180), width=1)


# ── PCA helpers ───────────────────────────────────────────────────────────────
def fit_pca(val_data):
    keys = sorted(list(val_data.keys()))
    all_slots = np.concatenate([val_data[k]['slots'] for k in keys], axis=0)
    pca_a = PCA(n_components=4); pca_a.fit(all_slots[:, 0, :])
    pca_b = PCA(n_components=4); pca_b.fit(all_slots[:, 1, :])

    def extents(pca, data):
        proj = pca.transform(data)   # (N, 4)
        lo = proj[:, 1:3].min(0) - 0.15
        hi = proj[:, 1:3].max(0) + 0.15
        return lo, hi

    alo, ahi = extents(pca_a, all_slots[:, 0, :])
    blo, bhi = extents(pca_b, all_slots[:, 1, :])
    return pca_a, pca_b, alo, ahi, blo, bhi


def to_canvas(slot_vec, pca, lo, hi, size=SIZE, margin=30):
    proj = pca.transform(slot_vec.reshape(1, -1))[0]
    x = int(np.clip((proj[1] - lo[0]) / (hi[0] - lo[0] + 1e-8), 0, 1) * (size - 2 * margin) + margin)
    y = int(np.clip(1 - (proj[2] - lo[1]) / (hi[1] - lo[1] + 1e-8), 0, 1) * (size - 2 * margin) + margin)
    return x, y


def slot_angle(slot_vec, pca):
    """Estimate block rotation angle from PCA projection."""
    proj = pca.transform(slot_vec.reshape(1, -1))[0]
    return math.atan2(proj[3], proj[2]) if len(proj) > 3 else 0.0


# ── Single-panel frame (3 overlaid: real + frozen + abm) ─────────────────────
def make_overlay_frame(
    real_b, real_a, real_ang,
    froz_b, froz_ang,
    abm_b,  abm_ang,
    goal_b, goal_ang,
    trail_r, trail_f, trail_a,
    adapted, surp_f, surp_a, step_i, total,
    size=SIZE,
):
    """RGBA PIL image. All positions are (x,y) ints already in canvas coords."""
    img = Image.new('RGBA', (size, size + INFO_H), BG + (255,))
    draw = ImageDraw.Draw(img, 'RGBA')

    # Faint grid
    for i in range(0, size, 40):
        draw.line([(i, 0), (i, size)], fill=GRID_COL + (255,), width=1)
        draw.line([(0, i), (size, i)], fill=GRID_COL + (255,), width=1)

    # Goal (ghost)
    if goal_b:
        draw_block(draw, *goal_b, goal_ang, GOAL_COL, outline_only=True, scale=BLOCK_SCALE - 4, width=2)

    # Trails (thin fading lines)
    def draw_trail(pts, col):
        for i in range(1, len(pts)):
            alpha = int(50 + 170 * i / len(pts))
            draw.line([pts[i-1], pts[i]], fill=col + (alpha,), width=2)

    draw_trail(trail_r, BLOCK_REAL)
    draw_trail(trail_f, BLOCK_FROZ)
    draw_trail(trail_a, BLOCK_ABM)

    # Frozen prediction (outline)
    if froz_b:
        draw_block(draw, *froz_b, froz_ang, BLOCK_FROZ, outline_only=True, width=2)
    # A-B-M prediction (outline, slightly smaller)
    if abm_b:
        draw_block(draw, *abm_b, abm_ang, BLOCK_ABM, outline_only=True, scale=BLOCK_SCALE - 3, width=2)
    # Real (filled, on top)
    if real_b:
        draw_block(draw, *real_b, real_ang, BLOCK_REAL)
    if real_a:
        draw_agent(draw, *real_a, AGENT_REAL)

    # Adaptation border flash
    if adapted:
        for w in range(4):
            draw.rectangle([w, w, size - 1 - w, size - 1 - w],
                           outline=ADAPT_FLASH + (200 - w * 40,), width=1)

    # Progress bar
    bar_w = int((step_i / max(total - 1, 1)) * (size - 4))
    draw.rectangle([2, size - 3, 2 + bar_w, size - 1], fill=(100, 100, 120, 200))

    # ── Info strip ────────────────────────────────────────────────────────
    draw.rectangle([0, size, size, size + INFO_H], fill=INFO_BG + (255,))
    draw.line([(0, size), (size, size)], fill=(80, 80, 100, 255), width=1)

    # Colour legend dots
    dot_r = 5
    for i, (col, lbl) in enumerate([
        (BLOCK_REAL, 'Real (GT)'),
        (BLOCK_FROZ, 'Frozen pred'),
        (BLOCK_ABM,  'A-B-M pred'),
    ]):
        x = 10 + i * 125
        draw.ellipse([x, size + 10, x + dot_r * 2, size + 10 + dot_r * 2], fill=col + (255,))
        draw.text((x + dot_r * 2 + 4, size + 8), lbl, fill=col + (255,))

    # Metrics
    impr = surp_f / (surp_a + 1e-8)
    draw.text((10, size + 30),
              f'Surprise  Frozen: {surp_f:.4f}   A-B-M: {surp_a:.4f}  ({impr:.1f}× better)',
              fill=TEXT_COL + (200,))
    if adapted:
        draw.text((size - 130, size + 30), '⚡ ADAPTING', fill=ADAPT_FLASH + (255,))

    return img


# ── Side-by-side frame ────────────────────────────────────────────────────────
def make_side_frame(left_img, right_img, size=SIZE):
    gap = 4
    total_w = size * 2 + gap
    total_h = size + INFO_H
    out = Image.new('RGBA', (total_w, total_h), (200, 200, 200, 255))
    out.paste(left_img, (0, 0))
    out.paste(right_img, (size + gap, 0))
    return out


def make_panel_frame(
    real_b, real_a, real_ang,
    pred_b, pred_ang,
    goal_b, goal_ang,
    trail_r, trail_p,
    adapted, surp, step_i, total, label, label_col,
    size=SIZE,
):
    """One panel (Frozen or A-B-M)."""
    img = Image.new('RGBA', (size, size + INFO_H), BG + (255,))
    draw = ImageDraw.Draw(img, 'RGBA')

    for i in range(0, size, 40):
        draw.line([(i, 0), (i, size)], fill=GRID_COL + (255,), width=1)
        draw.line([(0, i), (size, i)], fill=GRID_COL + (255,), width=1)

    if goal_b:
        draw_block(draw, *goal_b, goal_ang, GOAL_COL, outline_only=True, scale=BLOCK_SCALE - 4, width=2)

    def draw_trail(pts, col):
        for i in range(1, len(pts)):
            alpha = int(50 + 170 * i / len(pts))
            draw.line([pts[i-1], pts[i]], fill=col + (alpha,), width=2)

    draw_trail(trail_r, BLOCK_REAL)
    draw_trail(trail_p, label_col)

    if pred_b:
        draw_block(draw, *pred_b, pred_ang, label_col, outline_only=True, width=2)
    if real_b:
        draw_block(draw, *real_b, real_ang, BLOCK_REAL)
    if real_a:
        draw_agent(draw, *real_a, AGENT_REAL)

    if adapted:
        for w in range(4):
            draw.rectangle([w, w, size - 1 - w, size - 1 - w],
                           outline=ADAPT_FLASH + (200 - w * 40,), width=1)

    bar_w = int((step_i / max(total - 1, 1)) * (size - 4))
    draw.rectangle([2, size - 3, 2 + bar_w, size - 1], fill=(100, 100, 120, 200))

    draw.rectangle([0, size, size, size + INFO_H], fill=INFO_BG + (255,))
    draw.line([(0, size), (size, size)], fill=(80, 80, 100, 255), width=1)

    # Header label
    draw.text((8, size + 8),  label,               fill=label_col + (255,))
    draw.text((8, size + 28), f'Surprise: {surp:.4f}', fill=TEXT_COL + (200,))

    if adapted:
        draw.text((size - 100, size + 28), '⚡ ADAPT', fill=ADAPT_FLASH + (255,))

    return img


# ── Main episode run ──────────────────────────────────────────────────────────
def run_episode(ep_slots, ep_actions, goal_slots_t,
                frozen_model, abm_model, frozen_plan, abm_plan, optimizer,
                pca_a, pca_b, alo, ahi, blo, bhi,
                goal_b_pos, goal_b_ang,
                device, threshold=0.015, adapt_steps=3):

    T = ep_slots.shape[0]
    hist_size = 3
    overlay_frames, side_frames = [], []
    trails_r, trails_f, trails_a = [], [], []
    trail_r_l, trail_f_l, trail_a_l = [], [], []

    hist_slots, hist_acts = [], []

    for t in range(T - 1):
        curr = torch.from_numpy(ep_slots[t]).float().unsqueeze(0).to(device)
        hist_slots.append(curr)
        if len(hist_slots) > hist_size: hist_slots.pop(0)

        if len(hist_acts) < hist_size:
            hist_acts.insert(0, torch.from_numpy(ep_actions[max(0, t-1)]).float().unsqueeze(0).to(device))
        if len(hist_acts) > hist_size: hist_acts.pop(0)

        next_real = torch.from_numpy(ep_slots[t+1]).float().unsqueeze(0).to(device)

        # Canvas positions for REAL
        real_b = to_canvas(ep_slots[t, 1, :], pca_b, blo, bhi, SIZE)
        real_a = to_canvas(ep_slots[t, 0, :], pca_a, alo, ahi, SIZE)
        real_ang = slot_angle(ep_slots[t, 1, :], pca_b)

        trail_r_l.append(real_b)
        if len(trail_r_l) > TRAIL_LEN: trail_r_l.pop(0)

        if len(hist_slots) < hist_size:
            # Not enough history — just show real
            img_ov = make_overlay_frame(
                real_b, real_a, real_ang,
                None, 0.0, None, 0.0,
                goal_b_pos, goal_b_ang,
                list(trail_r_l), [], [],
                False, 0.0, 0.0, t, T-1,
            )
            overlay_frames.append(img_ov)
            # Side panels
            lp = make_panel_frame(real_b, real_a, real_ang, None, 0.0, goal_b_pos, goal_b_ang,
                                   list(trail_r_l), [], False, 0.0, t, T-1, 'Frozen Model', LABEL_FROZ)
            rp = make_panel_frame(real_b, real_a, real_ang, None, 0.0, goal_b_pos, goal_b_ang,
                                   list(trail_r_l), [], False, 0.0, t, T-1, 'A-B-M Agent', LABEL_ABM)
            side_frames.append(make_side_frame(lp, rp))
            continue

        hist_t = torch.cat(hist_slots, dim=0).unsqueeze(0)
        hact_t = torch.cat(hist_acts,  dim=0).unsqueeze(0)

        # ── Frozen ───────────────────────────────────────────────────────────
        with torch.no_grad():
            pred_f = frozen_model.inference(hist_t, actions=hact_t).squeeze(1)
        surp_f = F.mse_loss(pred_f, next_real).item()

        froz_b   = to_canvas(pred_f[0, 1, :].cpu().numpy(), pca_b, blo, bhi, SIZE)
        froz_ang = slot_angle(pred_f[0, 1, :].cpu().numpy(), pca_b)
        trail_f_l.append(froz_b)
        if len(trail_f_l) > TRAIL_LEN: trail_f_l.pop(0)

        # ── A-B-M ────────────────────────────────────────────────────────────
        with torch.no_grad():
            pred_a = abm_model.inference(hist_t, actions=hact_t).squeeze(1)
        surp_a = F.mse_loss(pred_a, next_real).item()
        adapted = False

        if surp_a > threshold and optimizer is not None:
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
            surp_a = F.mse_loss(pred_a, next_real).item()

        abm_b   = to_canvas(pred_a[0, 1, :].cpu().numpy(), pca_b, blo, bhi, SIZE)
        abm_ang = slot_angle(pred_a[0, 1, :].cpu().numpy(), pca_b)
        trail_a_l.append(abm_b)
        if len(trail_a_l) > TRAIL_LEN: trail_a_l.pop(0)

        # Advance action buffer
        hist_acts.append(torch.from_numpy(ep_actions[t]).float().unsqueeze(0).to(device))
        if len(hist_acts) > hist_size: hist_acts.pop(0)

        # ── Compose frames ────────────────────────────────────────────────────
        img_ov = make_overlay_frame(
            real_b, real_a, real_ang,
            froz_b, froz_ang,
            abm_b,  abm_ang,
            goal_b_pos, goal_b_ang,
            list(trail_r_l), list(trail_f_l), list(trail_a_l),
            adapted, surp_f, surp_a, t, T-1,
        )
        overlay_frames.append(img_ov)

        lp = make_panel_frame(real_b, real_a, real_ang, froz_b, froz_ang,
                               goal_b_pos, goal_b_ang, list(trail_r_l), list(trail_f_l),
                               False, surp_f, t, T-1, 'Frozen Model', LABEL_FROZ)
        rp = make_panel_frame(real_b, real_a, real_ang, abm_b, abm_ang,
                               goal_b_pos, goal_b_ang, list(trail_r_l), list(trail_a_l),
                               adapted, surp_a, t, T-1, 'A-B-M Agent', LABEL_ABM)
        side_frames.append(make_side_frame(lp, rp))

    return overlay_frames, side_frames


def save_gif(frames, path, loop=0):
    if not frames:
        return
    rgb = [f.convert('RGB') if f.mode == 'RGBA' else f for f in frames]
    rgb[0].save(
        path,
        save_all=True,
        append_images=rgb[1:],
        duration=FPS_DELAY,
        loop=loop,
        optimize=False,
    )
    logging.info(f'Saved {path}  ({len(frames)} frames)')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt',      default='/workspace/checkpoints/mechjepa_pusht_act_best.ckpt')
    parser.add_argument('--data',      default='/workspace/data/pusht_slots_actions.pkl')
    parser.add_argument('--out_dir',   default='/workspace/results/gifs')
    parser.add_argument('--episodes',  type=int, default=5)
    parser.add_argument('--threshold', type=float, default=0.015)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    logging.info('Loading data + fitting PCA...')
    with open(args.data, 'rb') as f:
        data = pkl.load(f)
    val = data['val']
    ep_keys = sorted(list(val.keys()))[:args.episodes]

    pca_a, pca_b, alo, ahi, blo, bhi = fit_pca(val)

    goal_slot_np = val[ep_keys[0]]['slots'][-1]
    goal_b_pos   = to_canvas(goal_slot_np[1], pca_b, blo, bhi, SIZE)
    goal_b_ang   = slot_angle(goal_slot_np[1], pca_b)
    goal_slots_t = torch.from_numpy(goal_slot_np).float()

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

    all_overlay, all_side = [], []

    for ep_i, key in enumerate(ep_keys):
        logging.info(f'▶ Episode {ep_i+1}/{len(ep_keys)}: {key}')
        ep = val[key]

        ov_frames, side_frames = run_episode(
            ep['slots'], ep['actions'], goal_slots_t,
            frozen_model, abm_model, frozen_plan, abm_plan, optimizer,
            pca_a, pca_b, alo, ahi, blo, bhi,
            goal_b_pos, goal_b_ang,
            device=device, threshold=args.threshold,
        )

        save_gif(ov_frames,   os.path.join(args.out_dir, f'pusht_overlay_ep{ep_i+1:02d}.gif'))
        save_gif(side_frames, os.path.join(args.out_dir, f'pusht_compare_ep{ep_i+1:02d}.gif'))
        all_overlay.extend(ov_frames)
        all_side.extend(side_frames)

    save_gif(all_overlay, os.path.join(args.out_dir, 'pusht_overlay_all.gif'))
    save_gif(all_side,    os.path.join(args.out_dir, 'pusht_compare_all.gif'))
    logging.info(f'All GIFs saved to {args.out_dir}/')


if __name__ == '__main__':
    main()
