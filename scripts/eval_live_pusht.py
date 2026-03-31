"""
eval_live_pusht.py — Unified closed-loop Push-T evaluation with real env rendering.

Pipeline per step (identical to LeWorldModel):
  frame = env.render()               # Real Push-T pixels
  slots = encoder.encode(frame)       # VideoSAUR: pixels → 4 slots
  action = cem_planner.plan(...)      # CEM in latent slot space
  env.step(action)                    # Execute in real physics
  frames_list.append(frame)           # Record real frame for video

Three conditions run in parallel (same initial state):
  1. Expert   — replays stored expert actions
  2. Frozen   — MechJEPA CEM, no adaptation
  3. A-B-M    — MechJEPA CEM + System M (surprise-triggered online update)

OOD distribution shift via real physics (SWM variation space):
  --ood_block_scale 42   # 40% bigger T-block (default is 30)

Output: side-by-side GIF of REAL environment renders.

Usage:
  # In-distribution
  python scripts/eval_live_pusht.py --episodes 3

  # OOD: bigger block
  python scripts/eval_live_pusht.py --episodes 3 --ood_block_scale 42
"""

import os
import sys
import argparse
import pickle as pkl

os.environ["SDL_VIDEODRIVER"] = "offscreen"

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont
from loguru import logger as logging

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from mechjepa.model import MechJEPA
from mechjepa.encoder import VideoSAUREncoder
from mechjepa.planner import CEMPlanner

import stable_worldmodel as swm
from stable_worldmodel.policy import Policy

# ── Defaults ──────────────────────────────────────────────────────────────────
MAX_STEPS = 200
FPS_DELAY = 80  # ms per frame in GIF (~12.5 fps)
RENDER_SIZE = 256  # env render resolution per panel
INFO_BAR_H = 40
FONT_SIZE = 14

MODEL_CFG = dict(
    num_slots=4, slot_dim=128, num_mechanisms=8,
    history_frames=3, pred_frames=1, action_dim=2,
    transformer_depth=6, transformer_heads=16,
    transformer_dim_head=64, transformer_mlp_dim=2048,
    edge_hidden_dim=256,
)


# ══════════════════════════════════════════════════════════════════════════════
# Agent wrappers
# ══════════════════════════════════════════════════════════════════════════════

class ExpertPolicy(Policy):
    """Replays stored expert actions."""
    def __init__(self, actions: np.ndarray):
        self.actions = actions
        self.t = 0

    def get_action(self, infos):
        a = self.actions[min(self.t, len(self.actions) - 1)]
        self.t += 1
        return a.reshape(1, 2).astype(np.float32)


class CEMAgentPolicy(Policy):
    """Wraps a closed-loop CEM agent for use with swm.World."""
    def __init__(self):
        self._action = np.zeros(2, dtype=np.float32)

    def get_action(self, infos):
        return self._action.reshape(1, 2)

    def set_action(self, action: np.ndarray):
        self._action = action


class ClosedLoopAgent:
    """
    Full observe-encode-plan-act loop.

    At each step:
      1. Encode current env frame → slots via VideoSAUR
      2. Maintain rolling history buffer (3 frames)
      3. Compute surprise (prediction error)
      4. If system_m and surprise > threshold: adapt model online
      5. Run CEM to get planned action
    """
    def __init__(self, encoder, model, planner, goal_slots, device,
                 system_m=False, optimizer=None, threshold=0.015, adapt_steps=3):
        self.encoder = encoder
        self.model = model
        self.planner = planner
        self.goal_slots = goal_slots.to(device)
        self.device = device
        self.system_m = system_m
        self.optimizer = optimizer
        self.threshold = threshold
        self.adapt_steps = adapt_steps

        self.hist_slots = []
        self.hist_acts = []
        self.last_action = np.zeros(2, dtype=np.float32)

        # Logs
        self.surprise_log = []
        self.adapted_log = []

    def reset(self):
        self.hist_slots.clear()
        self.hist_acts.clear()
        self.last_action = np.zeros(2, dtype=np.float32)
        self.surprise_log.clear()
        self.adapted_log.clear()

    def step(self, frame_rgb: np.ndarray):
        """
        Args:
            frame_rgb: (H, W, 3) uint8 from env.render()
        Returns:
            action: (2,) numpy
            surprise: float
            adapted: bool
        """
        # 1. Encode
        slots = self.encoder.encode(frame_rgb)  # (4, 128)
        curr = slots.unsqueeze(0)  # (1, 4, 128)

        self.hist_slots.append(curr)
        if len(self.hist_slots) > 3:
            self.hist_slots.pop(0)

        act_t = torch.from_numpy(self.last_action).float().unsqueeze(0).to(self.device)
        self.hist_acts.append(act_t)
        if len(self.hist_acts) > 3:
            self.hist_acts.pop(0)

        # Need 3 frames of history before we can plan
        if len(self.hist_slots) < 3:
            self.surprise_log.append(0.0)
            self.adapted_log.append(False)
            return np.zeros(2, dtype=np.float32), 0.0, False

        hist_t = torch.cat(self.hist_slots, dim=0).unsqueeze(0)  # (1, 3, 4, 128)
        hact_t = torch.cat(self.hist_acts, dim=0).unsqueeze(0)   # (1, 3, 2)

        # 2. Surprise: how wrong was our prediction?
        with torch.no_grad():
            pred = self.model.inference(hist_t, actions=hact_t).squeeze(1)
        surprise = F.mse_loss(pred, curr).item()
        adapted = False

        # 3. System M: adapt if surprised
        if self.system_m and self.optimizer and surprise > self.threshold:
            adapted = True
            self.model.train()
            for _ in range(self.adapt_steps):
                self.optimizer.zero_grad()
                p = self.model.differentiable_inference(hist_t, actions=hact_t).squeeze(1)
                loss = F.mse_loss(p, curr)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.1)
                self.optimizer.step()
            self.model.eval()

        # 4. Plan
        planned = self.planner.plan(hist_t, hact_t, self.goal_slots)  # (1, H, 2)
        action = planned[0, 0, :].detach().cpu().numpy()

        # Update action history with the planned action
        self.hist_acts[-1] = torch.from_numpy(action).float().unsqueeze(0).to(self.device)
        self.last_action = action

        self.surprise_log.append(surprise)
        self.adapted_log.append(adapted)
        return action, surprise, adapted


# ══════════════════════════════════════════════════════════════════════════════
# Video assembly
# ══════════════════════════════════════════════════════════════════════════════

def add_info_bar(frame: np.ndarray, label: str, surprise: float | None = None,
                 adapted: bool = False, step: int = 0, total: int = 1) -> Image.Image:
    """Add an info bar below the real env frame."""
    h, w = frame.shape[:2]
    img = Image.new("RGB", (w, h + INFO_BAR_H), (30, 30, 40))
    img.paste(Image.fromarray(frame), (0, 0))
    draw = ImageDraw.Draw(img)

    # Adapted border
    if adapted:
        for i in range(3):
            draw.rectangle([i, i, w - 1 - i, h - 1 - i], outline=(255, 155, 0), width=1)

    # Progress bar
    bar_w = int(step / max(total - 1, 1) * (w - 4))
    draw.rectangle([2, h - 2, 2 + bar_w, h], fill=(100, 100, 130))

    # Info text
    draw.text((6, h + 4), label, fill=(220, 220, 220))
    if surprise is not None:
        draw.text((6, h + 20), f"Pred err: {surprise:.4f}", fill=(160, 160, 175))
    if adapted:
        draw.text((w - 90, h + 20), "⚡ ADAPT", fill=(255, 155, 0))

    return img


def stitch_panels(panels: list[Image.Image], gap: int = 3) -> Image.Image:
    """Stitch multiple panels side-by-side."""
    widths = [p.width for p in panels]
    h = max(p.height for p in panels)
    total_w = sum(widths) + gap * (len(panels) - 1)
    out = Image.new("RGB", (total_w, h), (50, 50, 60))
    x = 0
    for p in panels:
        out.paste(p, (x, 0))
        x += p.width + gap
    return out


def save_gif(frames: list[Image.Image], path: str):
    frames[0].save(
        path, save_all=True, append_images=frames[1:],
        duration=FPS_DELAY, loop=0, optimize=False,
    )
    dur = len(frames) * FPS_DELAY / 1000
    logging.info(f"Saved {path} ({len(frames)} frames, {dur:.1f}s)")


# ══════════════════════════════════════════════════════════════════════════════
# Episode runner
# ══════════════════════════════════════════════════════════════════════════════

def run_episode(
    ep_actions: np.ndarray,
    init_state: np.ndarray,
    goal_slots: torch.Tensor,
    encoder: VideoSAUREncoder,
    frozen_agent: ClosedLoopAgent,
    abm_agent: ClosedLoopAgent,
    max_steps: int = MAX_STEPS,
    render_size: int = RENDER_SIZE,
    ood_block_scale: float | None = None,
):
    """Run 3 parallel envs and capture real rendered frames."""

    # Build reset options
    reset_opts = {"state": init_state}
    if ood_block_scale is not None:
        reset_opts["block"] = {"scale": float(ood_block_scale)}

    expert_pol = ExpertPolicy(ep_actions)
    frozen_pol = CEMAgentPolicy()
    abm_pol = CEMAgentPolicy()

    def make_env(pol):
        e = swm.World(
            "swm/PushT-v1", num_envs=1,
            image_shape=(render_size, render_size),
            max_episode_steps=max_steps + 10,
            verbose=0,
        )
        e.set_policy(pol)
        e.reset(options=reset_opts)
        return e

    env_expert = make_env(expert_pol)
    env_frozen = make_env(frozen_pol)
    env_abm = make_env(abm_pol)

    frozen_agent.reset()
    abm_agent.reset()

    frames = []

    for step_i in range(max_steps):
        # Render real frames from each env
        frame_e = env_expert.envs.envs[0].render()  # (H, W, 3) uint8
        frame_f = env_frozen.envs.envs[0].render()
        frame_a = env_abm.envs.envs[0].render()

        # CEM agents observe and plan
        fz_action, surp_f, _ = frozen_agent.step(frame_f.astype(np.uint8))
        ab_action, surp_a, adapted = abm_agent.step(frame_a.astype(np.uint8))

        frozen_pol.set_action(fz_action)
        abm_pol.set_action(ab_action)

        # Build side-by-side frame from REAL renders
        panel_e = add_info_bar(frame_e, "Expert", step=step_i, total=max_steps)
        panel_f = add_info_bar(frame_f, "Frozen", surp_f, step=step_i, total=max_steps)
        panel_a = add_info_bar(frame_a, "A-B-M", surp_a, adapted, step=step_i, total=max_steps)
        frames.append(stitch_panels([panel_e, panel_f, panel_a]))

        # Step all envs (check terminated first)
        done_e = env_expert.terminateds is not None and bool(env_expert.terminateds[0])
        done_f = env_frozen.terminateds is not None and bool(env_frozen.terminateds[0])
        done_a = env_abm.terminateds is not None and bool(env_abm.terminateds[0])

        if not done_e:
            env_expert.step()
        if not done_f:
            env_frozen.step()
        if not done_a:
            env_abm.step()

        if done_e and done_f and done_a:
            break

    env_expert.close()
    env_frozen.close()
    env_abm.close()

    stats = {
        "frozen_surprise": float(np.mean(frozen_agent.surprise_log)) if frozen_agent.surprise_log else 0,
        "abm_surprise": float(np.mean(abm_agent.surprise_log)) if abm_agent.surprise_log else 0,
        "adaptations": int(sum(abm_agent.adapted_log)),
        "steps": step_i + 1,
    }
    return frames, stats


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Closed-loop Push-T evaluation")
    parser.add_argument("--ckpt", default="/workspace/checkpoints/mechjepa_pusht_act_best.ckpt",
                        help="MechJEPA world model checkpoint")
    parser.add_argument("--encoder", default="/workspace/data/pusht_videosaur_model.ckpt",
                        help="VideoSAUR encoder checkpoint")
    parser.add_argument("--data", default="/workspace/data/pusht_slots_actions.pkl",
                        help="Slot+action dataset (for expert actions and goal slots)")
    parser.add_argument("--state_meta", default="/workspace/data/pusht_expert_state_meta.pkl",
                        help="Expert state metadata (for initial states)")
    parser.add_argument("--out_dir", default="/workspace/results/eval_live",
                        help="Output directory for GIFs")
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--max_steps", type=int, default=MAX_STEPS)
    parser.add_argument("--render_size", type=int, default=RENDER_SIZE)
    parser.add_argument("--threshold", type=float, default=0.015,
                        help="System M surprise threshold")
    parser.add_argument("--ood_block_scale", type=float, default=None,
                        help="OOD block scale (default=30, try 42 for +40%%)")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ── Load data ─────────────────────────────────────────────────────────────
    logging.info("Loading data...")
    with open(args.data, "rb") as f:
        data = pkl.load(f)
    with open(args.state_meta, "rb") as f:
        state_meta = pkl.load(f)

    val = data["val"]
    val_stat = state_meta["val"]
    ep_keys = sorted([k for k in val.keys() if k in val_stat])[:args.episodes]

    # Goal: last frame slots of first episode
    goal_slots = torch.from_numpy(val[ep_keys[0]]["slots"][-1]).float()

    # ── Load encoder ──────────────────────────────────────────────────────────
    logging.info("Loading VideoSAUR encoder...")
    encoder = VideoSAUREncoder.from_ckpt(args.encoder, device=device)

    # ── Load world models ─────────────────────────────────────────────────────
    logging.info("Loading MechJEPA world model...")

    def load_model(ckpt_path):
        m = MechJEPA(**MODEL_CFG)
        m.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=False))
        return m.to(device)

    frozen_model = load_model(args.ckpt).eval()
    frozen_planner = CEMPlanner(
        frozen_model, horizon=10, num_samples=256, num_iterations=5, device=device,
    )

    abm_model = load_model(args.ckpt)
    abm_model.eval()
    abm_optimizer = torch.optim.AdamW([
        {"params": abm_model.codebook.parameters(), "lr": 5e-4},
        {"params": abm_model.predictor.parameters(), "lr": 1e-4},
    ])
    abm_planner = CEMPlanner(
        abm_model, horizon=10, num_samples=256, num_iterations=5, device=device,
    )

    # ── Run episodes ──────────────────────────────────────────────────────────
    all_frames = []
    for ep_i, key in enumerate(ep_keys):
        ep_actions = val[key]["actions"]
        ep_states = val_stat[key]
        init_state = ep_states[0]

        ood_str = f" OOD block_scale={args.ood_block_scale}" if args.ood_block_scale else ""
        logging.info(
            f"▶ Episode {ep_i + 1}/{len(ep_keys)}: {key}{ood_str}\n"
            f"  Init: agent=({init_state[0]:.0f},{init_state[1]:.0f}) "
            f"block=({init_state[2]:.0f},{init_state[3]:.0f})"
        )

        frozen_agent = ClosedLoopAgent(
            encoder, frozen_model, frozen_planner, goal_slots, device,
        )
        abm_agent = ClosedLoopAgent(
            encoder, abm_model, abm_planner, goal_slots, device,
            system_m=True, optimizer=abm_optimizer, threshold=args.threshold,
        )

        frames, stats = run_episode(
            ep_actions, init_state, goal_slots,
            encoder, frozen_agent, abm_agent,
            max_steps=args.max_steps,
            render_size=args.render_size,
            ood_block_scale=args.ood_block_scale,
        )

        logging.info(
            f"  {stats['steps']} steps | "
            f"Frozen surp={stats['frozen_surprise']:.4f}  "
            f"A-B-M surp={stats['abm_surprise']:.4f}  "
            f"adaptations={stats['adaptations']}"
        )

        tag = f"_ood{int(args.ood_block_scale)}" if args.ood_block_scale else ""
        save_gif(frames, os.path.join(args.out_dir, f"pusht_ep{ep_i + 1:02d}{tag}.gif"))
        all_frames.extend(frames)

    save_gif(all_frames, os.path.join(args.out_dir, f"pusht_all{tag if args.ood_block_scale else ''}.gif"))
    logging.info(f"Done. All GIFs saved to {args.out_dir}/")


if __name__ == "__main__":
    main()
