"""
eval_live_pusht.py — Push-T evaluation using SWM infrastructure.

Uses SWM's CEMSolver, WorldModelPolicy, evaluate_from_dataset(), and
record_video() for standardised, comparable evaluation.

Two conditions:
  1. Frozen: WorldModelPolicy + CEMSolver (no adaptation)
  2. A-B-M:  ABMPolicy + CEMSolver (surprise-triggered System M)

Usage:
  # Standard evaluation (3 episodes, with video)
  python scripts/eval_live_pusht.py --episodes 3

  # OOD: 40% bigger block
  python scripts/eval_live_pusht.py --episodes 3 --ood_block_scale 42

  # Just record a video (no metrics)
  python scripts/eval_live_pusht.py --video_only --episodes 1
"""

import os
import sys
import argparse

os.environ["SDL_VIDEODRIVER"] = "offscreen"

import numpy as np
import torch
from loguru import logger as logging

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from mechjepa.model import MechJEPA
from mechjepa.encoder import VideoSAUREncoder
from mechjepa.cost_model import MechJEPACostModel
from mechjepa.abm_policy import ABMPolicy

import stable_worldmodel as swm
from stable_worldmodel.solver import CEMSolver
from stable_worldmodel.policy import WorldModelPolicy, PlanConfig

# ── Model config (must match training) ────────────────────────────────────────
MODEL_CFG = dict(
    num_slots=4, slot_dim=128, num_mechanisms=8,
    history_frames=3, pred_frames=1, action_dim=2,
    transformer_depth=6, transformer_heads=16,
    transformer_dim_head=64, transformer_mlp_dim=2048,
    edge_hidden_dim=256,
)


def load_model(ckpt_path: str, device: str) -> MechJEPA:
    """Load a MechJEPA model from checkpoint."""
    m = MechJEPA(**MODEL_CFG)
    m.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=False))
    return m.to(device)


def main():
    parser = argparse.ArgumentParser(description="Push-T evaluation with SWM")
    parser.add_argument("--ckpt", default="/workspace/checkpoints/mechjepa_pusht_act_best.ckpt")
    parser.add_argument("--encoder", default="/workspace/data/pusht_videosaur_model.ckpt")
    parser.add_argument("--out_dir", default="/workspace/results/eval_swm")
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--max_steps", type=int, default=200)
    parser.add_argument("--render_size", type=int, default=256)
    parser.add_argument("--threshold", type=float, default=0.015)
    parser.add_argument("--ood_block_scale", type=float, default=None)
    parser.add_argument("--video_only", action="store_true")
    # CEM hyperparameters
    parser.add_argument("--horizon", type=int, default=10)
    parser.add_argument("--cem_samples", type=int, default=300)
    parser.add_argument("--cem_steps", type=int, default=5)
    parser.add_argument("--cem_topk", type=int, default=30)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    img_shape = (args.render_size, args.render_size)

    # ── Load encoder ──────────────────────────────────────────────────────────
    logging.info("Loading VideoSAUR encoder...")
    encoder = VideoSAUREncoder.from_ckpt(args.encoder, device=device)

    # ── Load world models ─────────────────────────────────────────────────────
    logging.info("Loading MechJEPA world model...")
    frozen_model = load_model(args.ckpt, device).eval()
    abm_model = load_model(args.ckpt, device).eval()

    # ── Build SWM cost models ─────────────────────────────────────────────────
    frozen_cost = MechJEPACostModel(frozen_model, encoder)
    abm_cost = MechJEPACostModel(abm_model, encoder)

    # ── Build SWM solvers ─────────────────────────────────────────────────────
    frozen_solver = CEMSolver(
        model=frozen_cost,
        num_samples=args.cem_samples,
        n_steps=args.cem_steps,
        topk=args.cem_topk,
        device=device,
    )
    abm_solver = CEMSolver(
        model=abm_cost,
        num_samples=args.cem_samples,
        n_steps=args.cem_steps,
        topk=args.cem_topk,
        device=device,
    )

    # ── Build SWM policies ────────────────────────────────────────────────────
    plan_config = PlanConfig(
        horizon=args.horizon,
        receding_horizon=1,
        history_len=3,
        warm_start=True,
    )

    frozen_policy = WorldModelPolicy(solver=frozen_solver, config=plan_config)
    abm_policy = ABMPolicy(
        solver=abm_solver,
        config=plan_config,
        cost_model=abm_cost,
        world_model=abm_model,
        threshold=args.threshold,
    )

    # ── Reset options (OOD variation) ─────────────────────────────────────────
    reset_opts = {}
    if args.ood_block_scale is not None:
        reset_opts["variation"] = ["block.scale"]
        logging.info(f"OOD mode: block.scale = {args.ood_block_scale}")

    def make_world():
        return swm.World(
            "swm/PushT-v1",
            num_envs=1,
            image_shape=img_shape,
            max_episode_steps=args.max_steps,
            goal_conditioned=True,
            verbose=0,
        )

    # ══════════════════════════════════════════════════════════════════════════
    # Condition 1: Frozen model (no System M)
    # ══════════════════════════════════════════════════════════════════════════
    logging.info("▶ Condition 1: Frozen model (no System M)")
    world_frozen = make_world()
    world_frozen.set_policy(frozen_policy)

    if args.video_only:
        frozen_vid = os.path.join(args.out_dir, "frozen")
        os.makedirs(frozen_vid, exist_ok=True)
        world_frozen.record_video(
            frozen_vid, max_steps=args.max_steps, fps=12,
            extension="gif", options=reset_opts or None,
        )
        logging.info(f"Saved Frozen video to {frozen_vid}/")
    else:
        # Standard SWM online evaluation (same as LeWorldModel benchmark)
        results_frozen = world_frozen.evaluate(
            episodes=args.episodes,
            seed=42,
            options=reset_opts or None,
        )
        logging.info(f"Frozen results: {results_frozen}")

    # Also record a video for visualization
    frozen_vid = os.path.join(args.out_dir, "frozen")
    os.makedirs(frozen_vid, exist_ok=True)
    world_frozen.record_video(
        frozen_vid, max_steps=args.max_steps, fps=12,
        extension="gif", options=reset_opts or None,
    )
    logging.info(f"Saved Frozen video to {frozen_vid}/")
    world_frozen.close()

    # ══════════════════════════════════════════════════════════════════════════
    # Condition 2: A-B-M model (System M active)
    # ══════════════════════════════════════════════════════════════════════════
    logging.info("▶ Condition 2: A-B-M Agent (System M active)")
    world_abm = make_world()
    world_abm.set_policy(abm_policy)

    if args.video_only:
        abm_vid = os.path.join(args.out_dir, "abm")
        os.makedirs(abm_vid, exist_ok=True)
        world_abm.record_video(
            abm_vid, max_steps=args.max_steps, fps=12,
            extension="gif", options=reset_opts or None,
        )
        logging.info(f"Saved A-B-M video to {abm_vid}/")
    else:
        results_abm = world_abm.evaluate(
            episodes=args.episodes,
            seed=42,
            options=reset_opts or None,
        )
        logging.info(f"A-B-M results: {results_abm}")

    # Also record a video
    abm_vid = os.path.join(args.out_dir, "abm")
    os.makedirs(abm_vid, exist_ok=True)
    world_abm.record_video(
        abm_vid, max_steps=args.max_steps, fps=12,
        extension="gif", options=reset_opts or None,
    )
    logging.info(f"Saved A-B-M video to {abm_vid}/")
    world_abm.close()

    # ── Summary ───────────────────────────────────────────────────────────────
    if not args.video_only:
        abm_metrics = abm_policy.get_metrics()
        logging.info(f"System M metrics: {abm_metrics}")

        logging.info("\n" + "=" * 60)
        logging.info("SUMMARY")
        logging.info("=" * 60)
        if 'success_rate' in results_frozen:
            logging.info(f"  Frozen success rate: {results_frozen['success_rate']:.1f}%")
            logging.info(f"  A-B-M  success rate: {results_abm['success_rate']:.1f}%")
        logging.info(f"  A-B-M mean surprise: {abm_metrics['mean_surprise']:.4f}")
        logging.info(f"  A-B-M adaptations:   {abm_metrics['total_adaptations']}")
        logging.info("=" * 60)

    logging.info(f"All outputs saved to {args.out_dir}/")


if __name__ == "__main__":
    main()
