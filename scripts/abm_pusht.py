import torch
import torch.nn as nn
import numpy as np
import gymnasium as gym
import pickle as pkl
import os
from loguru import logger as logging
from tqdm import tqdm
import stable_worldmodel as swm

from mechjepa.model import MechJEPA
from mechjepa.planner import CEMPlanner
from mechjepa.encoder import VideoSAUREncoder
from mechjepa.system_m import compute_surprise_from_prediction

def online_adaptation_step(model, optimizer, history_slots, hist_actions, actual_next, steps=5):
    """
    Run gradient updates on the surprise boundary.
    history_slots: (1, T_hist, S, D)
    hist_actions: (1, T_hist, action_dim)
    actual_next: (1, S, D)
    """
    model.train()
    for _ in range(steps):
        optimizer.zero_grad()
        
        # Predict next frame using action-conditioned dynamics
        pred_next = model.inference(history_slots, actions=hist_actions) # (1, 1, S, D)
        pred_next = pred_next.squeeze(1) # (1, S, D)
        
        # Simple MSE loss against the real observation
        loss = torch.nn.functional.mse_loss(pred_next, actual_next)
        loss.backward()
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
        optimizer.step()
        
    model.eval()

def run_abm_demo(
    model_path, 
    encoder_path, 
    data_path, 
    num_episodes=5, 
    max_steps=200, 
    horizon=16, 
    adaptation_steps=5,
    surprise_threshold=0.05,
    device="cuda"
):
    """
    Run the full A-B-M loop:
    1. System A: World Model
    2. System B: Planner
    3. System M: Surprise monitor -> Adaptation trigger
    """
    logging.info("Loading models...")
    world_model = MechJEPA(
        num_slots=4, slot_dim=128, num_mechanisms=8, 
        history_frames=3, pred_frames=1, action_dim=2
    )
    if os.path.exists(model_path):
        world_model.load_state_dict(torch.load(model_path, map_location=device))
    world_model.to(device).eval()
    
    # We only optimize the bottleneck (and optionally predictor) during adaptation
    # Often, updating just the codebook is faster and prevents catastrophic forgetting
    optimizer = torch.optim.AdamW([
        {'params': world_model.codebook.parameters(), 'lr': 1e-3},
        {'params': world_model.predictor.parameters(), 'lr': 1e-4}
    ])
    
    encoder = VideoSAUREncoder(img_size=96, num_slots=4, slot_dim=128)
    if os.path.exists(encoder_path):
        # Implementation depends on exact ckpt mapping
        pass 
    encoder.to(device).eval()
    
    with open(data_path, "rb") as f:
        data = pkl.load(f)
    val_data = data.get("val", data)
    ep_keys = sorted(list(val_data.keys()))
    goal_slots = torch.from_numpy(val_data[ep_keys[0]]["slots"][-1]).float().to(device)
    
    # Setup OOD Environment (e.g. 2x block size to simulate mass/friction change if physics isn't exported)
    logging.info("Setting up Push-T OOD environment (Double block scale)...")
    world = swm.World('swm/PushT-v1', num_envs=1, image_shape=(96, 96), goal_conditioned=False)
    
    # Let's apply severe distribution shift: scale block up and shift start position
    from stable_worldmodel.world import PlanConfig
    ood_cfg = PlanConfig.get_default('swm/PushT-v1')
    ood_cfg.block.scale.range = [40.0, 60.0]  # Standard is usually 20-30
    
    system_m = world_model.system_m
    system_m.surprise_threshold = surprise_threshold
    planner = CEMPlanner(world_model, horizon=horizon, device=device)
    
    successes = 0
    total_adaptations = 0
    
    for ep_idx in range(num_episodes):
        obs, _ = world.reset(options={'config': ood_cfg})
        history_slots = []
        history_actions = []
        
        logging.info(f"Starting OOD Episode {ep_idx+1}")
        pbar = tqdm(range(max_steps))
        
        for step in pbar:
            # 1. Observe
            pixels = torch.from_numpy(obs['pixels']).float().to(device)
            pixels = pixels.permute(0, 3, 1, 2) / 255.0
            
            with torch.no_grad():
                curr_slots = encoder(pixels)
            
            # --- System M Monitoring (evaluate previous action's outcome) ---
            if len(history_slots) == 3 and len(history_actions) == 3:
                hist_tensor = torch.cat(history_slots, dim=0).unsqueeze(0)
                hist_act_tensor = torch.cat(history_actions, dim=0).unsqueeze(0)
                
                # Check surprise between our expected outcome and reality
                surprise_result = compute_surprise_from_prediction(
                    world_model.predictor, 
                    world_model.codebook,
                    hist_tensor,
                    curr_slots.squeeze(0) # actual next state
                )
                
                max_surprise = surprise_result["max_surprise"].item()
                
                # If surprising, trigger online adaptation
                if system_m.should_learn(max_surprise):
                    total_adaptations += 1
                    pbar.set_description(f"ADAPTING (Surprise: {max_surprise:.3f})")
                    online_adaptation_step(
                        world_model, optimizer, 
                        hist_tensor, hist_act_tensor, curr_slots.squeeze(0),
                        steps=adaptation_steps
                    )
                    system_m.update_threshold() # auto-adjust threshold
            
            # Update history buffers for the *current* frame
            history_slots.append(curr_slots)
            if len(history_slots) > 3:
                history_slots.pop(0)
            
            # 2. Plan and Act
            if len(history_slots) < 3:
                action = torch.zeros(1, 2, device=device)
            else:
                while len(history_actions) < 3:
                    history_actions.insert(0, torch.zeros(1, 2, device=device))
                    
                hist_tensor = torch.cat(history_slots, dim=0).unsqueeze(0)
                hist_act_tensor = torch.cat(history_actions[-3:], dim=0).unsqueeze(0)
                
                planned_actions = planner.plan(hist_tensor, hist_act_tensor, goal_slots)
                action = planned_actions[:, 0, :]
            
            history_actions.append(action)
            if len(history_actions) > 3:
                history_actions.pop(0)
                
            obs, reward, terminated, truncated, _ = world.step(action)
            
            pbar.set_postfix({"reward": f"{reward.mean().item():.3f}", "adapts": total_adaptations})
            if terminated.any() or truncated.any():
                if reward.mean() > 0.9:
                    successes += 1
                break
        
    logging.info(f"OOD Eval complete. Success: {successes}/{num_episodes}. Total Adaptations: {total_adaptations}")
    world.close()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, default="/workspace/checkpoints/mechjepa_pusht_act_best.ckpt")
    parser.add_argument("--encoder", type=str, default="/workspace/data/pusht_videosaur_model.ckpt")
    parser.add_argument("--data", type=str, default="/workspace/data/pusht_slots_actions.pkl")
    args = parser.parse_args()
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    run_abm_demo(args.ckpt, args.encoder, args.data, device=device)
