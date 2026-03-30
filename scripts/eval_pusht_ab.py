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

def run_ab_loop(
    model_path, 
    encoder_path, 
    data_path, 
    num_episodes=5, 
    max_steps=200, 
    horizon=16, 
    device="cuda"
):
    """
    Run the A-B loop: observe -> encode -> plan -> act.
    """
    # 1. Load Models
    logging.info("Loading models...")
    # These parameters should match your training setup
    world_model = MechJEPA(
        num_slots=4, slot_dim=128, num_mechanisms=8, 
        history_frames=3, pred_frames=1, action_dim=2
    )
    if os.path.exists(model_path):
        world_model.load_state_dict(torch.load(model_path, map_location=device))
    world_model.to(device).eval()
    
    encoder = VideoSAUREncoder(img_size=96, num_slots=4, slot_dim=128)
    # Note: official C-JEPA Push-T uses 96x96 pixels for slots
    if os.path.exists(encoder_path):
        # Placeholder: load weights logic
        pass 
    encoder.to(device).eval()
    
    # 2. Get a goal configuration from the validation data
    with open(data_path, "rb") as f:
        data = pkl.load(f)
    val_data = data.get("val", data)
    ep_keys = sorted(list(val_data.keys()))
    goal_ep = val_data[ep_keys[0]]
    # Typically, the goal in Push-T is the last frame of an expert episode
    goal_slots = torch.from_numpy(goal_ep["slots"][-1]).float().to(device) # (S, D)
    
    # 3. Setup Environment
    logging.info("Setting up Push-T environment...")
    world = swm.World('swm/PushT-v1', num_envs=1, image_shape=(96, 96), goal_conditioned=False)
    
    planner = CEMPlanner(world_model, horizon=horizon, device=device)
    
    successes = 0
    for ep_idx in range(num_episodes):
        obs, _ = world.reset()
        history_slots = []
        history_actions = []
        
        logging.info(f"Starting Episode {ep_idx+1}")
        pbar = tqdm(range(max_steps))
        for step in pbar:
            # 4. Observe and Encode
            pixels = torch.from_numpy(obs['pixels']).float().to(device) # (1, 96, 96, 3)
            pixels = pixels.permute(0, 3, 1, 2) / 255.0 # (1, 3, 96, 96)
            
            with torch.no_grad():
                curr_slots = encoder(pixels) # (1, S, D)
            
            history_slots.append(curr_slots)
            if len(history_slots) > 3:
                history_slots.pop(0)
            
            if len(history_slots) < 3:
                # Need history to plan
                action = torch.zeros(1, 2, device=device)
            else:
                while len(history_actions) < 3:
                    history_actions.insert(0, torch.zeros(1, 2, device=device))
                    
                # 5. Plan
                hist_tensor = torch.cat(history_slots, dim=0).unsqueeze(0) # (1, T_hist, S, D)
                hist_act_tensor = torch.cat(history_actions[-3:], dim=0).unsqueeze(0) # (1, T_hist, 2)
                
                planned_actions = planner.plan(hist_tensor, hist_act_tensor, goal_slots)
                action = planned_actions[:, 0, :] # Execute first action
            
            history_actions.append(action)
            if len(history_actions) > 3:
                history_actions.pop(0)
                
            # 6. Step
            obs, reward, terminated, truncated, _ = world.step(action)
            
            pbar.set_postfix({"reward": f"{reward.mean().item():.4f}"})
            if terminated.any() or truncated.any():
                if reward.mean() > 0.9: # Threshold for success
                    successes += 1
                break
        
    logging.info(f"Evaluation complete. Success rate: {successes/num_episodes:.2f}")
    world.close()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, default="/workspace/checkpoints/mechjepa_pusht_act_best.ckpt")
    parser.add_argument("--encoder", type=str, default="/workspace/data/pusht_videosaur_model.ckpt")
    parser.add_argument("--data", type=str, default="/workspace/data/pusht_slots_actions.pkl")
    args = parser.parse_args()
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    run_ab_loop(args.ckpt, args.encoder, args.data, device=device)
