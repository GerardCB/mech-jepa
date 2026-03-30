import torch
import torch.nn as nn
import numpy as np
import pickle as pkl
import os
from loguru import logger as logging
from tqdm import tqdm
from omegaconf import OmegaConf

from mechjepa.model import MechJEPA
from mechjepa.planner import CEMPlanner

def load_model(ckpt_path, device="cuda"):
    """Load model from checkpoint."""
    # Build default config (as used in train_pusht.py)
    # This should match your pusht.yaml
    model = MechJEPA(
        num_slots=4,
        slot_dim=128,
        num_mechanisms=8,
        history_frames=3,
        pred_frames=1,
        edge_hidden_dim=256,
        transformer_depth=6,
        transformer_heads=16,
        transformer_dim_head=64,
        transformer_mlp_dim=2048,
        action_dim=2,
    )
    if os.path.exists(ckpt_path):
        state_dict = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(state_dict)
        logging.info(f"Loaded checkpoint from {ckpt_path}")
    else:
        logging.warning(f"No checkpoint found at {ckpt_path}, using random initialization")
    model.to(device).eval()
    return model

def run_planning_playback(
    model, 
    data_path, 
    episode_id=0, 
    horizon=16, 
    num_samples=512, 
    device="cuda"
):
    """
    Run 'playback' planning: 
    1. Take real history from episode_id
    2. Set real future frame as goal
    3. Plan actions with model
    4. Compare predicted outcome with real outcome
    """
    with open(data_path, "rb") as f:
        data = pkl.load(f)
    
    val_data = data.get("val", data)
    ep_keys = sorted(list(val_data.keys()))
    if episode_id >= len(ep_keys):
        logging.error(f"Episode {episode_id} not found in validation data")
        return
    
    key = ep_keys[episode_id]
    ep = val_data[key]
    slots = torch.from_numpy(ep["slots"]).float().to(device) # (T, S, D)
    actions = torch.from_numpy(ep["actions"]).float().to(device) # (T, 2)
    
    T = slots.shape[0]
    history_size = 3
    
    # Start planning from t=history_size
    # Goal is the state at t=history_size + horizon
    start_t = 0
    goal_t = min(start_t + history_size + horizon, T - 1)
    
    history = slots[start_t:start_t + history_size].unsqueeze(0) # (1, T_hist, S, D)
    hist_actions = actions[start_t:start_t + history_size].unsqueeze(0) # (1, T_hist, action_dim)
    goal_slots = slots[goal_t] # (S, D)
    
    planner = CEMPlanner(
        model, 
        horizon=goal_t - (start_t + history_size), 
        num_samples=num_samples,
        device=device
    )
    
    logging.info(f"Planning from episode {key}, goal at t={goal_t}")
    planned_actions = planner.plan(history, hist_actions, goal_slots)
    
    # Validation metrics
    # Compare with ground truth actions if possible, 
    # but CEM finds *any* sequence to reach goal, not necessarily the expert's one.
    # The key is to check if rolling out these actions leads to the goal in latent space.
    
    with torch.no_grad():
        curr_hist = history
        curr_acts = hist_actions
        for h in range(planned_actions.shape[1]):
            # Action for current step
            step_act = planned_actions[:, h:h+1, :] # (1, 1, 2)
            
            # Shift action buffer
            curr_acts = torch.cat([curr_acts[:, 1:, :], step_act], dim=1) # (1, T_hist, 2)
            
            next_pred = model.inference(curr_hist, actions=curr_acts)
            curr_hist = torch.cat([curr_hist[:, 1:], next_pred], dim=1)
            
        final_pred = curr_hist[:, -1, :, :]
        error = (final_pred - goal_slots).pow(2).mean().item()
        logging.info(f"Planning done. Final latent error: {error:.6f}")
        
    return error

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, default="/workspace/checkpoints/mechjepa_pusht_act_best.ckpt")
    parser.add_argument("--data", type=str, default="/workspace/data/pusht_slots_actions.pkl")
    parser.add_argument("--ep", type=int, default=0)
    parser.add_argument("--horizon", type=int, default=10)
    args = parser.parse_args()
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_model(args.ckpt, device=device)
    run_planning_playback(model, args.data, episode_id=args.ep, horizon=args.horizon, device=device)
