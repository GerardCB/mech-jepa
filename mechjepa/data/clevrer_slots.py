"""
CLEVRER Slot Dataset for MechJEPA.

Loads pre-extracted VideoSAUR slot embeddings from pickle files.
Same data format and pipeline as the C-JEPA baseline for fair comparison.
"""

import torch
import numpy as np


class ClevrerSlotDataset(torch.utils.data.Dataset):
    """
    Dataset for pre-extracted slot embeddings from CLEVRER videos.

    Each video maps to a slot tensor of shape [num_frames, num_slots, slot_dim].
    We extract clips with frameskip for training.

    Args:
        data: dict mapping video names to slot tensors
        split: 'train', 'val', or 'test'
        history_size: number of history frames per clip
        num_preds: number of future frames to predict
        frameskip: temporal stride for frame sampling
    """

    def __init__(
        self,
        data: dict,
        split: str,
        history_size: int = 3,
        num_preds: int = 1,
        frameskip: int = 5,
    ):
        super().__init__()
        self.data = data
        self.split = split
        self.history_size = history_size
        self.num_preds = num_preds
        self.frameskip = frameskip

        # Total frames needed per clip (before frameskip)
        self.num_steps = history_size + num_preds
        self.clip_len = self.frameskip * self.num_steps

        # Build index: list of (video_key, start_frame) tuples
        self.video_keys = list(self.data.keys())
        self.samples = []

        for video_key in self.video_keys:
            slots = self.data[video_key]
            num_frames = slots.shape[0]
            num_valid_starts = max(0, num_frames - self.clip_len + 1)

            for start in range(num_valid_starts):
                self.samples.append((video_key, start))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        video_key, start_frame = self.samples[idx]
        slots = self.data[video_key]

        # Extract clip with frameskip
        end_frame = start_frame + self.clip_len
        clip_slots = slots[start_frame:end_frame:self.frameskip]

        if not isinstance(clip_slots, torch.Tensor):
            clip_slots = torch.tensor(clip_slots, dtype=torch.float32)

        return {"embed": clip_slots}  # [history_size + num_preds, num_slots, slot_dim]


class PushTSlotDataset(torch.utils.data.Dataset):
    """
    Dataset for pre-extracted slot embeddings + actions from Push-T.

    Handles action-frameskip alignment: with frameskip=5, each transition
    between subsampled slot frames spans 5 raw actions. These are averaged
    into action blocks following LeWorldModel's approach.

    Returns:
        {"embed": (T, S, D), "actions": (T, action_dim)}
        where each action[t] is the averaged action for the transition
        from slot frame t to slot frame t+1.

    Args:
        slots_data: dict mapping episode names to slot tensors (T_raw, S, D)
        actions_data: dict mapping episode names to action tensors (T_raw, action_dim)
                      If None, returns zero actions (for backward compat)
        split: 'train', 'val', or 'test'
        history_size: number of history frames
        num_preds: number of prediction frames
        frameskip: temporal stride (actions between frames are averaged)
        action_dim: dimension of action space (default 2 for Push-T)
    """

    def __init__(
        self,
        slots_data: dict,
        actions_data: dict | None = None,
        split: str = "train",
        history_size: int = 3,
        num_preds: int = 1,
        frameskip: int = 5,
        action_dim: int = 2,
    ):
        super().__init__()
        self.slots_data = slots_data
        self.actions_data = actions_data
        self.split = split
        self.history_size = history_size
        self.num_preds = num_preds
        self.frameskip = frameskip
        self.action_dim = action_dim

        self.num_steps = history_size + num_preds
        self.clip_len = self.frameskip * self.num_steps

        self.episode_keys = list(self.slots_data.keys())
        self.samples = []

        for key in self.episode_keys:
            slots = self.slots_data[key]
            num_frames = slots.shape[0]
            num_valid = max(0, num_frames - self.clip_len + 1)
            for start in range(num_valid):
                self.samples.append((key, start))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        key, start = self.samples[idx]
        slots = self.slots_data[key]

        # Extract slot clip with frameskip
        end = start + self.clip_len
        clip = slots[start:end:self.frameskip]  # (num_steps, S, D)

        if not isinstance(clip, torch.Tensor):
            clip = torch.tensor(clip, dtype=torch.float32)

        result = {"embed": clip}

        # Extract and aggregate actions
        if self.actions_data is not None and key in self.actions_data:
            actions = self.actions_data[key]

            # For each transition between subsampled slot frames,
            # average the frameskip raw actions in between
            action_blocks = []
            for t in range(self.num_steps):
                raw_start = start + t * self.frameskip
                raw_end = min(raw_start + self.frameskip, len(actions))

                if raw_start < len(actions):
                    block = actions[raw_start:raw_end]
                    if not isinstance(block, torch.Tensor):
                        block = torch.tensor(block, dtype=torch.float32)
                    # Average the actions in this block
                    avg_action = block.mean(dim=0)
                else:
                    avg_action = torch.zeros(self.action_dim)

                action_blocks.append(avg_action)

            result["actions"] = torch.stack(action_blocks)  # (num_steps, action_dim)
        else:
            # No actions: return zeros
            result["actions"] = torch.zeros(self.num_steps, self.action_dim)

        return result
