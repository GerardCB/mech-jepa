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
    Dataset for pre-extracted slot embeddings from Push-T environment.

    Same interface as ClevrerSlotDataset for consistency.

    Args:
        data: dict mapping episode names to slot tensors
        split: 'train', 'val', or 'test'
        history_size: number of history frames
        num_preds: number of prediction frames
        frameskip: temporal stride
    """

    def __init__(
        self,
        data: dict,
        split: str,
        history_size: int = 3,
        num_preds: int = 1,
        frameskip: int = 1,
    ):
        super().__init__()
        self.data = data
        self.split = split
        self.history_size = history_size
        self.num_preds = num_preds
        self.frameskip = frameskip

        self.num_steps = history_size + num_preds
        self.clip_len = self.frameskip * self.num_steps

        self.episode_keys = list(self.data.keys())
        self.samples = []

        for key in self.episode_keys:
            slots = self.data[key]
            num_frames = slots.shape[0]
            num_valid = max(0, num_frames - self.clip_len + 1)
            for start in range(num_valid):
                self.samples.append((key, start))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        key, start = self.samples[idx]
        slots = self.data[key]

        end = start + self.clip_len
        clip = slots[start:end:self.frameskip]

        if not isinstance(clip, torch.Tensor):
            clip = torch.tensor(clip, dtype=torch.float32)

        return {"embed": clip}
