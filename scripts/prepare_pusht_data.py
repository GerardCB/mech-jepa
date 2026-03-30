"""
Push-T Data Preparation: Merge slot embeddings with actions.

Downloads C-JEPA's pre-extracted Push-T VideoSAUR slots from HuggingFace
and extracts actions from the stable_worldmodel HDF5 dataset. Produces a
combined pickle with both slots and actions aligned by episode.

Usage (on RunPod):
    python scripts/prepare_pusht_data.py \
        --output /workspace/data/pusht_slots_actions.pkl

The script will:
1. Download pusht_videosaur_slots.pkl from HuggingFace (~500MB)
2. Download pusht_expert HDF5 from stable_worldmodel (~2GB)
3. Align actions with slot episodes by filename/index
4. Save combined pickle: {"train": {"slots": {...}, "actions": {...}}, "val": {...}}
"""

import argparse
import os
import pickle
import sys

import numpy as np
import torch


def download_slots(output_dir: str) -> str:
    """Download Push-T VideoSAUR slots from C-JEPA's HuggingFace."""
    slots_path = os.path.join(output_dir, "pusht_videosaur_slots.pkl")
    if os.path.exists(slots_path):
        print(f"Slots already downloaded: {slots_path}")
        return slots_path

    url = "https://huggingface.co/HazelNam/CJEPA/resolve/main/pusht_videosaur_slots.pkl"
    print(f"Downloading Push-T slots from HuggingFace...")
    os.system(f"wget -q --show-progress -O {slots_path} '{url}'")

    if not os.path.exists(slots_path) or os.path.getsize(slots_path) < 1000:
        print("ERROR: Download failed. Try manual download:")
        print(f"  wget -O {slots_path} '{url}'")
        sys.exit(1)

    return slots_path


def download_pusht_actions(output_dir: str) -> str:
    """
    Download Push-T expert demonstrations from stable_worldmodel.

    The data is in HDF5 format with episodes containing:
    - pixels: (T, C, H, W) video frames
    - action: (T, 2) push actions
    - state: (T, 5) agent+block state [agent_x, agent_y, block_x, block_y, block_angle]
    """
    data_dir = os.path.join(output_dir, "pusht_raw")
    os.makedirs(data_dir, exist_ok=True)

    # Try to use stable_worldmodel's download utility
    try:
        import stable_worldmodel as swm
        dataset = swm.data.HDF5Dataset(
            dataset="pusht_expert",
            root="~/.stable_worldmodel",
            keys_to_load=["action"],
        )
        print(f"Loaded Push-T actions via stable_worldmodel: {len(dataset)} episodes")
        return "swm"
    except ImportError:
        print("stable_worldmodel not available, trying alternative sources...")

    # Alternative: download from Diffusion Policy's zarr
    zarr_path = os.path.join(data_dir, "pusht_cchi_v7_replay.zarr")
    if not os.path.exists(zarr_path):
        url = "https://diffusion-policy.cs.columbia.edu/data/training/pusht.zip"
        zip_path = os.path.join(data_dir, "pusht.zip")
        print(f"Downloading Push-T data from Diffusion Policy...")
        os.system(f"wget -q --show-progress -O {zip_path} '{url}'")
        os.system(f"cd {data_dir} && python3 -c 'import zipfile; zipfile.ZipFile(\"{zip_path}\").extractall()'")
        os.remove(zip_path) if os.path.exists(zip_path) else None

    return zarr_path


def extract_actions_from_zarr(zarr_path: str) -> dict:
    """Extract per-episode actions from the Diffusion Policy zarr."""
    import zarr

    root = zarr.open(zarr_path, "r")
    print(f"Zarr keys: {list(root.keys())}")
    print(f"Data keys: {list(root['data'].keys())}")

    # Get episode boundaries
    actions = np.array(root["data"]["action"])
    episode_ends = np.array(root["meta"]["episode_ends"])
    n_episodes = len(episode_ends)

    print(f"Total timesteps: {len(actions)}, Episodes: {n_episodes}")

    episodes = {}
    start = 0
    for i, end in enumerate(episode_ends):
        episodes[i] = actions[start:end]
        start = end

    return episodes


def extract_actions_from_swm() -> dict:
    """Extract per-episode actions from stable_worldmodel."""
    import stable_worldmodel as swm

    dataset = swm.data.HDF5Dataset(
        dataset="pusht_expert",
        root="~/.stable_worldmodel",
        keys_to_load=["action"],
        sequence_length=None,  # full episodes
    )

    episodes = {}
    for i in range(len(dataset)):
        sample = dataset[i]
        episodes[i] = np.array(sample["action"])

    return episodes


def align_slots_actions(slots_data: dict, action_episodes: dict) -> dict:
    """
    Align slot embeddings with actions by episode index.

    Slot pickle keys are like '0_pixels.mp4', '1_pixels.mp4', etc.
    Action episodes are indexed by integer.

    The slot frames correspond to video frames at some frameskip — the raw
    actions have one action per raw frame. We store the raw actions and let
    the PushTSlotDataset handle frameskip averaging.
    """
    aligned = {}
    unmatched = 0

    for key in sorted(slots_data.keys()):
        # Extract episode index from key like '0_pixels.mp4' or '0.mp4'
        try:
            ep_idx = int(key.split("_")[0].split(".")[0])
        except (ValueError, IndexError):
            unmatched += 1
            continue

        if ep_idx in action_episodes:
            slots = slots_data[key]
            actions = action_episodes[ep_idx]

            # Verify temporal alignment: slots may be subsampled
            T_slots = slots.shape[0]
            T_actions = len(actions)

            if T_actions < T_slots:
                print(f"  WARNING: ep {ep_idx}: actions ({T_actions}) < slots ({T_slots}), padding")
                pad = np.zeros((T_slots - T_actions, actions.shape[-1]))
                actions = np.concatenate([actions, pad])

            aligned[key] = {
                "slots": slots,  # (T_slots, N, 128)
                "actions": actions[:T_slots * 1],  # raw actions (may be more than T_slots)
            }
        else:
            unmatched += 1

    print(f"  Aligned {len(aligned)} episodes, {unmatched} unmatched")
    return aligned


def main():
    parser = argparse.ArgumentParser(description="Prepare Push-T slot + action data")
    parser.add_argument("--output", type=str, default="/workspace/data/pusht_slots_actions.pkl")
    parser.add_argument("--slots_only", action="store_true", help="Skip action extraction (testing)")
    args = parser.parse_args()

    output_dir = os.path.dirname(args.output)
    os.makedirs(output_dir, exist_ok=True)

    # 1. Download slots
    print("\n[1/3] Downloading Push-T VideoSAUR slots...")
    slots_path = download_slots(output_dir)

    with open(slots_path, "rb") as f:
        slots_data = pickle.load(f)

    print(f"  Splits: {list(slots_data.keys())}")
    for split, data in slots_data.items():
        keys = list(data.keys())
        print(f"  {split}: {len(keys)} episodes")
        if keys:
            sample = data[keys[0]]
            print(f"    Sample shape: {sample.shape}")  # (T, N, 128)

    if args.slots_only:
        print("\n--slots_only: Saving slots without actions")
        combined = {}
        for split in slots_data:
            combined[split] = {"slots": slots_data[split], "actions": None}
        with open(args.output, "wb") as f:
            pickle.dump(combined, f)
        print(f"Saved to {args.output}")
        return

    # 2. Get actions
    print("\n[2/3] Getting Push-T actions...")
    action_source = download_pusht_actions(output_dir)

    if action_source == "swm":
        action_episodes = extract_actions_from_swm()
    else:
        action_episodes = extract_actions_from_zarr(action_source)

    print(f"  Extracted {len(action_episodes)} action episodes")
    if action_episodes:
        sample_key = list(action_episodes.keys())[0]
        print(f"  Sample: ep {sample_key}, shape {action_episodes[sample_key].shape}")

    # 3. Align and combine
    print("\n[3/3] Aligning slots with actions...")
    combined = {}
    for split in slots_data:
        print(f"  Processing {split}...")
        combined[split] = align_slots_actions(slots_data[split], action_episodes)

    # Save
    with open(args.output, "wb") as f:
        pickle.dump(combined, f, protocol=pickle.HIGHEST_PROTOCOL)

    file_size = os.path.getsize(args.output) / 1e9
    print(f"\nSaved to {args.output} ({file_size:.2f} GB)")

    # Summary
    for split in combined:
        n = len(combined[split])
        if n > 0:
            sample_key = list(combined[split].keys())[0]
            sample = combined[split][sample_key]
            print(f"  {split}: {n} episodes, slots={sample['slots'].shape}, actions={sample['actions'].shape}")


if __name__ == "__main__":
    main()
