"""
Bottleneck Analysis: h_ij collision vs. free-flight differentiation.

This script determines whether the continuous mechanism bottleneck learns
to differentiate between interaction types. It extracts h_ij (bottleneck
representations) for all slot pairs across validation videos, labels each
pair-frame as "collision" or "free-flight" using CLEVRER ground-truth
annotations, and produces:

  1. t-SNE/PCA plot of h_ij colored by collision label
  2. Per-dimension mean activation bar chart (collision vs no-collision)
  3. Summary statistics

Usage (on RunPod after training):
    python scripts/analyze_bottleneck.py \
        --checkpoint /workspace/checkpoints/mechjepa_clevrer_best.ckpt \
        --slots_pkl /workspace/data/clevrer_videosaur_slots.pkl \
        --annotations_dir /workspace/data/clevrer_annotations \
        --output_dir /workspace/analysis \
        --num_videos 200 \
        --subsample 50000
"""

import argparse
import glob
import json
import os
import pickle

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score

from mechjepa.codebook import MechanismCodebook


def load_annotations(annotations_dir: str) -> dict[int, list[dict]]:
    """
    Load CLEVRER collision annotations.

    Returns:
        dict mapping scene_index → list of collision events
        Each event: {"object_ids": [i, j], "frame_id": int, "location": [...]}
    """
    ann_files = sorted(glob.glob(os.path.join(annotations_dir, "*", "annotation_*.json")))
    print(f"Found {len(ann_files)} annotation files")

    annotations = {}
    for fpath in ann_files:
        with open(fpath) as f:
            ann = json.load(f)
        scene_idx = ann["scene_index"]
        annotations[scene_idx] = ann.get("collision", [])

    total_collisions = sum(len(v) for v in annotations.values())
    print(f"Total collision events: {total_collisions} across {len(annotations)} videos")
    return annotations


def extract_bottleneck_vectors(
    codebook: MechanismCodebook,
    slots_dict: dict,
    annotations: dict,
    num_videos: int = 200,
    frame_window: int = 2,
    device: str = "cuda",
) -> tuple[np.ndarray, np.ndarray]:
    """
    Extract h_ij vectors and collision labels for validation videos.

    Uses frame-level collision labeling: a slot pair at frame t is labeled
    as "collision" if any collision event occurs within ±frame_window frames.
    This is noisier than pair-level matching but much simpler and sufficient
    for showing separation.

    Args:
        codebook: trained MechanismCodebook
        slots_dict: val split of slot pickle {video_key: (T, K, D)}
        annotations: {scene_index: [collision events]}
        num_videos: how many videos to process
        frame_window: ±frames around collision to label as positive
        device: compute device

    Returns:
        all_h: (N, bottleneck_dim) — h_ij vectors
        all_labels: (N,) — 0 = free-flight, 1 = collision frame
    """
    codebook = codebook.to(device).eval()

    all_h = []
    all_labels = []
    n_collision_frames = 0
    n_total_frames = 0

    video_keys = sorted(slots_dict.keys())[:num_videos]

    for vid_key in video_keys:
        # Extract scene index from key like "10000_pixels.mp4"
        scene_idx = int(vid_key.split("_")[0])

        if scene_idx not in annotations:
            continue

        slots = slots_dict[vid_key]  # (T, K, D)
        T, K, D = slots.shape
        collisions = annotations[scene_idx]

        # Build collision frame set (±window)
        collision_frames = set()
        for coll in collisions:
            frame = coll["frame_id"]
            for f in range(max(0, frame - frame_window), min(T, frame + frame_window + 1)):
                collision_frames.add(f)

        # CLEVRER runs at ~24 fps, 128 frames per video. VideoSAUR uses frameskip=2.
        # Slot frame t corresponds to video frame t * frameskip
        # Adjust collision frames to slot frame indices
        frameskip = 2
        collision_slot_frames = set()
        for cf in collision_frames:
            sf = cf // frameskip
            if sf < T:
                collision_slot_frames.add(sf)

        # Process all frames through codebook
        z = torch.from_numpy(slots).float().to(device)  # (T, K, D)

        with torch.no_grad():
            # Process in batches to avoid OOM
            batch_size = 32
            for start in range(0, T, batch_size):
                end = min(start + batch_size, T)
                z_batch = z[start:end]  # (bs, K, D)
                result = codebook(z_batch)
                h_ij = result["h_ij"].cpu().numpy()  # (bs, K, K, N)

                # Flatten to (bs * K * K, N) and create labels
                bs = h_ij.shape[0]
                h_flat = h_ij.reshape(bs, K * K, -1)

                for t_offset in range(bs):
                    t = start + t_offset
                    label = 1 if t in collision_slot_frames else 0
                    all_h.append(h_flat[t_offset])  # (K*K, N)
                    all_labels.extend([label] * (K * K))

                    n_total_frames += 1
                    if label == 1:
                        n_collision_frames += 1

    all_h = np.concatenate(all_h, axis=0)  # (total_pairs, N)
    all_labels = np.array(all_labels)

    print(f"\nExtracted {all_h.shape[0]} pair vectors from {n_total_frames} frames")
    print(f"Collision frames: {n_collision_frames} ({100*n_collision_frames/max(n_total_frames,1):.1f}%)")
    print(f"Collision pairs: {all_labels.sum()} ({100*all_labels.mean():.1f}%)")

    return all_h, all_labels


def plot_tsne(h: np.ndarray, labels: np.ndarray, output_path: str):
    """t-SNE visualization of h_ij colored by collision label."""
    print(f"Running t-SNE on {h.shape[0]} points...")
    tsne = TSNE(n_components=2, perplexity=30, random_state=42, n_iter=1000)
    h_2d = tsne.fit_transform(h)

    fig, ax = plt.subplots(1, 1, figsize=(10, 8))

    # Plot free-flight first (background), then collisions (foreground)
    mask_0 = labels == 0
    mask_1 = labels == 1

    ax.scatter(h_2d[mask_0, 0], h_2d[mask_0, 1],
               c="#4A90D9", alpha=0.3, s=4, label=f"Free-flight ({mask_0.sum()})")
    ax.scatter(h_2d[mask_1, 0], h_2d[mask_1, 1],
               c="#E74C3C", alpha=0.6, s=8, label=f"Collision ({mask_1.sum()})")

    ax.set_title("MechJEPA Bottleneck Representations (h_ij)\nt-SNE colored by collision ground truth", fontsize=14)
    ax.legend(fontsize=12)
    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Saved t-SNE plot to {output_path}")
    plt.close()


def plot_pca(h: np.ndarray, labels: np.ndarray, output_path: str):
    """PCA visualization of h_ij colored by collision label."""
    pca = PCA(n_components=2)
    h_2d = pca.fit_transform(h)

    fig, ax = plt.subplots(1, 1, figsize=(10, 8))

    mask_0 = labels == 0
    mask_1 = labels == 1

    ax.scatter(h_2d[mask_0, 0], h_2d[mask_0, 1],
               c="#4A90D9", alpha=0.3, s=4, label=f"Free-flight ({mask_0.sum()})")
    ax.scatter(h_2d[mask_1, 0], h_2d[mask_1, 1],
               c="#E74C3C", alpha=0.6, s=8, label=f"Collision ({mask_1.sum()})")

    var_explained = pca.explained_variance_ratio_
    ax.set_title(f"MechJEPA Bottleneck Representations (h_ij)\nPCA (var explained: {var_explained[0]:.1%}, {var_explained[1]:.1%})", fontsize=14)
    ax.legend(fontsize=12)
    ax.set_xlabel(f"PC1 ({var_explained[0]:.1%})")
    ax.set_ylabel(f"PC2 ({var_explained[1]:.1%})")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Saved PCA plot to {output_path}")
    plt.close()


def plot_dimension_activations(h: np.ndarray, labels: np.ndarray, output_path: str):
    """
    Bar chart of mean h_ij activation per bottleneck dimension,
    split by collision vs free-flight.
    """
    N = h.shape[1]
    mask_coll = labels == 1
    mask_free = labels == 0

    mean_coll = h[mask_coll].mean(axis=0) if mask_coll.any() else np.zeros(N)
    mean_free = h[mask_free].mean(axis=0) if mask_free.any() else np.zeros(N)

    # Compute absolute difference for ranking
    diff = np.abs(mean_coll - mean_free)

    fig, axes = plt.subplots(2, 1, figsize=(12, 8))

    x = np.arange(N)
    width = 0.35

    # Top: mean activations
    axes[0].bar(x - width/2, mean_coll, width, label="Collision", color="#E74C3C", alpha=0.8)
    axes[0].bar(x + width/2, mean_free, width, label="Free-flight", color="#4A90D9", alpha=0.8)
    axes[0].set_title("Mean Bottleneck Activation by Dimension", fontsize=14)
    axes[0].set_xlabel("Bottleneck Dimension")
    axes[0].set_ylabel("Mean Activation")
    axes[0].legend()
    axes[0].set_xticks(x)

    # Bottom: absolute difference (mechanism discriminability)
    colors = plt.cm.RdYlGn(diff / (diff.max() + 1e-8))
    axes[1].bar(x, diff, color=colors, edgecolor="gray", linewidth=0.5)
    axes[1].set_title("Collision Discriminability by Dimension (|mean_coll - mean_free|)", fontsize=14)
    axes[1].set_xlabel("Bottleneck Dimension")
    axes[1].set_ylabel("|Δ mean activation|")
    axes[1].set_xticks(x)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Saved dimension activation plot to {output_path}")
    plt.close()


def compute_linear_separability(h: np.ndarray, labels: np.ndarray) -> dict:
    """
    Compute linear separability metrics for collision vs free-flight.

    Returns per-dimension AUC and overall statistics.
    """
    N = h.shape[1]
    results = {}

    # Per-dimension AUC (how well each dimension alone separates the classes)
    dim_aucs = []
    for d in range(N):
        try:
            auc = roc_auc_score(labels, h[:, d])
            dim_aucs.append(auc)
        except ValueError:
            dim_aucs.append(0.5)

    results["dim_aucs"] = dim_aucs
    results["best_dim"] = int(np.argmax([abs(a - 0.5) for a in dim_aucs]))
    results["best_dim_auc"] = max(dim_aucs, key=lambda a: abs(a - 0.5))

    # Mean/std by class
    mask_coll = labels == 1
    mask_free = labels == 0
    results["mean_norm_coll"] = float(np.linalg.norm(h[mask_coll].mean(axis=0)))
    results["mean_norm_free"] = float(np.linalg.norm(h[mask_free].mean(axis=0)))

    # Cosine distance between class centroids
    centroid_coll = h[mask_coll].mean(axis=0)
    centroid_free = h[mask_free].mean(axis=0)
    cos_sim = np.dot(centroid_coll, centroid_free) / (
        np.linalg.norm(centroid_coll) * np.linalg.norm(centroid_free) + 1e-8
    )
    results["centroid_cosine_similarity"] = float(cos_sim)
    results["centroid_cosine_distance"] = float(1 - cos_sim)

    return results


def main():
    parser = argparse.ArgumentParser(description="Analyze MechJEPA bottleneck representations")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--slots_pkl", type=str, required=True, help="Path to slot embeddings pickle")
    parser.add_argument("--annotations_dir", type=str, required=True, help="Path to CLEVRER annotations dir")
    parser.add_argument("--output_dir", type=str, default="./analysis", help="Output directory for plots")
    parser.add_argument("--num_videos", type=int, default=200, help="Number of validation videos to process")
    parser.add_argument("--subsample", type=int, default=50000, help="Max points for visualization")
    parser.add_argument("--device", type=str, default="cuda", help="Compute device")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # 1. Load codebook from checkpoint
    print("=" * 60)
    print("MechJEPA Bottleneck Analysis")
    print("=" * 60)
    print(f"\nLoading checkpoint: {args.checkpoint}")

    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)

    # Extract codebook config from checkpoint
    # The codebook has: edge_mlp, encode, decode
    # Infer dimensions from weight shapes
    encode_weight = ckpt["codebook.encode.weight"]  # (N, D)
    N, D = encode_weight.shape
    print(f"Bottleneck dim N={N}, slot dim D={D}")

    # Find edge_hidden_dim from edge MLP weights
    edge_w0 = ckpt["codebook.edge_mlp.0.weight"]  # (hidden, 2*D)
    edge_hidden = edge_w0.shape[0]

    codebook = MechanismCodebook(num_mechanisms=N, slot_dim=D, edge_hidden_dim=edge_hidden)

    # Load only codebook weights
    codebook_state = {k.replace("codebook.", ""): v for k, v in ckpt.items() if k.startswith("codebook.")}
    codebook.load_state_dict(codebook_state)
    print("Codebook loaded successfully")

    # 2. Load slot embeddings
    print(f"\nLoading slot embeddings: {args.slots_pkl}")
    with open(args.slots_pkl, "rb") as f:
        all_slots = pickle.load(f)
    val_slots = all_slots["val"]
    print(f"Validation videos: {len(val_slots)}")

    # 3. Load annotations
    print(f"\nLoading annotations: {args.annotations_dir}")
    annotations = load_annotations(args.annotations_dir)

    # 4. Extract h_ij vectors
    print(f"\nExtracting bottleneck vectors from {args.num_videos} videos...")
    all_h, all_labels = extract_bottleneck_vectors(
        codebook, val_slots, annotations,
        num_videos=args.num_videos, device=args.device,
    )

    # 5. Subsample for visualization
    if len(all_h) > args.subsample:
        print(f"\nSubsampling {len(all_h)} → {args.subsample} points...")
        # Stratified subsample to preserve collision ratio
        idx_coll = np.where(all_labels == 1)[0]
        idx_free = np.where(all_labels == 0)[0]
        coll_ratio = len(idx_coll) / len(all_labels)
        n_coll = min(len(idx_coll), int(args.subsample * coll_ratio))
        n_free = args.subsample - n_coll
        n_free = min(n_free, len(idx_free))

        idx = np.concatenate([
            np.random.choice(idx_coll, n_coll, replace=False),
            np.random.choice(idx_free, n_free, replace=False),
        ])
        np.random.shuffle(idx)
        h_vis = all_h[idx]
        labels_vis = all_labels[idx]
    else:
        h_vis = all_h
        labels_vis = all_labels

    # 6. Generate plots
    print(f"\nGenerating visualizations...")
    plot_tsne(h_vis, labels_vis, os.path.join(args.output_dir, "tsne_collision.png"))
    plot_pca(h_vis, labels_vis, os.path.join(args.output_dir, "pca_collision.png"))
    plot_dimension_activations(all_h, all_labels, os.path.join(args.output_dir, "dim_activations.png"))

    # 7. Linear separability analysis
    print(f"\nComputing separability metrics...")
    sep = compute_linear_separability(all_h, all_labels)

    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"Centroid cosine distance: {sep['centroid_cosine_distance']:.4f}")
    print(f"  (0 = identical centroids, 1 = orthogonal, 2 = opposite)")
    print(f"Best single-dimension AUC: dim {sep['best_dim']} → AUC={sep['best_dim_auc']:.4f}")
    print(f"  (0.5 = random, >0.6 = some signal, >0.7 = clear separation)")
    print(f"\nPer-dimension AUCs:")
    for d, auc in enumerate(sep["dim_aucs"]):
        marker = " ← BEST" if d == sep["best_dim"] else ""
        marker = " ← GOOD" if abs(auc - 0.5) > 0.1 and d != sep["best_dim"] else marker
        print(f"  dim {d:2d}: AUC={auc:.4f}{marker}")

    # Save results
    results_path = os.path.join(args.output_dir, "results.json")
    with open(results_path, "w") as f:
        json.dump(sep, f, indent=2)
    print(f"\nSaved results to {results_path}")
    print(f"Plots saved to {args.output_dir}/")
    print("\nDone!")


if __name__ == "__main__":
    main()
