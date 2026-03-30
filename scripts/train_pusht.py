"""
MechJEPA Training Script for Push-T.

Action-conditioned world model training on pre-extracted VideoSAUR slot
embeddings with actions. Uses AdaLN for action conditioning.

Usage:
    python scripts/train_pusht.py embedding_dir=/path/to/pusht_slots_actions.pkl

    # Single GPU, no wandb (debug)
    python scripts/train_pusht.py embedding_dir=/path/to/data.pkl \
        wandb.enable=false trainer.max_epochs=5 batch_size=32

    # Multi-GPU with wandb
    torchrun --nproc_per_node=4 scripts/train_pusht.py \
        embedding_dir=/path/to/data.pkl wandb.entity=your-entity
"""

import math
import os
import pickle as pkl
from pathlib import Path

import hydra
import numpy as np
import torch
import torch.distributed as dist
from loguru import logger as logging
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, DistributedSampler
from tqdm import tqdm

import wandb as wandb_lib

from mechjepa.model import MechJEPA
from mechjepa.data.clevrer_slots import PushTSlotDataset


# ===========================================================================
# Distributed Setup
# ===========================================================================


def setup_distributed():
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ["LOCAL_RANK"])
        dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)
        torch.cuda.set_device(local_rank)
        return True, rank, world_size, local_rank
    return False, 0, 1, 0


def cleanup_distributed():
    if dist.is_initialized():
        dist.destroy_process_group()


def is_main(rank):
    return rank == 0


# ===========================================================================
# Data
# ===========================================================================


def get_data(cfg, is_ddp, world_size, rank):
    """Load Push-T slots + actions pickle."""
    with open(cfg.embedding_dir, "rb") as f:
        data = pkl.load(f)

    action_dim = cfg.get("action_dim", 2)

    # Handle two pickle formats:
    # Format 1: {"train": {"ep_key": {"slots": ..., "actions": ...}, ...}, "val": ...}
    # Format 2: {"train": {"slots": {...}, "actions": {...}}, "val": ...}
    def extract_split(split_data):
        if not split_data:
            return {}, {}

        sample_key = list(split_data.keys())[0]
        sample = split_data[sample_key]

        if isinstance(sample, dict) and "slots" in sample:
            # Format 1: per-episode dicts
            slots = {k: v["slots"] for k, v in split_data.items()}
            actions = {k: v["actions"] for k, v in split_data.items()}
            return slots, actions
        elif isinstance(sample, np.ndarray) or isinstance(sample, torch.Tensor):
            # Format 2: slots-only (like C-JEPA's raw pickle)
            return split_data, None
        elif sample_key in ("slots", "actions"):
            # Format 3: {"slots": {...}, "actions": {...}}
            return split_data.get("slots", {}), split_data.get("actions", None)
        else:
            return split_data, None

    train_slots, train_actions = extract_split(data.get("train", {}))
    val_slots, val_actions = extract_split(data.get("val", {}))

    train_ds = PushTSlotDataset(
        train_slots, actions_data=train_actions, split="train",
        history_size=cfg.dinowm.history_size,
        num_preds=cfg.dinowm.num_preds,
        frameskip=cfg.frameskip,
        action_dim=action_dim,
    )
    val_ds = PushTSlotDataset(
        val_slots, actions_data=val_actions, split="val",
        history_size=cfg.dinowm.history_size,
        num_preds=cfg.dinowm.num_preds,
        frameskip=cfg.frameskip,
        action_dim=action_dim,
    )

    train_sampler = DistributedSampler(train_ds, num_replicas=world_size, rank=rank, shuffle=True) if is_ddp else None
    val_sampler = DistributedSampler(val_ds, num_replicas=world_size, rank=rank, shuffle=False) if is_ddp else None

    train_loader = DataLoader(
        train_ds, batch_size=cfg.batch_size, shuffle=(train_sampler is None),
        sampler=train_sampler, num_workers=cfg.num_workers, pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=cfg.batch_size, shuffle=False,
        sampler=val_sampler, num_workers=cfg.num_workers, pin_memory=True, drop_last=False,
    )

    return train_loader, val_loader, data, train_sampler


# ===========================================================================
# Model
# ===========================================================================


def build_model(cfg):
    """Build MechJEPA model with action conditioning."""
    return MechJEPA(
        num_slots=cfg.videosaur.NUM_SLOTS,
        slot_dim=cfg.videosaur.SLOT_DIM,
        num_mechanisms=cfg.codebook.num_mechanisms,
        history_frames=cfg.dinowm.history_size,
        pred_frames=cfg.dinowm.num_preds,
        num_masked_slots=cfg.get("num_masked_slots", 2),
        edge_hidden_dim=cfg.codebook.edge_hidden_dim,
        transformer_depth=cfg.predictor.depth,
        transformer_heads=cfg.predictor.heads,
        transformer_dim_head=cfg.predictor.dim_head,
        transformer_mlp_dim=cfg.predictor.mlp_dim,
        dropout=cfg.predictor.dropout,
        seed=cfg.seed,
        surprise_threshold=cfg.system_m.get("surprise_threshold", 1.0),
        action_dim=cfg.get("action_dim", None),
    )


# ===========================================================================
# Loss Computation
# ===========================================================================


def compute_loss(model, batch, cfg, device, epoch=0, inference=False):
    """Compute MechJEPA losses with action conditioning."""
    embed = batch["embed"].to(device)
    actions = batch.get("actions")
    if actions is not None:
        actions = actions.to(device)

    history = embed[:, :cfg.dinowm.history_size, :, :]
    target = embed[:, cfg.dinowm.history_size:cfg.dinowm.history_size + cfg.dinowm.num_preds, :, :]

    # Actions for history timesteps
    history_actions = actions[:, :cfg.dinowm.history_size] if actions is not None else None

    if inference:
        pred_future = model.inference(history, actions=history_actions)
        loss_future = torch.nn.functional.mse_loss(pred_future, target.detach())
        return {
            "loss": loss_future,
            "loss_future": loss_future,
            "loss_masked_history": torch.tensor(0.0, device=device),
        }

    # Full forward pass with actions
    outputs = model(history, actions=history_actions)

    loss_cfg = {
        "history_size": cfg.dinowm.history_size,
        "num_preds": cfg.dinowm.num_preds,
        "bottleneck_recon_weight": cfg.codebook.get("bottleneck_recon_weight", 0.0),
    }

    losses = model.compute_loss(outputs, history, target, cfg=loss_cfg)
    return losses


# ===========================================================================
# Validation
# ===========================================================================


@torch.no_grad()
def validate(model, val_loader, cfg, device, world_size):
    model.eval()
    total_loss = 0.0
    total_future = 0.0
    n = 0

    for batch in val_loader:
        losses = compute_loss(model, batch, cfg, device, inference=True)
        total_loss += losses["loss"].item()
        total_future += losses["loss_future"].item()
        n += 1

    avg = total_loss / max(n, 1)
    avg_future = total_future / max(n, 1)

    if world_size > 1:
        t = torch.tensor([avg, avg_future, n], device=device)
        dist.all_reduce(t, op=dist.ReduceOp.SUM)
        avg = t[0].item() / world_size
        avg_future = t[1].item() / world_size

    return {"val/loss": avg, "val/loss_future": avg_future}


# ===========================================================================
# Main Training Loop
# ===========================================================================


@hydra.main(version_base=None, config_path="../configs", config_name="pusht")
def run(cfg):
    is_ddp, rank, world_size, local_rank = setup_distributed()
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")

    if is_main(rank):
        logging.info(f"DDP={is_ddp}, Rank={rank}, World={world_size}, Device={device}")
        logging.info(f"Config:\n{OmegaConf.to_yaml(cfg)}")

    # Wandb
    wandb_logger = None
    if cfg.wandb.enable and is_main(rank):
        wandb_lib.init(
            name=cfg.wandb.get("name", "mechjepa_pusht"),
            project=cfg.wandb.project,
            entity=cfg.wandb.get("entity", None),
            config=OmegaConf.to_container(cfg, resolve=True),
        )
        wandb_logger = wandb_lib

    # Data
    train_loader, val_loader, data, train_sampler = get_data(cfg, is_ddp, world_size, rank)
    if is_main(rank):
        logging.info(f"Train: {len(train_loader.dataset)}, Val: {len(val_loader.dataset)}")

    # Model
    model = build_model(cfg).to(device)
    if is_main(rank):
        param_counts = model.get_parameter_count()
        logging.info(f"Parameter counts: {param_counts}")
        logging.info(f"Action dim: {cfg.get('action_dim', None)}")
        if wandb_logger:
            wandb_logger.log(param_counts)

    if is_ddp:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[local_rank], output_device=local_rank,
        )

    raw_model = model.module if is_ddp else model

    # Optimizer (separate LR for codebook)
    codebook_params = list(raw_model.codebook.parameters())
    predictor_params = list(raw_model.predictor.parameters())
    optimizer = torch.optim.AdamW([
        {"params": predictor_params, "lr": cfg.predictor_lr},
        {"params": codebook_params, "lr": cfg.codebook_lr},
    ])

    # Cosine LR schedule with linear warmup
    warmup_epochs = cfg.get("warmup_epochs", 10)
    total_epochs = cfg.trainer.max_epochs

    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return epoch / max(warmup_epochs, 1)
        progress = (epoch - warmup_epochs) / max(total_epochs - warmup_epochs, 1)
        return 0.5 * (1 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    max_grad_norm = cfg.get("max_grad_norm", 0.05)

    # ── Training Loop ──
    cache_dir = cfg.cache_dir or os.path.expanduser("~/.cache/mechjepa")
    os.makedirs(cache_dir, exist_ok=True)
    global_step = 0
    best_val_loss = float("inf")

    # Mixed precision
    use_amp = (
        cfg.trainer.get("precision", "32") == "16-mixed"
        and torch.cuda.is_available()
    )
    scaler = torch.amp.GradScaler(enabled=use_amp)
    amp_dtype = torch.float16 if use_amp else torch.float32
    if is_main(rank):
        logging.info(f"Mixed precision: {'fp16' if use_amp else 'fp32'}")

    for epoch in range(total_epochs):
        if train_sampler:
            train_sampler.set_epoch(epoch)

        model.train()
        epoch_loss = 0.0
        epoch_future = 0.0
        n_batches = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}", disable=not is_main(rank))
        for batch in pbar:
            optimizer.zero_grad()

            with torch.amp.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
                losses = compute_loss(raw_model, batch, cfg, device, epoch=epoch)

            scaler.scale(losses["loss"]).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(raw_model.parameters(), max_norm=max_grad_norm)
            scaler.step(optimizer)
            scaler.update()

            epoch_loss += losses["loss"].item()
            epoch_future += losses["loss_future"].item()
            n_batches += 1
            global_step += 1

            pbar.set_postfix({
                "loss": f"{losses['loss'].item():.4f}",
                "future": f"{losses['loss_future'].item():.4f}",
            })

            if wandb_logger and global_step % cfg.get("trainer", {}).get("log_every_n_steps", 10) == 0:
                wandb_logger.log({
                    "train/loss": losses["loss"].item(),
                    "train/loss_future": losses["loss_future"].item(),
                    "train/step": global_step,
                    "train/epoch": epoch,
                })

        avg_loss = epoch_loss / max(n_batches, 1)

        if is_main(rank):
            logging.info(
                f"Epoch {epoch+1}: loss={avg_loss:.4f} "
                f"future={epoch_future/max(n_batches,1):.4f}"
            )

            diagnostics = raw_model.get_diagnostics()
            if 'codebook/dim_importance' in diagnostics:
                dim_imp = diagnostics['codebook/dim_importance']
                imp_str = ' '.join(f'{v:.3f}' for v in dim_imp.tolist())
                logging.info(
                    f"  bottleneck dims: [{imp_str}] "
                    f"ratio={dim_imp.max()/(dim_imp.min()+1e-8):.1f}x"
                )

        scheduler.step()

        # Validation
        val_metrics = validate(raw_model, val_loader, cfg, device, world_size)

        if is_main(rank):
            logging.info(
                f"Epoch {epoch+1}: val_loss={val_metrics['val/loss']:.4f}  "
                f"lr={scheduler.get_last_lr()[0]:.2e}"
            )

            diagnostics = raw_model.get_diagnostics()
            diag_log = {}
            for k, v in diagnostics.items():
                if isinstance(v, torch.Tensor) and v.dim() == 0:
                    diag_log[k] = v.item()
                elif not isinstance(v, torch.Tensor):
                    diag_log[k] = float(v)

            if wandb_logger:
                wandb_logger.log({
                    **val_metrics, **diag_log,
                    "epoch": epoch + 1,
                    "train/epoch_loss": avg_loss,
                    "train/lr": scheduler.get_last_lr()[0],
                })

            # Save checkpoint
            ckpt_path = os.path.join(cache_dir, f"{cfg.output_model_name}_epoch_{epoch+1}.ckpt")
            torch.save(raw_model.state_dict(), ckpt_path)

            if val_metrics["val/loss"] < best_val_loss:
                best_val_loss = val_metrics["val/loss"]
                best_path = os.path.join(cache_dir, f"{cfg.output_model_name}_best.ckpt")
                torch.save(raw_model.state_dict(), best_path)
                logging.info(f"New best model: val_loss={best_val_loss:.4f}")

    if is_main(rank):
        final_path = os.path.join(cache_dir, f"{cfg.output_model_name}_final.ckpt")
        torch.save(raw_model.state_dict(), final_path)
        logging.info(f"Saved final model to {final_path}")

    if wandb_logger:
        wandb_lib.finish()
    cleanup_distributed()


if __name__ == "__main__":
    run()
