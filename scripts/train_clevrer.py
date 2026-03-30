"""
MechJEPA Training Script for CLEVRER.

Trains the MechJEPA model (codebook + mechanism-biased dynamics) on
pre-extracted VideoSAUR slot embeddings. Compatible with DDP for
multi-GPU training and wandb for experiment tracking.

Usage:
    python scripts/train_clevrer.py embedding_dir=/path/to/clevrer_slots.pkl

    # Single GPU, no wandb (debug)
    python scripts/train_clevrer.py embedding_dir=/path/to/slots.pkl \
        wandb.enable=false trainer.max_epochs=5 batch_size=32

    # Multi-GPU with wandb
    torchrun --nproc_per_node=4 scripts/train_clevrer.py \
        embedding_dir=/path/to/slots.pkl wandb.entity=your-entity
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
from mechjepa.data.clevrer_slots import ClevrerSlotDataset
from mechjepa.system_m import compute_surprise_from_prediction


# Constants (same as C-JEPA for compatibility)
OBS_FRAMES = 128
TARGET_LEN = 160


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
    with open(cfg.embedding_dir, "rb") as f:
        data = pkl.load(f)

    train_ds = ClevrerSlotDataset(
        data["train"], "train",
        history_size=cfg.dinowm.history_size,
        num_preds=cfg.dinowm.num_preds,
        frameskip=cfg.frameskip,
    )
    val_ds = ClevrerSlotDataset(
        data["val"], "val",
        history_size=cfg.dinowm.history_size,
        num_preds=cfg.dinowm.num_preds,
        frameskip=cfg.frameskip,
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
    """Build MechJEPA model from config."""
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
    )


# ===========================================================================
# Loss Computation
# ===========================================================================


def compute_loss(model, batch, cfg, device, epoch=0, inference=False):
    """Compute MechJEPA losses."""
    embed = batch["embed"].to(device)

    history = embed[:, :cfg.dinowm.history_size, :, :]
    target = embed[:, cfg.dinowm.history_size:cfg.dinowm.history_size + cfg.dinowm.num_preds, :, :]

    if inference:
        pred_future = model.inference(history)
        loss_future = torch.nn.functional.mse_loss(pred_future, target.detach())
        return {
            "loss": loss_future,
            "loss_future": loss_future,
            "loss_masked_history": torch.tensor(0.0, device=device),
        }

    # Full forward pass
    outputs = model(history)

    # Simple loss config
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
# Slot Rollout (128 → 160 frames)
# ===========================================================================


@torch.no_grad()
def rollout_video_slots(model, pre_slots, cfg, device, batch_size=None):
    """Autoregressively extend slots from OBS_FRAMES to TARGET_LEN."""
    model.eval()
    history_len = cfg.dinowm.history_size
    pred_len = cfg.dinowm.num_preds
    frameskip = cfg.frameskip

    if batch_size is None:
        batch_size = max(1, torch.cuda.device_count())

    all_fn = list(pre_slots.keys())
    all_slots = {}

    for batch_start in tqdm(range(0, len(all_fn), batch_size), desc="Rollout"):
        batch_end = min(batch_start + batch_size, len(all_fn))
        batch_fns = all_fn[batch_start:batch_end]
        bs = len(batch_fns)

        batch_list = []
        for fn in batch_fns:
            s = pre_slots[fn]
            if isinstance(s, np.ndarray):
                s = torch.from_numpy(s)
            batch_list.append(s.float())

        batch_slots = torch.stack(batch_list, dim=0).to(device)
        num_slots, slot_dim = batch_slots.shape[2], batch_slots.shape[3]

        extended = torch.zeros(bs, TARGET_LEN, num_slots, slot_dim, device=device)
        extended[:, :OBS_FRAMES] = batch_slots

        frames_to_predict = TARGET_LEN - OBS_FRAMES
        all_pred = []

        for off_idx in range(frameskip):
            start = OBS_FRAMES - history_len * frameskip + off_idx
            hist_indices = list(range(start, OBS_FRAMES, frameskip))
            history = extended[:, hist_indices]

            num_needed = (frames_to_predict + frameskip - 1) // frameskip
            preds = []
            current = history

            while len(preds) < num_needed:
                pred = model.inference(current)
                preds.append(pred)
                if current.shape[1] > pred_len:
                    current = torch.cat([current[:, pred_len:], pred], dim=1)
                else:
                    current = pred[:, -history_len:]

            all_preds_off = torch.cat(preds, dim=1)
            all_pred.append(all_preds_off[:, :num_needed])

        for i in range(frames_to_predict):
            off = i % frameskip
            pos = i // frameskip
            if pos < all_pred[off].shape[1]:
                extended[:, OBS_FRAMES + i] = all_pred[off][:, pos]

        for bi, fn in enumerate(batch_fns):
            all_slots[fn] = extended[bi].cpu().numpy()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return all_slots


# ===========================================================================
# Main Training Loop
# ===========================================================================


@hydra.main(version_base=None, config_path="../configs", config_name="clevrer")
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
            name=cfg.wandb.get("name", "mechjepa"),
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
    warmup_epochs = cfg.get("warmup_epochs", 5)
    total_epochs = cfg.trainer.max_epochs

    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return epoch / max(warmup_epochs, 1)
        progress = (epoch - warmup_epochs) / max(total_epochs - warmup_epochs, 1)
        return 0.5 * (1 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # Gradient clipping (matching C-JEPA)
    max_grad_norm = cfg.get("max_grad_norm", 0.05)

    # Rollout-only mode
    if cfg.rollout.get("rollout_only", False):
        ckpt_path = cfg.rollout.get("rollout_checkpoint")
        assert ckpt_path, "rollout_checkpoint required for rollout_only"
        state = torch.load(ckpt_path, map_location=device)
        raw_model.load_state_dict(state)
        logging.info(f"Loaded {ckpt_path} for rollout")

        rollout_data = {}
        for split in ["train", "val", "test"]:
            if split in data:
                rollout_data[split] = rollout_video_slots(
                    raw_model, data[split], cfg, device,
                    batch_size=cfg.rollout.get("rollout_batch_size"),
                )

        emb_path = Path(cfg.embedding_dir)
        out_path = emb_path.parent / f"rollout_mechjepa_{emb_path.stem}.pkl"
        with open(out_path, "wb") as f:
            pkl.dump(rollout_data, f)
        logging.info(f"Saved rollout to {out_path}")

        if wandb_logger:
            wandb_lib.finish()
        cleanup_distributed()
        return

    # ── Training Loop ──
    cache_dir = cfg.cache_dir or os.path.expanduser("~/.cache/mechjepa")
    os.makedirs(cache_dir, exist_ok=True)
    global_step = 0
    best_val_loss = float("inf")

    # Mixed precision (AMP)
    use_amp = (
        cfg.trainer.get("precision", "32") == "16-mixed"
        and torch.cuda.is_available()
    )
    scaler = torch.amp.GradScaler(enabled=use_amp)
    amp_dtype = torch.float16 if use_amp else torch.float32
    if is_main(rank):
        logging.info(f"Mixed precision: {'fp16' if use_amp else 'fp32'}")

    for epoch in range(cfg.trainer.max_epochs):
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

            # Step-level logging
            if wandb_logger and global_step % cfg.get("trainer", {}).get("log_every_n_steps", 10) == 0:
                log = {
                    "train/loss": losses["loss"].item(),
                    "train/loss_future": losses["loss_future"].item(),
                    "train/loss_masked_history": losses["loss_masked_history"].item(),
                    "train/step": global_step,
                    "train/epoch": epoch,
                }
                wandb_logger.log(log)

        # Epoch-level logging
        avg_loss = epoch_loss / max(n_batches, 1)

        if is_main(rank):
            logging.info(
                f"Epoch {epoch+1}: loss={avg_loss:.4f} "
                f"future={epoch_future/max(n_batches,1):.4f}"
            )

            # Bottleneck diagnostic: encode/decode weight norms show dimension usage
            diagnostics = raw_model.get_diagnostics()
            if 'codebook/dim_importance' in diagnostics:
                dim_imp = diagnostics['codebook/dim_importance']
                imp_str = ' '.join(f'{v:.3f}' for v in dim_imp.tolist())
                logging.info(
                    f"  bottleneck dims: [{imp_str}] "
                    f"ratio={dim_imp.max()/(dim_imp.min()+1e-8):.1f}x"
                )

        # Step LR scheduler
        scheduler.step()

        # Validation
        val_metrics = validate(raw_model, val_loader, cfg, device, world_size)

        if is_main(rank):
            logging.info(
                f"Epoch {epoch+1}: val_loss={val_metrics['val/loss']:.4f}  "
                f"lr={scheduler.get_last_lr()[0]:.2e}"
            )

            # Codebook diagnostics
            diagnostics = raw_model.get_diagnostics()
            diag_log = {}
            for k, v in diagnostics.items():
                if isinstance(v, torch.Tensor):
                    if v.dim() == 0:
                        diag_log[k] = v.item()
                else:
                    diag_log[k] = float(v)

            if wandb_logger:
                wandb_logger.log({
                    **val_metrics,
                    **diag_log,
                    "epoch": epoch + 1,
                    "train/epoch_loss": avg_loss,
                    "train/lr": scheduler.get_last_lr()[0],
                })

            # Save checkpoint
            ckpt_path = os.path.join(cache_dir, f"{cfg.output_model_name}_epoch_{epoch+1}.ckpt")
            torch.save(raw_model.state_dict(), ckpt_path)

            # Save best
            if val_metrics["val/loss"] < best_val_loss:
                best_val_loss = val_metrics["val/loss"]
                best_path = os.path.join(cache_dir, f"{cfg.output_model_name}_best.ckpt")
                torch.save(raw_model.state_dict(), best_path)
                logging.info(f"New best model: val_loss={best_val_loss:.4f}")

        # (No codebook maintenance needed — continuous bottleneck has no dead entries)

    # Final save
    if is_main(rank):
        final_path = os.path.join(cache_dir, f"{cfg.output_model_name}_final.ckpt")
        torch.save(raw_model.state_dict(), final_path)
        logging.info(f"Saved final model to {final_path}")

    # Rollout
    if cfg.rollout.get("save_rollout", False) and is_main(rank):
        logging.info("Starting slot rollout (128 → 160)...")
        rollout_data = {}
        for split in ["train", "val", "test"]:
            if split in data:
                logging.info(f"Rolling out {split}...")
                rollout_data[split] = rollout_video_slots(
                    raw_model, data[split], cfg, device,
                    batch_size=cfg.rollout.get("rollout_batch_size"),
                )

        emb_path = Path(cfg.embedding_dir)
        out_path = emb_path.parent / f"rollout_mechjepa_{emb_path.stem}.pkl"
        with open(out_path, "wb") as f:
            pkl.dump(rollout_data, f)
        logging.info(f"Saved rollout to {out_path}")

    if wandb_logger:
        wandb_lib.finish()
    cleanup_distributed()


if __name__ == "__main__":
    run()
