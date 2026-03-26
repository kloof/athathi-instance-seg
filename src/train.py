"""Training loop for wall segmentation."""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import time
from pathlib import Path

from src.gpu_augment import augment_batch_gpu, compute_features_batch_gpu


def compute_metrics(
    predictions: np.ndarray,
    targets: np.ndarray,
    num_classes: int = 2,
) -> dict:
    """Compute segmentation metrics."""
    oa = (predictions == targets).mean()

    ious = []
    for c in range(num_classes):
        pred_c = predictions == c
        true_c = targets == c
        intersection = (pred_c & true_c).sum()
        union = (pred_c | true_c).sum()
        iou = intersection / max(union, 1)
        ious.append(iou)

    wall_pred = predictions == 1
    wall_true = targets == 1
    tp = (wall_pred & wall_true).sum()
    wall_precision = tp / max(wall_pred.sum(), 1)
    wall_recall = tp / max(wall_true.sum(), 1)

    return {
        "overall_accuracy": float(oa),
        "iou_nonwall": float(ious[0]),
        "iou_wall": float(ious[1]),
        "miou": float(np.mean(ious)),
        "wall_precision": float(wall_precision),
        "wall_recall": float(wall_recall),
    }


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    scaler: torch.amp.GradScaler | None = None,
    augment: bool = True,
    aug_cfg: dict | None = None,
) -> float:
    """Train for one epoch. Augmentation + features computed on GPU."""
    model.train()
    total_loss = 0.0
    num_batches = 0
    total = len(loader)
    use_amp = scaler is not None
    epoch_start = time.time()
    if aug_cfg is None:
        aug_cfg = {}

    for batch in loader:
        points, labels, room_mins, room_maxs = batch
        points = points.to(device, dtype=torch.float32)
        labels = labels.to(device, dtype=torch.int64)
        room_mins = room_mins.to(device, dtype=torch.float32)
        room_maxs = room_maxs.to(device, dtype=torch.float32)

        # Augmentation on GPU (densify + noise + rotation + tilt + scale)
        if augment:
            points, labels = augment_batch_gpu(
                points, labels,
                densify=aug_cfg.get("densify", True),
                densify_multiplier=aug_cfg.get("densify_multiplier", 6),
                noise_min=aug_cfg.get("noise_min", 0.005),
                noise_max=aug_cfg.get("noise_max", 0.020),
                dropout_keep=aug_cfg.get("dropout_keep", 0.85),
            )

        # Feature computation on GPU
        features = compute_features_batch_gpu(points, room_mins, room_maxs)

        optimizer.zero_grad()

        with torch.amp.autocast("cuda", enabled=use_amp):
            logits = model(features)
            loss = criterion(logits, labels)

        if use_amp:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        total_loss += loss.item()
        num_batches += 1

        if num_batches % 50 == 0 or num_batches == total:
            avg_loss = total_loss / num_batches
            preds = logits.argmax(dim=1)
            acc = (preds == labels).float().mean().item()
            wall_pred = (preds == 1).sum().item()
            wall_true = (labels == 1).sum().item()
            elapsed = time.time() - epoch_start
            eta_epoch = elapsed / num_batches * (total - num_batches)
            print(
                f"  step {num_batches:>5d}/{total} | "
                f"loss={loss.item():.4f} | "
                f"avg_loss={avg_loss:.4f} | "
                f"acc={acc:.3f} | "
                f"wall_pred={wall_pred} wall_true={wall_true} | "
                f"eta={eta_epoch:.0f}s",
                flush=True,
            )

    return total_loss / max(num_batches, 1)


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> dict:
    """Evaluate model on a dataset. Streams metrics to avoid OOM."""
    model.eval()
    total = len(loader)

    # Accumulate confusion matrix incrementally (no giant arrays)
    tp = 0  # true positive (wall predicted as wall)
    fp = 0  # false positive (non-wall predicted as wall)
    fn = 0  # false negative (wall predicted as non-wall)
    tn = 0  # true negative
    total_pts = 0

    for batch_idx, batch in enumerate(loader, 1):
        points, labels, room_mins, room_maxs = batch
        points = points.to(device, dtype=torch.float32)
        room_mins = room_mins.to(device, dtype=torch.float32)
        room_maxs = room_maxs.to(device, dtype=torch.float32)

        with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
            features = compute_features_batch_gpu(points, room_mins, room_maxs)
            logits = model(features)

        preds = logits.argmax(dim=1).cpu()
        labels = labels.to(torch.int64)

        wall_pred = preds == 1
        wall_true = labels == 1
        tp += (wall_pred & wall_true).sum().item()
        fp += (wall_pred & ~wall_true).sum().item()
        fn += (~wall_pred & wall_true).sum().item()
        tn += (~wall_pred & ~wall_true).sum().item()
        total_pts += labels.numel()

        if batch_idx % 50 == 0 or batch_idx == total:
            print(f"  eval {batch_idx}/{total}", end="\r", flush=True)

    # Compute metrics from confusion matrix
    oa = (tp + tn) / max(total_pts, 1)
    iou_wall = tp / max(tp + fp + fn, 1)
    iou_nonwall = tn / max(tn + fp + fn, 1)
    wall_precision = tp / max(tp + fp, 1)
    wall_recall = tp / max(tp + fn, 1)

    return {
        "overall_accuracy": float(oa),
        "iou_nonwall": float(iou_nonwall),
        "iou_wall": float(iou_wall),
        "miou": float((iou_wall + iou_nonwall) / 2),
        "wall_precision": float(wall_precision),
        "wall_recall": float(wall_recall),
    }


def train(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    cfg: dict,
    device: torch.device,
    checkpoint_dir: Path,
    resume_checkpoint: str | None = None,
) -> None:
    """Full training loop with validation and checkpointing."""
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    wall_weight = cfg["training"]["wall_class_weight"]
    class_weights = torch.tensor([1.0, wall_weight], dtype=torch.float32).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    optimizer = optim.Adam(
        model.parameters(),
        lr=cfg["training"]["learning_rate"],
        weight_decay=cfg["training"]["weight_decay"],
    )

    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg["training"]["epochs"],
    )

    best_miou = 0.0
    start_epoch = 1

    # Resume from checkpoint
    if resume_checkpoint:
        ckpt = torch.load(resume_checkpoint, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        start_epoch = ckpt["epoch"] + 1
        best_miou = ckpt.get("val_metrics", {}).get("miou", 0.0)
        # Advance scheduler to the correct epoch
        for _ in range(ckpt["epoch"]):
            scheduler.step()
        print(f"Resumed from epoch {ckpt['epoch']}, best mIoU={best_miou:.4f}")

    # Mixed precision for faster training
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda") if use_amp else None
    if use_amp:
        print("Using mixed precision (FP16)")

    for epoch in range(start_epoch, cfg["training"]["epochs"] + 1):
        t0 = time.time()
        aug_cfg = cfg.get("augmentation", {})
        train_loss = train_one_epoch(
            model, train_loader, optimizer, criterion, device, scaler,
            augment=aug_cfg.get("enabled", True),
            aug_cfg=aug_cfg,
        )
        scheduler.step()

        elapsed = time.time() - t0
        total_epochs = cfg["training"]["epochs"]
        remaining_epochs = total_epochs - epoch
        eta_total = elapsed * remaining_epochs
        eta_min = eta_total / 60

        eval_every = cfg["training"].get("eval_every", 5)
        if epoch % eval_every == 0 or epoch == start_epoch:
            val_metrics = evaluate(model, val_loader, device)
            lr = optimizer.param_groups[0]["lr"]
            print(
                f"Epoch {epoch:3d}/{total_epochs} | "
                f"loss={train_loss:.4f} | "
                f"val_mIoU={val_metrics['miou']:.4f} | "
                f"wall_IoU={val_metrics['iou_wall']:.4f} | "
                f"OA={val_metrics['overall_accuracy']:.4f} | "
                f"lr={lr:.6f} | "
                f"{elapsed:.0f}s | "
                f"ETA={eta_min:.0f}min"
            )

            if val_metrics["miou"] > best_miou:
                best_miou = val_metrics["miou"]
                torch.save(model.state_dict(), checkpoint_dir / "best_model.pth")
                print(f"  -> New best mIoU: {best_miou:.4f}")
        else:
            print(
                f"Epoch {epoch:3d}/{total_epochs} | "
                f"loss={train_loss:.4f} | "
                f"{elapsed:.0f}s | "
                f"ETA={eta_min:.0f}min"
            )

        if epoch % cfg["training"].get("save_every", 10) == 0:
            ckpt_path = checkpoint_dir / f"checkpoint_epoch_{epoch}.pth"
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_metrics": val_metrics,
            }, ckpt_path)
            print(f"  -> Saved checkpoint: {ckpt_path}")

    torch.save(model.state_dict(), checkpoint_dir / "final_model.pth")
    print(f"\nTraining complete. Best mIoU: {best_miou:.4f}")
    print(f"Checkpoints saved in: {checkpoint_dir}")
