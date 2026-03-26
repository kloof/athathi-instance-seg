"""Training loop for instance segmentation (PTv2 dual-head).

Loss = semantic CE + offset L1 (for thing points only).
Pipeline: load raw → densify (keep all) → augment on GPU → features → model.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time
import math

from src.gpu_augment import compute_features_batch_gpu, densify_batch_gpu

# Thing class IDs (have instances)
THING_CLASS_IDS = {3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14}


def _gpu_augment(points, offsets, device):
    """GPU augmentation: rotation, flip, scale, noise.

    Applies identical transform to both points and offset vectors so they
    stay consistent. Noise is only added to points (not offsets).
    """
    B, N, _ = points.shape

    # Z rotation — apply to both points and offsets
    theta = torch.rand(B, device=device) * 2 * math.pi
    cos_t, sin_t = torch.cos(theta), torch.sin(theta)

    px, py = points[:, :, 0].clone(), points[:, :, 1].clone()
    points[:, :, 0] = px * cos_t.unsqueeze(1) - py * sin_t.unsqueeze(1)
    points[:, :, 1] = px * sin_t.unsqueeze(1) + py * cos_t.unsqueeze(1)

    ox, oy = offsets[:, :, 0].clone(), offsets[:, :, 1].clone()
    offsets[:, :, 0] = ox * cos_t.unsqueeze(1) - oy * sin_t.unsqueeze(1)
    offsets[:, :, 1] = ox * sin_t.unsqueeze(1) + oy * cos_t.unsqueeze(1)

    # Random flip X/Y — apply to both
    flip_x = (torch.rand(B, device=device) > 0.5).float() * 2 - 1
    flip_y = (torch.rand(B, device=device) > 0.5).float() * 2 - 1
    points[:, :, 0] *= flip_x.unsqueeze(1)
    points[:, :, 1] *= flip_y.unsqueeze(1)
    offsets[:, :, 0] *= flip_x.unsqueeze(1)
    offsets[:, :, 1] *= flip_y.unsqueeze(1)

    # Scale [0.9, 1.1] — apply to both
    scale = 0.9 + torch.rand(B, 1, 1, device=device) * 0.2
    points = points * scale
    offsets = offsets * scale

    # Noise 10-30mm — points only (offsets are direction vectors, don't add noise)
    noise_mag = torch.rand(B, N, 1, device=device) * 0.020 + 0.010
    noise_dir = torch.randn_like(points)
    noise_dir = noise_dir / (noise_dir.norm(dim=2, keepdim=True) + 1e-8)
    points = points + noise_dir * noise_mag

    return points, offsets


def train_one_epoch(model, loader, optimizer, criterion_sem, device, scaler=None,
                    densify=True, densify_multiplier=10, augment=True,
                    offset_weight=1.0, max_points=200000):
    """Train one epoch. Pipeline: densify (keep all) → augment → features → model."""
    model.train()
    total_loss = 0.0
    total_sem_loss = 0.0
    total_off_loss = 0.0
    num_batches = 0
    total = len(loader)
    use_amp = scaler is not None
    t0 = time.time()

    for batch in loader:
        points = batch["points"].to(device)
        sem_labels = batch["sem_labels"].to(device)
        inst_labels = batch["inst_labels"].to(device)
        offsets_gt = batch["offsets"].to(device)       # (B, N, 3)

        # Densify (keep all) then cap to max_points for GPU memory
        if densify:
            points, sem_labels = densify_batch_gpu(
                points, sem_labels, target_multiplier=densify_multiplier, keep_all=True,
            )
            # Expand instance labels to match dense points
            B_orig, N_orig = inst_labels.shape
            factor = densify_multiplier
            inst_labels = inst_labels.repeat(1, factor)[:, :points.shape[1]]

        # Cap total points to fit in GPU memory
        B, N_cur, _ = points.shape
        need_recompute = densify
        if N_cur > max_points:
            idx = torch.stack([torch.randperm(N_cur, device=device)[:max_points] for _ in range(B)])
            points = torch.gather(points, 1, idx.unsqueeze(-1).expand(-1, -1, 3))
            sem_labels = torch.gather(sem_labels, 1, idx)
            inst_labels = torch.gather(inst_labels, 1, idx)
            need_recompute = True

        # Recompute offsets only if points changed (densify or cap)
        if need_recompute:
            with torch.no_grad():
                B, N, _ = points.shape
                offsets_gt = torch.zeros(B, N, 3, device=device)
                for b in range(B):
                    ids = inst_labels[b]  # (N,)
                    unique_ids = ids.unique()
                    for iid in unique_ids:
                        if iid == 0:
                            continue
                        mask = ids == iid
                        centroid = points[b, mask].mean(dim=0)
                        offsets_gt[b, mask] = centroid - points[b, mask]

        # Augment — transform both points and offset vectors consistently
        if augment:
            points, offsets_gt = _gpu_augment(points, offsets_gt, device)

        B, N, _ = points.shape

        # Features
        room_mins = points.min(dim=1).values
        room_maxs = points.max(dim=1).values
        features = compute_features_batch_gpu(points, room_mins, room_maxs)

        optimizer.zero_grad()

        with torch.amp.autocast("cuda", enabled=use_amp):
            sem_logits, offset_pred = model(features)

            # Semantic loss
            sem_loss = criterion_sem(sem_logits, sem_labels)

            # Offset loss (only for thing points with instance > 0)
            thing_mask = inst_labels > 0  # (B, N)
            if thing_mask.any():
                off_pred = offset_pred.permute(0, 2, 1)  # (B, N, 3)
                off_loss = F.l1_loss(
                    off_pred[thing_mask], offsets_gt[thing_mask],
                )
            else:
                off_loss = torch.tensor(0.0, device=device)

            loss = sem_loss + offset_weight * off_loss

        if torch.isnan(loss):
            print(f"  WARNING: NaN loss at batch {num_batches}, skipping")
            optimizer.zero_grad()
            continue

        if use_amp:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        total_loss += loss.item()
        total_sem_loss += sem_loss.item()
        total_off_loss += off_loss.item()
        num_batches += 1

        # Free GPU memory
        del points, sem_labels, inst_labels, offsets_gt, features
        del sem_logits, offset_pred, loss, sem_loss, off_loss

        if num_batches % 20 == 0 or num_batches == total:
            avg = total_loss / num_batches
            avg_s = total_sem_loss / num_batches
            avg_o = total_off_loss / num_batches
            elapsed = time.time() - t0
            eta = elapsed / num_batches * (total - num_batches)
            print(
                f"  step {num_batches:>4d}/{total} | "
                f"loss={avg:.4f} sem={avg_s:.4f} off={avg_o:.4f} | "
                f"eta={eta:.0f}s",
                flush=True,
            )

    return total_loss / max(num_batches, 1)


@torch.no_grad()
def evaluate(model, loader, device, num_classes=15):
    """Evaluate semantic mIoU."""
    model.eval()
    confusion = np.zeros((num_classes, num_classes), dtype=np.int64)

    for batch in loader:
        points = batch["points"].to(device)
        sem_labels = batch["sem_labels"]

        B, N, _ = points.shape
        room_mins = points.min(dim=1).values
        room_maxs = points.max(dim=1).values

        with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
            features = compute_features_batch_gpu(points, room_mins, room_maxs)
            sem_logits, _ = model(features)

        preds = sem_logits.argmax(dim=1).cpu().numpy().flatten()
        targets = sem_labels.numpy().flatten()

        for p, t in zip(preds, targets):
            if t < num_classes:
                confusion[t, p] += 1

    # Per-class IoU
    ious = []
    for c in range(num_classes):
        tp = confusion[c, c]
        fp = confusion[:, c].sum() - tp
        fn = confusion[c, :].sum() - tp
        iou = tp / max(tp + fp + fn, 1)
        ious.append(iou)

    return {
        "mIoU": float(np.mean(ious)),
        "per_class_iou": {i: float(ious[i]) for i in range(num_classes)},
        "overall_accuracy": float(np.diag(confusion).sum() / max(confusion.sum(), 1)),
    }


def _check_flash_attention():
    """Check which SDPA backend is available."""
    if not torch.cuda.is_available():
        print("Flash attention: N/A (no CUDA)")
        return
    backends = []
    if hasattr(torch.backends.cuda, 'flash_sdp_enabled'):
        if torch.backends.cuda.flash_sdp_enabled():
            backends.append("flash")
        if torch.backends.cuda.mem_efficient_sdp_enabled():
            backends.append("mem_efficient")
        if torch.backends.cuda.math_sdp_enabled():
            backends.append("math")
    if backends:
        print(f"SDPA backends: {', '.join(backends)}")
    else:
        print("SDPA backends: unknown (PyTorch < 2.0?)")

    # Quick test to see which one actually runs
    try:
        q = torch.randn(1, 4, 64, 24, device="cuda", dtype=torch.float16)
        k = torch.randn(1, 4, 64, 24, device="cuda", dtype=torch.float16)
        v = torch.randn(1, 4, 64, 24, device="cuda", dtype=torch.float16)
        with torch.backends.cuda.sdp_kernel(enable_flash=True, enable_math=False, enable_mem_efficient=False):
            F.scaled_dot_product_attention(q, k, v)
            print("Flash attention: ACTIVE")
    except Exception:
        print("Flash attention: NOT available (using math/mem_efficient fallback)")


def train(model, train_loader, val_loader, cfg, device, checkpoint_dir,
          resume_checkpoint=None, test_loader=None):
    """Full training loop."""
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    _check_flash_attention()

    num_classes = cfg["model"]["num_classes"]
    criterion_sem = nn.CrossEntropyLoss()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg["training"]["learning_rate"],
        weight_decay=cfg["training"]["weight_decay"],
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg["training"]["epochs"],
    )

    aug_cfg = cfg.get("augmentation", {})
    do_densify = aug_cfg.get("densify", True)
    densify_mult = aug_cfg.get("densify_multiplier", 10)
    do_augment = aug_cfg.get("enabled", True)
    offset_weight = cfg["training"].get("offset_weight", 1.0)

    if do_densify:
        print(f"GPU densification: {densify_mult}x (keep all)")
    if do_augment:
        print(f"GPU augmentation: rotation + flip + scale + noise")

    best_miou = 0.0
    start_epoch = 1

    if resume_checkpoint:
        ckpt = torch.load(resume_checkpoint, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        start_epoch = ckpt["epoch"] + 1
        best_miou = ckpt.get("best_miou", 0.0)
        for _ in range(ckpt["epoch"]):
            scheduler.step()
        print(f"Resumed from epoch {ckpt['epoch']}, best mIoU={best_miou:.4f}")

    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda") if use_amp else None
    if use_amp:
        print("Using mixed precision (FP16)")

    total_epochs = cfg["training"]["epochs"]
    CLASS_NAMES = [
        "wall", "floor", "ceiling", "door", "window",
        "cabinet", "bed", "chair", "sofa", "table",
        "bookshelf", "desk", "dresser", "toilet", "sink",
    ]

    for epoch in range(start_epoch, total_epochs + 1):
        t0 = time.time()
        train_loss = train_one_epoch(
            model, train_loader, optimizer, criterion_sem, device, scaler,
            densify=do_densify, densify_multiplier=densify_mult,
            augment=do_augment, offset_weight=offset_weight,
            max_points=cfg["training"].get("max_points", 200000),
        )
        scheduler.step()

        elapsed = time.time() - t0
        remaining = (total_epochs - epoch) * elapsed
        eta_min = remaining / 60

        eval_every = cfg["training"].get("eval_every", 5)
        if epoch % eval_every == 0 or epoch == start_epoch or epoch == 1:
            val_metrics = evaluate(model, val_loader, device, num_classes)
            miou = val_metrics["mIoU"]
            oa = val_metrics["overall_accuracy"]
            lr = optimizer.param_groups[0]["lr"]
            print(
                f"Epoch {epoch:3d}/{total_epochs} | "
                f"loss={train_loss:.4f} | "
                f"mIoU={miou:.4f} | OA={oa:.4f} | "
                f"lr={lr:.6f} | {elapsed:.0f}s | ETA={eta_min:.0f}min"
            )
            # Print per-class IoU for classes with > 0
            for c, iou in val_metrics["per_class_iou"].items():
                if iou > 0:
                    print(f"    {CLASS_NAMES[c]:12s}: {iou:.4f}")

            if miou > best_miou:
                best_miou = miou
                torch.save(model.state_dict(), checkpoint_dir / "best_model.pth")
                print(f"  -> New best mIoU: {best_miou:.4f}")
        else:
            print(
                f"Epoch {epoch:3d}/{total_epochs} | "
                f"loss={train_loss:.4f} | "
                f"{elapsed:.0f}s | ETA={eta_min:.0f}min"
            )

        save_every = cfg["training"].get("save_every", 5)
        if epoch == 1 or epoch % save_every == 0:
            ckpt_path = checkpoint_dir / f"checkpoint_epoch_{epoch}.pth"
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_miou": best_miou,
            }, ckpt_path)
            print(f"  -> Saved: {ckpt_path.name}")

    torch.save(model.state_dict(), checkpoint_dir / "final_model.pth")
    print(f"\nTraining complete. Best mIoU: {best_miou:.4f}")

    if test_loader:
        print("\nEvaluating on test set...")
        test_metrics = evaluate(model, test_loader, device, num_classes)
        print(f"Test mIoU: {test_metrics['mIoU']:.4f} | OA: {test_metrics['overall_accuracy']:.4f}")
