"""Train the wall segmentation model.

Usage:
    python scripts/run_training.py --config config.yaml
    python scripts/run_training.py --config config.yaml --name my_experiment
    python scripts/run_training.py --config config.yaml --resume checkpoints/run_003/checkpoint_epoch_50.pth
"""

import argparse
import yaml
import torch
from torch.utils.data import DataLoader
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.model import PointNetSegmentation, DGCNNSegmentation, RandLANetSegmentation
from src.dataset import WallSegDataset
from src.train import train


def next_run_dir(base: Path) -> Path:
    """Find next available run_XXX directory."""
    base.mkdir(parents=True, exist_ok=True)
    existing = sorted(base.glob("run_*"))
    if not existing:
        return base / "run_001"
    last_num = max(int(d.name.split("_")[1]) for d in existing)
    return base / f"run_{last_num + 1:03d}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--name", type=str, default=None,
                        help="Custom run name (default: auto-increment run_XXX)")
    parser.add_argument("--model", type=str, default="randla", choices=["dgcnn", "pointnet", "randla"],
                        help="Model architecture (default: randla)")
    parser.add_argument("--k", type=int, default=20,
                        help="KNN neighbors for DGCNN (default: 20)")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Create unique run directory
    base = Path("checkpoints")
    if args.name:
        run_dir = base / args.name
    else:
        run_dir = next_run_dir(base)
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"Run directory: {run_dir}")

    # Save config copy for reproducibility
    with open(run_dir / "config.yaml", "w") as f:
        yaml.dump(cfg, f)

    processed = Path(cfg["data"]["processed_root"])
    aug_cfg = cfg.get("augmentation", {})

    print("Loading datasets...")
    train_ds = WallSegDataset(
        processed / "train",
        num_points=cfg["data"]["num_points"],
        block_size=cfg["data"]["block_size"],
        augment=aug_cfg.get("enabled", True),
        augment_cfg=aug_cfg,
        min_points=cfg["data"]["min_points_per_block"],
    )

    val_ds = WallSegDataset(
        processed / "val",
        num_points=cfg["data"]["num_points"],
        block_size=cfg["data"]["block_size"],
        augment=False,
    )

    print(f"Train blocks: {len(train_ds):,} | Val blocks: {len(val_ds):,}")

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg["training"]["batch_size"],
        shuffle=False,  # IterableDataset shuffles internally
        num_workers=cfg["training"]["num_workers"],
        pin_memory=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=cfg["training"]["batch_size"],
        shuffle=False,
        num_workers=cfg["training"]["num_workers"],
        pin_memory=True,
    )

    if args.model == "dgcnn":
        model = DGCNNSegmentation(
            input_dim=cfg["model"]["input_dim"],
            num_classes=cfg["model"]["num_classes"],
            k=args.k,
        ).to(device)
    elif args.model == "randla":
        model = RandLANetSegmentation(
            input_dim=cfg["model"]["input_dim"],
            num_classes=cfg["model"]["num_classes"],
        ).to(device)
    else:
        model = PointNetSegmentation(
            input_dim=cfg["model"]["input_dim"],
            num_classes=cfg["model"]["num_classes"],
        ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {args.model} | Parameters: {total_params:,} | K={args.k if args.model == 'dgcnn' else 'N/A'}")

    train(model, train_loader, val_loader, cfg, device, run_dir, resume_checkpoint=args.resume)


if __name__ == "__main__":
    main()
