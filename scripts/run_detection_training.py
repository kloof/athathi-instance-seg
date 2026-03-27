"""Train instance segmentation (PTv2 dual-head: semantic + offset).

Usage:
    python scripts/run_detection_training.py
    python scripts/run_detection_training.py --config config_detection.yaml
    python scripts/run_detection_training.py --resume checkpoints/det_run_001/checkpoint_epoch_1.pth
"""

import argparse
import yaml
import torch
from torch.utils.data import DataLoader
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.detection_model import LitePTInstanceSeg
from src.detection_dataset import InstanceSegDataset, instance_collate
from src.detection_train import train
from src.ptv3_wrapper import PTv3InstanceSeg


def next_run_dir(base: Path) -> Path:
    base.mkdir(parents=True, exist_ok=True)
    existing = sorted(base.glob("det_run_*"))
    if not existing:
        return base / "det_run_001"
    last_num = max(int(d.name.split("_")[-1]) for d in existing)
    return base / f"det_run_{last_num + 1:03d}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config_detection.yaml")
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--name", type=str, default=None)
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    base = Path("checkpoints")
    run_dir = base / args.name if args.name else next_run_dir(base)
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"Run directory: {run_dir}")

    with open(run_dir / "config.yaml", "w") as f:
        yaml.dump(cfg, f)

    data_root = Path(cfg["data"]["processed_root"])

    print("Loading datasets...")
    max_points = cfg["training"].get("max_points", 100000)
    train_ds = InstanceSegDataset(data_root / "train", max_points=max_points)
    val_ds = InstanceSegDataset(data_root / "val", max_points=max_points)
    test_ds = InstanceSegDataset(data_root / "test", max_points=max_points)
    print(f"Train: {len(train_ds):,} | Val: {len(val_ds):,} | Test: {len(test_ds):,}")

    batch_size = cfg["training"]["batch_size"]
    num_workers = cfg["training"].get("num_workers", 2)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True, drop_last=True,
        collate_fn=instance_collate,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
        collate_fn=instance_collate,
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
        collate_fn=instance_collate,
    )

    model_type = cfg["model"].get("type", "litept")
    if model_type == "ptv3":
        model = PTv3InstanceSeg(
            input_dim=cfg["model"]["input_dim"],
            num_classes=cfg["model"]["num_classes"],
        ).to(device)
        model_name = "PTv3InstanceSeg"
    else:
        model = LitePTInstanceSeg(
            input_dim=cfg["model"]["input_dim"],
            num_classes=cfg["model"]["num_classes"],
        ).to(device)
        model_name = "LitePTInstanceSeg"

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {model_name} | Parameters: {total_params:,}")

    train(model, train_loader, val_loader, cfg, device, run_dir,
          resume_checkpoint=args.resume, test_loader=test_loader)


if __name__ == "__main__":
    main()
