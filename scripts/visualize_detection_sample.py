"""Visualize instance segmentation data pipeline.

Saves 3 PLY files:
  1. raw.ply        — points colored by semantic class + instance boundaries
  2. densified.ply  — 10x densified, colored by instance ID
  3. augmented.ply  — densified + augmented (what model sees)

Usage:
    python scripts/visualize_detection_sample.py
    python scripts/visualize_detection_sample.py --room data/detection_small/train/scene_00123_room1195
"""

import argparse
import numpy as np
import torch
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.gpu_augment import densify_batch_gpu
from src.detection_train import _gpu_augment

CLASS_NAMES = [
    "wall", "floor", "ceiling", "door", "window",
    "cabinet", "bed", "chair", "sofa", "table",
    "bookshelf", "desk", "dresser", "toilet", "sink",
]

SEM_COLORS = np.array([
    [180, 180, 180], [120, 80, 40], [200, 200, 255], [0, 200, 0], [0, 200, 255],
    [255, 128, 0], [255, 0, 0], [0, 0, 255], [128, 0, 255], [255, 255, 0],
    [0, 128, 0], [128, 128, 0], [200, 100, 50], [255, 200, 200], [0, 128, 128],
], dtype=np.uint8)

# Random bright colors for instances
def instance_colors(inst_labels):
    """Assign a distinct color per instance ID. 0 = gray (stuff)."""
    rng = np.random.RandomState(42)
    max_id = inst_labels.max() + 1
    palette = rng.randint(50, 255, (max_id, 3)).astype(np.uint8)
    palette[0] = [140, 140, 140]  # stuff = gray
    return palette[inst_labels]


def write_ply(path, points, colors):
    N = len(points)
    with open(path, "wb") as f:
        header = (
            f"ply\nformat binary_little_endian 1.0\n"
            f"element vertex {N}\n"
            f"property float x\nproperty float y\nproperty float z\n"
            f"property uchar red\nproperty uchar green\nproperty uchar blue\n"
            f"end_header\n"
        )
        f.write(header.encode())
        dt = np.dtype([("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
                        ("r", "u1"), ("g", "u1"), ("b", "u1")])
        arr = np.zeros(N, dtype=dt)
        arr["x"] = points[:, 0]; arr["y"] = points[:, 1]; arr["z"] = points[:, 2]
        arr["r"] = colors[:, 0]; arr["g"] = colors[:, 1]; arr["b"] = colors[:, 2]
        f.write(arr.tobytes())


def sem_to_colors(sem_labels):
    colors = np.zeros((len(sem_labels), 3), dtype=np.uint8)
    for c in range(len(SEM_COLORS)):
        colors[sem_labels == c] = SEM_COLORS[c]
    return colors


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--room", default=None)
    parser.add_argument("--densify", type=int, default=10)
    parser.add_argument("--output", default="data/detection_small/viz")
    args = parser.parse_args()

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.room:
        room_dir = Path(args.room)
    else:
        train_dir = Path("data/detection_small/train")
        for r in sorted(train_dir.iterdir()):
            if r.is_dir() and (r / "instance.npy").exists():
                inst = np.load(str(r / "instance.npy"))
                if len(set(inst) - {0}) >= 3:
                    room_dir = r
                    break

    pts = np.load(str(room_dir / "coord.npy"))
    sem = np.load(str(room_dir / "semantic.npy"))
    inst = np.load(str(room_dir / "instance.npy"))

    n_instances = len(set(inst) - {0})
    print(f"Room: {room_dir.name}")
    print(f"  {len(pts):,} points, {n_instances} instances")
    for iid in sorted(set(inst) - {0}):
        mask = inst == iid
        cls = sem[mask][0] if mask.any() else 0
        name = CLASS_NAMES[cls] if cls < len(CLASS_NAMES) else f"cls{cls}"
        print(f"    inst {iid}: {name} ({mask.sum():,} pts)")

    # Center XY
    center_xy = pts[:, :2].mean(axis=0)
    pts[:, 0] -= center_xy[0]
    pts[:, 1] -= center_xy[1]

    # 1. RAW — semantic colors
    write_ply(str(out_dir / "1_raw_semantic.ply"), pts, sem_to_colors(sem))
    print(f"  -> 1_raw_semantic.ply ({len(pts):,} pts)")

    # 1b. RAW — instance colors
    write_ply(str(out_dir / "1_raw_instance.ply"), pts, instance_colors(inst))
    print(f"  -> 1_raw_instance.ply ({len(pts):,} pts, {n_instances} instances)")

    # 2. DENSIFIED — keep all
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pts_t = torch.from_numpy(pts).unsqueeze(0).to(device)
    sem_t = torch.from_numpy(sem).unsqueeze(0).to(device).long()

    with torch.no_grad():
        dense_pts, dense_sem = densify_batch_gpu(
            pts_t, sem_t, target_multiplier=args.densify, keep_all=True,
        )
    pts_d = dense_pts[0].cpu().numpy()
    sem_d = dense_sem[0].cpu().numpy()

    # Expand instance labels to match dense points
    inst_d = np.tile(inst, args.densify)[:len(pts_d)]

    write_ply(str(out_dir / "2_densified.ply"), pts_d, instance_colors(inst_d))
    print(f"  -> 2_densified.ply ({len(pts_d):,} pts, instance colors)")

    # 3. AUGMENTED
    with torch.no_grad():
        aug_pts, aug_sem = densify_batch_gpu(
            pts_t, sem_t, target_multiplier=args.densify, keep_all=True,
        )
        aug_pts = _gpu_augment(aug_pts, device)

    pts_a = aug_pts[0].cpu().numpy()
    sem_a = aug_sem[0].cpu().numpy()

    write_ply(str(out_dir / "3_augmented.ply"), pts_a, sem_to_colors(sem_a))
    print(f"  -> 3_augmented.ply ({len(pts_a):,} pts, augmented)")

    print(f"\nAll saved to {out_dir}/")


if __name__ == "__main__":
    main()
