"""Build a small instance segmentation dataset from panorama_00.zip (~1115 rooms).

Extracts point clouds with 15 semantic classes + per-point instance IDs from
panorama depth/semantic/instance images. Splits into train/val/test.

Each room gets:
  - coord.npy (N, 3) float32 — 3D points
  - semantic.npy (N,) uint8 — semantic class [0-14]
  - instance.npy (N,) int32 — instance ID per point (0 = no instance / stuff class)

Bounding boxes are derived at inference time from segmented instances.

Usage:
    python scripts/build_small_detection_dataset.py
"""

import zipfile
import io
import json
import math
import numpy as np
from PIL import Image
from pathlib import Path
from collections import Counter
import time
import argparse
import random

# --- Config ---
RAW_DIR = Path(r"C:\Users\klof\Desktop\New folder (4)\structured3d_raw")
OUTPUT_DIR = Path(r"C:\Users\klof\Desktop\New folder (4)\wall-seg\data\detection_small")
PANORAMA_ZIP = "Structured3D_panorama_00.zip"
BBOX_ZIP = "Structured3D_bbox.zip"  # only for instance.png, not bbox_3d.json
VOXEL_SIZE = 0.03
MIN_POINTS = 5000

TRAIN_RATIO = 0.72
VAL_RATIO = 0.13
TEST_RATIO = 0.15

# NYU-40 → our class IDs
# Everything not explicitly mapped → class 15 (other/background)
NYU40_TO_DET = {
    1: 0, 2: 1, 22: 2, 8: 3, 9: 4,       # structural
    3: 5, 4: 6, 5: 7, 6: 8, 7: 9,         # furniture
    10: 10, 14: 11, 17: 12, 33: 13, 34: 14,
}
NUM_CLASSES = 16  # 15 named classes + 1 "other"
OTHER_CLASS_ID = 15
CLASS_NAMES = [
    "wall", "floor", "ceiling", "door", "window",
    "cabinet", "bed", "chair", "sofa", "table",
    "bookshelf", "desk", "dresser", "toilet", "sink",
    "other",
]
# "Thing" classes get instance IDs; "stuff" classes (wall/floor/ceiling/other) don't
THING_CLASS_IDS = {3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14}


def backproject_panorama(depth, cam, H, W):
    u, v = np.meshgrid(
        np.arange(W, dtype=np.float32) + 0.5,
        np.arange(H, dtype=np.float32) + 0.5,
    )
    phi = u / W * 2 * np.pi
    theta = v / H * np.pi
    d = depth.astype(np.float32) / 1000.0
    x = d * np.sin(theta) * np.cos(phi) + cam[0]
    y = d * np.sin(theta) * np.sin(phi) + cam[1]
    z = d * np.cos(theta) + cam[2]
    valid = depth > 0
    pts = np.stack([x[valid], y[valid], z[valid]], axis=-1)
    return pts, valid


def voxel_downsample_with_instances(points, sem_labels, inst_labels, voxel_size):
    """Voxel downsample with majority-vote for both semantic and instance labels."""
    if len(points) == 0:
        return points, sem_labels, inst_labels

    voxel_idx = np.floor(points / voxel_size).astype(np.int32)
    _, unique_idx, inverse = np.unique(voxel_idx, axis=0, return_index=True, return_inverse=True)
    n_voxels = len(unique_idx)
    counts = np.bincount(inverse, minlength=n_voxels).astype(np.float64)

    # Average positions
    ds_points = np.zeros((n_voxels, 3), dtype=np.float64)
    for d in range(3):
        ds_points[:, d] = np.bincount(inverse, weights=points[:, d].astype(np.float64), minlength=n_voxels)
    ds_points /= counts[:, None]

    # Majority vote semantic labels
    best_votes = np.zeros(n_voxels, dtype=np.float64)
    ds_sem = np.zeros(n_voxels, dtype=np.uint8)
    for c in range(NUM_CLASSES):
        c_votes = np.bincount(inverse, weights=(sem_labels == c).astype(np.float64), minlength=n_voxels)
        better = c_votes > best_votes
        ds_sem[better] = c
        best_votes = np.maximum(best_votes, c_votes)

    # Majority vote instance labels (use the most frequent instance ID per voxel)
    # Simple approach: pick the instance ID of the point nearest to voxel center
    ds_inst = inst_labels[unique_idx].copy()

    return ds_points.astype(np.float32), ds_sem, ds_inst


def process_room(scene_name, room_id, panorama_zf, bbox_zf, bbox_namelist):
    try:
        cam_txt = panorama_zf.read(
            f"Structured3D/{scene_name}/2D_rendering/{room_id}/panorama/camera_xyz.txt"
        ).decode().strip()
        cx, cy, cz = [float(x) for x in cam_txt.split()]
        cam = np.array([cx, cy, cz]) / 1000.0

        depth = np.array(Image.open(io.BytesIO(
            panorama_zf.read(f"Structured3D/{scene_name}/2D_rendering/{room_id}/panorama/full/depth.png")
        )))
        sem_img = np.array(Image.open(io.BytesIO(
            panorama_zf.read(f"Structured3D/{scene_name}/2D_rendering/{room_id}/panorama/full/semantic.png")
        )))

        H, W = depth.shape
        pts, valid = backproject_panorama(depth, cam, H, W)
        raw_sem = sem_img[valid]

        # Map NYU-40 → our classes. Unmapped → "other" (class 15). Keep ALL points.
        mapped_sem = np.full(len(raw_sem), OTHER_CLASS_ID, dtype=np.uint8)
        for nyu_id, det_id in NYU40_TO_DET.items():
            mapped_sem[raw_sem == nyu_id] = det_id

        if len(pts) < MIN_POINTS:
            return None, None, None, {}

        # Instance labels from instance.png
        instance_path = f"Structured3D/{scene_name}/2D_rendering/{room_id}/panorama/full/instance.png"
        if instance_path in bbox_namelist:
            instance_img = np.array(Image.open(io.BytesIO(bbox_zf.read(instance_path))))
            raw_inst = instance_img[valid].astype(np.int32)
        else:
            raw_inst = np.zeros(len(pts), dtype=np.int32)

        # Zero out instance IDs for "stuff" classes (wall/floor/ceiling) — only "things" get instances
        for i in range(len(raw_inst)):
            if mapped_sem[i] not in THING_CLASS_IDS:
                raw_inst[i] = 0

        # Also zero out tiny instances (< 50 points)
        inst_ids, inst_counts = np.unique(raw_inst, return_counts=True)
        small_insts = set(inst_ids[inst_counts < 50]) | {0}
        for i in range(len(raw_inst)):
            if raw_inst[i] in small_insts:
                raw_inst[i] = 0

        # Renumber instances to be contiguous (1, 2, 3, ...)
        unique_insts = sorted(set(raw_inst) - {0})
        remap = {0: 0}
        for new_id, old_id in enumerate(unique_insts, start=1):
            remap[old_id] = new_id
        inst_remapped = np.array([remap[x] for x in raw_inst], dtype=np.int32)

        # Voxel downsample
        pts, mapped_sem, inst_remapped = voxel_downsample_with_instances(
            pts, mapped_sem, inst_remapped, VOXEL_SIZE
        )
        if len(pts) < MIN_POINTS:
            return None, None, None, {}

        n_instances = len(set(inst_remapped) - {0})
        stats = {
            "points": len(pts),
            "instances": n_instances,
        }

        return pts, mapped_sem, inst_remapped, stats

    except KeyboardInterrupt:
        raise
    except Exception as e:
        print(f"  WARNING: Failed {scene_name}/{room_id}: {e}")
        return None, None, None, {}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    panorama_path = RAW_DIR / PANORAMA_ZIP
    bbox_path = RAW_DIR / BBOX_ZIP
    assert panorama_path.exists(), f"Not found: {panorama_path}"
    assert bbox_path.exists(), f"Not found: {bbox_path}"

    for split in ["train", "val", "test"]:
        (OUTPUT_DIR / split).mkdir(parents=True, exist_ok=True)

    print(f"Opening {PANORAMA_ZIP}...")
    panorama_zf = zipfile.ZipFile(str(panorama_path))
    bbox_zf = zipfile.ZipFile(str(bbox_path))
    bbox_namelist = set(bbox_zf.namelist())

    scene_rooms = {}
    for name in panorama_zf.namelist():
        parts = name.split("/")
        if (len(parts) >= 5 and parts[1].startswith("scene_")
                and parts[2] == "2D_rendering" and parts[3].isdigit()):
            scene_rooms.setdefault(parts[1], set()).add(parts[3])

    all_pairs = []
    for scene in sorted(scene_rooms):
        for room_id in sorted(scene_rooms[scene]):
            all_pairs.append((scene, room_id))

    print(f"Found {len(all_pairs)} rooms in {len(scene_rooms)} scenes")

    random.shuffle(all_pairs)
    n = len(all_pairs)
    n_train = int(n * TRAIN_RATIO)
    n_val = int(n * VAL_RATIO)
    splits = {
        "train": all_pairs[:n_train],
        "val": all_pairs[n_train:n_train + n_val],
        "test": all_pairs[n_train + n_val:],
    }
    for s, pairs in splits.items():
        print(f"  {s}: {len(pairs)} rooms")

    t0 = time.time()
    total_processed = 0
    total_skipped = 0
    total_instances = 0
    class_counts = Counter()

    for split_name, pairs in splits.items():
        print(f"\nProcessing {split_name} ({len(pairs)} rooms)...")
        split_dir = OUTPUT_DIR / split_name

        for i, (scene, room_id) in enumerate(pairs):
            out_name = f"{scene}_room{room_id}"
            out_path = split_dir / out_name

            if out_path.exists() and (out_path / "coord.npy").exists():
                total_processed += 1
                continue

            pts, sem, inst, stats = process_room(scene, room_id, panorama_zf, bbox_zf, bbox_namelist)

            if pts is None:
                total_skipped += 1
                continue

            out_path.mkdir(parents=True, exist_ok=True)
            np.save(str(out_path / "coord.npy"), pts)
            np.save(str(out_path / "semantic.npy"), sem)
            np.save(str(out_path / "instance.npy"), inst)

            total_processed += 1
            total_instances += stats["instances"]

            if (i + 1) % 50 == 0:
                elapsed = time.time() - t0
                print(f"  {i+1}/{len(pairs)} | {total_processed} saved | {elapsed:.0f}s", flush=True)

    panorama_zf.close()
    bbox_zf.close()

    elapsed = time.time() - t0
    print(f"\n{'='*60}")
    print(f"Extraction done! {total_processed} rooms, {total_skipped} skipped ({elapsed:.0f}s)")
    print(f"Total thing instances: {total_instances}")

    for split_name in ["train", "val", "test"]:
        split_dir = OUTPUT_DIR / split_name
        rooms = [d for d in split_dir.iterdir() if d.is_dir()]
        n_inst = sum(
            len(set(np.load(str(d / "instance.npy"))) - {0})
            for d in rooms if (d / "instance.npy").exists()
        )
        print(f"  {split_name}: {len(rooms)} rooms, {n_inst} instances")

    print(f"\nOutput: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
