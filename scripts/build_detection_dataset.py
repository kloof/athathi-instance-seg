"""Build the FULL instance segmentation dataset from all 18 Structured3D panorama zips.

Auto-downloads any missing zips, then extracts all rooms with 16 semantic classes
(15 named + "other") + per-point instance IDs. Splits into train/val/test.

Usage:
    python scripts/build_detection_dataset.py
    python scripts/build_detection_dataset.py --skip-download
    python scripts/build_detection_dataset.py --connections 32
"""

import zipfile
import numpy as np
from pathlib import Path
from collections import Counter, defaultdict
import time
import argparse
import random
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))

from build_small_detection_dataset import (
    process_room, CLASS_NAMES, VOXEL_SIZE, MIN_POINTS,
)

RAW_DIR = Path(r"C:\Users\klof\Desktop\New folder (4)\structured3d_raw")
OUTPUT_DIR = Path(r"C:\Users\klof\Desktop\New folder (4)\wall-seg\data\detection_full")

BASE_URL = "https://zju-kjl-jointlab-azure.kujiale.com/Structured3D"
PANORAMA_ZIPS = [f"Structured3D_panorama_{i:02d}.zip" for i in range(18)]
BBOX_ZIP = "Structured3D_bbox.zip"

TRAIN_RATIO = 0.72
VAL_RATIO = 0.13


def download_if_missing(filename, raw_dir, num_connections=16):
    filepath = raw_dir / filename
    if filepath.exists():
        return True
    url = f"{BASE_URL}/{filename}"
    print(f"\n  Downloading {filename} ({num_connections} connections)...")
    from parallel_download import download_file
    return download_file(url, str(filepath), num_connections=num_connections)


def ensure_all_data(raw_dir, num_connections=16, skip_download=False):
    all_needed = PANORAMA_ZIPS + [BBOX_ZIP]
    missing = [f for f in all_needed if not (raw_dir / f).exists()]
    if not missing:
        print(f"All {len(all_needed)} required zips found")
        return True
    if skip_download:
        print(f"Missing {len(missing)} files (--skip-download set)")
        for f in missing:
            print(f"  {f}")
        return False
    for f in missing:
        if not download_if_missing(f, raw_dir, num_connections):
            return False
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--connections", type=int, default=16)
    parser.add_argument("--skip-download", action="store_true")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    if not ensure_all_data(RAW_DIR, args.connections, args.skip_download):
        return

    bbox_zf = zipfile.ZipFile(str(RAW_DIR / BBOX_ZIP))
    bbox_namelist = set(bbox_zf.namelist())

    # Scan all zips
    all_rooms = []
    for zip_idx, zip_name in enumerate(PANORAMA_ZIPS):
        zf = zipfile.ZipFile(str(RAW_DIR / zip_name))
        scene_rooms = {}
        for name in zf.namelist():
            parts = name.split("/")
            if (len(parts) >= 5 and parts[1].startswith("scene_")
                    and parts[2] == "2D_rendering" and parts[3].isdigit()):
                scene_rooms.setdefault(parts[1], set()).add(parts[3])
        zf.close()
        n = sum(len(r) for r in scene_rooms.values())
        for scene in sorted(scene_rooms):
            for room_id in sorted(scene_rooms[scene]):
                all_rooms.append((zip_idx, scene, room_id))
        print(f"  {zip_name}: {len(scene_rooms)} scenes, {n} rooms")

    print(f"\nTotal: {len(all_rooms)} rooms")

    random.shuffle(all_rooms)
    n = len(all_rooms)
    n_train = int(n * TRAIN_RATIO)
    n_val = int(n * VAL_RATIO)
    splits = {
        "train": all_rooms[:n_train],
        "val": all_rooms[n_train:n_train + n_val],
        "test": all_rooms[n_train + n_val:],
    }
    for split in ["train", "val", "test"]:
        (OUTPUT_DIR / split).mkdir(parents=True, exist_ok=True)
        print(f"  {split}: {len(splits[split])} rooms")

    # Group by zip for efficient processing
    zip_to_tasks = defaultdict(list)
    for split_name, rooms in splits.items():
        for zip_idx, scene, room_id in rooms:
            zip_to_tasks[zip_idx].append((split_name, scene, room_id))

    t0 = time.time()
    total_processed = 0
    total_skipped = 0
    total_instances = 0

    for zip_idx in range(18):
        tasks = zip_to_tasks.get(zip_idx, [])
        if not tasks:
            continue
        zip_name = PANORAMA_ZIPS[zip_idx]
        print(f"\n  {zip_name} ({len(tasks)} rooms)...", flush=True)
        panorama_zf = zipfile.ZipFile(str(RAW_DIR / zip_name))

        for i, (split_name, scene, room_id) in enumerate(tasks):
            out_path = OUTPUT_DIR / split_name / f"{scene}_room{room_id}"
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

            if (i + 1) % 100 == 0:
                elapsed = time.time() - t0
                rate = total_processed / max(elapsed, 1)
                remaining = (len(all_rooms) - total_processed - total_skipped) / max(rate, 0.01)
                print(f"    {i+1}/{len(tasks)} | {total_processed} total | ETA {remaining/60:.0f}min", flush=True)

        panorama_zf.close()

    bbox_zf.close()
    elapsed = time.time() - t0
    print(f"\nDone! {total_processed} rooms, {total_skipped} skipped ({elapsed/60:.1f}min)")
    print(f"Total instances: {total_instances}")


if __name__ == "__main__":
    main()
