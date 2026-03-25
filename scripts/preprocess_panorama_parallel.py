"""Parallel preprocessing of Structured3D panorama depth maps.

Processes multiple zip files in parallel, with thread-pool parallelism
for rooms within each zip.

Usage:
    python scripts/preprocess_panorama_parallel.py --zip-workers 4 --room-workers 4
"""

import zipfile
import io
import os
import sys
import argparse
import numpy as np
from PIL import Image
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import time

RAW_DIR = Path(r"C:\Users\klof\Desktop\New folder (4)\structured3d_raw")
OUTPUT_DIR = Path(r"C:\Users\klof\Desktop\New folder (4)\wall-seg\data\structured3d_panorama_processed")
VOXEL_SIZE = 0.03
WALL_LABEL = 1
MIN_POINTS = 5000


def backproject_panorama(depth, cam):
    H, W = depth.shape
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
    return np.stack([x[valid], y[valid], z[valid]], axis=-1), valid


def voxel_downsample(points, labels, voxel_size):
    if len(points) == 0:
        return points, labels
    voxel_idx = np.floor(points / voxel_size).astype(np.int32)
    _, unique_idx, inverse = np.unique(voxel_idx, axis=0, return_index=True, return_inverse=True)
    n = len(unique_idx)
    counts = np.bincount(inverse, minlength=n).astype(np.float64)
    ds = np.zeros((n, 3), dtype=np.float64)
    for d in range(3):
        ds[:, d] = np.bincount(inverse, weights=points[:, d].astype(np.float64), minlength=n)
    ds /= counts[:, None]
    wall_votes = np.bincount(inverse, weights=(labels == 1).astype(np.float64), minlength=n)
    ds_labels = (wall_votes > counts / 2).astype(np.int64)
    return ds.astype(np.float32), ds_labels


def process_single_zip(args):
    """Process all rooms in a single zip file. Runs in its own process."""
    zip_path, room_workers = args
    zip_name = Path(zip_path).name

    zf = zipfile.ZipFile(zip_path)

    # Find all scene/room pairs
    scene_rooms = {}
    for name in zf.namelist():
        parts = name.split("/")
        if (len(parts) >= 5 and parts[1].startswith("scene_")
                and parts[2] == "2D_rendering" and parts[3].isdigit()):
            scene_rooms.setdefault(parts[1], set()).add(parts[3])

    # Flatten to list of (scene, room) pairs
    tasks = []
    for scene in sorted(scene_rooms):
        for room in sorted(scene_rooms[scene]):
            out_name = f"{scene}_room{room}"
            out_path = OUTPUT_DIR / out_name
            if out_path.exists() and (out_path / "coord.npy").exists():
                continue
            tasks.append((scene, room))

    if not tasks:
        print(f"  {zip_name}: all rooms already done", flush=True)
        zf.close()
        return 0, 0

    # Pre-read all needed data from zip (zip reading is not thread-safe)
    room_data = {}
    for scene, room in tasks:
        try:
            cam_txt = zf.read(
                f"Structured3D/{scene}/2D_rendering/{room}/panorama/camera_xyz.txt"
            ).decode().strip()
            depth_bytes = zf.read(
                f"Structured3D/{scene}/2D_rendering/{room}/panorama/full/depth.png"
            )
            sem_bytes = zf.read(
                f"Structured3D/{scene}/2D_rendering/{room}/panorama/full/semantic.png"
            )
            room_data[(scene, room)] = (cam_txt, depth_bytes, sem_bytes)
        except Exception:
            pass

    zf.close()

    def process_room(key):
        scene, room = key
        if key not in room_data:
            return 0
        cam_txt, depth_bytes, sem_bytes = room_data[key]

        cx, cy, cz = [float(x) for x in cam_txt.split()]
        cam = np.array([cx, cy, cz]) / 1000.0

        depth = np.array(Image.open(io.BytesIO(depth_bytes)))
        sem = np.array(Image.open(io.BytesIO(sem_bytes)))

        pts, valid = backproject_panorama(depth, cam)
        binary = (sem[valid] == WALL_LABEL).astype(np.int64)
        pts, binary = voxel_downsample(pts, binary, VOXEL_SIZE)

        if len(pts) < MIN_POINTS:
            return 0

        out_name = f"{scene}_room{room}"
        out_path = OUTPUT_DIR / out_name
        out_path.mkdir(parents=True, exist_ok=True)
        np.save(str(out_path / "coord.npy"), pts)
        np.save(str(out_path / "label.npy"), binary)
        return len(pts)

    # Process rooms in parallel threads
    saved = 0
    total_pts = 0
    with ThreadPoolExecutor(max_workers=room_workers) as pool:
        for n_pts in pool.map(process_room, list(room_data.keys())):
            if n_pts > 0:
                saved += 1
                total_pts += n_pts

    print(f"  {zip_name}: {saved} rooms saved ({total_pts:,} pts)", flush=True)
    return saved, total_pts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--zip-workers", type=int, default=4,
                        help="Number of zip files to process in parallel")
    parser.add_argument("--room-workers", type=int, default=4,
                        help="Number of rooms to process in parallel within each zip")
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    zip_files = sorted(RAW_DIR.glob("Structured3D_panorama_*.zip"))
    print(f"Found {len(zip_files)} panorama zips")
    print(f"Parallelism: {args.zip_workers} zip workers x {args.room_workers} room workers")

    t0 = time.time()
    tasks = [(str(zf), args.room_workers) for zf in zip_files]

    total_rooms = 0
    total_pts = 0

    with ProcessPoolExecutor(max_workers=args.zip_workers) as pool:
        for saved, pts in pool.map(process_single_zip, tasks):
            total_rooms += saved
            total_pts += pts

    elapsed = time.time() - t0
    print(f"\nDone! {total_rooms:,} rooms, {total_pts:,} points ({elapsed:.0f}s)")
    print(f"Output: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
