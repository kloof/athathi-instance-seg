"""Preprocess Structured3D panorama depth maps into wall-seg training data.

Back-projects equirectangular depth maps to 3D point clouds using the 'full'
renders (with furniture). Uses semantic.png for wall labels (label 1 = wall).

This produces realistic training data with furniture occlusion, unlike the
structural-only polygon approach.

Usage:
    python scripts/preprocess_panorama.py
"""

import zipfile
import io
import os
import sys
import numpy as np
from PIL import Image
from pathlib import Path
from collections import Counter
import time

# --- Config ---
RAW_DIR = Path(r"C:\Users\klof\Desktop\New folder (4)\structured3d_raw")
OUTPUT_DIR = Path(r"C:\Users\klof\Desktop\New folder (4)\wall-seg\data\structured3d_panorama_processed")
VOXEL_SIZE = 0.03  # meters
WALL_LABEL = 1     # Structured3D semantic label for wall
MIN_POINTS = 5000


def backproject_panorama(depth, cam, H, W):
    """Back-project equirectangular depth to 3D points.

    Args:
        depth: (H, W) uint16 depth in mm
        cam: (3,) camera position in meters
        H, W: image dimensions

    Returns:
        (N, 3) float32 points in world coordinates
        (N,) bool valid mask
    """
    u, v = np.meshgrid(np.arange(W, dtype=np.float32) + 0.5,
                        np.arange(H, dtype=np.float32) + 0.5)
    phi = u / W * 2 * np.pi
    theta = v / H * np.pi
    d = depth.astype(np.float32) / 1000.0  # mm -> m

    x = d * np.sin(theta) * np.cos(phi) + cam[0]
    y = d * np.sin(theta) * np.sin(phi) + cam[1]
    z = d * np.cos(theta) + cam[2]

    valid = depth > 0
    pts = np.stack([x[valid], y[valid], z[valid]], axis=-1)
    return pts, valid


def voxel_downsample(points, labels, voxel_size):
    """Voxel downsample with majority-vote labels."""
    if len(points) == 0:
        return points, labels

    voxel_idx = np.floor(points / voxel_size).astype(np.int32)
    _, unique_idx, inverse = np.unique(voxel_idx, axis=0, return_index=True, return_inverse=True)
    n_voxels = len(unique_idx)
    counts = np.bincount(inverse, minlength=n_voxels).astype(np.float64)

    ds_points = np.zeros((n_voxels, 3), dtype=np.float64)
    for d in range(3):
        ds_points[:, d] = np.bincount(inverse, weights=points[:, d].astype(np.float64), minlength=n_voxels)
    ds_points /= counts[:, None]

    wall_votes = np.bincount(inverse, weights=(labels == 1).astype(np.float64), minlength=n_voxels)
    ds_labels = (wall_votes > counts / 2).astype(np.int64)

    return ds_points.astype(np.float32), ds_labels


def process_room(scene_name, room_id, zf):
    """Process one room from a panorama depth map.

    Returns:
        (points, labels, stats) or (None, None, {})
    """
    try:
        cam_txt = zf.read(
            f"Structured3D/{scene_name}/2D_rendering/{room_id}/panorama/camera_xyz.txt"
        ).decode().strip()
        cx, cy, cz = [float(x) for x in cam_txt.split()]
        cam = np.array([cx, cy, cz]) / 1000.0

        depth = np.array(Image.open(io.BytesIO(
            zf.read(f"Structured3D/{scene_name}/2D_rendering/{room_id}/panorama/full/depth.png")
        )))

        sem = np.array(Image.open(io.BytesIO(
            zf.read(f"Structured3D/{scene_name}/2D_rendering/{room_id}/panorama/full/semantic.png")
        )))

        H, W = depth.shape
        pts, valid = backproject_panorama(depth, cam, H, W)
        lbl = sem[valid]

        # Binary: wall=1, everything else=0
        binary = (lbl == WALL_LABEL).astype(np.int64)

        # Voxel downsample
        pts, binary = voxel_downsample(pts, binary, VOXEL_SIZE)

        if len(pts) < MIN_POINTS:
            return None, None, {}

        stats = {
            "total_points": len(pts),
            "wall_points": int((binary == 1).sum()),
            "wall_ratio": float((binary == 1).mean()),
        }

        return pts, binary, stats
    except Exception as e:
        return None, None, {}


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    zip_files = sorted(RAW_DIR.glob("Structured3D_panorama_*.zip"))
    print(f"Found {len(zip_files)} panorama zips")

    total_stats = Counter()
    good_rooms = 0
    skipped = 0
    t0 = time.time()

    for zf_path in zip_files:
        print(f"\nProcessing {zf_path.name}...", flush=True)
        zf = zipfile.ZipFile(str(zf_path))

        # Find all scene/room pairs in this zip
        scene_rooms = {}
        for name in zf.namelist():
            parts = name.split("/")
            if (len(parts) >= 5 and parts[1].startswith("scene_")
                    and parts[2] == "2D_rendering" and parts[3].isdigit()):
                scene_rooms.setdefault(parts[1], set()).add(parts[3])

        total_rooms_in_zip = sum(len(rooms) for rooms in scene_rooms.values())
        print(f"  {len(scene_rooms)} scenes, {total_rooms_in_zip} rooms", flush=True)

        room_count = 0
        for scene in sorted(scene_rooms):
            for room_id in sorted(scene_rooms[scene]):
                # Save as scene_XXXXX_roomYYYYYY
                out_name = f"{scene}_room{room_id}"
                out_path = OUTPUT_DIR / out_name

                if out_path.exists() and (out_path / "coord.npy").exists():
                    good_rooms += 1
                    room_count += 1
                    continue

                points, labels, stats = process_room(scene, room_id, zf)

                if points is None:
                    skipped += 1
                    room_count += 1
                    continue

                out_path.mkdir(parents=True, exist_ok=True)
                np.save(str(out_path / "coord.npy"), points)
                np.save(str(out_path / "label.npy"), labels)

                good_rooms += 1
                total_stats["total_points"] += stats["total_points"]
                total_stats["wall_points"] += stats["wall_points"]
                room_count += 1

                if room_count % 100 == 0:
                    elapsed = time.time() - t0
                    print(
                        f"    {room_count}/{total_rooms_in_zip} rooms, "
                        f"{good_rooms:,} saved, "
                        f"{elapsed:.0f}s elapsed", flush=True
                    )

        zf.close()

    elapsed = time.time() - t0
    print(f"\nDone! {good_rooms:,} rooms saved, {skipped} skipped ({elapsed:.0f}s)")
    print(f"Total points: {total_stats['total_points']:,}")
    if total_stats["total_points"] > 0:
        print(f"Wall ratio: {total_stats['wall_points']/total_stats['total_points']:.1%}")
    print(f"Output: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
