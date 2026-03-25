"""Preprocess Pointcept Structured3D data into training blocks.

Reads coord.npy + segment.npy from the Pointcept format,
voxel downsamples, splits into blocks, saves as chunk files.

Usage:
    python scripts/preprocess_pointcept.py --workers 8
"""

import argparse
import numpy as np
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import time
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.preprocess import voxel_downsample, split_into_blocks, sample_block


POINTCEPT_ROOT = Path("data/structured3d_pointcept")
OUT_ROOT = Path("data/processed")
WALL_LABEL = 1
VOXEL_SIZE = 0.03
BLOCK_SIZE = 1.0
NUM_POINTS = 4096
MIN_POINTS = 100
BLOCKS_PER_CHUNK = 2000


def process_room(room_dir: str) -> list:
    """Load one room, voxel downsample, split into blocks, return list of (pts, lbl, rmin, rmax)."""
    room_dir = Path(room_dir)
    try:
        coords = np.load(room_dir / "coord.npy")           # (N, 3) float32
        segment = np.load(room_dir / "segment.npy").flatten()  # (N,) int16

        # Binary labels: wall=1, everything else=0
        labels = (segment == WALL_LABEL).astype(np.int64)

        # Filter void (label 0)
        valid = segment != 0
        coords = coords[valid]
        labels = labels[valid]

        if coords.shape[0] < MIN_POINTS:
            return []

        # Voxel downsample
        coords, labels = voxel_downsample(coords, labels, voxel_size=VOXEL_SIZE)

        if coords.shape[0] < MIN_POINTS:
            return []

        room_min = coords.min(axis=0)
        room_max = coords.max(axis=0)

        # Split into blocks
        blocks = split_into_blocks(coords, labels, BLOCK_SIZE, MIN_POINTS)

        result = []
        for block_pts, block_lbl in blocks:
            block_pts, block_lbl = sample_block(block_pts, block_lbl, NUM_POINTS)
            result.append((block_pts, block_lbl, room_min, room_max))

        return result
    except Exception as e:
        return []


def process_split(split: str, workers: int):
    split_dir = POINTCEPT_ROOT / split
    if not split_dir.exists():
        print(f"  {split}: directory not found")
        return

    out_dir = OUT_ROOT / split / "blocks"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Skip if already done
    if list(out_dir.glob("blocks_*.npz")):
        print(f"  {split}: already preprocessed")
        return

    # Collect all room directories
    room_dirs = []
    for scene_dir in sorted(split_dir.iterdir()):
        if not scene_dir.is_dir():
            continue
        for room_dir in sorted(scene_dir.iterdir()):
            if not room_dir.is_dir():
                continue
            room_dirs.append(str(room_dir))

    print(f"  {split}: {len(room_dirs)} rooms, processing with {workers} workers...")

    t0 = time.time()
    all_points = []
    all_labels = []
    all_room_mins = []
    all_room_maxs = []
    chunk_idx = 0
    total_blocks = 0

    with ProcessPoolExecutor(max_workers=workers) as pool:
        for i, room_blocks in enumerate(pool.map(process_room, room_dirs, chunksize=8)):
            for pts, lbl, rmin, rmax in room_blocks:
                all_points.append(pts)
                all_labels.append(lbl)
                all_room_mins.append(rmin)
                all_room_maxs.append(rmax)
                total_blocks += 1

                if len(all_points) >= BLOCKS_PER_CHUNK:
                    np.savez(
                        out_dir / f"blocks_{chunk_idx:04d}.npz",
                        points=np.stack(all_points),
                        labels=np.stack(all_labels),
                        room_mins=np.stack(all_room_mins),
                        room_maxs=np.stack(all_room_maxs),
                    )
                    chunk_idx += 1
                    all_points, all_labels, all_room_mins, all_room_maxs = [], [], [], []

            if (i + 1) % 500 == 0:
                elapsed = time.time() - t0
                rate = (i + 1) / elapsed
                eta = (len(room_dirs) - i - 1) / rate
                print(f"    {i+1}/{len(room_dirs)} rooms, {total_blocks:,} blocks, ~{eta:.0f}s left")

    # Save remaining
    if all_points:
        np.savez(
            out_dir / f"blocks_{chunk_idx:04d}.npz",
            points=np.stack(all_points),
            labels=np.stack(all_labels),
            room_mins=np.stack(all_room_mins),
            room_maxs=np.stack(all_room_maxs),
        )
        chunk_idx += 1

    elapsed = time.time() - t0
    print(f"  {split}: {total_blocks:,} blocks in {chunk_idx} chunks ({elapsed:.0f}s)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    for split in ["train", "val", "test"]:
        process_split(split, args.workers)


if __name__ == "__main__":
    main()
