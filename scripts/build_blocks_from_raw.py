"""Build training blocks from raw Structured3D processed scenes.

Reads coord.npy + label.npy per scene (from preprocess_structured3d_raw.py),
splits into 1m blocks, saves as chunked .npz files for training.

Usage:
    python scripts/build_blocks_from_raw.py --workers 8
"""

import argparse
import numpy as np
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import time
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.preprocess import split_into_blocks, sample_block

RAW_PROCESSED = Path(r"C:\Users\klof\Desktop\New folder (4)\wall-seg\data\structured3d_raw_processed")
OUT_ROOT = Path(r"C:\Users\klof\Desktop\New folder (4)\wall-seg\data\processed_raw")
BLOCK_SIZE = 1.0
NUM_POINTS = 4096
MIN_POINTS = 100
BLOCKS_PER_CHUNK = 2000

# Train/val split: 3150 train, 350 val (90/10)
VAL_SCENES = 350


def process_scene(scene_dir: str) -> list:
    """Load one scene, split into blocks."""
    scene_dir = Path(scene_dir)
    try:
        coords = np.load(scene_dir / "coord.npy")   # (N, 3) float32
        labels = np.load(scene_dir / "label.npy")    # (N,) int64

        if coords.shape[0] < MIN_POINTS:
            return []

        room_min = coords.min(axis=0)
        room_max = coords.max(axis=0)

        blocks = split_into_blocks(coords, labels, BLOCK_SIZE, MIN_POINTS)

        result = []
        rng = np.random.default_rng(42)
        for block_pts, block_lbl in blocks:
            block_pts, block_lbl = sample_block(block_pts, block_lbl, NUM_POINTS, rng)
            result.append((block_pts, block_lbl, room_min, room_max))

        return result
    except Exception as e:
        print(f"  Error in {scene_dir.name}: {e}")
        return []


def save_blocks(blocks_list, out_dir, split_name):
    """Save blocks as chunked .npz files."""
    out_dir.mkdir(parents=True, exist_ok=True)

    all_points = []
    all_labels = []
    all_room_mins = []
    all_room_maxs = []
    chunk_idx = 0
    total_blocks = 0

    for pts, lbl, rmin, rmax in blocks_list:
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

    if all_points:
        np.savez(
            out_dir / f"blocks_{chunk_idx:04d}.npz",
            points=np.stack(all_points),
            labels=np.stack(all_labels),
            room_mins=np.stack(all_room_mins),
            room_maxs=np.stack(all_room_maxs),
        )
        chunk_idx += 1

    print(f"  {split_name}: {total_blocks:,} blocks in {chunk_idx} chunks")
    return total_blocks


def process_split_streaming(split_name, dirs, workers):
    """Process scenes and save blocks incrementally (no full-memory collect)."""
    out_dir = OUT_ROOT / split_name / "blocks"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Skip if already done
    if list(out_dir.glob("blocks_*.npz")):
        existing = list(out_dir.glob("blocks_*.npz"))
        total = sum(np.load(str(f))["points"].shape[0] for f in existing)
        print(f"  {split_name}: already done ({total:,} blocks in {len(existing)} chunks)")
        return

    t0 = time.time()
    all_points = []
    all_labels = []
    all_room_mins = []
    all_room_maxs = []
    chunk_idx = 0
    total_blocks = 0

    with ProcessPoolExecutor(max_workers=workers) as pool:
        for i, scene_blocks in enumerate(pool.map(process_scene, dirs, chunksize=4)):
            for pts, lbl, rmin, rmax in scene_blocks:
                all_points.append(pts)
                all_labels.append(lbl)
                all_room_mins.append(rmin)
                all_room_maxs.append(rmax)
                total_blocks += 1

                if len(all_points) >= BLOCKS_PER_CHUNK:
                    # Shuffle within chunk
                    order = np.random.permutation(len(all_points))
                    np.savez(
                        out_dir / f"blocks_{chunk_idx:04d}.npz",
                        points=np.stack([all_points[j] for j in order]),
                        labels=np.stack([all_labels[j] for j in order]),
                        room_mins=np.stack([all_room_mins[j] for j in order]),
                        room_maxs=np.stack([all_room_maxs[j] for j in order]),
                    )
                    chunk_idx += 1
                    all_points, all_labels, all_room_mins, all_room_maxs = [], [], [], []

            if (i + 1) % 200 == 0:
                elapsed = time.time() - t0
                rate = (i + 1) / elapsed
                eta = (len(dirs) - i - 1) / rate
                print(f"  {i+1}/{len(dirs)} scenes, {total_blocks:,} blocks, ~{eta:.0f}s left", flush=True)

    # Save remaining
    if all_points:
        order = np.random.permutation(len(all_points))
        np.savez(
            out_dir / f"blocks_{chunk_idx:04d}.npz",
            points=np.stack([all_points[j] for j in order]),
            labels=np.stack([all_labels[j] for j in order]),
            room_mins=np.stack([all_room_mins[j] for j in order]),
            room_maxs=np.stack([all_room_maxs[j] for j in order]),
        )
        chunk_idx += 1

    elapsed = time.time() - t0
    print(f"  {split_name}: {total_blocks:,} blocks in {chunk_idx} chunks ({elapsed:.0f}s)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    # Get all scene directories
    scene_dirs = sorted([
        str(d) for d in RAW_PROCESSED.iterdir()
        if d.is_dir() and d.name.startswith("scene_")
    ])
    print(f"Found {len(scene_dirs)} scenes")

    # Split into train/val
    rng = np.random.default_rng(42)
    indices = rng.permutation(len(scene_dirs))
    val_idx = set(indices[:VAL_SCENES])

    train_dirs = [d for i, d in enumerate(scene_dirs) if i not in val_idx]
    val_dirs = [d for i, d in enumerate(scene_dirs) if i in val_idx]
    print(f"Train: {len(train_dirs)} scenes, Val: {len(val_dirs)} scenes")

    for split_name, dirs in [("train", train_dirs), ("val", val_dirs)]:
        print(f"\nProcessing {split_name}...", flush=True)
        process_split_streaming(split_name, dirs, args.workers)


if __name__ == "__main__":
    main()
