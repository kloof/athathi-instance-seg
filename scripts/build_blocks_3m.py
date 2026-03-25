"""Build 3m training blocks from panorama-processed rooms.

Larger blocks (3m vs 1m) give much more spatial context for wall detection.
16384 points per block instead of 4096.

Usage:
    python scripts/build_blocks_3m.py --workers 8
"""

import numpy as np
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import time
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

RAW_PROCESSED = Path(r"C:\Users\klof\Desktop\New folder (4)\wall-seg\data\structured3d_panorama_processed")
OUT_ROOT = Path(r"C:\Users\klof\Desktop\New folder (4)\wall-seg\data\processed_3m")
BLOCK_SIZE = 3.0
NUM_POINTS = 16384
MIN_POINTS = 500
BLOCKS_PER_CHUNK = 500  # fewer blocks per chunk since each is 4x larger
VAL_ROOMS = 2000


def split_into_blocks(points, labels, block_size, min_points):
    min_xy = points[:, :2].min(axis=0)
    block_idx = np.floor((points[:, :2] - min_xy) / block_size).astype(np.int32)
    block_keys = block_idx[:, 0] * 100000 + block_idx[:, 1]
    unique_keys = np.unique(block_keys)
    blocks = []
    for key in unique_keys:
        mask = block_keys == key
        if mask.sum() >= min_points:
            blocks.append((points[mask].copy(), labels[mask].copy()))
    return blocks


def sample_block(points, labels, num_points, rng=None):
    if rng is None:
        rng = np.random.default_rng()
    N = points.shape[0]
    if N >= num_points:
        idx = rng.choice(N, num_points, replace=False)
    else:
        pad = rng.choice(N, num_points - N, replace=True)
        idx = np.concatenate([np.arange(N), pad])
    return points[idx], labels[idx]


def process_scene(scene_dir: str) -> list:
    scene_dir = Path(scene_dir)
    try:
        coords = np.load(scene_dir / "coord.npy")
        labels = np.load(scene_dir / "label.npy")
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
    except Exception:
        return []


def process_split_streaming(split_name, dirs, workers):
    out_dir = OUT_ROOT / split_name / "blocks"
    out_dir.mkdir(parents=True, exist_ok=True)

    if list(out_dir.glob("blocks_*.npz")):
        existing = list(out_dir.glob("blocks_*.npz"))
        total = sum(np.load(str(f))["points"].shape[0] for f in existing)
        print(f"  {split_name}: already done ({total:,} blocks)")
        return

    t0 = time.time()
    all_points, all_labels, all_room_mins, all_room_maxs = [], [], [], []
    chunk_idx = 0
    total_blocks = 0

    with ProcessPoolExecutor(max_workers=workers) as pool:
        for i, scene_blocks in enumerate(pool.map(process_scene, dirs, chunksize=8)):
            for pts, lbl, rmin, rmax in scene_blocks:
                all_points.append(pts)
                all_labels.append(lbl)
                all_room_mins.append(rmin)
                all_room_maxs.append(rmax)
                total_blocks += 1

                if len(all_points) >= BLOCKS_PER_CHUNK:
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

            if (i + 1) % 1000 == 0:
                elapsed = time.time() - t0
                rate = (i + 1) / elapsed
                eta = (len(dirs) - i - 1) / rate
                print(f"  {i+1}/{len(dirs)} rooms, {total_blocks:,} blocks, ~{eta:.0f}s left", flush=True)

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

    print(f"  {split_name}: {total_blocks:,} blocks in {chunk_idx} chunks ({time.time()-t0:.0f}s)")


def main():
    scene_dirs = sorted([
        str(d) for d in RAW_PROCESSED.iterdir()
        if d.is_dir() and d.name.startswith("scene_")
    ])
    print(f"Found {len(scene_dirs)} rooms")

    rng = np.random.default_rng(42)
    indices = rng.permutation(len(scene_dirs))
    val_idx = set(indices[:VAL_ROOMS])

    train_dirs = [d for i, d in enumerate(scene_dirs) if i not in val_idx]
    val_dirs = [d for i, d in enumerate(scene_dirs) if i in val_idx]
    print(f"Train: {len(train_dirs)}, Val: {len(val_dirs)}")
    print(f"Block size: {BLOCK_SIZE}m, Points per block: {NUM_POINTS}")

    for split_name, dirs in [("train", train_dirs), ("val", val_dirs)]:
        print(f"\nProcessing {split_name}...", flush=True)
        process_split_streaming(split_name, dirs, 8)


if __name__ == "__main__":
    main()
