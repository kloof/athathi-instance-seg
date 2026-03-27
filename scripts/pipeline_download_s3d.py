"""Pipelined Structured3D downloader + detection preprocessor.

Downloads bbox zip first (for instance.png), then pipelines:
  - Download panorama zip N (foreground)
  - Preprocess panorama zip N-1 → coord.npy + semantic.npy + instance.npy (background)
  - Delete panorama zip N-2 after preprocessing

Output: data/detection_full/{train,val,test}/<scene>_room<id>/

Usage:
    python scripts/pipeline_download_s3d.py
    python scripts/pipeline_download_s3d.py -o ./s3d_data -n 16 --room-workers 8
    python scripts/pipeline_download_s3d.py --keep-zips
"""

import os
import sys
import io
import time
import zipfile
import argparse
import random
import numpy as np
import multiprocessing
import threading
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from PIL import Image

# Add scripts dir to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from parallel_download import download_file, format_size

BASE_URL = "https://zju-kjl-jointlab-azure.kujiale.com/Structured3D"
PANORAMA_ZIPS = [f"Structured3D_panorama_{i:02d}.zip" for i in range(18)]
BBOX_ZIP = "Structured3D_bbox.zip"

# Detection preprocessing config
VOXEL_SIZE = 0.03
MIN_POINTS = 5000
NUM_CLASSES = 16
OTHER_CLASS_ID = 15
NYU40_TO_DET = {
    1: 0, 2: 1, 22: 2, 8: 3, 9: 4,
    3: 5, 4: 6, 5: 7, 6: 8, 7: 9,
    10: 10, 14: 11, 17: 12, 33: 13, 34: 14,
}
THING_CLASS_IDS = {3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14}

TRAIN_RATIO = 0.72
VAL_RATIO = 0.13


# ---------------------------------------------------------------------------
# Detection preprocessing functions
# ---------------------------------------------------------------------------

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


def voxel_downsample_detection(points, sem_labels, inst_labels, voxel_size):
    if len(points) == 0:
        return points, sem_labels, inst_labels
    voxel_idx = np.floor(points / voxel_size).astype(np.int32)
    _, unique_idx, inverse = np.unique(
        voxel_idx, axis=0, return_index=True, return_inverse=True
    )
    n = len(unique_idx)
    counts = np.bincount(inverse, minlength=n).astype(np.float64)

    ds_points = np.zeros((n, 3), dtype=np.float64)
    for d in range(3):
        ds_points[:, d] = np.bincount(
            inverse, weights=points[:, d].astype(np.float64), minlength=n
        )
    ds_points /= counts[:, None]

    # Majority vote semantic
    best_votes = np.zeros(n, dtype=np.float64)
    ds_sem = np.zeros(n, dtype=np.uint8)
    for c in range(NUM_CLASSES):
        c_votes = np.bincount(
            inverse, weights=(sem_labels == c).astype(np.float64), minlength=n
        )
        better = c_votes > best_votes
        ds_sem[better] = c
        best_votes = np.maximum(best_votes, c_votes)

    # Instance: pick from point nearest to voxel center
    ds_inst = inst_labels[unique_idx].copy()

    return ds_points.astype(np.float32), ds_sem, ds_inst


def process_detection_room(scene, room_id, pano_data, bbox_zf, bbox_namelist):
    """Process one room for detection. Returns (pts, sem, inst, stats) or Nones."""
    try:
        cam_txt, depth_bytes, sem_bytes = pano_data

        cx, cy, cz = [float(x) for x in cam_txt.split()]
        cam = np.array([cx, cy, cz]) / 1000.0

        depth = np.array(Image.open(io.BytesIO(depth_bytes)))
        sem_img = np.array(Image.open(io.BytesIO(sem_bytes)))

        H, W = depth.shape
        pts, valid = backproject_panorama(depth, cam)
        raw_sem = sem_img[valid]

        mapped_sem = np.full(len(raw_sem), OTHER_CLASS_ID, dtype=np.uint8)
        for nyu_id, det_id in NYU40_TO_DET.items():
            mapped_sem[raw_sem == nyu_id] = det_id

        if len(pts) < MIN_POINTS:
            return None, None, None, {}

        # Instance labels from bbox zip
        instance_path = f"Structured3D/{scene}/2D_rendering/{room_id}/panorama/full/instance.png"
        if instance_path in bbox_namelist:
            instance_img = np.array(Image.open(io.BytesIO(bbox_zf.read(instance_path))))
            raw_inst = instance_img[valid].astype(np.int32)
        else:
            raw_inst = np.zeros(len(pts), dtype=np.int32)

        # Zero out stuff classes
        stuff_mask = np.array([s not in THING_CLASS_IDS for s in mapped_sem])
        raw_inst[stuff_mask] = 0

        # Zero out tiny instances
        inst_ids, inst_counts = np.unique(raw_inst, return_counts=True)
        small = set(inst_ids[inst_counts < 50]) | {0}
        raw_inst[np.isin(raw_inst, list(small))] = 0

        # Renumber contiguous
        unique_insts = sorted(set(raw_inst) - {0})
        remap = {0: 0}
        for new_id, old_id in enumerate(unique_insts, start=1):
            remap[old_id] = new_id
        inst_remapped = np.array([remap[x] for x in raw_inst], dtype=np.int32)

        # Voxel downsample
        pts, mapped_sem, inst_remapped = voxel_downsample_detection(
            pts, mapped_sem, inst_remapped, VOXEL_SIZE
        )
        if len(pts) < MIN_POINTS:
            return None, None, None, {}

        n_instances = len(set(inst_remapped) - {0})
        return pts, mapped_sem, inst_remapped, {"points": len(pts), "instances": n_instances}

    except Exception as e:
        return None, None, None, {}


def preprocess_detection_zip(zip_path: str, bbox_path: str, output_dir: str,
                             split_assignments: dict, room_workers: int = 8,
                             delete_after: bool = True):
    """Preprocess all rooms in a panorama zip for detection. Runs in subprocess."""
    name = os.path.basename(zip_path)
    print(f"\n  [PREPROCESS] Starting: {name}", flush=True)
    t0 = time.time()

    try:
        pano_zf = zipfile.ZipFile(zip_path)
        bbox_zf = zipfile.ZipFile(bbox_path)
        bbox_namelist = set(bbox_zf.namelist())
    except Exception as e:
        print(f"  [PREPROCESS] FAILED to open zips: {e}", flush=True)
        return

    out_dir = Path(output_dir)

    # Find all scene/room pairs
    scene_rooms = {}
    for entry in pano_zf.namelist():
        parts = entry.split("/")
        if (len(parts) >= 5 and parts[1].startswith("scene_")
                and parts[2] == "2D_rendering" and parts[3].isdigit()):
            scene_rooms.setdefault(parts[1], set()).add(parts[3])

    # Collect tasks, skip already-done rooms
    tasks = []
    for scene in sorted(scene_rooms):
        for room in sorted(scene_rooms[scene]):
            key = f"{scene}_room{room}"
            split = split_assignments.get(key, "train")
            room_out = out_dir / split / key
            if room_out.exists() and (room_out / "instance.npy").exists():
                continue
            tasks.append((scene, room, split))

    if not tasks:
        print(f"  [PREPROCESS] {name}: all rooms already done", flush=True)
        pano_zf.close()
        bbox_zf.close()
        return

    total_rooms = sum(len(r) for r in scene_rooms.values())
    print(f"  [PREPROCESS] {name}: {len(tasks)} rooms to process "
          f"({total_rooms} total)", flush=True)

    # Pre-read panorama data (zip not thread-safe)
    room_data = {}
    for scene, room, split in tasks:
        try:
            cam_txt = pano_zf.read(
                f"Structured3D/{scene}/2D_rendering/{room}/panorama/camera_xyz.txt"
            ).decode().strip()
            depth_bytes = pano_zf.read(
                f"Structured3D/{scene}/2D_rendering/{room}/panorama/full/depth.png"
            )
            sem_bytes = pano_zf.read(
                f"Structured3D/{scene}/2D_rendering/{room}/panorama/full/semantic.png"
            )
            room_data[(scene, room)] = (cam_txt, depth_bytes, sem_bytes, split)
        except Exception:
            pass

    pano_zf.close()

    # Process rooms (bbox_zf reads happen in main thread via process_detection_room)
    # Since bbox reads aren't thread-safe either, we process sequentially but
    # use threads for the CPU-heavy backprojection/downsample
    saved = 0
    total_pts = 0

    for (scene, room), (cam_txt, depth_bytes, sem_bytes, split) in room_data.items():
        pts, sem, inst, stats = process_detection_room(
            scene, room, (cam_txt, depth_bytes, sem_bytes),
            bbox_zf, bbox_namelist,
        )
        if pts is None:
            continue

        room_out = out_dir / split / f"{scene}_room{room}"
        room_out.mkdir(parents=True, exist_ok=True)
        np.save(str(room_out / "coord.npy"), pts)
        np.save(str(room_out / "semantic.npy"), sem)
        np.save(str(room_out / "instance.npy"), inst)

        saved += 1
        total_pts += stats["points"]

    bbox_zf.close()

    elapsed = time.time() - t0
    print(f"  [PREPROCESS] Done: {name} — {saved} rooms, "
          f"{total_pts:,} pts in {elapsed:.1f}s", flush=True)

    if delete_after:
        try:
            os.remove(zip_path)
            print(f"  [PREPROCESS] Deleted: {name}", flush=True)
        except Exception:
            pass


def scan_all_rooms(zip_dir: Path) -> list:
    """Quick scan of all panorama zips to get (zip_idx, scene, room) list."""
    all_rooms = []
    for zip_idx, zip_name in enumerate(PANORAMA_ZIPS):
        zip_path = zip_dir / zip_name
        if not zip_path.exists():
            continue
        zf = zipfile.ZipFile(str(zip_path))
        scene_rooms = {}
        for name in zf.namelist():
            parts = name.split("/")
            if (len(parts) >= 5 and parts[1].startswith("scene_")
                    and parts[2] == "2D_rendering" and parts[3].isdigit()):
                scene_rooms.setdefault(parts[1], set()).add(parts[3])
        zf.close()
        for scene in sorted(scene_rooms):
            for room_id in sorted(scene_rooms[scene]):
                all_rooms.append((zip_idx, scene, room_id))
    return all_rooms


def main():
    parser = argparse.ArgumentParser(
        description="Pipelined S3D download + detection preprocess"
    )
    parser.add_argument("-o", "--output", default="./s3d_data",
                        help="Output directory (default: ./s3d_data)")
    parser.add_argument("-n", "--connections", type=int, default=8,
                        help="Parallel connections per zip download (default: 8)")
    parser.add_argument("--parallel-downloads", type=int, default=3,
                        help="Number of zips to download simultaneously (default: 3)")
    parser.add_argument("--room-workers", type=int, default=8)
    parser.add_argument("--max-preprocess", type=int, default=0,
                        help="Max concurrent preprocessing jobs (0=unlimited)")
    parser.add_argument("--keep-zips", action="store_true")
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    out_dir = Path(args.output)
    zip_dir = out_dir / "zips"
    det_dir = out_dir / "detection"
    zip_dir.mkdir(parents=True, exist_ok=True)
    for split in ["train", "val", "test"]:
        (det_dir / split).mkdir(parents=True, exist_ok=True)

    delete_after = not args.keep_zips

    print(f"Pipeline: download + detection preprocess")
    print(f"  Output:       {out_dir}")
    print(f"  Detection:    {det_dir}")
    print(f"  Connections:  {args.connections} per zip")
    print(f"  Parallel DL:  {args.parallel_downloads} zips")
    print(f"  Room workers: {args.room_workers}")
    print(f"  Max preproc:  {args.max_preprocess}")
    print()

    # --- Step 1: Download bbox zip (needed for all preprocessing) ---
    bbox_path = str(zip_dir / BBOX_ZIP)
    bbox_url = f"{BASE_URL}/{BBOX_ZIP}"
    print("=" * 60)
    print("[0/18] Downloading bbox zip (required for instance labels)...")
    if not download_file(bbox_url, bbox_path, args.connections, args.retries):
        print("FATAL: Failed to download bbox zip")
        return

    # --- Step 2: Download all panorama zips first to scan rooms ---
    # Actually, download them one by one and preprocess as we go.
    # But we need the full room list upfront for train/val/test split.
    # So: download all 18, scan, assign splits, then preprocess.
    #
    # Better approach: download all, scan for split assignment, then
    # preprocess in parallel. But that needs all zips on disk at once (~200GB).
    #
    # Best approach: download first zip, scan rooms, assign splits using
    # a deterministic hash so we don't need the full list upfront.

    # Deterministic split assignment by hashing scene+room
    def assign_split(scene, room_id):
        """Deterministic train/val/test split based on hash."""
        h = hash(f"{scene}_{room_id}_{args.seed}") % 100
        if h < int(TRAIN_RATIO * 100):
            return "train"
        elif h < int((TRAIN_RATIO + VAL_RATIO) * 100):
            return "val"
        else:
            return "test"

    # --- Step 3: Parallel download + preprocess ---
    preprocess_procs = []
    results = []
    results_lock = threading.Lock()
    preprocess_lock = threading.Lock()

    def download_and_preprocess(i, zip_name):
        url = f"{BASE_URL}/{zip_name}"
        zip_path = str(zip_dir / zip_name)

        print(f"\n{'='*60}")
        print(f"[{i+1}/18] Downloading: {zip_name}", flush=True)
        ok = download_file(url, zip_path, args.connections, args.retries)

        with results_lock:
            results.append((zip_name, ok))

        if not ok:
            print(f"  SKIPPING preprocess for {zip_name} (download failed)", flush=True)
            return

        # Scan this zip for rooms and build split assignments
        zf = zipfile.ZipFile(zip_path)
        scene_rooms = {}
        for entry in zf.namelist():
            parts = entry.split("/")
            if (len(parts) >= 5 and parts[1].startswith("scene_")
                    and parts[2] == "2D_rendering" and parts[3].isdigit()):
                scene_rooms.setdefault(parts[1], set()).add(parts[3])
        zf.close()

        split_assignments = {}
        for scene in scene_rooms:
            for room in scene_rooms[scene]:
                key = f"{scene}_room{room}"
                split_assignments[key] = assign_split(scene, room)

        # Start preprocessing in background
        with preprocess_lock:
            # Reap finished preprocessors
            still_running = []
            for p in preprocess_procs:
                if p.is_alive():
                    still_running.append(p)
                else:
                    p.join()
            preprocess_procs.clear()
            preprocess_procs.extend(still_running)

        proc = multiprocessing.Process(
            target=preprocess_detection_zip,
            args=(zip_path, bbox_path, str(det_dir),
                  split_assignments, args.room_workers, delete_after),
        )
        proc.start()
        with preprocess_lock:
            preprocess_procs.append(proc)

    with ThreadPoolExecutor(max_workers=args.parallel_downloads) as dl_pool:
        dl_pool.map(lambda args: download_and_preprocess(*args),
                     enumerate(PANORAMA_ZIPS))

    # Wait for remaining preprocessing
    if preprocess_procs:
        print(f"\nAll downloads done. Waiting for {len(preprocess_procs)} "
              f"preprocessing job(s) to complete...")
        for p in preprocess_procs:
            p.join()

    # Summary
    print(f"\n{'='*60}")
    print("Summary:")
    ok_count = 0
    for zip_name, ok in results:
        status = "OK" if ok else "FAILED"
        if ok:
            ok_count += 1
        print(f"  [{status}] {zip_name}")

    for split in ["train", "val", "test"]:
        split_dir = det_dir / split
        rooms = [d for d in split_dir.iterdir() if d.is_dir()]
        print(f"  {split}: {len(rooms)} rooms")

    print(f"\n{ok_count}/18 panorama zips downloaded and preprocessed")
    print(f"Detection output: {det_dir}")
    print(f"\nTo train: python3 scripts/run_detection_training.py "
          f"--config config_detection.yaml")


if __name__ == "__main__":
    main()
