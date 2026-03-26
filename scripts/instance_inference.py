"""Run instance segmentation on a real LiDAR .ply, derive bounding boxes from instances.

Pipeline:
  1. Load .ply
  2. Predict semantic class + offset per point
  3. Shift points by offset → cluster same-class shifted points (DBSCAN)
  4. Each cluster = one instance → compute tight oriented bounding box
  5. Save result .ply with instance colors + box wireframes

Usage:
    python scripts/instance_inference.py --input room.ply --checkpoint checkpoints/det_run_001/best_model.pth
"""

import argparse
import numpy as np
import torch
import math
import sys
from pathlib import Path
from collections import Counter

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.detection_model import LitePTInstanceSeg
from src.gpu_augment import compute_features_batch_gpu

CLASS_NAMES = [
    "wall", "floor", "ceiling", "door", "window",
    "cabinet", "bed", "chair", "sofa", "table",
    "bookshelf", "desk", "dresser", "toilet", "sink",
]
THING_CLASS_IDS = {3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14}


def load_ply(path):
    with open(path, "rb") as f:
        header = b""
        while True:
            line = f.readline()
            header += line
            if b"end_header" in line:
                break
        header_str = header.decode("ascii", errors="replace")
        lines = header_str.strip().split("\n")
        n_vertices = 0
        props = []
        for line in lines:
            if line.startswith("element vertex"):
                n_vertices = int(line.split()[-1])
            if line.startswith("property"):
                parts = line.split()
                props.append((parts[1], parts[2]))
        type_map = {"float": np.float32, "float32": np.float32, "double": np.float64,
                     "uchar": np.uint8, "uint8": np.uint8, "int": np.int32, "int32": np.int32,
                     "short": np.int16, "int16": np.int16, "ushort": np.uint16, "uint16": np.uint16}
        for typ, name in props:
            if typ not in type_map:
                raise ValueError(f"Unknown PLY property type '{typ}' for '{name}'. Supported: {list(type_map.keys())}")
        dt = np.dtype([(name, type_map[typ]) for typ, name in props])
        data = np.frombuffer(f.read(n_vertices * dt.itemsize), dtype=dt, count=n_vertices)

    pts = np.column_stack([data["x"].astype(np.float32), data["y"].astype(np.float32), data["z"].astype(np.float32)])
    prop_names = [name for _, name in props]
    colors = None
    for r, g, b in [("red", "green", "blue"), ("r", "g", "b")]:
        if r in prop_names:
            colors = np.column_stack([data[r].astype(np.uint8), data[g].astype(np.uint8), data[b].astype(np.uint8)])
            break
    return pts, colors


def write_ply(path, points, colors):
    N = len(points)
    with open(path, "wb") as f:
        header = f"ply\nformat binary_little_endian 1.0\nelement vertex {N}\nproperty float x\nproperty float y\nproperty float z\nproperty uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n"
        f.write(header.encode())
        dt = np.dtype([("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("r", "u1"), ("g", "u1"), ("b", "u1")])
        arr = np.zeros(N, dtype=dt)
        arr["x"] = points[:, 0]; arr["y"] = points[:, 1]; arr["z"] = points[:, 2]
        arr["r"] = colors[:, 0]; arr["g"] = colors[:, 1]; arr["b"] = colors[:, 2]
        f.write(arr.tobytes())


def dbscan_cluster(points, eps=0.3, min_samples=50):
    """Simple grid-based clustering (no sklearn needed)."""
    if len(points) == 0:
        return np.zeros(0, dtype=np.int32)

    labels = -np.ones(len(points), dtype=np.int32)
    grid = {}
    cell_size = eps

    # Assign points to grid cells
    cells = np.floor(points / cell_size).astype(np.int64)
    for i, c in enumerate(cells):
        key = tuple(c)
        grid.setdefault(key, []).append(i)

    # BFS flood fill through adjacent cells
    cluster_id = 0
    visited_cells = set()

    for cell_key, indices in grid.items():
        if cell_key in visited_cells:
            continue

        # Start new cluster
        queue = [cell_key]
        visited_cells.add(cell_key)
        cluster_pts = []

        while queue:
            ck = queue.pop(0)
            if ck in grid:
                cluster_pts.extend(grid[ck])
                # Check 26 neighbors
                for dx in [-1, 0, 1]:
                    for dy in [-1, 0, 1]:
                        for dz in [-1, 0, 1]:
                            nk = (ck[0]+dx, ck[1]+dy, ck[2]+dz)
                            if nk not in visited_cells and nk in grid:
                                visited_cells.add(nk)
                                queue.append(nk)

        if len(cluster_pts) >= min_samples:
            for idx in cluster_pts:
                labels[idx] = cluster_id
            cluster_id += 1

    return labels


def compute_obb(pts):
    """Compute oriented bounding box from points. Returns (cx,cy,cz,w,h,d,heading)."""
    center = pts.mean(axis=0)
    xy = pts[:, :2] - center[:2]
    cov = np.cov(xy.T) if len(xy) > 2 else np.eye(2)
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    primary = eigenvectors[:, np.argmax(eigenvalues)]
    heading = math.atan2(primary[1], primary[0])

    cos_h, sin_h = np.cos(-heading), np.sin(-heading)
    local_x = xy[:, 0] * cos_h - xy[:, 1] * sin_h
    local_y = xy[:, 0] * sin_h + xy[:, 1] * cos_h

    w = max(local_x.max() - local_x.min(), 0.05)
    h = max(local_y.max() - local_y.min(), 0.05)
    d = max(pts[:, 2].max() - pts[:, 2].min(), 0.05)
    return center[0], center[1], center[2], w, h, d, heading


def box_wireframe(center, size, heading, n=30):
    w, h, d = np.array(size) / 2
    corners_local = np.array([
        [-w,-h,-d],[+w,-h,-d],[+w,+h,-d],[-w,+h,-d],
        [-w,-h,+d],[+w,-h,+d],[+w,+h,+d],[-w,+h,+d],
    ], dtype=np.float32)
    cos_h, sin_h = np.cos(heading), np.sin(heading)
    rot = np.array([[cos_h,-sin_h,0],[sin_h,cos_h,0],[0,0,1]], dtype=np.float32)
    corners = corners_local @ rot.T + center
    edges = [(0,1),(1,2),(2,3),(3,0),(4,5),(5,6),(6,7),(7,4),(0,4),(1,5),(2,6),(3,7)]
    pts = []
    for i, j in edges:
        t = np.linspace(0, 1, n).reshape(-1, 1)
        pts.append(corners[i]*(1-t) + corners[j]*t)
    return np.concatenate(pts)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", default=None)
    parser.add_argument("--config", default="config_detection.yaml")
    parser.add_argument("--cluster_eps", type=float, default=0.3)
    parser.add_argument("--min_instance_pts", type=int, default=100)
    args = parser.parse_args()

    import yaml
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model
    model = LitePTInstanceSeg(
        input_dim=cfg["model"]["input_dim"],
        num_classes=cfg["model"]["num_classes"],
    ).to(device)

    ckpt = torch.load(args.checkpoint, map_location=device)
    state = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
    model.load_state_dict(state)
    model.eval()
    print(f"Model loaded ({sum(p.numel() for p in model.parameters()):,} params)")

    # Load point cloud
    pts, colors = load_ply(args.input)
    print(f"Loaded {len(pts):,} points from {args.input}")

    # Center XY
    center_xy = pts[:, :2].mean(axis=0)
    pts[:, 0] -= center_xy[0]
    pts[:, 1] -= center_xy[1]

    # Predict
    pts_t = torch.from_numpy(pts).unsqueeze(0).to(device)
    room_mins = pts_t.min(dim=1).values
    room_maxs = pts_t.max(dim=1).values

    with torch.no_grad():
        features = compute_features_batch_gpu(pts_t, room_mins, room_maxs)
        sem_logits, offset_pred = model(features)

    sem_pred = sem_logits.argmax(dim=1)[0].cpu().numpy()
    offsets = offset_pred[0].permute(1, 0).cpu().numpy()  # (N, 3)

    # Cluster in centered frame (offsets were predicted in centered coords)
    # Then shift results back to world coords after clustering

    # Instance grouping: for each thing class, cluster (point + offset)
    instance_labels = np.zeros(len(pts), dtype=np.int32)
    next_id = 1
    boxes = []

    for cls_id in THING_CLASS_IDS:
        cls_mask = sem_pred == cls_id
        if cls_mask.sum() < args.min_instance_pts:
            continue

        cls_pts = pts[cls_mask]  # still in centered coords
        cls_offsets = offsets[cls_mask]
        shifted = cls_pts + cls_offsets  # cluster near instance centers (centered frame)

        clusters = dbscan_cluster(shifted, eps=args.cluster_eps, min_samples=args.min_instance_pts)

        cls_indices = np.where(cls_mask)[0]
        for cid in range(clusters.max() + 1):
            cmask = clusters == cid
            if cmask.sum() < args.min_instance_pts:
                continue

            inst_pts = cls_pts[cmask]
            for idx in cls_indices[cmask]:
                instance_labels[idx] = next_id

            cx, cy, cz, w, h, d, heading = compute_obb(inst_pts)
            boxes.append((cx, cy, cz, w, h, d, heading, cls_id, next_id))
            next_id += 1

    # Shift everything back to world coordinates
    pts[:, 0] += center_xy[0]
    pts[:, 1] += center_xy[1]
    boxes = [(cx + center_xy[0], cy + center_xy[1], cz, w, h, d, heading, cls_id, iid)
             for cx, cy, cz, w, h, d, heading, cls_id, iid in boxes]

    # Print results
    print(f"\nDetected {len(boxes)} instances:")
    for cx, cy, cz, w, h, d, heading, cls_id, iid in boxes:
        name = CLASS_NAMES[cls_id]
        n_pts = (instance_labels == iid).sum()
        print(f"  [{iid:3d}] {name:12s} | {n_pts:,} pts | "
              f"center=({cx:.2f},{cy:.2f},{cz:.2f}) | "
              f"size=({w:.2f}x{h:.2f}x{d:.2f}) | heading={math.degrees(heading):.0f}")

    # Save visualization
    if args.output is None:
        inp = Path(args.input)
        args.output = str(inp.parent / f"{inp.stem}_instances.ply")

    # Instance-colored points + box wireframes
    rng = np.random.RandomState(42)
    palette = rng.randint(50, 255, (next_id, 3)).astype(np.uint8)
    palette[0] = [140, 140, 140]

    out_pts = [pts]
    out_colors = [palette[instance_labels]]

    for cx, cy, cz, w, h, d, heading, cls_id, iid in boxes:
        wire = box_wireframe(np.array([cx,cy,cz]), np.array([w,h,d]), heading)
        out_pts.append(wire)
        out_colors.append(np.tile(palette[iid], (len(wire), 1)))

    write_ply(args.output, np.concatenate(out_pts).astype(np.float32),
              np.concatenate(out_colors).astype(np.uint8))
    print(f"\nSaved: {args.output}")


if __name__ == "__main__":
    main()
