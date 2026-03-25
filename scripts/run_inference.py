"""Run wall segmentation on a point cloud file.

Usage:
    python scripts/run_inference.py input.ply --model checkpoints/pointnet_wall_int8.onnx --output walls.ply

Supports: .ply, .pcd, .xyz, .npy, .npz input formats.
"""

import argparse
import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.infer import WallSegPipeline


def load_points(path: str) -> np.ndarray:
    """Load point cloud from various formats."""
    path = Path(path)

    if path.suffix == ".npy":
        return np.load(path).astype(np.float32)

    if path.suffix == ".npz":
        data = np.load(path)
        return data["points"].astype(np.float32)

    if path.suffix in (".ply", ".pcd"):
        try:
            import open3d as o3d
            pcd = o3d.io.read_point_cloud(str(path))
            return np.asarray(pcd.points, dtype=np.float32)
        except ImportError:
            raise ImportError("Install open3d to read .ply/.pcd files: pip install open3d")

    if path.suffix in (".xyz", ".txt", ".csv"):
        return np.loadtxt(path, dtype=np.float32, delimiter=None, usecols=(0, 1, 2))

    raise ValueError(f"Unsupported format: {path.suffix}")


def save_wall_points(points: np.ndarray, path: str) -> None:
    """Save wall-only points."""
    path = Path(path)

    if path.suffix == ".npy":
        np.save(path, points)
    elif path.suffix == ".ply":
        try:
            import open3d as o3d
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(points)
            o3d.io.write_point_cloud(str(path), pcd)
        except ImportError:
            raise ImportError("Install open3d to write .ply files")
    else:
        np.savetxt(path, points, fmt="%.6f")

    print(f"Saved {points.shape[0]:,} wall points to {path}")


def main():
    parser = argparse.ArgumentParser(description="Wall segmentation inference")
    parser.add_argument("input", help="Input point cloud file")
    parser.add_argument("--model", default="checkpoints/pointnet_wall_int8.onnx")
    parser.add_argument("--output", default="walls.ply")
    parser.add_argument("--voxel-size", type=float, default=0.03)
    parser.add_argument("--block-size", type=float, default=3.0)
    parser.add_argument("--num-points", type=int, default=16384)
    parser.add_argument("--overlap", type=float, default=0.5)
    parser.add_argument("--no-level", action="store_true",
                        help="Skip auto-leveling (use if scan is already leveled)")
    args = parser.parse_args()

    points = load_points(args.input)
    print(f"Loaded {points.shape[0]:,} points from {args.input}")

    pipeline = WallSegPipeline(
        args.model,
        voxel_size=args.voxel_size,
        block_size=args.block_size,
        num_points=args.num_points,
        overlap=args.overlap,
    )

    predictions = pipeline.predict(points, level=not args.no_level)

    wall_mask = predictions == 1
    wall_points = points[wall_mask]
    print(f"Wall points: {wall_points.shape[0]:,} / {points.shape[0]:,} ({wall_mask.mean():.1%})")

    save_wall_points(wall_points, args.output)


if __name__ == "__main__":
    main()
