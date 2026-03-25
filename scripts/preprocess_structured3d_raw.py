"""Preprocess raw Structured3D annotation_3d.json into wall-seg training data.

Extracts wall/floor/ceiling polygons from 3D annotations, samples points on them,
and saves as (coord.npy, label.npy) per scene. Labels: 1=wall, 0=not-wall.

This gives FULL-HEIGHT wall labels (unlike the broken Pointcept labels that only
covered baseboards).

Usage:
    python scripts/preprocess_structured3d_raw.py
"""

import zipfile
import json
import os
import sys
import numpy as np
from pathlib import Path
from collections import Counter

# --- Config ---
RAW_DIR = r"C:\Users\klof\Desktop\New folder (4)\structured3d_raw"
OUTPUT_DIR = r"C:\Users\klof\Desktop\New folder (4)\wall-seg\data\structured3d_raw_processed"
ANNOTATION_ZIP = os.path.join(RAW_DIR, "Structured3D_annotation_3d.zip")

# Points per square meter of surface area
POINTS_PER_SQM = 400  # ~0.05m spacing, similar to Pointcept density
MIN_POINTS_PER_SCENE = 5000
VOXEL_SIZE = 0.03  # meters, for final downsampling


def get_plane_polygon(plane_id, planes, junctions, lines,
                      plane_line_mat, line_junc_mat):
    """Get the polygon vertices for a plane from its lines and junctions.

    Returns an ordered list of 3D junction coordinates forming the polygon.
    """
    # Find lines belonging to this plane
    plane_line_ids = np.where(plane_line_mat[plane_id] == 1)[0]
    if len(plane_line_ids) == 0:
        return None

    # Collect all unique junctions for this plane
    junc_ids = set()
    edges = []
    for lid in plane_line_ids:
        jids = np.where(line_junc_mat[lid] == 1)[0]
        if len(jids) == 2:
            edges.append((jids[0], jids[1]))
            junc_ids.update(jids)

    if len(junc_ids) < 3:
        return None

    # Get junction coordinates (convert mm to meters)
    coords = {}
    for jid in junc_ids:
        c = junctions[jid]["coordinate"]
        coords[jid] = np.array(c, dtype=np.float64) / 1000.0  # mm -> m

    # Order junctions into a polygon by following edges
    ordered = order_polygon(list(junc_ids), edges)
    if ordered is None:
        # Fallback: just return unordered
        return np.array([coords[jid] for jid in junc_ids])

    return np.array([coords[jid] for jid in ordered])


def order_polygon(junc_ids, edges):
    """Order junction IDs into a polygon loop by following edges."""
    if len(edges) < 3:
        return None

    # Build adjacency
    adj = {}
    for a, b in edges:
        adj.setdefault(a, []).append(b)
        adj.setdefault(b, []).append(a)

    # Walk the polygon
    start = junc_ids[0]
    ordered = [start]
    visited = {start}
    current = start

    for _ in range(len(junc_ids) * 2):
        neighbors = adj.get(current, [])
        found_next = False
        for n in neighbors:
            if n not in visited:
                ordered.append(n)
                visited.add(n)
                current = n
                found_next = True
                break
        if not found_next:
            break

    if len(ordered) >= 3:
        return ordered
    return None


def sample_polygon_points(vertices, density_per_sqm):
    """Sample points uniformly on a planar polygon using triangulation.

    Args:
        vertices: (K, 3) polygon vertices in order
        density_per_sqm: target points per square meter

    Returns:
        (M, 3) sampled points
    """
    if len(vertices) < 3:
        return np.zeros((0, 3))

    # Fan triangulation from first vertex
    all_points = []
    total_area = 0.0

    for i in range(1, len(vertices) - 1):
        v0 = vertices[0]
        v1 = vertices[i]
        v2 = vertices[i + 1]

        # Triangle area
        edge1 = v1 - v0
        edge2 = v2 - v0
        area = 0.5 * np.linalg.norm(np.cross(edge1, edge2))
        total_area += area

        if area < 1e-6:
            continue

        # Number of points for this triangle
        n_pts = max(1, int(area * density_per_sqm))

        # Random barycentric coordinates
        r1 = np.random.rand(n_pts)
        r2 = np.random.rand(n_pts)
        # Fold to stay in triangle
        mask = r1 + r2 > 1
        r1[mask] = 1 - r1[mask]
        r2[mask] = 1 - r2[mask]

        points = (1 - r1 - r2)[:, None] * v0 + r1[:, None] * v1 + r2[:, None] * v2
        all_points.append(points)

    if len(all_points) == 0:
        return np.zeros((0, 3))

    return np.concatenate(all_points, axis=0).astype(np.float32)


def voxel_downsample(points, labels, voxel_size):
    """Voxel downsample, picking one point per voxel (majority label)."""
    if len(points) == 0:
        return points, labels

    voxel_idx = np.floor(points / voxel_size).astype(np.int32)
    # Unique voxels
    _, unique_idx, inverse = np.unique(
        voxel_idx, axis=0, return_index=True, return_inverse=True
    )

    n_voxels = len(unique_idx)

    # Average position per voxel
    ds_points = np.zeros((n_voxels, 3), dtype=np.float64)
    counts = np.bincount(inverse, minlength=n_voxels).astype(np.float64)
    for d in range(3):
        ds_points[:, d] = np.bincount(
            inverse, weights=points[:, d].astype(np.float64), minlength=n_voxels
        )
    ds_points /= counts[:, None]

    # Majority label per voxel
    ds_labels = np.zeros(n_voxels, dtype=np.int64)
    # Count wall votes per voxel
    wall_votes = np.bincount(inverse, weights=(labels == 1).astype(np.float64), minlength=n_voxels)
    ds_labels[wall_votes > counts / 2] = 1

    return ds_points.astype(np.float32), ds_labels


def process_scene(scene_name, annotation_zip):
    """Process one scene: extract polygons, sample points, assign wall labels."""
    data = json.loads(annotation_zip.read(f"Structured3D/{scene_name}/annotation_3d.json"))

    planes = data["planes"]
    junctions = data["junctions"]
    lines = data["lines"]
    plane_line_mat = np.array(data["planeLineMatrix"])
    line_junc_mat = np.array(data["lineJunctionMatrix"])
    semantics = data["semantics"]

    # Identify which planes belong to doors/windows (we'll exclude these from walls)
    door_window_plane_ids = set()
    for sem in semantics:
        if sem["type"] in ("door", "window"):
            door_window_plane_ids.update(sem["planeID"])

    all_points = []
    all_labels = []

    for plane in planes:
        pid = plane["ID"]
        ptype = plane["type"]

        # Skip door/window planes
        if pid in door_window_plane_ids:
            continue

        # Get polygon
        polygon = get_plane_polygon(
            pid, planes, junctions, lines, plane_line_mat, line_junc_mat
        )
        if polygon is None or len(polygon) < 3:
            continue

        # Sample points on polygon
        pts = sample_polygon_points(polygon, POINTS_PER_SQM)
        if len(pts) == 0:
            continue

        # Label: 1=wall, 0=not-wall (floor/ceiling)
        label = 1 if ptype == "wall" else 0
        labels = np.full(len(pts), label, dtype=np.int64)

        all_points.append(pts)
        all_labels.append(labels)

    if len(all_points) == 0:
        return None, None, {}

    points = np.concatenate(all_points, axis=0)
    labels = np.concatenate(all_labels, axis=0)

    # Voxel downsample
    points, labels = voxel_downsample(points, labels, VOXEL_SIZE)

    stats = {
        "total_points": len(points),
        "wall_points": int((labels == 1).sum()),
        "non_wall_points": int((labels == 0).sum()),
        "wall_ratio": float((labels == 1).mean()) if len(labels) > 0 else 0,
    }

    return points, labels, stats


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print(f"Reading annotations from: {ANNOTATION_ZIP}")
    z = zipfile.ZipFile(ANNOTATION_ZIP)

    # Find all scenes
    scenes = sorted(set(
        name.split("/")[1]
        for name in z.namelist()
        if name.count("/") >= 2 and name.split("/")[1].startswith("scene_")
    ))
    print(f"Found {len(scenes)} scenes")

    total_stats = Counter()
    good_scenes = 0
    skipped = 0

    for i, scene in enumerate(scenes):
        try:
            points, labels, stats = process_scene(scene, z)
        except Exception as e:
            print(f"  [{i+1}/{len(scenes)}] {scene}: ERROR - {e}")
            skipped += 1
            continue

        if points is None or len(points) < MIN_POINTS_PER_SCENE:
            skipped += 1
            if (i + 1) % 100 == 0:
                print(f"  [{i+1}/{len(scenes)}] {scene}: skipped ({0 if points is None else len(points)} pts)")
            continue

        # Save
        scene_dir = os.path.join(OUTPUT_DIR, scene)
        os.makedirs(scene_dir, exist_ok=True)
        np.save(os.path.join(scene_dir, "coord.npy"), points)
        np.save(os.path.join(scene_dir, "label.npy"), labels)

        good_scenes += 1
        total_stats["total_points"] += stats["total_points"]
        total_stats["wall_points"] += stats["wall_points"]

        if (i + 1) % 100 == 0 or i < 5:
            print(
                f"  [{i+1}/{len(scenes)}] {scene}: "
                f"{stats['total_points']:,} pts, "
                f"{stats['wall_ratio']:.1%} wall"
            )

    z.close()

    print(f"\nDone! {good_scenes} scenes saved, {skipped} skipped")
    print(f"Total points: {total_stats['total_points']:,}")
    print(f"Wall points: {total_stats['wall_points']:,}")
    if total_stats["total_points"] > 0:
        print(f"Overall wall ratio: {total_stats['wall_points']/total_stats['total_points']:.1%}")
    print(f"Output: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
