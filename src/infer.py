"""Full inference pipeline for wall segmentation -- RPi4 compatible.

Dependencies: numpy, onnxruntime (no PyTorch, no Open3D).

Pipeline:
    1. Voxel downsample (numpy)
    2. Split into 3m overlapping blocks (50% overlap)
    3. Sample each block to 16384 points
    4. Run PointNet ONNX on each block
    5. Vote-merge overlapping predictions
    6. Back-map to original points
"""

import numpy as np
import onnxruntime as ort
import time


def auto_level(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Auto-level a point cloud so Z points up, using RANSAC floor detection."""
    z_thresh = np.percentile(points[:, 2], 20)
    floor_candidates = points[points[:, 2] <= z_thresh]

    if len(floor_candidates) < 100:
        return points.copy(), np.eye(3, dtype=np.float32)

    best_normal = np.array([0, 0, 1], dtype=np.float32)
    best_inliers = 0
    rng = np.random.default_rng(42)

    n_candidates = len(floor_candidates)
    for _ in range(200):
        idx = rng.choice(n_candidates, 3, replace=False)
        p0, p1, p2 = floor_candidates[idx]
        normal = np.cross(p1 - p0, p2 - p0)
        norm_len = np.linalg.norm(normal)
        if norm_len < 1e-8:
            continue
        normal = normal / norm_len
        if normal[2] < 0:
            normal = -normal
        dists = np.abs((floor_candidates - p0) @ normal)
        inliers = (dists < 0.02).sum()
        if inliers > best_inliers:
            best_inliers = inliers
            best_normal = normal

    target = np.array([0, 0, 1], dtype=np.float32)
    v = np.cross(best_normal, target)
    c = np.dot(best_normal, target)

    if np.linalg.norm(v) < 1e-8:
        rot = np.eye(3, dtype=np.float32)
    else:
        vx = np.array([
            [0, -v[2], v[1]],
            [v[2], 0, -v[0]],
            [-v[1], v[0], 0],
        ], dtype=np.float32)
        rot = np.eye(3, dtype=np.float32) + vx + vx @ vx / (1 + c)

    leveled = (points @ rot.T).astype(np.float32)
    floor_z = np.percentile(leveled[:, 2], 5)
    leveled[:, 2] -= floor_z

    tilt_deg = np.degrees(np.arccos(np.clip(c, -1, 1)))
    print(f"Auto-level: tilt={tilt_deg:.1f}°, floor_inliers={best_inliers}/{n_candidates}")

    return leveled, rot


class WallSegPipeline:
    """Wall segmentation with overlapping 3m blocks and vote merging.

    Args:
        model_path: path to ONNX model.
        voxel_size: voxel grid size for downsampling (meters).
        block_size: XY block size (meters).
        num_points: points per block (must match training).
        overlap: fraction of overlap between blocks (0.5 = 50%).
    """

    def __init__(
        self,
        model_path: str,
        voxel_size: float = 0.03,
        block_size: float = 3.0,
        num_points: int = 16384,
        assumed_height: float = 3.5,
        overlap: float = 0.5,
    ):
        self.voxel_size = voxel_size
        self.block_size = block_size
        self.num_points = num_points
        self.assumed_height = assumed_height
        self.overlap = overlap
        self._rng = np.random.default_rng(42)

        self.session = ort.InferenceSession(
            model_path,
            providers=["CPUExecutionProvider"],
        )

    def _voxel_downsample(self, points):
        """Voxel downsample. Returns (downsampled_points, inverse_indices)."""
        voxel_idx = np.floor(points / self.voxel_size).astype(np.int32)
        _, unique_idx, inverse = np.unique(
            voxel_idx, axis=0, return_index=True, return_inverse=True,
        )
        num_voxels = len(unique_idx)
        counts = np.bincount(inverse, minlength=num_voxels).astype(np.float64)
        ds_points = np.zeros((num_voxels, 3), dtype=np.float64)
        for dim in range(3):
            ds_points[:, dim] = np.bincount(
                inverse, weights=points[:, dim].astype(np.float64), minlength=num_voxels,
            )
        ds_points /= counts[:, None]
        return ds_points.astype(np.float32), inverse

    def _split_overlapping_blocks(self, points):
        """Split into overlapping blocks with stride = block_size * (1 - overlap).

        Returns list of (block_points, global_indices).
        """
        min_xy = points[:, :2].min(axis=0)
        max_xy = points[:, :2].max(axis=0)
        stride = self.block_size * (1.0 - self.overlap)

        blocks = []
        x = min_xy[0]
        while x < max_xy[0]:
            y = min_xy[1]
            while y < max_xy[1]:
                mask = (
                    (points[:, 0] >= x) & (points[:, 0] < x + self.block_size) &
                    (points[:, 1] >= y) & (points[:, 1] < y + self.block_size)
                )
                indices = np.where(mask)[0]
                if len(indices) >= 10:
                    blocks.append((points[indices], indices))
                y += stride
            x += stride

        return blocks

    def _compute_features(self, sampled):
        """Compute 10-dim features for a block of points."""
        centroid = sampled.mean(axis=0)
        local_xyz = sampled - centroid

        block_min = sampled.min(axis=0)
        block_max = sampled.max(axis=0)

        xy_extent = np.clip(block_max[:2] - block_min[:2], 1e-6, None)
        norm_xy = (sampled[:, :2] - block_min[:2]) / xy_extent

        height_abs = np.clip(sampled[:, 2:3] / self.assumed_height, 0, 1.5)

        num_z_bins = 32
        z_span = max(block_max[2] - block_min[2], 1e-6)
        z_bin = np.clip(
            ((sampled[:, 2] - block_min[2]) / z_span * num_z_bins).astype(np.int32),
            0, num_z_bins - 1,
        )
        bin_counts = np.bincount(z_bin, minlength=num_z_bins).astype(np.float32)
        height_density = (bin_counts[z_bin] / self.num_points).reshape(-1, 1)

        z_range = z_span / self.assumed_height
        z_extent = np.full((self.num_points, 1), z_range, dtype=np.float32)

        xy_range = block_max[:2] - block_min[:2]
        max_xy = max(xy_range.max(), 1e-6)
        aspect = np.full((self.num_points, 1), z_span / max_xy, dtype=np.float32)

        dists = np.linalg.norm(local_xyz, axis=1, keepdims=True)
        mean_dist = max(dists.mean(), 1e-6)
        density = np.clip(dists / mean_dist, 0, 5)

        features = np.concatenate(
            [local_xyz, norm_xy, height_abs, height_density,
             z_extent, aspect, density], axis=1,
        ).astype(np.float32)
        return features.T[np.newaxis, ...]  # (1, 10, N)

    def _predict_block(self, block_points):
        """Run model on a single block. Returns per-point predictions for original indices."""
        N = block_points.shape[0]

        if N >= self.num_points:
            idx = self._rng.choice(N, self.num_points, replace=False)
        else:
            pad = self._rng.choice(N, self.num_points - N, replace=True)
            idx = np.concatenate([np.arange(N), pad])

        sampled = block_points[idx]
        features = self._compute_features(sampled)

        logits = self.session.run(None, {"input": features})[0]  # (1, 2, N)
        preds = logits[0].argmax(axis=0)  # (N,)

        # Map back — for duplicated indices, last write wins (fine for voting)
        block_preds = np.zeros(N, dtype=np.int64)
        for i, orig_idx in enumerate(idx):
            block_preds[orig_idx] = preds[i]

        return block_preds

    def predict(self, points: np.ndarray, level: bool = True) -> np.ndarray:
        """Run full pipeline with overlapping blocks and vote merging.

        Args:
            points: (N, 3) float32 XYZ coordinates.
            level: if True, auto-level the scan before processing.

        Returns:
            predictions: (N,) int -- 1=wall, 0=not-wall for each original point.
        """
        t0 = time.time()

        if level:
            points, _ = auto_level(points)

        # Step 1: Voxel downsample
        ds_points, voxel_inverse = self._voxel_downsample(points)
        t1 = time.time()

        # Step 2: Split into overlapping blocks
        blocks = self._split_overlapping_blocks(ds_points)
        t2 = time.time()

        # Step 3: Predict each block, accumulate votes
        wall_votes = np.zeros(ds_points.shape[0], dtype=np.float32)
        vote_counts = np.zeros(ds_points.shape[0], dtype=np.float32)

        for block_pts, block_indices in blocks:
            block_preds = self._predict_block(block_pts)
            wall_votes[block_indices] += block_preds.astype(np.float32)
            vote_counts[block_indices] += 1.0

        t3 = time.time()

        # Step 4: Majority vote — wall if >50% of overlapping blocks say wall
        vote_counts = np.clip(vote_counts, 1, None)
        ds_predictions = (wall_votes / vote_counts > 0.5).astype(np.int64)

        # Step 5: Back-map to original resolution
        predictions = ds_predictions[voxel_inverse]
        t4 = time.time()

        print(
            f"Inference: {points.shape[0]:,} pts | "
            f"downsample={t1-t0:.2f}s | "
            f"split={t2-t1:.2f}s ({len(blocks)} blocks, {self.overlap*100:.0f}% overlap) | "
            f"predict={t3-t2:.2f}s | "
            f"vote+backmap={t4-t3:.2f}s | "
            f"total={t4-t0:.2f}s"
        )

        return predictions
