"""GPU-based augmentation and feature computation using torch. Runs on batches.

Augmentation pipeline:
  1. Densification (x6): KNN interpolation between same-label neighbors
  2. LiDAR noise (5-20mm random per point)
  3. Point dropout (sensor occlusion)
  4. Z-axis rotation + tilt + anisotropic scale
"""

import torch
import math


def densify_batch_gpu(
    points: torch.Tensor,
    labels: torch.Tensor,
    target_multiplier: int = 6,
    max_dist: float = 0.30,
    K: int = 12,
    keep_all: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Densify point clouds via torch_cluster KNN interpolation on GPU.

    For each point, finds K true nearest neighbors via CUDA KNN,
    picks a random same-label neighbor within max_dist, and interpolates.

    Args:
        points: (B, N, 3) float32 on GPU.
        labels: (B, N) int64 on GPU.
        target_multiplier: density factor (default 6x).
        max_dist: max distance for interpolation pairs (meters).
        K: number of KNN neighbors.
        keep_all: if True, return all N*factor points.

    Returns:
        keep_all=False: (B, N, 3), (B, N)
        keep_all=True:  (B, N*factor, 3), (B, N*factor)
    """
    from torch_cluster import knn as tc_knn

    B, N, _ = points.shape
    device = points.device
    new_rounds = target_multiplier - 1

    if new_rounds <= 0:
        return points, labels

    # Run torch_cluster KNN once (CUDA optimized)
    pts_flat = points.reshape(B * N, 3)
    batch_vec = torch.arange(B, device=device).repeat_interleave(N)
    edge_index = tc_knn(pts_flat, pts_flat, K + 1, batch_vec, batch_vec)  # +1 for self

    # Reshape to (B, N, K+1) and remove self-neighbor
    knn_idx = edge_index[1].reshape(B, N, K + 1)
    knn_idx = knn_idx[:, :, 1:]  # skip self (index 0), keep K neighbors
    # Convert global to per-sample indices
    offsets = torch.arange(B, device=device).unsqueeze(1).unsqueeze(2) * N
    knn_idx = knn_idx - offsets  # (B, N, K)

    # Gather neighbor points and labels
    knn_pts = torch.gather(
        points, 1,
        knn_idx.reshape(B, N * K).unsqueeze(2).expand(-1, -1, 3)
    ).reshape(B, N, K, 3)
    knn_lbl = torch.gather(
        labels, 1,
        knn_idx.reshape(B, N * K)
    ).reshape(B, N, K)

    # Distances to neighbors
    knn_dists = (knn_pts - points.unsqueeze(2)).norm(dim=3)  # (B, N, K)

    # Mask: same label + within max_dist
    valid = (knn_lbl == labels.unsqueeze(2)) & (knn_dists < max_dist) & (knn_dists > 1e-6)

    has_any_valid = valid.any(dim=2)  # (B, N)
    has_two_valid = valid.sum(dim=2) >= 2  # (B, N) — at least 2 valid neighbors

    all_new_pts = [points]
    all_new_lbl = [labels]

    for _ in range(new_rounds):
        # Pick two different random valid neighbors
        rand1 = torch.rand(B, N, K, device=device)
        rand1 = torch.where(valid, rand1, torch.full_like(rand1, -1.0))
        k1 = rand1.argmax(dim=2)  # (B, N)

        rand2 = torch.rand(B, N, K, device=device)
        rand2 = torch.where(valid, rand2, torch.full_like(rand2, -1.0))
        # Make sure k2 != k1 by zeroing out k1's position
        rand2.scatter_(2, k1.unsqueeze(2), -1.0)
        k2 = rand2.argmax(dim=2)  # (B, N)

        idx1 = torch.gather(knn_idx, 2, k1.unsqueeze(2)).squeeze(2)
        idx2 = torch.gather(knn_idx, 2, k2.unsqueeze(2)).squeeze(2)
        nb1 = torch.gather(points, 1, idx1.unsqueeze(2).expand(-1, -1, 3))
        nb2 = torch.gather(points, 1, idx2.unsqueeze(2).expand(-1, -1, 3))

        # Barycentric interpolation between (point, neighbor1, neighbor2)
        # Random weights that sum to 1
        w = torch.rand(B, N, 3, device=device)
        w = w / w.sum(dim=2, keepdim=True)

        new_pts = (
            points * w[:, :, 0:1] +
            nb1 * w[:, :, 1:2] +
            nb2 * w[:, :, 2:3]
        )

        # If only 1 valid neighbor, do regular 2-point interpolation
        t = torch.rand(B, N, 1, device=device) * 0.8 + 0.1
        two_pt = points * (1 - t) + nb1 * t

        new_pts = torch.where(
            has_two_valid.unsqueeze(2).expand(-1, -1, 3),
            new_pts,
            two_pt,
        )

        # No valid neighbor at all: tiny jitter
        new_pts = torch.where(
            has_any_valid.unsqueeze(2).expand(-1, -1, 3),
            new_pts,
            points + torch.randn_like(points) * 0.001,
        )

        all_new_pts.append(new_pts)
        all_new_lbl.append(labels)

    dense_pts = torch.cat(all_new_pts, dim=1)
    dense_lbl = torch.cat(all_new_lbl, dim=1)

    if keep_all:
        return dense_pts, dense_lbl

    # Subsample back to N
    N_dense = dense_pts.shape[1]
    perm = torch.stack([torch.randperm(N_dense, device=device) for _ in range(B)])
    idx = perm[:, :N]
    dense_pts = torch.gather(dense_pts, 1, idx.unsqueeze(-1).expand(-1, -1, 3))
    dense_lbl = torch.gather(dense_lbl, 1, idx)

    return dense_pts, dense_lbl


def augment_batch_gpu(
    points: torch.Tensor,
    labels: torch.Tensor,
    jitter_std: float = 0.002,
    rotation: bool = True,
    dropout_keep: float = 0.85,
    densify: bool = True,
    densify_multiplier: int = 6,
    noise_min: float = 0.005,
    noise_max: float = 0.020,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply augmentation pipeline to a batch on GPU.

    Pipeline:
      1. Densify via KNN interpolation (x6)
      2. LiDAR-like noise (5-20mm random per point)
      3. Point dropout (sensor occlusion) — subsample back to N
      4. Z-axis rotation
      5. Tilt (imperfect leveling)
      6. Anisotropic scale

    Args:
        points: (B, N, 3) float32 on GPU.
        labels: (B, N) int64 on GPU.
        jitter_std: unused, kept for API compat. Noise is now 5-20mm.
        rotation: random Z-axis rotation per sample.
        dropout_keep: fraction of points to keep from densified cloud.
        densify: whether to apply densification.
        densify_multiplier: densification factor (default 6x).
        noise_min: min noise magnitude in meters (default 5mm).
        noise_max: max noise magnitude in meters (default 20mm).
    """
    B, N, _ = points.shape

    # 1. Densify — KNN interpolation then subsample back to N
    if densify:
        points, labels = densify_batch_gpu(
            points, labels, target_multiplier=densify_multiplier
        )

    # 2. LiDAR-like noise — random 5-20mm per point
    B, N, _ = points.shape  # refresh N after potential densify
    if noise_max > 0:
        noise_mag = torch.rand(B, N, 1, device=points.device) * (noise_max - noise_min) + noise_min
        noise_dir = torch.randn_like(points)
        noise_dir = noise_dir / (torch.norm(noise_dir, dim=2, keepdim=True) + 1e-8)
        points = points + noise_dir * noise_mag

    # 3. Point dropout — randomly drop points to simulate occlusion
    if dropout_keep < 1.0:
        n_keep = max(int(N * dropout_keep), 1)
        idx = torch.stack([torch.randperm(N, device=points.device)[:n_keep] for _ in range(B)])
        points = torch.gather(points, 1, idx.unsqueeze(-1).expand(-1, -1, 3))
        labels = torch.gather(labels, 1, idx)
        B, N, _ = points.shape  # update N

    # 3.5. Random flip X and/or Y
    flip_x = (torch.rand(B, device=points.device) > 0.5).float() * 2 - 1  # -1 or 1
    flip_y = (torch.rand(B, device=points.device) > 0.5).float() * 2 - 1
    points[:, :, 0] *= flip_x.unsqueeze(1)
    points[:, :, 1] *= flip_y.unsqueeze(1)

    # 4. Random Z-axis rotation
    if rotation:
        theta = torch.rand(B, device=points.device) * 2 * math.pi
        cos_t = torch.cos(theta)
        sin_t = torch.sin(theta)
        rot_z = torch.zeros(B, 3, 3, device=points.device)
        rot_z[:, 0, 0] = cos_t
        rot_z[:, 0, 1] = -sin_t
        rot_z[:, 1, 0] = sin_t
        rot_z[:, 1, 1] = cos_t
        rot_z[:, 2, 2] = 1.0
        points = torch.bmm(points, rot_z.transpose(1, 2))

    # 5. Random tilt ±2° — simulate imperfect leveling
    max_tilt = math.radians(2)
    tilt_x = (torch.rand(B, device=points.device) * 2 - 1) * max_tilt
    tilt_y = (torch.rand(B, device=points.device) * 2 - 1) * max_tilt

    cx, sx = torch.cos(tilt_x), torch.sin(tilt_x)
    rot_x = torch.zeros(B, 3, 3, device=points.device)
    rot_x[:, 0, 0] = 1.0
    rot_x[:, 1, 1] = cx
    rot_x[:, 1, 2] = -sx
    rot_x[:, 2, 1] = sx
    rot_x[:, 2, 2] = cx

    cy, sy = torch.cos(tilt_y), torch.sin(tilt_y)
    rot_y = torch.zeros(B, 3, 3, device=points.device)
    rot_y[:, 0, 0] = cy
    rot_y[:, 0, 2] = sy
    rot_y[:, 1, 1] = 1.0
    rot_y[:, 2, 0] = -sy
    rot_y[:, 2, 2] = cy

    rot_tilt = torch.bmm(rot_x, rot_y)
    points = torch.bmm(points, rot_tilt.transpose(1, 2))

    # 6. Anisotropic scale — slight per-axis stretch [0.95, 1.05]
    scale = 0.95 + torch.rand(B, 1, 3, device=points.device) * 0.1
    points = points * scale

    return points, labels


def compute_features_batch_gpu(
    points: torch.Tensor,
    room_mins: torch.Tensor,
    room_maxs: torch.Tensor,
    assumed_height: float = 3.5,
    num_z_bins: int = 32,
) -> torch.Tensor:
    """Compute 10-dim features for a batch on GPU.

    Features:
        - local_xyz (3): relative to block centroid
        - norm_xy (2): per-block [0,1] normalization in XY
        - height_abs (1): absolute Z / assumed_height
        - height_density (1): fraction of block points at same Z level
          KEY DISCRIMINATOR: floor/ceiling → high (dense), walls → low (sparse)
        - z_extent (1): block Z range / assumed_height
        - block_aspect (1): Z_extent / max(XY_extent)
        - density (1): relative distance to centroid

    Args:
        points: (B, N, 3) float32.
        room_mins: (B, 3) — unused, kept for API compat.
        room_maxs: (B, 3) — unused, kept for API compat.

    Returns:
        (B, 10, N) features — channels first for Conv1d.
    """
    B, N, _ = points.shape

    # 1. Local XYZ: relative to block centroid
    centroid = points.mean(dim=1, keepdim=True)  # (B, 1, 3)
    local_xyz = points - centroid                 # (B, N, 3)

    # 2. Per-block normalized XY (room-size agnostic)
    block_mins = points.min(dim=1, keepdim=True).values
    block_maxs = points.max(dim=1, keepdim=True).values
    xy_extent = (block_maxs[:, :, :2] - block_mins[:, :, :2]).clamp(min=1e-6)
    norm_xy = (points[:, :, :2] - block_mins[:, :, :2]) / xy_extent

    # 3. Absolute Z height normalized by assumed ceiling height
    height_abs = (points[:, :, 2:3] / assumed_height).clamp(0, 1.5)

    # 4. Height density: what fraction of block points are at this Z level?
    #    Floor/ceiling points → high value (dense clusters)
    #    Wall points → low value (spread across many Z levels)
    z_min = block_mins[:, :, 2:3]  # (B, 1, 1)
    z_max = block_maxs[:, :, 2:3]  # (B, 1, 1)
    z_span = (z_max - z_min).clamp(min=1e-6)
    # Bin each point's Z
    z_bin = ((points[:, :, 2:3] - z_min) / z_span * num_z_bins).long().clamp(0, num_z_bins - 1)
    # Count points per bin using scatter_add
    bin_counts = torch.zeros(B, num_z_bins, dtype=torch.float32, device=points.device)
    bin_counts.scatter_add_(1, z_bin.squeeze(2), torch.ones(B, N, device=points.device))
    # Look up each point's bin count, normalize by total
    height_density = torch.gather(bin_counts, 1, z_bin.squeeze(2))  # (B, N)
    height_density = (height_density / N).unsqueeze(2)  # (B, N, 1)

    # 5. Block Z extent (walls are tall, floors are flat)
    z_range = z_span / assumed_height
    z_extent = z_range.expand(B, N, 1)

    # 6. Block aspect ratio: Z_range / max(XY_range)
    xy_range = block_maxs[:, :, :2] - block_mins[:, :, :2]
    max_xy = xy_range.max(dim=2, keepdim=True).values.clamp(min=1e-6)
    aspect = (z_span / max_xy).expand(B, N, 1)

    # 7. Local point density proxy
    dists = torch.norm(local_xyz, dim=2, keepdim=True)
    mean_dist = dists.mean(dim=1, keepdim=True)
    density = (dists / mean_dist.clamp(min=1e-6)).clamp(0, 5)

    # Concatenate: (B, N, 10) -> (B, 10, N)
    features = torch.cat([
        local_xyz, norm_xy, height_abs, height_density,
        z_extent, aspect, density,
    ], dim=2)
    features = features.permute(0, 2, 1)

    return features
