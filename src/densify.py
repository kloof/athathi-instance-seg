"""GPU point cloud densification via KNN midpoint interpolation.

Bridges the density gap between synthetic Structured3D data (~0.05m NN distance)
and real merged LiDAR scans (~0.007m NN distance) by inserting interpolated
points between existing nearby points.

Approach: for each point, find its K nearest neighbors that share the same label,
then insert linearly interpolated points along those edges. This preserves local
geometry and label boundaries -- no points are generated across class boundaries.

Usage in training pipeline:
    from src.densify import densify_points_gpu

    # In train_one_epoch, after moving to GPU and before augmentation:
    points, labels = densify_points_gpu(points, labels, factor=2)
    points, labels = augment_batch_gpu(points, labels, ...)
    features = compute_features_batch_gpu(points, room_mins, room_maxs)

    # The model (DGCNN/PointNet) uses Conv1d and handles variable N.
    # With factor=2, N goes from 4096 -> 8192 points per block.
    # This increases GPU memory and compute -- adjust batch size accordingly.
"""

import torch


def _pairwise_distances(points: torch.Tensor) -> torch.Tensor:
    """Compute pairwise squared distances.

    Args:
        points: (B, N, 3)

    Returns:
        (B, N, N) squared distances.
    """
    # ||a - b||^2 = ||a||^2 + ||b||^2 - 2 a.b
    xx = (points * points).sum(dim=2, keepdim=True)  # (B, N, 1)
    inner = torch.bmm(points, points.transpose(1, 2))  # (B, N, N)
    dists = xx + xx.transpose(1, 2) - 2.0 * inner  # (B, N, N)
    return dists.clamp(min=0.0)


def densify_points_gpu(
    points: torch.Tensor,
    labels: torch.Tensor,
    factor: int = 2,
    max_k: int = 6,
    jitter_std: float = 0.002,
    max_dist: float = 0.1,
    same_label_only: bool = True,
    output_n: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Densify a point cloud by inserting interpolated points between neighbors.

    For each original point, generates (factor - 1) new points by:
      1. Finding its K nearest neighbors (optionally restricted to same label)
      2. Randomly selecting one neighbor per new point
      3. Interpolating at a random position along the edge (uniform in [0.3, 0.7])
      4. Adding tiny jitter to avoid exact collinearity

    New points inherit the label of their source (original) point.

    The output contains all original points followed by the new interpolated
    points. If output_n is specified, the result is randomly subsampled (or
    padded by duplication) to exactly output_n points.

    Args:
        points: (B, N, 3) float32 on GPU.
        labels: (B, N) int64 on GPU.
        factor: density multiplier. 2 = double the points, 3 = triple, etc.
        max_k: number of nearest neighbors to consider for interpolation.
        jitter_std: small Gaussian noise added to interpolated points (meters).
            Prevents degenerate collinear patterns. 0.002m = 2mm.
        same_label_only: if True, only interpolate between points with the
            same label. This prevents blurring of class boundaries.
        output_n: if set, subsample/pad the output to exactly this many points.
            Useful to keep a fixed tensor size for batched training.
            If None, output has N * factor points.

    Returns:
        (dense_points, dense_labels):
            dense_points: (B, M, 3) where M = output_n or N * factor
            dense_labels: (B, M) int64
    """
    B, N, _ = points.shape
    device = points.device
    num_new = factor - 1  # new points to generate per original point

    if num_new <= 0:
        if output_n is not None and output_n != N:
            return _subsample_or_pad(points, labels, output_n)
        return points, labels

    # --- Step 1: Find K nearest neighbors ---
    # Compute pairwise distances: (B, N, N)
    sq_dists = _pairwise_distances(points)

    # Mask out self-distance (set to inf so self is never chosen as neighbor)
    eye = torch.eye(N, device=device, dtype=torch.bool).unsqueeze(0)
    sq_dists = sq_dists.masked_fill(eye, float("inf"))

    # Optionally mask cross-label pairs
    if same_label_only:
        # (B, N, 1) == (B, 1, N) -> (B, N, N) same-label mask
        label_match = labels.unsqueeze(2) == labels.unsqueeze(1)
        sq_dists = sq_dists.masked_fill(~label_match, float("inf"))

    # K nearest neighbors: (B, N, K) indices
    K = min(max_k, N - 1)
    _, nn_idx = sq_dists.topk(K, dim=2, largest=False)  # (B, N, K)

    # --- Step 2: Generate interpolated points ---
    all_new_points = []
    all_new_labels = []

    for _ in range(num_new):
        # Pick a random neighbor index (one of K) for each point
        rand_k = torch.randint(0, K, (B, N), device=device)  # (B, N)
        # Gather the chosen neighbor index
        chosen_nn = torch.gather(nn_idx, 2, rand_k.unsqueeze(2)).squeeze(2)  # (B, N)

        # Gather neighbor coordinates: (B, N, 3)
        nn_points = torch.gather(
            points, 1, chosen_nn.unsqueeze(2).expand(-1, -1, 3)
        )

        # Random interpolation weight in [0.3, 0.7] -- avoids clustering at
        # endpoints (which would just duplicate existing points)
        t = 0.3 + 0.4 * torch.rand(B, N, 1, device=device)

        # Interpolated position
        new_pts = points + t * (nn_points - points)  # (B, N, 3)

        # Skip pairs that are too far apart (across gaps)
        pair_dist = torch.norm(nn_points - points, dim=2, keepdim=True)  # (B, N, 1)
        too_far = pair_dist > max_dist
        fallback = points + torch.randn_like(points) * jitter_std
        new_pts = torch.where(too_far, fallback, new_pts)

        # Small jitter to break collinearity
        if jitter_std > 0:
            new_pts = new_pts + torch.randn_like(new_pts) * jitter_std

        all_new_points.append(new_pts)
        all_new_labels.append(labels)  # inherit source label

    # --- Step 3: Concatenate original + new points ---
    new_points = torch.cat(all_new_points, dim=1)  # (B, N * num_new, 3)
    new_labels = torch.cat(all_new_labels, dim=1)  # (B, N * num_new)

    dense_points = torch.cat([points, new_points], dim=1)  # (B, N * factor, 3)
    dense_labels = torch.cat([labels, new_labels], dim=1)  # (B, N * factor)

    # --- Step 4: Optional fixed-size output ---
    if output_n is not None:
        dense_points, dense_labels = _subsample_or_pad(
            dense_points, dense_labels, output_n
        )

    return dense_points, dense_labels


def _subsample_or_pad(
    points: torch.Tensor,
    labels: torch.Tensor,
    target_n: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Randomly subsample or pad-by-duplication to reach target_n points.

    Args:
        points: (B, M, 3)
        labels: (B, M)
        target_n: desired number of points.

    Returns:
        (B, target_n, 3) and (B, target_n).
    """
    B, M, _ = points.shape
    device = points.device

    if M == target_n:
        return points, labels

    if M > target_n:
        # Random subsample
        idx = torch.stack([
            torch.randperm(M, device=device)[:target_n] for _ in range(B)
        ])
    else:
        # Keep all, then randomly duplicate to fill
        base = torch.arange(M, device=device).unsqueeze(0).expand(B, -1)
        extra = torch.randint(0, M, (B, target_n - M), device=device)
        idx = torch.cat([base, extra], dim=1)

    points_out = torch.gather(points, 1, idx.unsqueeze(2).expand(-1, -1, 3))
    labels_out = torch.gather(labels, 1, idx)
    return points_out, labels_out


def densify_points_gpu_fast(
    points: torch.Tensor,
    labels: torch.Tensor,
    factor: int = 2,
    jitter_std: float = 0.002,
    max_dist: float = 0.1,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fast approximate densification without explicit KNN.

    Instead of computing full pairwise distances (O(N^2) memory), this uses a
    random-shift strategy: for each point, it finds a nearby point by sorting
    along a random projection axis. Adjacent points in this sorted order are
    likely spatial neighbors.

    Only interpolates between points within max_dist of each other.
    Points whose nearest sorted neighbor is too far get a jittered duplicate
    instead (no bridging across gaps).

    Args:
        points: (B, N, 3) float32 on GPU.
        labels: (B, N) int64 on GPU.
        factor: density multiplier (2-4).
        jitter_std: Gaussian noise std for interpolated points (meters).
        max_dist: maximum distance (meters) between points to interpolate.
            Pairs farther apart get a jittered duplicate instead.

    Returns:
        (dense_points, dense_labels) with M = N * factor points.
    """
    B, N, _ = points.shape
    device = points.device
    num_new = factor - 1

    if num_new <= 0:
        return points, labels

    all_new_points = []
    all_new_labels = []

    for _ in range(num_new):
        # Random projection direction (unit vector per batch)
        direction = torch.randn(B, 1, 3, device=device)
        direction = direction / direction.norm(dim=2, keepdim=True).clamp(min=1e-8)

        # Project points onto the random axis: (B, N)
        proj = (points * direction).sum(dim=2)

        # Sort by projection value -> spatially nearby points become adjacent
        sort_idx = proj.argsort(dim=1)  # (B, N)

        # Neighbor = next point in sorted order (wrap around)
        neighbor_sort_idx = torch.roll(sort_idx, -1, dims=1)  # (B, N)

        # Unsort to get neighbor index for each original point
        # inv_sort_idx[sort_idx[i]] = i, so neighbor of point j = neighbor_sort_idx[inv[j]]
        inv_sort_idx = torch.empty_like(sort_idx)
        batch_range = torch.arange(B, device=device).unsqueeze(1).expand_as(sort_idx)
        inv_sort_idx[batch_range, sort_idx] = torch.arange(N, device=device).unsqueeze(0).expand(B, -1)

        # For each original point, find its neighbor in the sorted order
        sorted_pos = inv_sort_idx  # position of each point in sorted order: (B, N)
        nn_in_sorted = torch.gather(neighbor_sort_idx, 1, sorted_pos)  # (B, N)

        # Gather neighbor coordinates
        nn_points = torch.gather(
            points, 1, nn_in_sorted.unsqueeze(2).expand(-1, -1, 3)
        )
        nn_labels = torch.gather(labels, 1, nn_in_sorted)

        # Interpolation weight
        t = 0.3 + 0.4 * torch.rand(B, N, 1, device=device)
        new_pts = points + t * (nn_points - points)

        # Compute actual distance to neighbor
        pair_dist = torch.norm(nn_points - points, dim=2)  # (B, N)

        # Fall back to jittered duplicate if labels differ OR neighbor too far
        label_mismatch = (labels != nn_labels)  # (B, N)
        too_far = pair_dist > max_dist  # (B, N)
        skip_mask = (label_mismatch | too_far).unsqueeze(2)  # (B, N, 1)

        fallback = points + torch.randn_like(points) * jitter_std
        new_pts = torch.where(skip_mask, fallback, new_pts)

        # Add jitter
        if jitter_std > 0:
            new_pts = new_pts + torch.randn_like(new_pts) * jitter_std

        all_new_points.append(new_pts)
        all_new_labels.append(labels)

    new_points = torch.cat(all_new_points, dim=1)
    new_labels = torch.cat(all_new_labels, dim=1)

    dense_points = torch.cat([points, new_points], dim=1)
    dense_labels = torch.cat([labels, new_labels], dim=1)

    return dense_points, dense_labels
