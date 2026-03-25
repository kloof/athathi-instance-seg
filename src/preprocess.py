"""Point cloud preprocessing: voxel downsample, block split, feature computation."""

import numpy as np


def voxel_downsample(
    points: np.ndarray,
    labels: np.ndarray,
    voxel_size: float = 0.03,
) -> tuple[np.ndarray, np.ndarray]:
    """Voxel grid downsample using vectorized np.bincount (O(N)).

    For each occupied voxel, computes the centroid and majority-vote label.
    Pure-numpy — runs on RPi4.
    """
    voxel_indices = np.floor(points / voxel_size).astype(np.int32)

    _, unique_idx, inverse = np.unique(
        voxel_indices, axis=0, return_index=True, return_inverse=True,
    )

    num_voxels = len(unique_idx)
    counts = np.bincount(inverse, minlength=num_voxels).astype(np.float64)

    ds_points = np.zeros((num_voxels, 3), dtype=np.float64)
    for dim in range(3):
        ds_points[:, dim] = np.bincount(
            inverse, weights=points[:, dim].astype(np.float64), minlength=num_voxels,
        )
    ds_points /= counts[:, None]
    ds_points = ds_points.astype(np.float32)

    wall_counts = np.bincount(
        inverse, weights=(labels == 1).astype(np.float64), minlength=num_voxels,
    )
    ds_labels = (wall_counts > counts / 2).astype(np.int64)

    return ds_points, ds_labels


def split_into_blocks(
    points: np.ndarray,
    labels: np.ndarray,
    block_size: float = 1.0,
    min_points: int = 100,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Split point cloud into block_size x block_size vertical columns."""
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


def sample_block(
    points: np.ndarray,
    labels: np.ndarray,
    num_points: int = 4096,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Sample or pad a block to exactly num_points points."""
    if rng is None:
        rng = np.random.default_rng()

    N = points.shape[0]
    if N >= num_points:
        idx = rng.choice(N, num_points, replace=False)
    else:
        pad = rng.choice(N, num_points - N, replace=True)
        idx = np.concatenate([np.arange(N), pad])

    return points[idx], labels[idx]


def compute_block_features(
    block_points: np.ndarray,
    room_min: np.ndarray,
    room_max: np.ndarray,
) -> np.ndarray:
    """Compute 6-dim features: local XYZ (relative to centroid) + normalized XYZ (room bounds)."""
    centroid = block_points.mean(axis=0)
    local_xyz = block_points - centroid

    room_extent = room_max - room_min
    room_extent = np.clip(room_extent, 1e-6, None)
    norm_xyz = (block_points - room_min) / room_extent

    features = np.concatenate([local_xyz, norm_xyz], axis=1).astype(np.float32)
    return features
