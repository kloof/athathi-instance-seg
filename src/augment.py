"""Noise augmentation to close the synthetic-to-real domain gap."""

import numpy as np


def augment_points(
    points: np.ndarray,
    labels: np.ndarray,
    jitter_std: float | None = 0.005,
    dropout_range: list[float] | None = None,
    rotation: bool = True,
    outlier_ratio: list[float] | None = None,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply noise augmentation to a point cloud.

    All augmentations are optional and controlled by their parameters.

    Args:
        points: (N, 3) XYZ.
        labels: (N,) binary labels.
        jitter_std: std dev of Gaussian noise in meters. None to skip.
        dropout_range: [min_keep, max_keep] fraction. None to skip.
        rotation: if True, random rotation around Z (gravity) axis.
        outlier_ratio: [min_ratio, max_ratio] of outlier points to inject. None to skip.
        rng: numpy random generator.

    Returns:
        Augmented (points, labels).
    """
    if rng is None:
        rng = np.random.default_rng()

    points = points.copy()
    labels = labels.copy()

    # 1. Gaussian jitter
    if jitter_std is not None and jitter_std > 0:
        noise = rng.normal(0, jitter_std, size=points.shape).astype(points.dtype)
        points += noise

    # 2. Random dropout
    if dropout_range is not None:
        keep_ratio = rng.uniform(dropout_range[0], dropout_range[1])
        N = points.shape[0]
        keep_count = max(1, int(N * keep_ratio))
        idx = rng.choice(N, keep_count, replace=False)
        points = points[idx]
        labels = labels[idx]

    # 3. Random rotation around Z axis (gravity-aligned)
    if rotation:
        theta = rng.uniform(0, 2 * np.pi)
        cos_t, sin_t = np.cos(theta), np.sin(theta)
        rotation_matrix = np.array([
            [cos_t, -sin_t, 0],
            [sin_t,  cos_t, 0],
            [0,      0,     1],
        ], dtype=np.float64)
        points = (points.astype(np.float64) @ rotation_matrix.T).astype(np.float32)

    # 4. Outlier injection
    if outlier_ratio is not None:
        ratio = rng.uniform(outlier_ratio[0], outlier_ratio[1])
        num_outliers = max(1, int(points.shape[0] * ratio))
        bbox_min = points.min(axis=0)
        bbox_max = points.max(axis=0)
        outliers = rng.uniform(bbox_min, bbox_max, (num_outliers, 3)).astype(np.float32)
        outlier_labels = np.zeros(num_outliers, dtype=labels.dtype)
        points = np.concatenate([points, outliers], axis=0)
        labels = np.concatenate([labels, outlier_labels], axis=0)

    return points, labels
