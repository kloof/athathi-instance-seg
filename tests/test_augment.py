import numpy as np
import pytest


def test_augment_jitter_changes_points():
    from src.augment import augment_points

    rng = np.random.default_rng(42)
    points = rng.random((1000, 3)).astype(np.float32)
    labels = np.ones(1000, dtype=np.int64)

    aug_points, _ = augment_points(
        points.copy(), labels.copy(),
        jitter_std=0.01, dropout_range=None, rotation=False, outlier_ratio=None,
        rng=np.random.default_rng(42),
    )

    assert not np.allclose(points, aug_points[:1000])
    diff = np.abs(aug_points[:1000] - points)
    assert diff.max() < 0.1


def test_augment_dropout_reduces_points():
    from src.augment import augment_points

    points = np.random.default_rng(42).random((10_000, 3)).astype(np.float32)
    labels = np.zeros(10_000, dtype=np.int64)

    aug_points, aug_labels = augment_points(
        points.copy(), labels.copy(),
        jitter_std=0.0, dropout_range=[0.7, 0.8], rotation=False, outlier_ratio=None,
        rng=np.random.default_rng(42),
    )

    assert aug_points.shape[0] < 10_000
    assert aug_points.shape[0] > 5_000


def test_augment_outlier_injection():
    from src.augment import augment_points

    points = np.zeros((1000, 3), dtype=np.float32)
    points[:, 0] = np.linspace(-1, 1, 1000)  # spread out for bbox
    labels = np.ones(1000, dtype=np.int64)

    aug_points, aug_labels = augment_points(
        points.copy(), labels.copy(),
        jitter_std=0.0, dropout_range=None, rotation=False,
        outlier_ratio=[0.1, 0.1],
        rng=np.random.default_rng(42),
    )

    num_outliers = aug_points.shape[0] - 1000
    assert num_outliers > 0
    assert (aug_labels[1000:] == 0).all()


def test_augment_rotation_preserves_distances():
    from src.augment import augment_points

    rng = np.random.default_rng(42)
    points = rng.uniform(-5, 5, (500, 3)).astype(np.float32)
    labels = np.zeros(500, dtype=np.int64)

    orig_dists = np.linalg.norm(points, axis=1)

    aug_points, _ = augment_points(
        points.copy(), labels.copy(),
        jitter_std=0.0, dropout_range=None, rotation=True, outlier_ratio=None,
        rng=np.random.default_rng(42),
    )

    aug_dists = np.linalg.norm(aug_points, axis=1)
    np.testing.assert_allclose(orig_dists, aug_dists, atol=1e-5)
