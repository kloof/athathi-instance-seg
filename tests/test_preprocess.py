import numpy as np
import pytest


def test_voxel_downsample_reduces_points():
    from src.preprocess import voxel_downsample

    rng = np.random.default_rng(42)
    points = rng.uniform([0, 0, 0], [10, 10, 3], size=(100_000, 3)).astype(np.float32)
    labels = rng.integers(0, 2, size=100_000).astype(np.int64)

    ds_points, ds_labels = voxel_downsample(points, labels, voxel_size=0.05)

    assert ds_points.shape[0] < points.shape[0]
    assert ds_points.shape[0] == ds_labels.shape[0]
    assert ds_points.shape[1] == 3


def test_split_into_blocks_covers_all_points():
    from src.preprocess import split_into_blocks

    rng = np.random.default_rng(42)
    points = rng.uniform([0, 0, 0], [5, 5, 3], size=(10_000, 3)).astype(np.float32)
    labels = rng.integers(0, 2, size=10_000).astype(np.int64)

    blocks = split_into_blocks(points, labels, block_size=1.0, min_points=1)

    total_points = sum(b[0].shape[0] for b in blocks)
    assert total_points == points.shape[0]
    assert 20 <= len(blocks) <= 30


def test_sample_block_exact_size():
    from src.preprocess import sample_block

    rng = np.random.default_rng(42)

    small_pts = rng.random((100, 3)).astype(np.float32)
    small_lbl = np.ones(100, dtype=np.int64)
    out_pts, out_lbl = sample_block(small_pts, small_lbl, num_points=4096)
    assert out_pts.shape == (4096, 3)
    assert out_lbl.shape == (4096,)

    big_pts = rng.random((10_000, 3)).astype(np.float32)
    big_lbl = np.zeros(10_000, dtype=np.int64)
    out_pts, out_lbl = sample_block(big_pts, big_lbl, num_points=4096)
    assert out_pts.shape == (4096, 3)
    assert out_lbl.shape == (4096,)


def test_compute_features_shape_and_range():
    from src.preprocess import compute_block_features

    rng = np.random.default_rng(42)
    block_points = rng.uniform([4.5, 2.5, 0], [5.5, 3.5, 3], size=(4096, 3)).astype(np.float32)
    room_min = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    room_max = np.array([10.0, 8.0, 3.0], dtype=np.float32)

    features = compute_block_features(block_points, room_min, room_max)

    assert features.shape == (4096, 6)
    local_xyz = features[:, :3]
    assert abs(local_xyz.mean()) < 0.5
    norm_xyz = features[:, 3:6]
    assert norm_xyz.min() >= -0.01
    assert norm_xyz.max() <= 1.01
