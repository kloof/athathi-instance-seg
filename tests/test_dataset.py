import numpy as np
import pytest
import tempfile
from pathlib import Path


@pytest.fixture
def fake_data_dir():
    """Create a temp directory with fake precomputed block files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        blocks_dir = Path(tmpdir) / "blocks"
        blocks_dir.mkdir()

        rng = np.random.default_rng(42)
        num_blocks = 50
        num_points = 4096

        points = rng.uniform(-5, 5, (num_blocks, num_points, 3)).astype(np.float32)
        labels = rng.integers(0, 2, (num_blocks, num_points)).astype(np.int64)
        room_mins = np.full((num_blocks, 3), -5.0, dtype=np.float32)
        room_maxs = np.full((num_blocks, 3), 5.0, dtype=np.float32)

        np.savez(blocks_dir / "blocks_0000.npz",
                 points=points, labels=labels,
                 room_mins=room_mins, room_maxs=room_maxs)
        yield tmpdir


def test_dataset_length(fake_data_dir):
    from src.dataset import WallSegDataset
    ds = WallSegDataset(fake_data_dir, num_points=4096, augment=False)
    assert len(ds) == 50


def test_dataset_item_shapes(fake_data_dir):
    from src.dataset import WallSegDataset
    ds = WallSegDataset(fake_data_dir, num_points=4096, augment=False)
    points, labels, room_mins, room_maxs = ds[0]

    assert points.shape == (4096, 3)
    assert labels.shape == (4096,)
    assert room_mins.shape == (3,)
    assert room_maxs.shape == (3,)


def test_dataset_labels_are_binary(fake_data_dir):
    from src.dataset import WallSegDataset
    ds = WallSegDataset(fake_data_dir, num_points=4096, augment=False)
    _, labels, _, _ = ds[0]
    unique = labels.unique().tolist()
    assert all(l in [0, 1] for l in unique)
