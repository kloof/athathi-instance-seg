import numpy as np
import pytest
import tempfile
from pathlib import Path


@pytest.fixture
def dummy_onnx_model():
    """Create a real (untrained) ONNX model for testing the pipeline."""
    from src.model import PointNetSegmentation
    from src.export import export_to_onnx

    model = PointNetSegmentation(input_dim=6, num_classes=2)
    model.eval()

    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "test.onnx"
        export_to_onnx(model, path, input_dim=6)
        yield str(path)


def test_infer_pipeline_output_shape(dummy_onnx_model):
    from src.infer import WallSegPipeline

    pipeline = WallSegPipeline(dummy_onnx_model, voxel_size=0.05, block_size=2.0, num_points=512)

    rng = np.random.default_rng(42)
    points = rng.uniform(-3, 3, (50_000, 3)).astype(np.float32)

    predictions = pipeline.predict(points)

    assert predictions.shape == (50_000,), f"Got {predictions.shape}"
    assert set(np.unique(predictions)).issubset({0, 1})


def test_infer_pipeline_small_cloud(dummy_onnx_model):
    from src.infer import WallSegPipeline

    pipeline = WallSegPipeline(dummy_onnx_model, voxel_size=0.1, block_size=10.0, num_points=512)

    points = np.random.default_rng(42).uniform(-1, 1, (500, 3)).astype(np.float32)
    predictions = pipeline.predict(points)

    assert predictions.shape == (500,)
