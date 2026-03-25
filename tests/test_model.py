import torch
import pytest


def test_pointnet_output_shape():
    from src.model import PointNetSegmentation

    model = PointNetSegmentation(input_dim=6, num_classes=2)
    x = torch.randn(4, 6, 4096)
    logits = model(x)
    assert logits.shape == (4, 2, 4096)


def test_pointnet_different_point_counts():
    from src.model import PointNetSegmentation

    model = PointNetSegmentation(input_dim=6, num_classes=2)

    for N in [1024, 4096, 8192]:
        x = torch.randn(2, 6, N)
        logits = model(x)
        assert logits.shape == (2, 2, N)


def test_pointnet_parameter_count():
    from src.model import PointNetSegmentation

    model = PointNetSegmentation(input_dim=6, num_classes=2)
    total_params = sum(p.numel() for p in model.parameters())

    assert total_params < 1_500_000, f"Too many params: {total_params:,}"
    print(f"Parameter count: {total_params:,}")


def test_pointnet_backward_pass():
    from src.model import PointNetSegmentation

    model = PointNetSegmentation(input_dim=6, num_classes=2)
    x = torch.randn(2, 6, 4096)
    labels = torch.randint(0, 2, (2, 4096))

    logits = model(x)
    loss = torch.nn.functional.cross_entropy(logits, labels)
    loss.backward()

    for name, param in model.named_parameters():
        assert param.grad is not None, f"No gradient for {name}"
