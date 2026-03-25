"""ONNX export and INT8 quantization for RPi4 deployment."""

import torch
import numpy as np
from pathlib import Path

from src.model import PointNetSegmentation


def export_to_onnx(
    model: PointNetSegmentation,
    output_path: str | Path,
    input_dim: int = 6,
    num_points: int = 4096,
) -> None:
    """Export PyTorch model to ONNX with dynamic point count."""
    model.eval()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    dummy_input = torch.randn(1, input_dim, num_points)

    torch.onnx.export(
        model,
        dummy_input,
        str(output_path),
        input_names=["input"],
        output_names=["logits"],
        dynamic_axes={
            "input": {0: "batch", 2: "num_points"},
            "logits": {0: "batch", 2: "num_points"},
        },
        opset_version=17,
        do_constant_folding=True,
    )

    size_mb = output_path.stat().st_size / (1024 * 1024)
    print(f"Exported ONNX model: {output_path} ({size_mb:.1f} MB)")


def quantize_onnx(
    input_path: str | Path,
    output_path: str | Path,
) -> bool:
    """Apply dynamic INT8 quantization to an ONNX model.

    Returns True if quantization succeeded, False if it failed
    (e.g., ConvInteger not supported on this ONNX Runtime build).
    The FP32 model is only 3.3 MB, so INT8 is optional.
    """
    try:
        from onnxruntime.quantization import quantize_dynamic, QuantType

        quantize_dynamic(
            str(input_path),
            str(output_path),
            weight_type=QuantType.QInt8,
        )

        # Verify the quantized model actually loads
        import onnxruntime as ort
        ort.InferenceSession(str(output_path), providers=["CPUExecutionProvider"])

        orig_mb = Path(input_path).stat().st_size / (1024 * 1024)
        quant_mb = Path(output_path).stat().st_size / (1024 * 1024)
        print(f"Quantized: {orig_mb:.1f} MB -> {quant_mb:.1f} MB ({quant_mb/orig_mb:.0%})")
        return True

    except Exception as e:
        print(f"INT8 quantization failed ({e}), using FP32 model instead (3.3 MB is fine)")
        import shutil
        shutil.copy2(str(input_path), str(output_path))
        return False
