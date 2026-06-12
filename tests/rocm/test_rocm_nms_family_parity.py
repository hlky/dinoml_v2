from __future__ import annotations

import os
import shutil

import numpy as np
import pytest

import dinoml as dml
from dinoml.backends import rocm as rocm_backend
from dinoml.runtime import load
from tests.ir.test_nms_family import (
    _batched_nms_boxes,
    _efficient_nms_inputs,
    _nms_inputs,
    _torch_batched_keep_oracle,
    _torch_efficient_nms_oracle,
    _torch_nms_boxes_oracle,
)


pytestmark = pytest.mark.skipif(
    os.environ.get("DINOML_RUN_ROCM_CONTRACTS") != "1",
    reason="set DINOML_RUN_ROCM_CONTRACTS=1 in the ROCm venv to compile/run ROCm artifacts",
)


def _rocm_module_compile_toolchain_available() -> bool:
    if rocm_backend._rocm_sdk_command() is not None:
        return True
    if shutil.which("hipconfig") is not None:
        return True
    return bool(os.environ.get("ROCM_PATH") or os.environ.get("HIP_PATH"))


def test_rocm_nms_parity(tmp_path):
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("Torch HIP/CUDA device is not available")
    if not _rocm_module_compile_toolchain_available():
        pytest.skip("ROCm module compile toolchain not found from active Python/PATH")

    class Tiny(dml.nn.Module):
        def forward(self, boxes, scores):
            return dml.ops.output(dml.ops.nms(boxes, scores, pre_nms_top=4, max_output=3, iou_threshold=0.5, min_box_size=0.5), "y")

    spec = dml.trace(
        Tiny(),
        inputs={"boxes": dml.TensorSpec([2, 5, 4], "float32"), "scores": dml.TensorSpec([2, 5], "float32")},
        name="rocm_nms_parity",
    )
    boxes, scores = _nms_inputs()
    artifact = dml.compile(spec, dml.Target("rocm"), tmp_path / "nms_rocm.dinoml")
    module = load(artifact.path)
    session = module.create_session()
    try:
        actual = session.run_numpy({"boxes": boxes, "scores": scores})["y"]
    finally:
        session.close()
        module.close()
    expected = _torch_nms_boxes_oracle(boxes, scores, pre_nms_top=4, max_output=3, iou_threshold=0.5, min_box_size=0.5)
    np.testing.assert_allclose(actual, expected, atol=1e-6, rtol=1e-6)


def test_rocm_batched_nms_parity(tmp_path):
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("Torch HIP/CUDA device is not available")
    if not _rocm_module_compile_toolchain_available():
        pytest.skip("ROCm module compile toolchain not found from active Python/PATH")

    class Tiny(dml.nn.Module):
        def forward(self, boxes):
            return dml.ops.output(dml.ops.batched_nms(boxes, iou_threshold=0.5, keep_n=3), "y")

    spec = dml.trace(Tiny(), inputs={"boxes": dml.TensorSpec([5, 4], "float32")}, name="rocm_batched_nms_parity")
    boxes = _batched_nms_boxes()
    artifact = dml.compile(spec, dml.Target("rocm"), tmp_path / "batched_nms_rocm.dinoml")
    module = load(artifact.path)
    session = module.create_session()
    try:
        actual = session.run_numpy({"boxes": boxes})["y"]
    finally:
        session.close()
        module.close()
    expected = _torch_batched_keep_oracle(boxes, iou_threshold=0.5, keep_n=3)
    np.testing.assert_array_equal(actual, expected)


def test_rocm_efficient_nms_parity(tmp_path):
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("Torch HIP/CUDA device is not available")
    if not _rocm_module_compile_toolchain_available():
        pytest.skip("ROCm module compile toolchain not found from active Python/PATH")

    class Tiny(dml.nn.Module):
        def forward(self, boxes, scores):
            num_det, det_boxes, det_scores, det_classes = dml.ops.efficient_nms(
                boxes,
                scores,
                pre_nms_top=5,
                max_output=3,
                iou_threshold=0.5,
                min_box_size=0.5,
            )
            return {
                "num_detections": dml.ops.output(num_det, "num_detections"),
                "detection_boxes": dml.ops.output(det_boxes, "detection_boxes"),
                "detection_scores": dml.ops.output(det_scores, "detection_scores"),
                "detection_classes": dml.ops.output(det_classes, "detection_classes"),
            }

    spec = dml.trace(
        Tiny(),
        inputs={"boxes": dml.TensorSpec([2, 4, 3, 4], "float32"), "scores": dml.TensorSpec([2, 4, 3], "float32")},
        name="rocm_efficient_nms_parity",
    )
    boxes, scores = _efficient_nms_inputs()
    artifact = dml.compile(spec, dml.Target("rocm"), tmp_path / "efficient_nms_rocm.dinoml")
    module = load(artifact.path)
    session = module.create_session()
    try:
        actual = session.run_numpy({"boxes": boxes, "scores": scores})
    finally:
        session.close()
        module.close()
    expected = _torch_efficient_nms_oracle(
        boxes,
        scores,
        pre_nms_top=5,
        max_output=3,
        iou_threshold=0.5,
        min_box_size=0.5,
    )
    np.testing.assert_array_equal(actual["num_detections"], expected[0])
    np.testing.assert_allclose(actual["detection_boxes"], expected[1], atol=1e-6, rtol=1e-6)
    np.testing.assert_allclose(actual["detection_scores"], expected[2], atol=1e-6, rtol=1e-6)
    np.testing.assert_array_equal(actual["detection_classes"], expected[3])
