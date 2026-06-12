from __future__ import annotations

import pytest

import dinoml as dml
from dinoml.compiler import _validate_mvp_runtime_contract
from dinoml.lowering.ops import render_generated_kernels, render_launch


def _spec():
    class Tiny(dml.nn.Module):
        def forward(self, nms_boxes, nms_scores, sorted_boxes, eff_boxes, eff_scores):
            num_det, det_boxes, det_scores, det_classes = dml.ops.efficient_nms(
                eff_boxes,
                eff_scores,
                pre_nms_top=5,
                max_output=3,
            )
            return {
                "nms": dml.ops.output(dml.ops.nms(nms_boxes, nms_scores, pre_nms_top=4, max_output=3), "nms"),
                "keep": dml.ops.output(dml.ops.batched_nms(sorted_boxes, keep_n=3), "keep"),
                "num_det": dml.ops.output(num_det, "num_det"),
                "det_boxes": dml.ops.output(det_boxes, "det_boxes"),
                "det_scores": dml.ops.output(det_scores, "det_scores"),
                "det_classes": dml.ops.output(det_classes, "det_classes"),
            }

    return dml.trace(
        Tiny(),
        inputs={
            "nms_boxes": dml.TensorSpec([2, 5, 4], "float32"),
            "nms_scores": dml.TensorSpec([2, 5], "float32"),
            "sorted_boxes": dml.TensorSpec([5, 4], "float32"),
            "eff_boxes": dml.TensorSpec([2, 4, 3, 4], "float32"),
            "eff_scores": dml.TensorSpec([2, 4, 3], "float32"),
        },
        name="nms_family_lowering",
    )


def test_nms_family_cpu_generated_sources_and_launches_render():
    spec = _spec()
    tensor_map = {tensor["name"]: tensor for tensor in spec.ir["tensors"]}
    sources = render_generated_kernels("cpu", spec.ir["nodes"], tensor_map)
    launches = [render_launch("cpu", node, tensor_map) for node in spec.ir["nodes"]]

    assert len(sources) == 3
    assert any("nms_" in source for source in sources)
    assert any("batched_nms_" in source for source in sources)
    assert any("efficient_nms_" in source for source in sources)
    assert any("runtime_numel_" in launch for launch in launches)


@pytest.mark.parametrize("target_name", ["cuda", "rocm"])
def test_nms_family_gpu_contracts_render_generated_sources(target_name):
    spec = _spec()
    tensor_map = {tensor["name"]: tensor for tensor in spec.ir["tensors"]}
    _validate_mvp_runtime_contract(spec.ir, dml.Target(target_name))
    sources = render_generated_kernels(target_name, spec.ir["nodes"], tensor_map)
    launches = [render_launch(target_name, node, tensor_map) for node in spec.ir["nodes"]]

    assert len(sources) == 3
    assert any("stream" in source for source in sources)
    assert any("__global__ void" in source for source in sources)
    assert all("session->stream" in launch for launch in launches)
