from __future__ import annotations

import dinoml as dml
from dinoml.lowering.gpu import render_gpu_module
from dinoml.passes import PassManager
from dinoml.shapes import Dim


class DenseElementwiseModule(dml.Module):
    def forward(self, x):
        return dml.ops.output(dml.ops.gelu_new(x), "output")


class GenericBroadcastModule(dml.Module):
    def forward(self, x, y):
        return dml.ops.output(x + y, "output")


def test_gpu_static_dense_elementwise_does_not_allocate_shape_buffers():
    spec = dml.trace(
        DenseElementwiseModule(),
        inputs={"x": dml.TensorSpec([2, 3], "float32")},
        name="dense_elementwise_shape_buffer_elision",
    )
    lowered, _ = PassManager().run(spec.ir)

    source = render_gpu_module("rocm", lowered)

    assert "int64_t* shape_" not in source
    assert "session->shape_" not in source
    assert "const int64_t shape_x_0 = inputs[0].shape[0];" in source


def test_gpu_generic_fused_elementwise_broadcast_keeps_required_shape_buffers():
    spec = dml.trace(
        GenericBroadcastModule(),
        inputs={
            "x": dml.TensorSpec([2, 1, 3], "float32"),
            "y": dml.TensorSpec([1, 4, 1], "float32"),
        },
        name="generic_broadcast_shape_buffer_required",
    )
    lowered, _ = PassManager().run(spec.ir)

    source = render_gpu_module("rocm", lowered)

    assert "int64_t* shape_x = nullptr;" in source
    assert "int64_t* shape_y = nullptr;" in source
    assert "int64_t* shape_t0 = nullptr;" in source
    assert "DINO_SESSION_CREATE_GPU_CHECK(hipMemcpy(" in source
    assert "hipMemcpy(session->shape_x, inputs[0].shape" not in source
    assert "hipMemcpy(session->shape_y, inputs[1].shape" not in source
    assert "hipMemcpy(session->shape_t0, outputs[0].shape" not in source


def test_gpu_dynamic_shape_buffers_still_refresh_during_run():
    batch = Dim("batch", min=1, max=4)
    spec = dml.trace(
        GenericBroadcastModule(),
        inputs={
            "x": dml.TensorSpec([batch, 1, 3], "float32"),
            "y": dml.TensorSpec([1, 4, 1], "float32"),
        },
        name="dynamic_broadcast_shape_buffer_required",
    )
    lowered, _ = PassManager().run(spec.ir)

    source = render_gpu_module("rocm", lowered)

    assert "int64_t* shape_x = nullptr;" in source
    assert "int64_t* shape_y = nullptr;" in source
    assert "int64_t* shape_t0 = nullptr;" in source
    assert "hipMemcpy(session->shape_x, inputs[0].shape" in source
    assert "hipMemcpy(session->shape_y, inputs[1].shape" not in source
    assert "hipMemcpy(session->shape_t0, outputs[0].shape" in source
