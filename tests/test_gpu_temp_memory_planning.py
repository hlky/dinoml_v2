from __future__ import annotations

import dinoml as dml
import pytest
from dinoml.ir import IR_SCHEMA_VERSION, VIEW_METADATA_VERSION
from dinoml.layout import dense_layout
from dinoml.lowering.gpu import render_gpu_module
from dinoml.passes import PassManager
from dinoml.passes.core import memory_plan


def _tensor(name: str, shape: list[int], dtype: str, kind: str = "temporary") -> dict[str, object]:
    nbytes = 4
    for dim in shape:
        nbytes *= int(dim)
    return {
        "name": name,
        "shape": list(shape),
        "shape_spec": list(shape),
        "layout": dense_layout(shape),
        "dtype": dtype,
        "kind": kind,
        "nbytes": nbytes,
    }


def test_memory_plan_reuses_non_overlapping_temporary_offsets():
    ir = {
        "schema_version": IR_SCHEMA_VERSION,
        "name": "reuse_non_overlapping_temporaries",
        "inputs": [
            {"name": "x", "tensor": "x", "shape": [4], "shape_spec": [4], "layout": dense_layout([4]), "dtype": "float32"},
            {"name": "y", "tensor": "y", "shape": [4], "shape_spec": [4], "layout": dense_layout([4]), "dtype": "float32"},
            {"name": "z", "tensor": "z", "shape": [4], "shape_spec": [4], "layout": dense_layout([4]), "dtype": "float32"},
        ],
        "states": [],
        "constants": [],
        "outputs": [
            {"name": "output", "tensor": "out", "shape": [4], "shape_spec": [4], "layout": dense_layout([4]), "dtype": "float32"}
        ],
        "nodes": [
            {"id": "n0", "op": "add", "inputs": ["x", "y"], "outputs": ["t0"], "attrs": {}},
            {"id": "n1", "op": "relu", "inputs": ["t0"], "outputs": ["t1"], "attrs": {}},
            {"id": "n2", "op": "add", "inputs": ["z", "y"], "outputs": ["t2"], "attrs": {}},
            {"id": "n3", "op": "relu", "inputs": ["t2"], "outputs": ["out"], "attrs": {}},
        ],
        "tensors": [
            _tensor("x", [4], "float32", kind="input"),
            _tensor("y", [4], "float32", kind="input"),
            _tensor("z", [4], "float32", kind="input"),
            _tensor("t0", [4], "float32"),
            _tensor("t1", [4], "float32"),
            _tensor("t2", [4], "float32"),
            _tensor("out", [4], "float32", kind="output"),
        ],
        "metadata": {},
    }

    planned = memory_plan(ir)["metadata"]["memory_plan"]
    temporaries = {str(item["tensor"]): item for item in planned["temporaries"]}
    total_aligned_nbytes = sum(int(item["aligned_nbytes"]) for item in planned["temporaries"])

    assert planned["allocation"] == "lifetime_planned_temporaries"
    assert planned["arena_nbytes"] < total_aligned_nbytes
    assert temporaries["t0"]["offset"] == temporaries["t2"]["offset"]


def test_memory_plan_keeps_output_view_source_live_until_materialization():
    ir = {
        "schema_version": IR_SCHEMA_VERSION,
        "name": "output_view_extends_source_lifetime",
        "inputs": [
            {"name": "x", "tensor": "x", "shape": [4], "shape_spec": [4], "layout": dense_layout([4]), "dtype": "float32"},
            {"name": "z", "tensor": "z", "shape": [4], "shape_spec": [4], "layout": dense_layout([4]), "dtype": "float32"},
        ],
        "states": [],
        "constants": [],
        "outputs": [
            {
                "name": "output",
                "tensor": "v0",
                "shape": [4],
                "shape_spec": [4],
                "layout": dense_layout([4]),
                "dtype": "float32",
            }
        ],
        "nodes": [
            {"id": "n0", "op": "relu", "inputs": ["x"], "outputs": ["t0"], "attrs": {}},
            {"id": "n1", "op": "relu", "inputs": ["z"], "outputs": ["t1"], "attrs": {}},
        ],
        "tensors": [
            _tensor("x", [4], "float32", kind="input"),
            _tensor("z", [4], "float32", kind="input"),
            _tensor("t0", [4], "float32"),
            _tensor("t1", [4], "float32"),
            _tensor("v0", [4], "float32", kind="output"),
        ],
        "metadata": {
            "views": {
                "version": VIEW_METADATA_VERSION,
                "views": [
                    {
                        "tensor": "v0",
                        "source": "t0",
                        "kind": "shape_view",
                        "transform": "identity",
                        "offset_elements": 0,
                        "shape": [4],
                        "shape_spec": [4],
                    }
                ],
            }
        },
    }

    planned = memory_plan(ir)["metadata"]["memory_plan"]
    temporaries = {str(item["tensor"]): item for item in planned["temporaries"]}

    assert temporaries["t0"]["last_use_node"] == 1
    assert temporaries["t0"]["offset"] != temporaries["t1"]["offset"]


class TwoLayerNormModule(dml.Module):
    def forward(self, x, weight, bias):
        hidden = dml.ops.layer_norm(x, weight, bias)
        return dml.ops.output(dml.ops.layer_norm(hidden, weight, bias), "output")


@pytest.mark.parametrize(
    ("target", "malloc_name"),
    [("rocm", "hipMalloc"), ("cuda", "cudaMalloc")],
)
def test_render_gpu_module_uses_shared_temp_arena(target: str, malloc_name: str):
    spec = dml.trace(
        TwoLayerNormModule(),
        inputs={
            "x": dml.TensorSpec([2, 4], "float32"),
            "weight": dml.TensorSpec([4], "float32"),
            "bias": dml.TensorSpec([4], "float32"),
        },
        name="shared_temp_arena_codegen",
    )
    lowered, _ = PassManager().run(spec.ir)

    source = render_gpu_module(target, lowered)

    assert "void* temp_arena = nullptr;" in source
    assert f"DINO_SESSION_CREATE_GPU_CHECK({malloc_name}(&session->temp_arena," in source
    assert "temp_arena_base + " in source
    assert "session->tmp_" not in source
