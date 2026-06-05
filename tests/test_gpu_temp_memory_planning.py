from __future__ import annotations

import dinoml as dml
import pytest
from dinoml.ir import IR_SCHEMA_VERSION, VIEW_METADATA_VERSION
from dinoml.layout import dense_layout
from dinoml.lowering.gpu import render_gpu_module
from dinoml.passes import PassManager
from dinoml.passes.core import memory_plan
from dinoml.shapes import Dim


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


class ThreeLayerNormModule(dml.Module):
    def forward(self, x, weight, bias):
        h0 = dml.ops.layer_norm(x, weight, bias)
        h1 = dml.ops.layer_norm(h0, weight, bias)
        return dml.ops.output(dml.ops.layer_norm(h1, weight, bias), "output")


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
    assert f"{malloc_name}(&next_temp_arena, target_nbytes)" in source
    assert "dino_session_prepare_temp_arena(session, 0, 0)" in source
    assert "temp_arena_base + " in source
    assert "session->tmp_" not in source


def test_memory_plan_emits_bucket_specific_temp_plans():
    batch = Dim("batch", min=2, max=8, buckets=(2, 4, 8))
    spec = dml.trace(
        TwoLayerNormModule(),
        inputs={
            "x": dml.TensorSpec([batch, 4], "float32"),
            "weight": dml.TensorSpec([4], "float32"),
            "bias": dml.TensorSpec([4], "float32"),
        },
        name="bucketed_temp_plan_metadata",
    )
    lowered, _ = PassManager().run(spec.ir)

    planned = lowered["metadata"]["memory_plan"]
    bucket_arenas = [int(plan["arena_nbytes"]) for plan in planned["bucket_plans"]]

    assert planned["planning_mode"] == "bucketed"
    assert planned["bucket_dimensions"][0]["name"] == "batch"
    assert planned["bucket_dimensions"][0]["min"] == 2
    assert planned["bucket_dimensions"][0]["max"] == 8
    assert planned["bucket_dimensions"][0]["buckets"] == [2, 4, 8]
    assert planned["bucket_dimensions"][0]["sources"][0] == {
        "section": "inputs",
        "name": "x",
        "tensor": "x",
        "axis": 0,
    }
    assert any(source["section"] == "outputs" and source["name"] == "output" for source in planned["bucket_dimensions"][0]["sources"])
    assert len(planned["bucket_plans"]) == 3
    assert min(bucket_arenas) < max(bucket_arenas)
    assert planned["max_bucket_arena_nbytes"] == max(bucket_arenas)
    assert planned["min_bucket_arena_nbytes"] == min(bucket_arenas)
    assert planned["arena_nbytes"] >= planned["max_bucket_arena_nbytes"]


@pytest.mark.parametrize(
    ("target", "malloc_name"),
    [("rocm", "hipMalloc"), ("cuda", "cudaMalloc")],
)
def test_render_gpu_module_switches_bucket_specific_temp_plans(target: str, malloc_name: str):
    batch = Dim("batch", min=2, max=8, buckets=(2, 4, 8))
    spec = dml.trace(
        ThreeLayerNormModule(),
        inputs={
            "x": dml.TensorSpec([batch, 4], "float32"),
            "weight": dml.TensorSpec([4], "float32"),
            "bias": dml.TensorSpec([4], "float32"),
        },
        name="bucketed_temp_plan_codegen",
    )
    lowered, _ = PassManager().run(spec.ir)

    source = render_gpu_module(target, lowered)

    assert "kDinoTempPlanCount = 3" in source
    assert "inputs[0].shape[0] <= 2" in source
    assert "inputs[0].shape[0] > 2 && inputs[0].shape[0] <= 4" in source
    assert "inputs[0].shape[0] > 4 && inputs[0].shape[0] <= 8" in source
    assert "dino_session_set_temp_arena_policy" in source
    assert "dino_session_release_temp_arena" in source
    assert f"{malloc_name}(&next_temp_arena, target_nbytes)" in source
    assert "if (required_nbytes == 0)" in source
    assert "kDinoTempPlanOffsets[selected_temp_plan_index][0]" in source
    assert "kDinoTempPlanOffsets[selected_temp_plan_index][1]" in source
    assert "session->tmp_" not in source
