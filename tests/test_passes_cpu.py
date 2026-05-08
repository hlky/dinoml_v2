import numpy as np
import pytest

from dinoml import Target, compile
from dinoml.backends.cpu import execute_cpu
from dinoml.ir import IR_SCHEMA_VERSION, ModelSpec
from dinoml.kernels.codegen import create_codegen_plan
from dinoml.kernels.manifest import build_external_kernel_plan, build_kernel_manifest
from dinoml.lowering.ops import collect_generated_sources, render_generated_kernels, render_launch
from dinoml.lowering.cuda import render_cuda_module
from dinoml.lowering.ops.fused_elementwise import _broadcast_function_name, _function_name
from dinoml.lowering.shape_buffers import dynamic_dim_sources, numel_expr, shape_buffer_context, shape_dim_expr
from dinoml.ops.definitions import OP_REGISTRY, get_op_def
from dinoml.passes import PassManager, validate_ir
from dinoml.passes.validation import ValidationError


def test_pass_manager_runs_expected_pipeline():
    from tests.models.fused_elementwise import build_spec

    spec = build_spec()
    lowered, reports = PassManager().run(spec.ir)
    validate_ir(lowered)
    assert [report.name for report in reports] == list(PassManager.DEFAULT_PIPELINE)
    assert "memory_plan" in lowered["metadata"]
    assert "fusion_groups" in lowered["metadata"]
    assert any(node["op"] == "fused_elementwise" for node in lowered["nodes"])


def test_memory_plan_records_shape_only_view_alias_metadata():
    ir = _shape_view_ir()

    lowered, _ = PassManager(pipeline=("memory_plan",)).run(ir)
    validate_ir(lowered)

    views = lowered["metadata"]["memory_plan"]["views"]["views"]
    assert views == [
        {
            "tensor": "y",
            "source": "x",
            "kind": "shape_view",
            "transform": "reshape",
            "offset_elements": 0,
            "shape": [3, 2],
            "shape_spec": [3, 2],
        }
    ]
    assert lowered["metadata"]["memory_plan"]["temporaries"] == []


def test_view_alias_validation_requires_shape_only_element_count():
    ir = _shape_view_ir(output_shape=[4, 2])

    with pytest.raises(ValidationError, match="preserve source element count"):
        PassManager(pipeline=("memory_plan",)).run(ir)


def test_compile_rejects_view_of_view_aliases(tmp_path):
    ir = _shape_view_ir()
    ir["outputs"] = [{"name": "z", "tensor": "z", "shape": [6], "shape_spec": [6], "dtype": "float32"}]
    ir["tensors"].append({"name": "z", "shape": [6], "shape_spec": [6], "dtype": "float32", "kind": "output", "nbytes": 24})
    ir["metadata"]["views"]["views"].append(
        {
            "tensor": "z",
            "source": "y",
            "kind": "shape_view",
            "transform": "flatten",
            "shape": [6],
            "shape_spec": [6],
        }
    )
    spec = ModelSpec("shape_view_of_view_alias", ir, constants={})

    with pytest.raises(NotImplementedError, match="View-of-view"):
        compile(spec, Target("cpu"), tmp_path / "shape_view_of_view_alias.dinoml")


def test_cuda_lowering_binds_and_materializes_shape_view_output_alias():
    lowered, _ = PassManager().run(_shape_view_ir())

    generated = render_cuda_module(lowered, generated_kernels=[])

    assert "const float* ptr_y = ptr_x;" in generated
    assert "cudaMemcpyAsync(outputs[0].data, ptr_y, runtime_numel_y * sizeof(float), cudaMemcpyDeviceToDevice, session->stream)" in generated


def test_cpu_reference_matches_numpy_formula():
    from tests.models.fused_elementwise import build_spec, build_validation_inputs, numpy_reference

    spec = build_spec()
    inputs = build_validation_inputs()
    actual = execute_cpu(spec, inputs)["y"]
    expected = numpy_reference(inputs)["y"]

    np.testing.assert_allclose(actual, expected, atol=1e-5, rtol=1e-5)


def _shape_view_ir(output_shape=None):
    output_shape = list(output_shape or [3, 2])
    output_nbytes = int(np.prod(output_shape, dtype=np.int64) * 4)
    return {
        "schema_version": IR_SCHEMA_VERSION,
        "name": "shape_view_alias",
        "inputs": [{"name": "x", "tensor": "x", "shape": [2, 3], "shape_spec": [2, 3], "dtype": "float32"}],
        "constants": [],
        "outputs": [{"name": "y", "tensor": "y", "shape": output_shape, "shape_spec": output_shape, "dtype": "float32"}],
        "nodes": [],
        "tensors": [
            {"name": "x", "shape": [2, 3], "shape_spec": [2, 3], "dtype": "float32", "kind": "input", "nbytes": 24},
            {
                "name": "y",
                "shape": output_shape,
                "shape_spec": output_shape,
                "dtype": "float32",
                "kind": "output",
                "nbytes": output_nbytes,
            },
        ],
        "metadata": {
            "views": {
                "version": 1,
                "views": [
                    {
                        "tensor": "y",
                        "source": "x",
                        "kind": "shape_view",
                        "transform": "reshape",
                        "shape": output_shape,
                        "shape_spec": output_shape,
                    }
                ],
            }
        },
    }


def test_elementwise_op_definitions_are_frontend_only_until_fused():
    add = get_op_def("add")
    fused = get_op_def("fused_elementwise")
    assert OP_REGISTRY.get_frontend("add") is add
    assert add.schema.inputs == ("x0", "x1")
    assert not add.backend_kernels
    assert fused.backend_kernels["cuda"].symbol == "generated_fused_elementwise"
    assert fused.backend_kernels["cpu"].symbol == "generated_fused_elementwise"


def test_kernel_manifest_lists_required_unique_kernels():
    from tests.models.fused_elementwise import build_spec

    spec = build_spec()
    lowered, _ = PassManager().run(spec.ir)
    manifest = build_kernel_manifest(lowered, {"name": "cpu", "arch": "native"})
    symbols = {item["kernel_symbol"] for item in manifest["required_kernels"]}
    assert symbols == {"generated_fused_elementwise"}
    model_generated = [item for item in manifest["required_kernels"] if item["kernel_library"] == "model"]
    assert model_generated and model_generated[0]["op"] == "fused_elementwise"
    assert manifest["support_cache_key"] != manifest["cache_key"]
    assert manifest["profile_cache_schema_version"] == 1
    plan = create_codegen_plan(manifest, "/tmp/dinoml-test-cache")
    assert plan.profiler_symbols == ()
    assert plan.support_cache_dir.name == manifest["support_cache_key"][:16]


def test_external_cuda_kernel_plan_lists_cutlass_gemm_families():
    plan = build_external_kernel_plan({"name": "cuda", "arch": "sm_86"})
    families = {family["op_name"]: family for family in plan["families"]}
    assert sorted(families) == ["gemm_rcr", "gemm_rrr"]
    assert families["gemm_rcr"]["provider"] == "cutlass"
    assert families["gemm_rcr"]["required_libraries"] == ["cutlass", "cublaslt"]
    assert families["gemm_rrr"]["attrs"]["b_layout"] == "row"
    assert plan["profiler_strategy"] == "generate_used_candidates_once_then_cache_results"
    assert len(plan["cache_key"]) == 64


def test_softmax_manifest_and_generated_sources_are_model_owned():
    import dinoml as dml

    class SoftmaxModel(dml.Module):
        def forward(self, x):
            return dml.ops.output(dml.ops.softmax(x), "y")

    spec = dml.trace(
        SoftmaxModel(),
        inputs={"x": dml.TensorSpec([256, 1024], "float32")},
        name="softmax_manifest",
    )
    lowered, _ = PassManager().run(spec.ir)
    manifest = build_kernel_manifest(lowered, {"name": "cpu", "arch": "native"})
    assert manifest["required_kernels"] == [
        {
            "op": "softmax",
            "kernel_symbol": "generated_softmax",
            "kernel_library": "model",
            "profiler_symbol": None,
            "has_profiler": False,
        }
    ]

    tensor_map = {tensor["name"]: tensor for tensor in lowered["tensors"]}
    sources = collect_generated_sources("cuda", lowered["nodes"], tensor_map)
    assert len(sources["kernels"]) == 1
    assert "expf" in sources["kernels"][0]
    assert "_packed_kernel" in sources["kernels"][0]
    assert "float4" in sources["kernels"][0]
    assert "<<<grid, block, 0, stream>>>" in sources["kernels"][0]


def test_softmax_cuda_source_policy_selects_warp_and_shared_paths():
    import dinoml as dml

    class SoftmaxModel(dml.Module):
        def forward(self, x):
            return dml.ops.output(dml.ops.softmax(x), "y")

    warp_spec = dml.trace(
        SoftmaxModel(),
        inputs={"x": dml.TensorSpec([8192, 77], "float32")},
        name="softmax_warp_policy",
    )
    warp_lowered, _ = PassManager().run(warp_spec.ir)
    warp_tensor_map = {tensor["name"]: tensor for tensor in warp_lowered["tensors"]}
    warp_sources = collect_generated_sources("cuda", warp_lowered["nodes"], warp_tensor_map)
    assert "_warp_kernel" in warp_sources["kernels"][0]
    assert "_packed_kernel" not in warp_sources["kernels"][0]

    shared_spec = dml.trace(
        SoftmaxModel(),
        inputs={"x": dml.TensorSpec([256, 4096], "float32")},
        name="softmax_shared_policy",
    )
    shared_lowered, _ = PassManager().run(shared_spec.ir)
    shared_tensor_map = {tensor["name"]: tensor for tensor in shared_lowered["tensors"]}
    shared_sources = collect_generated_sources("cuda", shared_lowered["nodes"], shared_tensor_map)
    assert "_warp_kernel" not in shared_sources["kernels"][0]
    assert "_packed_kernel" not in shared_sources["kernels"][0]
    assert "block * sizeof(float)" in shared_sources["kernels"][0]


def test_reduction_manifest_and_keepdim_shape_inference():
    import dinoml as dml

    class ReduceModel(dml.Module):
        def forward(self, x):
            return dml.ops.output(dml.ops.reduce_mean(x, keepdim=True), "y")

    spec = dml.trace(
        ReduceModel(),
        inputs={"x": dml.TensorSpec([2, 3, 4], "float32")},
        name="reduce_mean_manifest",
    )
    lowered, _ = PassManager().run(spec.ir)
    tensor_map = {tensor["name"]: tensor for tensor in lowered["tensors"]}
    output_tensor = tensor_map[lowered["outputs"][0]["tensor"]]
    assert output_tensor["shape"] == [2, 3, 1]
    assert output_tensor["shape_spec"] == [2, 3, 1]

    manifest = build_kernel_manifest(lowered, {"name": "cpu", "arch": "native"})
    assert manifest["required_kernels"] == [
        {
            "op": "reduce_mean",
            "kernel_symbol": "generated_reduction",
            "kernel_library": "model",
            "profiler_symbol": None,
            "has_profiler": False,
        }
    ]

    sources = collect_generated_sources("cuda", lowered["nodes"], tensor_map)
    assert len(sources["kernels"]) == 1
    assert "reduce_mean_" in sources["kernels"][0]
    assert "acc / 4.00000000f" in sources["kernels"][0]


def test_shape_type_infer_propagates_dynamic_shape_spec_through_elementwise_broadcast():
    batch = {"kind": "dim", "name": "batch", "min": 1, "max": 4}
    ir = {
        "schema_version": IR_SCHEMA_VERSION,
        "name": "dynamic_add_shape_spec",
        "inputs": [
            {
                "name": "x",
                "tensor": "x",
                "shape": [4, 16],
                "shape_spec": [batch, 16],
                "dtype": "float32",
            },
            {
                "name": "bias",
                "tensor": "bias",
                "shape": [1, 16],
                "shape_spec": [1, 16],
                "dtype": "float32",
            },
        ],
        "constants": [],
        "outputs": [{"name": "y", "tensor": "y", "shape": [4, 16], "shape_spec": [4, 16], "dtype": "float32"}],
        "nodes": [{"id": "n0", "op": "add", "inputs": ["x", "bias"], "outputs": ["y"], "attrs": {}}],
        "tensors": [
            {
                "name": "x",
                "shape": [4, 16],
                "shape_spec": [batch, 16],
                "dtype": "float32",
                "kind": "input",
                "nbytes": 256,
            },
            {
                "name": "bias",
                "shape": [1, 16],
                "shape_spec": [1, 16],
                "dtype": "float32",
                "kind": "input",
                "nbytes": 64,
            },
            {"name": "y", "shape": [4, 16], "shape_spec": [4, 16], "dtype": "float32", "kind": "output", "nbytes": 256},
        ],
        "metadata": {},
    }

    lowered, _ = PassManager(pipeline=("shape_type_infer",)).run(ir)

    tensors = {tensor["name"]: tensor for tensor in lowered["tensors"]}
    assert tensors["y"]["shape_spec"] == [batch, 16]
    assert lowered["outputs"][0]["shape_spec"] == [batch, 16]


def test_shape_type_infer_propagates_dynamic_shape_spec_through_reductions():
    batch = {"kind": "dim", "name": "batch", "min": 1, "max": 4}
    rows = {"kind": "dim", "name": "rows", "min": 2, "max": 8}
    ir = {
        "schema_version": IR_SCHEMA_VERSION,
        "name": "dynamic_reduce_shape_spec",
        "inputs": [
            {
                "name": "x",
                "tensor": "x",
                "shape": [4, 8, 16],
                "shape_spec": [batch, rows, 16],
                "dtype": "float32",
            }
        ],
        "constants": [],
        "outputs": [
            {"name": "sum", "tensor": "sum", "shape": [4, 8], "shape_spec": [4, 8], "dtype": "float32"},
            {"name": "mean", "tensor": "mean", "shape": [4, 8, 1], "shape_spec": [4, 8, 1], "dtype": "float32"},
        ],
        "nodes": [
            {
                "id": "n0",
                "op": "reduce_sum",
                "inputs": ["x"],
                "outputs": ["sum"],
                "attrs": {"dim": -1, "keepdim": False},
            },
            {
                "id": "n1",
                "op": "reduce_mean",
                "inputs": ["x"],
                "outputs": ["mean"],
                "attrs": {"dim": -1, "keepdim": True},
            },
        ],
        "tensors": [
            {
                "name": "x",
                "shape": [4, 8, 16],
                "shape_spec": [batch, rows, 16],
                "dtype": "float32",
                "kind": "input",
                "nbytes": 2048,
            },
            {
                "name": "sum",
                "shape": [4, 8],
                "shape_spec": [4, 8],
                "dtype": "float32",
                "kind": "output",
                "nbytes": 128,
            },
            {
                "name": "mean",
                "shape": [4, 8, 1],
                "shape_spec": [4, 8, 1],
                "dtype": "float32",
                "kind": "output",
                "nbytes": 128,
            },
        ],
        "metadata": {},
    }

    lowered, _ = PassManager(pipeline=("shape_type_infer",)).run(ir)

    tensors = {tensor["name"]: tensor for tensor in lowered["tensors"]}
    assert tensors["sum"]["shape_spec"] == [batch, rows]
    assert tensors["mean"]["shape_spec"] == [batch, rows, 1]
    assert {output["name"]: output["shape_spec"] for output in lowered["outputs"]} == {
        "sum": [batch, rows],
        "mean": [batch, rows, 1],
    }


def test_shape_buffer_helpers_materialize_dynamic_runtime_dims():
    tensor_map = {
        "x": {
            "name": "x",
            "shape": [4, 16],
            "shape_spec": [{"kind": "dim", "name": "batch", "min": 1, "max": 4}, 16],
        },
        "tmp": {
            "name": "tmp",
            "shape": [4, 16],
            "shape_spec": [{"kind": "dim", "name": "batch", "min": 1, "max": 4}, 16],
        },
    }
    sources = dynamic_dim_sources(input_map={"x": 0}, output_map={}, tensor_map=tensor_map)
    assert sources == {"batch": "inputs[0].shape[0]"}
    assert shape_dim_expr(tensor_map["tmp"], 0, sources) == "inputs[0].shape[0]"
    assert shape_dim_expr(tensor_map["tmp"], 1, sources) == "16"
    assert numel_expr("tmp", 2) == "shape_tmp_0 * shape_tmp_1"
    assert shape_buffer_context(tensor_map["tmp"]) == {"ident": "tmp", "rank": 2, "shape_literal": "4, 16"}


def test_fused_elementwise_function_names_are_stable_and_clean():
    node = {
        "id": "n0_n1_n2_fused",
        "op": "fused_elementwise",
        "inputs": ["x", "scale"],
        "outputs": ["y"],
        "attrs": {
            "sub_ops": [
                {"op": "mul", "inputs": ["x", "scale"], "outputs": ["t0"], "attrs": {}},
                {"op": "relu", "inputs": ["t0"], "outputs": ["y"], "attrs": {}},
            ]
        },
    }
    renamed_node = {**node, "id": "different_graph_node_id"}
    changed_node = {
        **node,
        "attrs": {
            "sub_ops": [
                {"op": "add", "inputs": ["x", "scale"], "outputs": ["t0"], "attrs": {}},
                {"op": "relu", "inputs": ["t0"], "outputs": ["y"], "attrs": {}},
            ]
        },
    }

    name = _function_name(node)

    assert name.startswith("fused_elementwise_")
    assert "dino_fused" not in name
    assert "n0_n1_n2" not in name
    assert _function_name(renamed_node) == name
    assert _function_name(changed_node) != name
    assert _broadcast_function_name(node, "x") == f"{name}_idx_x"


def test_render_generated_kernels_deduplicates_exact_fused_sources(tmp_path):
    node = {
        "id": "n0_n1_fused",
        "op": "fused_elementwise",
        "inputs": ["x", "scale"],
        "outputs": ["y"],
        "attrs": {
            "sub_ops": [
                {"op": "mul", "inputs": ["x", "scale"], "outputs": ["t0"], "attrs": {}},
                {"op": "relu", "inputs": ["t0"], "outputs": ["y"], "attrs": {}},
            ]
        },
    }
    tensor_map = {
        "x": {"name": "x", "shape": [4, 16], "dtype": "float32"},
        "scale": {"name": "scale", "shape": [16], "dtype": "float32"},
        "y": {"name": "y", "shape": [4, 16], "dtype": "float32", "kind": "output"},
    }
    nodes = [node, {**node, "id": "duplicate_fused"}]

    kernels = render_generated_kernels("cpu", nodes, tensor_map)
    generated_sources = collect_generated_sources("cpu", nodes, tensor_map, generated_src_dir=tmp_path)
    launches = [render_launch("cpu", item, tensor_map) for item in nodes]
    manifest = generated_sources["manifest"]
    manifest_sources = manifest["sources"]

    assert len(kernels) == 1
    assert kernels[0].count(f"int {_function_name(node)}(") == 1
    assert generated_sources["kernels"] == kernels
    assert (tmp_path / "source_manifest.json").exists()
    assert len(manifest_sources) == 2
    assert manifest_sources[0]["node_id"] == "n0_n1_fused"
    assert manifest_sources[0]["op"] == "fused_elementwise"
    assert manifest_sources[0]["target"] == "cpu"
    assert manifest_sources[0]["generated_function_name"] == _function_name(node)
    assert manifest_sources[0]["emitted_new_source"] is True
    assert manifest_sources[1]["node_id"] == "duplicate_fused"
    assert manifest_sources[1]["emitted_new_source"] is False
    assert manifest_sources[1]["emitted_source_path"] == manifest_sources[0]["emitted_source_path"]
    assert (tmp_path / manifest_sources[0]["emitted_source_path"]).exists()
    assert len(launches) == 2
