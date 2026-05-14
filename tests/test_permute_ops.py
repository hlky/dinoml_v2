import shutil

import numpy as np
import pytest

import dinoml as dml
from dinoml.backends.cpu import execute_cpu
from dinoml.ir import array_from_storage, array_to_storage, read_json
from dinoml.kernels.manifest import build_kernel_manifest
from dinoml.lowering.ops import collect_generated_sources, render_generated_kernels
from dinoml.ops.collections import SPECIALIZED_PERMUTE_DIMS
from dinoml.ops.definitions import OP_REGISTRY
from dinoml.passes import PassManager, validate_ir
from dinoml.passes.validation import ValidationError
from dinoml.runtime import load
from dinoml.shapes import Dim


SPECIALIZED_PERMUTE_CASES = [
    ("permute021", (2, 3, 4), (0, 2, 1), [2, 4, 3]),
    ("permute102", (2, 3, 4), (1, 0, 2), [3, 2, 4]),
    ("permute210", (2, 3, 4), (2, 1, 0), [4, 3, 2]),
    ("permute0213", (2, 3, 4, 5), (0, 2, 1, 3), [2, 4, 3, 5]),
]


class PermuteModule(dml.Module):
    def __init__(self, dims):
        self.dims = dims

    def forward(self, x):
        return dml.ops.output(dml.ops.permute(x, self.dims), "out")


class TransposeModule(dml.Module):
    def __init__(self, dim0, dim1):
        self.dim0 = dim0
        self.dim1 = dim1

    def forward(self, x):
        return dml.ops.output(dml.ops.transpose(x, self.dim0, self.dim1), "out")


class SpecializedPermuteModule(dml.Module):
    def __init__(self, op_name):
        self.op_name = op_name

    def forward(self, x):
        op = getattr(dml.ops, self.op_name)
        return dml.ops.output(op(x), "out")


class RegisteredSpecializedPermuteModule(dml.Module):
    def __init__(self, op_name):
        self.op_name = op_name

    def forward(self, x):
        return dml.ops.output(dml.ops.emit_registered_op(self.op_name, x), "out")


def _trace_permute(dtype="float32", dims=(2, 0, 1), shape=(2, 3, 4)):
    return dml.trace(PermuteModule(dims), inputs={"x": dml.TensorSpec(shape, dtype)}, name=f"permute_{dtype}")


def _trace_specialized_permute(op_name, dtype="float32", shape=(2, 3, 4)):
    return dml.trace(
        SpecializedPermuteModule(op_name),
        inputs={"x": dml.TensorSpec(shape, dtype)},
        name=f"{op_name}_{dtype}",
    )


def _trace_registered_specialized_permute(op_name, dtype="float32", shape=(2, 3, 4)):
    return dml.trace(
        RegisteredSpecializedPermuteModule(op_name),
        inputs={"x": dml.TensorSpec(shape, dtype)},
        name=f"{op_name}_registered_{dtype}",
    )


def _trace_transpose(dtype="float32", dim0=-1, dim1=0, shape=(2, 3, 4)):
    return dml.trace(
        TransposeModule(dim0, dim1),
        inputs={"x": dml.TensorSpec(shape, dtype)},
        name=f"transpose_{dtype}",
    )


def _storage_roundtrip(value, dtype):
    if dtype in {"float16", "bfloat16"}:
        return array_from_storage(array_to_storage(value, dtype), dtype)
    return np.asarray(value, dtype=np.bool_ if dtype == "bool" else np.float32)


def _input(dtype):
    if dtype == "bool":
        return np.array(
            [[[True, False], [False, True], [True, True]], [[False, False], [True, False], [False, True]]],
            dtype=np.bool_,
        )
    return np.arange(12, dtype=np.float32).reshape(2, 3, 2)


def test_permute_frontend_ir_normalizes_negative_dims_and_shape():
    spec = _trace_permute("float32", dims=(-1, 0, 1))

    assert spec.ir["outputs"][0]["shape"] == [4, 2, 3]
    assert spec.ir["outputs"][0]["shape_spec"] == [4, 2, 3]
    assert spec.ir["outputs"][0]["dtype"] == "float32"
    node = spec.ir["nodes"][0]
    assert node["op"] == "permute"
    assert node["inputs"] == ["x"]
    assert node["attrs"] == {"dims": [2, 0, 1]}


def test_transpose_frontend_emits_normalized_permute():
    spec = _trace_transpose("float32", dim0=-1, dim1=0)

    assert spec.ir["outputs"][0]["shape"] == [4, 3, 2]
    assert spec.ir["outputs"][0]["shape_spec"] == [4, 3, 2]
    node = spec.ir["nodes"][0]
    assert node["op"] == "permute"
    assert node["attrs"] == {"dims": [2, 1, 0]}


@pytest.mark.parametrize(("op_name", "shape", "dims", "out_shape"), SPECIALIZED_PERMUTE_CASES)
def test_specialized_permute_frontends_emit_specialized_ir(op_name, shape, dims, out_shape):
    spec = _trace_specialized_permute(op_name, shape=shape)

    assert op_name in OP_REGISTRY.frontend_names()
    assert spec.ir["outputs"][0]["shape"] == out_shape
    assert spec.ir["outputs"][0]["shape_spec"] == out_shape
    node = spec.ir["nodes"][0]
    assert node["op"] == op_name
    assert node["inputs"] == ["x"]
    assert node["attrs"] == {"dims": list(dims)}

    lowered, _ = PassManager().run(spec.ir)
    validate_ir(lowered)
    assert [lowered_node["op"] for lowered_node in lowered["nodes"]] == [op_name]


@pytest.mark.parametrize(("op_name", "shape", "dims", "out_shape"), SPECIALIZED_PERMUTE_CASES)
def test_specialized_permute_registry_defaults_match_schema(op_name, shape, dims, out_shape):
    op_def = OP_REGISTRY.get(op_name)

    [dims_attr] = op_def.schema.attrs
    assert dims_attr.name == "dims"
    assert dims_attr.type_name == "ints"
    assert tuple(dims_attr.default) == dims
    assert dims_attr.required is False
    assert op_def.frontend is not None
    assert op_def.frontend.default_attrs == {"dims": list(dims)}

    spec = _trace_registered_specialized_permute(op_name, shape=shape)
    assert spec.ir["outputs"][0]["shape"] == out_shape
    assert spec.ir["outputs"][0]["shape_spec"] == out_shape
    node = spec.ir["nodes"][0]
    assert node["op"] == op_name
    assert node["attrs"] == {"dims": list(dims)}


@pytest.mark.parametrize(("op_name", "shape", "dims", "_out_shape"), SPECIALIZED_PERMUTE_CASES)
def test_specialized_permute_manifest_and_generated_sources_are_model_owned(op_name, shape, dims, _out_shape):
    spec = _trace_specialized_permute(op_name, shape=shape)
    lowered, _ = PassManager().run(spec.ir)
    validate_ir(lowered)

    manifest = build_kernel_manifest(lowered, {"name": "cpu", "arch": "native"})
    [required] = manifest["required_kernels"]
    assert required["op"] == op_name
    assert required["kernel_symbol"] == f"generated_{op_name}"
    assert required["kernel_library"] == "model"
    assert required["profiler_symbol"] is None
    assert required["has_profiler"] is False
    assert required["generated_source"]["generated_function_name"].startswith(f"{op_name}_{''.join(map(str, dims))}_")
    assert required["generated_source"]["source_key"].startswith(f"cpu:{op_name}_{''.join(map(str, dims))}_")

    tensor_map = {tensor["name"]: tensor for tensor in lowered["tensors"]}
    sources = collect_generated_sources("cuda", lowered["nodes"], tensor_map)
    assert len(sources["kernels"]) == 1
    assert len(sources["manifest"]["sources"]) == 1
    assert sources["manifest"]["sources"][0]["op"] == op_name
    generated = sources["kernels"][0]
    assert f"static int {op_name}_{''.join(map(str, dims))}_" in generated
    assert "y[idx] = x[input_idx]" in generated
    assert "generated_permute" not in generated


@pytest.mark.parametrize(("op_name", "shape", "dims", "_out_shape"), SPECIALIZED_PERMUTE_CASES)
def test_cpu_reference_specialized_permute_float32(op_name, shape, dims, _out_shape):
    spec = _trace_specialized_permute(op_name, shape=shape)
    x = np.arange(np.prod(shape), dtype=np.float32).reshape(shape)

    actual = execute_cpu(spec, {"x": x})["out"]

    expected = np.transpose(x, axes=dims).copy()
    assert actual.dtype == np.float32
    np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize(
    ("op_name", "bad_shape"),
    [
        ("permute021", (2, 3)),
        ("permute102", (2, 3)),
        ("permute210", (2, 3)),
        ("permute0213", (2, 3, 4)),
    ],
)
def test_specialized_permute_frontends_reject_bad_rank(op_name, bad_shape):
    expected_rank = len(SPECIALIZED_PERMUTE_DIMS[op_name])
    with pytest.raises(ValueError, match=rf"rank-{expected_rank} input"):
        _trace_specialized_permute(op_name, shape=bad_shape)


@pytest.mark.parametrize(
    ("dtype", "expected_dtype"),
    [
        ("float32", np.float32),
        ("float16", np.float16),
        ("bfloat16", np.float32),
        ("bool", np.bool_),
    ],
)
def test_cpu_reference_permute(dtype, expected_dtype):
    spec = _trace_permute(dtype, dims=(2, 0, 1), shape=(2, 3, 2))
    x = _input(dtype)

    actual = execute_cpu(spec, {"x": x})["out"]

    expected = _storage_roundtrip(np.transpose(x, axes=(2, 0, 1)).copy(), dtype)
    assert actual.dtype == expected_dtype
    np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize(
    ("dtype", "expected_dtype"),
    [
        ("float32", np.float32),
        ("float16", np.float16),
        ("bfloat16", np.float32),
        ("bool", np.bool_),
    ],
)
def test_cpu_reference_transpose(dtype, expected_dtype):
    spec = _trace_transpose(dtype, dim0=-1, dim1=0, shape=(2, 3, 2))
    x = _input(dtype)

    actual = execute_cpu(spec, {"x": x})["out"]

    expected = _storage_roundtrip(np.swapaxes(x, -1, 0).copy(), dtype)
    assert actual.dtype == expected_dtype
    np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize("dtype", ["float32", "float16", "bfloat16", "bool"])
def test_permute_generated_cpu_source_and_runtime(tmp_path, dtype):
    spec = _trace_permute(dtype, dims=(2, 0, 1), shape=(2, 3, 2))
    lowered, _ = PassManager().run(spec.ir)
    validate_ir(lowered)
    assert [node["op"] for node in lowered["nodes"]] == ["permute"]
    tensor_map = {tensor["name"]: tensor for tensor in lowered["tensors"]}

    cpu_source = render_generated_kernels("cpu", lowered["nodes"], tensor_map)[0]

    assert "static int permute_" in cpu_source
    assert "input_idx += coord * 6" in cpu_source
    assert "input_idx += coord * 2" in cpu_source
    assert "y[idx] = x[input_idx]" in cpu_source
    if dtype == "float32":
        assert "const float* DINO_RESTRICT x" in cpu_source
        assert "float* DINO_RESTRICT y" in cpu_source

    artifact = dml.compile(spec, dml.Target("cpu"), tmp_path / f"permute_{dtype}_cpu.dinoml")
    session = load(artifact.path).create_session()
    x = _input(dtype)
    try:
        actual = session.run_numpy({"x": x})["out"]
    finally:
        session.close()

    expected = _storage_roundtrip(np.transpose(x, axes=(2, 0, 1)).copy(), dtype)
    np.testing.assert_array_equal(actual, expected)


def test_specialized_permute_cpu_artifact_manifest_and_source_manifest_use_named_op(tmp_path):
    spec = _trace_specialized_permute("permute0213", shape=(2, 3, 4, 5))
    artifact = dml.compile(spec, dml.Target("cpu"), tmp_path / "permute0213_cpu.dinoml")

    kernel_manifest = read_json(artifact.path / "kernel_manifest.json")
    source_manifest = read_json(artifact.path / "debug" / "generated_src" / "source_manifest.json")

    [required] = kernel_manifest["required_kernels"]
    [source] = source_manifest["sources"]
    assert required["op"] == "permute0213"
    assert required["kernel_symbol"] == "generated_permute0213"
    assert required["generated_source"]["generated_function_name"].startswith("permute0213_0213_")
    assert source["op"] == "permute0213"
    assert source["generated_function_name"].startswith("permute0213_0213_")


@pytest.mark.skipif(shutil.which("nvcc") is None, reason="nvcc is required")
def test_specialized_permute_cuda_artifact_compiles_with_named_generated_kernel(tmp_path):
    spec = _trace_specialized_permute("permute210", shape=(2, 3, 4))
    artifact = dml.compile(spec, dml.Target("cuda", arch="sm_86"), tmp_path / "permute210_cuda.dinoml")
    generated = (artifact.path / "debug" / "generated_src" / "module.cu").read_text(encoding="utf-8")
    assert "permute210_210_" in generated
    assert "generated_permute210" not in generated
    assert "y[idx] = x[input_idx]" in generated


def test_permute_generated_cuda_source_supports_reduced_precision_and_bool():
    for dtype, pointer_type in (
        ("float16", "const half* DINO_RESTRICT x"),
        ("bfloat16", "const __nv_bfloat16* DINO_RESTRICT x"),
        ("bool", "const bool* DINO_RESTRICT x"),
    ):
        spec = _trace_permute(dtype, dims=(2, 0, 1), shape=(2, 3, 2))
        lowered, _ = PassManager().run(spec.ir)
        tensor_map = {tensor["name"]: tensor for tensor in lowered["tensors"]}

        cuda_source = render_generated_kernels("cuda", lowered["nodes"], tensor_map)[0]

        assert pointer_type in cuda_source
        assert "permute_" in cuda_source
        assert "input_idx += coord * 6" in cuda_source
        assert "input_idx += coord * 2" in cuda_source
        assert "y[idx] = x[input_idx]" in cuda_source


def test_permute_frontend_rejects_dynamic_bad_dims_and_unsupported_dtype():
    class DynamicPermute(dml.Module):
        def forward(self, x):
            return dml.ops.permute(x, dims=(1, 0))

    with pytest.raises(ValueError, match="only static input shapes"):
        dml.trace(DynamicPermute(), inputs={"x": dml.TensorSpec([Dim("n", 1, 4), 3])})
    with pytest.raises(ValueError, match="length"):
        _trace_permute("float32", dims=(0, 1))
    with pytest.raises(ValueError, match="out of range"):
        _trace_permute("float32", dims=(0, 1, 3))
    with pytest.raises(ValueError, match="duplicates"):
        _trace_permute("float32", dims=(0, 1, -3))
    with pytest.raises(ValueError, match="integers"):
        _trace_permute("float32", dims=(0, "1", 2))
    with pytest.raises(ValueError, match="does not support dtype int64"):
        _trace_permute("int64", dims=(2, 0, 1))


def test_transpose_frontend_rejects_dynamic_bad_dims_and_unsupported_dtype():
    class DynamicTranspose(dml.Module):
        def forward(self, x):
            return dml.ops.transpose(x, -1, 0)

    with pytest.raises(ValueError, match="only static input shapes"):
        dml.trace(DynamicTranspose(), inputs={"x": dml.TensorSpec([Dim("n", 1, 4), 3])})
    with pytest.raises(ValueError, match="out of range"):
        _trace_transpose("float32", dim0=0, dim1=3)
    with pytest.raises(ValueError, match="integer"):
        _trace_transpose("float32", dim0="0", dim1=1)
    with pytest.raises(ValueError, match="does not support dtype int64"):
        _trace_transpose("int64", dim0=0, dim1=1)


def test_permute_validation_rejects_dynamic_shape_spec_bad_dims_shape_and_dtype():
    spec = _trace_permute("float32", dims=(2, 0, 1))
    spec.ir["inputs"][0]["shape_spec"] = [Dim("n", 1, 2).to_json(), 3, 4]
    input_tensor = next(tensor for tensor in spec.ir["tensors"] if tensor["name"] == "x")
    input_tensor["shape_spec"] = [Dim("n", 1, 2).to_json(), 3, 4]
    with pytest.raises(ValidationError, match="only static shapes"):
        validate_ir(spec.ir)

    spec = _trace_permute("float32", dims=(2, 0, 1))
    spec.ir["nodes"][0]["attrs"]["dims"] = [2, 0]
    with pytest.raises(ValidationError, match="length"):
        validate_ir(spec.ir)

    spec = _trace_permute("float32", dims=(2, 0, 1))
    spec.ir["nodes"][0]["attrs"]["dims"] = [2, 0, 3]
    with pytest.raises(ValidationError, match="out of range"):
        validate_ir(spec.ir)

    spec = _trace_permute("float32", dims=(2, 0, 1))
    spec.ir["nodes"][0]["attrs"]["dims"] = [2, 0, 0]
    with pytest.raises(ValidationError, match="duplicates"):
        validate_ir(spec.ir)

    spec = _trace_permute("float32", dims=(2, 0, 1))
    spec.ir["outputs"][0]["shape"] = [4, 2, 4]
    spec.ir["outputs"][0]["shape_spec"] = [4, 2, 4]
    output_tensor = next(tensor for tensor in spec.ir["tensors"] if tensor["name"] == spec.ir["outputs"][0]["tensor"])
    output_tensor["shape"] = [4, 2, 4]
    output_tensor["shape_spec"] = [4, 2, 4]
    output_tensor["layout"]["strides"] = [8, 4, 1]
    with pytest.raises(ValidationError, match=r"expected \[4, 2, 3\]"):
        validate_ir(spec.ir)

    spec = _trace_permute("float32", dims=(2, 0, 1))
    spec.ir["outputs"][0]["dtype"] = "bool"
    output_tensor = next(tensor for tensor in spec.ir["tensors"] if tensor["name"] == spec.ir["outputs"][0]["tensor"])
    output_tensor["dtype"] = "bool"
    with pytest.raises(ValidationError, match="expected float32"):
        validate_ir(spec.ir)


def test_specialized_permute_validation_rejects_attr_drift_from_fixed_dims():
    spec = _trace_specialized_permute("permute102", shape=(2, 3, 4))
    spec.ir["nodes"][0]["attrs"]["dims"] = [2, 0, 1]
    with pytest.raises(ValidationError, match=r"uses fixed dims \[1, 0, 2\]"):
        validate_ir(spec.ir)
