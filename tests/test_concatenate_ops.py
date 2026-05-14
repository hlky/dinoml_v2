import numpy as np
import pytest
import re
import shutil

import dinoml as dml
from dinoml import runtime
from dinoml.backends.cpu import execute_cpu
from dinoml.ir import array_from_storage, array_to_storage
from dinoml.lowering.ops import render_generated_kernels
from dinoml.passes import PassManager, validate_ir
from dinoml.passes.validation import ValidationError
from dinoml.runtime import load
from dinoml.shapes import Dim


class ConcatenateModule(dml.Module):
    def __init__(self, dim=0):
        self.dim = dim

    def forward(self, x, y, z):
        return dml.ops.output(dml.ops.concatenate([x, y, z], dim=self.dim), "out")


class FusedConcatenateModule(dml.Module):
    def forward(self, x):
        return dml.ops.output(dml.ops.concatenate([dml.ops.sin(x), dml.ops.cos(x)], dim=1), "out")


def _trace_concatenate(dtype="float32", dim=1, shapes=([2, 1, 4], [2, 2, 4], [2, 3, 4])):
    inputs = {
        "x": dml.TensorSpec(shapes[0], dtype),
        "y": dml.TensorSpec(shapes[1], dtype),
        "z": dml.TensorSpec(shapes[2], dtype),
    }
    return dml.trace(ConcatenateModule(dim), inputs=inputs, name=f"concatenate_{dtype}")


def _storage_roundtrip(value, dtype):
    if dtype in {"float16", "bfloat16"}:
        return array_from_storage(array_to_storage(value, dtype), dtype)
    return np.asarray(value, dtype=np.bool_ if dtype == "bool" else np.float32)


def _inputs(dtype):
    if dtype == "bool":
        return {
            "x": np.array([[[True], [False]]], dtype=np.bool_),
            "y": np.array([[[False], [True], [True]]], dtype=np.bool_),
            "z": np.array([[[True]]], dtype=np.bool_),
        }
    return {
        "x": np.arange(2, dtype=np.float32).reshape(1, 2, 1),
        "y": (10 + np.arange(3, dtype=np.float32)).reshape(1, 3, 1),
        "z": (20 + np.arange(1, dtype=np.float32)).reshape(1, 1, 1),
    }


def _trace_fused_concatenate(dtype="float32", shape=(2, 3)):
    return dml.trace(
        FusedConcatenateModule(),
        inputs={"x": dml.TensorSpec(shape, dtype)},
        name=f"fused_concatenate_{dtype}",
    )


def test_concatenate_frontend_ir_normalizes_negative_dim():
    spec = _trace_concatenate("float32", dim=-2)

    assert spec.ir["outputs"][0]["shape"] == [2, 6, 4]
    assert spec.ir["outputs"][0]["shape_spec"] == [2, 6, 4]
    assert spec.ir["outputs"][0]["dtype"] == "float32"
    node = spec.ir["nodes"][0]
    assert node["op"] == "concatenate"
    assert node["inputs"] == ["x", "y", "z"]
    assert node["attrs"] == {"dim": 1}


@pytest.mark.parametrize(
    ("dtype", "expected_dtype"),
    [
        ("float32", np.float32),
        ("float16", np.float16),
        ("bfloat16", np.float32),
        ("bool", np.bool_),
    ],
)
def test_cpu_reference_concatenate(dtype, expected_dtype):
    spec = _trace_concatenate(dtype, dim=1, shapes=([1, 2, 1], [1, 3, 1], [1, 1, 1]))
    inputs = _inputs(dtype)

    actual = execute_cpu(spec, inputs)["out"]

    expected = _storage_roundtrip(np.concatenate([inputs["x"], inputs["y"], inputs["z"]], axis=1), dtype)
    assert actual.dtype == expected_dtype
    np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize("dtype", ["float32", "float16", "bfloat16", "bool"])
def test_concatenate_generated_cpu_source_and_runtime(tmp_path, dtype):
    spec = _trace_concatenate(dtype, dim=1, shapes=([1, 2, 1], [1, 3, 1], [1, 1, 1]))
    lowered, _ = PassManager().run(spec.ir)
    validate_ir(lowered)
    assert [node["op"] for node in lowered["nodes"]] == ["concatenate"]
    tensor_map = {tensor["name"]: tensor for tensor in lowered["tensors"]}

    cpu_source = render_generated_kernels("cpu", lowered["nodes"], tensor_map)[0]

    assert "static int concatenate_" in cpu_source
    assert "concat_idx" in cpu_source
    if dtype == "float32":
        assert "const float* DINO_RESTRICT x0" in cpu_source
        assert "const float* DINO_RESTRICT x1" in cpu_source
        assert "float* DINO_RESTRICT y" in cpu_source

    artifact = dml.compile(spec, dml.Target("cpu"), tmp_path / f"concatenate_{dtype}_cpu.dinoml")
    session = load(artifact.path).create_session()
    inputs = _inputs(dtype)
    try:
        actual = session.run_numpy(inputs)["out"]
    finally:
        session.close()

    expected = _storage_roundtrip(np.concatenate([inputs["x"], inputs["y"], inputs["z"]], axis=1), dtype)
    np.testing.assert_array_equal(actual, expected)


def test_concatenate_generated_cuda_source_supports_reduced_precision_and_bool():
    for dtype, pointer_type in (
        ("float16", "const half* DINO_RESTRICT x0"),
        ("bfloat16", "const __nv_bfloat16* DINO_RESTRICT x0"),
        ("bool", "const bool* DINO_RESTRICT x0"),
    ):
        spec = _trace_concatenate(dtype, dim=1, shapes=([1, 2, 1], [1, 3, 1], [1, 1, 1]))
        lowered, _ = PassManager().run(spec.ir)
        tensor_map = {tensor["name"]: tensor for tensor in lowered["tensors"]}

        cuda_source = render_generated_kernels("cuda", lowered["nodes"], tensor_map)[0]

        assert pointer_type in cuda_source
        assert "concatenate_" in cuda_source
        assert "y[idx] = x" in cuda_source


def test_concatenate_generated_cuda_source_uses_wrapper_parameter_names_for_fused_inputs():
    spec = _trace_fused_concatenate("float32", shape=(2, 3))
    lowered, _ = PassManager().run(spec.ir)
    tensor_map = {tensor["name"]: tensor for tensor in lowered["tensors"]}

    cuda_source = "\n".join(render_generated_kernels("cuda", lowered["nodes"], tensor_map))

    assert "concatenate_" in cuda_source
    assert "fused_elementwise_" in cuda_source
    assert re.search(
        r"static int concatenate_[0-9a-f]+\([^)]*\)\s*\{.*?<<<grid, block, 0, stream>>>\(x0, x1, y, runtime_numel\);",
        cuda_source,
        re.DOTALL,
    )


@pytest.mark.skipif(shutil.which("nvcc") is None, reason="nvcc is required")
def test_cuda_artifact_compiles_and_runs_fused_elementwise_concatenate(tmp_path):
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA device is required")

    spec = _trace_fused_concatenate("float32", shape=(2, 3))
    artifact = dml.compile(spec, dml.Target("cuda", arch="sm_86"), tmp_path / "fused_concatenate_cuda.dinoml")
    generated = (artifact.path / "debug" / "generated_src" / "module.cu").read_text(encoding="utf-8")
    assert "concatenate_" in generated
    assert re.search(
        r"static int concatenate_[0-9a-f]+\([^)]*\)\s*\{.*?<<<grid, block, 0, stream>>>\(x0, x1, y, runtime_numel\);",
        generated,
        re.DOTALL,
    )

    x = np.array([[0.0, 0.25, 0.5], [1.0, -0.75, 2.0]], dtype=np.float32)
    expected = np.concatenate([np.sin(x), np.cos(x)], axis=1).astype(np.float32)

    module = runtime.load(artifact.path)
    session = module.create_session()
    actual = session.run_torch({"x": torch.tensor(x, device="cuda", dtype=torch.float32)})["out"]
    session.close()
    module.close()

    np.testing.assert_allclose(actual.float().cpu().numpy(), expected, atol=1e-6, rtol=1e-6)


def test_concatenate_frontend_rejects_invalid_inputs_and_dynamic_shapes():
    class DynamicConcatenate(dml.Module):
        def forward(self, x, y):
            return dml.ops.concatenate([x, y], dim=0)

    with pytest.raises(ValueError, match="non-empty sequence"):
        dml.ops.concatenate([], dim=0)
    with pytest.raises(ValueError, match="only static input shapes"):
        dml.trace(
            DynamicConcatenate(),
            inputs={"x": dml.TensorSpec([Dim("n", 1, 4), 3]), "y": dml.TensorSpec([2, 3])},
        )
    with pytest.raises(ValueError, match="out of range"):
        _trace_concatenate("float32", dim=3)
    with pytest.raises(ValueError, match="rank"):
        _trace_concatenate("float32", dim=0, shapes=([2, 3], [2, 3, 1], [2, 3]))
    with pytest.raises(ValueError, match="axis 0"):
        _trace_concatenate("float32", dim=1, shapes=([2, 1], [3, 2], [2, 1]))


def test_concatenate_validation_rejects_dynamic_shape_spec_bad_dim_and_dtype():
    spec = _trace_concatenate("float32", dim=1)
    spec.ir["inputs"][0]["shape_spec"] = [Dim("n", 1, 2).to_json(), 1, 4]
    input_tensor = next(tensor for tensor in spec.ir["tensors"] if tensor["name"] == "x")
    input_tensor["shape_spec"] = [Dim("n", 1, 2).to_json(), 1, 4]
    with pytest.raises(ValidationError, match="only static shapes"):
        validate_ir(spec.ir)

    spec = _trace_concatenate("float32", dim=1)
    spec.ir["nodes"][0]["attrs"]["dim"] = 4
    with pytest.raises(ValidationError, match="out of range"):
        validate_ir(spec.ir)

    spec = _trace_concatenate("float32", dim=1)
    spec.ir["outputs"][0]["shape"] = [2, 5, 4]
    spec.ir["outputs"][0]["shape_spec"] = [2, 5, 4]
    output_tensor = next(tensor for tensor in spec.ir["tensors"] if tensor["name"] == spec.ir["outputs"][0]["tensor"])
    output_tensor["shape"] = [2, 5, 4]
    output_tensor["shape_spec"] = [2, 5, 4]
    output_tensor["layout"]["strides"] = [20, 4, 1]
    with pytest.raises(ValidationError, match=r"expected \[2, 6, 4\]"):
        validate_ir(spec.ir)

    spec = _trace_concatenate("float32", dim=1)
    spec.ir["outputs"][0]["dtype"] = "bool"
    output_tensor = next(tensor for tensor in spec.ir["tensors"] if tensor["name"] == spec.ir["outputs"][0]["tensor"])
    output_tensor["dtype"] = "bool"
    with pytest.raises(ValidationError, match="expected float32"):
        validate_ir(spec.ir)
