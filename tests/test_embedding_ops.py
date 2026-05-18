import shutil

import numpy as np
import pytest

import dinoml as dml
from dinoml import runtime
from dinoml.reference import reference_numpy
from dinoml.frontend import GraphBuilder
from dinoml.ir import array_from_storage, array_to_storage
from dinoml.kernels.manifest import build_kernel_manifest
from dinoml.lowering.ops import collect_generated_sources, render_generated_kernels
from dinoml.ops.definitions import OP_REGISTRY
from dinoml.passes import PassManager, validate_ir
from dinoml.passes.validation import ValidationError


class EmbeddingModule(dml.Module):
    def forward(self, table, indices):
        return dml.ops.output(dml.ops.embedding(table, indices), "out")


def _trace_embedding(
    *,
    dtype: str = "float32",
    index_dtype: str = "int64",
    table_shape=(8, 4),
    index_shape=(2, 3),
):
    return dml.trace(
        EmbeddingModule(),
        inputs={
            "table": dml.TensorSpec(table_shape, dtype),
            "indices": dml.TensorSpec(index_shape, index_dtype),
        },
        name=f"embedding_{dtype}_{index_dtype}",
    )


def _storage_roundtrip(value, dtype: str) -> np.ndarray:
    if dtype == "float32":
        return np.asarray(value, dtype=np.float32)
    return array_from_storage(array_to_storage(np.asarray(value, dtype=np.float32), dtype), dtype)


def _table(dtype: str, shape=(8, 4)) -> np.ndarray:
    values = (np.arange(np.prod(shape), dtype=np.float32).reshape(shape) * 0.25) - 1.5
    return _storage_roundtrip(values, dtype)


def _indices(index_dtype: str = "int64", shape=(2, 3)) -> np.ndarray:
    values = np.array([[3, 0, 1], [1, 6, 2]], dtype=np.int64)
    if tuple(shape) != (2, 3):
        if len(shape) == 1:
            values = np.array([3, 0, 1, 6], dtype=np.int64)
        else:
            values = np.arange(np.prod(shape), dtype=np.int64).reshape(shape) % 7
    return values.astype(np.int64 if index_dtype == "int64" else np.int32)


def _reference_embedding(table: np.ndarray, indices: np.ndarray, *, dtype: str) -> np.ndarray:
    table_ref = _storage_roundtrip(table, dtype)
    index_values = np.asarray(indices, dtype=np.int64)
    return np.array(table_ref[index_values], copy=True)


def test_embedding_frontend_ir_registers_op_and_preserves_dynamic_index_shape_spec_and_dtype():
    batch = dml.Dim("batch", min=1, max=4)
    spec = dml.trace(
        EmbeddingModule(),
        inputs={
            "table": dml.TensorSpec([10, 5], "float32"),
            "indices": dml.TensorSpec([batch, 3], "int32"),
        },
        name="embedding_dynamic_batch",
    )

    assert "embedding" in OP_REGISTRY.frontend_names()
    node = spec.ir["nodes"][0]
    output = spec.ir["outputs"][0]
    assert node["op"] == "embedding"
    assert node["inputs"] == ["table", "indices"]
    assert node["attrs"] == {}
    assert output["shape"] == [4, 3, 5]
    assert output["shape_spec"] == [batch.to_json(), 3, 5]
    assert output["dtype"] == "float32"

    lowered, _ = PassManager().run(spec.ir)
    validate_ir(lowered)
    assert [lowered_node["op"] for lowered_node in lowered["nodes"]] == ["embedding"]


@pytest.mark.parametrize(
    ("dtype", "expected_dtype"),
    [("float32", np.float32), ("float16", np.float16), ("bfloat16", np.float32)],
)
def test_cpu_reference_embedding_matches_expected(dtype, expected_dtype):
    spec = _trace_embedding(dtype=dtype)
    table = _table(dtype)
    indices = _indices()

    actual = reference_numpy(spec, {"table": table, "indices": indices})["out"]
    expected = _reference_embedding(table, indices, dtype=dtype)

    assert actual.dtype == expected_dtype
    np.testing.assert_array_equal(actual, expected)


def test_cpu_reference_embedding_accepts_int32_indices():
    spec = _trace_embedding(dtype="float32", index_dtype="int32")
    table = _table("float32")
    indices = _indices("int32")

    actual = reference_numpy(spec, {"table": table, "indices": indices})["out"]

    np.testing.assert_array_equal(actual, _reference_embedding(table, indices, dtype="float32"))


def test_embedding_frontend_rejects_dynamic_table_bad_rank_and_unsupported_dtypes():
    vocab = dml.Dim("vocab", min=4, max=8)

    with pytest.raises(ValueError, match="static table shape"):
        dml.trace(
            EmbeddingModule(),
            inputs={
                "table": dml.TensorSpec([vocab, 4], "float32"),
                "indices": dml.TensorSpec([2, 3], "int64"),
            },
            name="embedding_dynamic_table",
        )

    with pytest.raises(ValueError, match="table rank 1 must be 2"):
        _trace_embedding(table_shape=(8,))

    with pytest.raises(ValueError, match="table rank 3 must be 2"):
        _trace_embedding(table_shape=(2, 4, 8))

    with pytest.raises(ValueError, match="embedding does not support dtype bool"):
        _trace_embedding(dtype="bool")

    with pytest.raises(ValueError, match="indices must have dtype int64 or int32, got bool"):
        _trace_embedding(index_dtype="bool")

    with pytest.raises(ValueError, match="different DinoML traces"):
        with GraphBuilder("embedding_table_builder") as table_builder:
            table = table_builder.input("table", dml.TensorSpec([8, 4], "float32"))
        with GraphBuilder("embedding_indices_builder") as index_builder:
            indices = index_builder.input("indices", dml.TensorSpec([2, 3], "int64"))
        with GraphBuilder("embedding_output_builder"):
            dml.ops.embedding(table, indices)


def test_embedding_validation_rejects_dynamic_table_bad_shape_and_dtype():
    spec = _trace_embedding(dtype="float32")
    spec.ir["inputs"][0]["shape_spec"] = [dml.Dim("vocab", min=1, max=8).to_json(), 4]
    table_tensor = next(tensor for tensor in spec.ir["tensors"] if tensor["name"] == "table")
    table_tensor["shape_spec"] = [dml.Dim("vocab", min=1, max=8).to_json(), 4]
    with pytest.raises(ValidationError, match="static table shape"):
        validate_ir(spec.ir)

    spec = _trace_embedding(dtype="float32")
    spec.ir["outputs"][0]["shape"] = [2, 3, 5]
    spec.ir["outputs"][0]["shape_spec"] = [2, 3, 5]
    output_tensor = next(tensor for tensor in spec.ir["tensors"] if tensor["name"] == spec.ir["outputs"][0]["tensor"])
    output_tensor["shape"] = [2, 3, 5]
    output_tensor["shape_spec"] = [2, 3, 5]
    output_tensor["layout"]["strides"] = [15, 5, 1]
    with pytest.raises(ValidationError, match=r"expected \[2, 3, 4\]"):
        validate_ir(spec.ir)

    spec = _trace_embedding(dtype="float32")
    spec.ir["outputs"][0]["dtype"] = "float16"
    output_tensor = next(tensor for tensor in spec.ir["tensors"] if tensor["name"] == spec.ir["outputs"][0]["tensor"])
    output_tensor["dtype"] = "float16"
    with pytest.raises(ValidationError, match="expected float32"):
        validate_ir(spec.ir)

    spec = _trace_embedding(dtype="float32")
    index_tensor = next(tensor for tensor in spec.ir["tensors"] if tensor["name"] == "indices")
    index_tensor["dtype"] = "float32"
    with pytest.raises(ValidationError, match="indices must have dtype int64 or int32"):
        validate_ir(spec.ir)


def test_embedding_manifest_and_generated_sources_are_model_owned():
    spec = _trace_embedding(dtype="float32", table_shape=(16, 257), index_shape=(4, 3))
    lowered, _ = PassManager().run(spec.ir)
    validate_ir(lowered)

    manifest = build_kernel_manifest(lowered, {"name": "cpu", "arch": "native"})
    [required] = manifest["required_kernels"]
    assert required["op"] == "embedding"
    assert required["kernel_symbol"] == "generated_embedding"
    assert required["kernel_library"] == "model"
    assert required["profiler_symbol"] is None
    assert required["has_profiler"] is False
    assert required["generated_source"]["generated_function_name"].startswith("embedding_")
    assert required["generated_source"]["source_key"].startswith("cpu:")

    tensor_map = {tensor["name"]: tensor for tensor in lowered["tensors"]}
    sources = collect_generated_sources("cuda", lowered["nodes"], tensor_map)
    assert len(sources["kernels"]) == 1
    generated = sources["kernels"][0]
    assert "embedding_" in generated
    assert "selected_index = static_cast<int64_t>(indices[row]);" in generated
    assert "runtime_numel_out != runtime_numel_indices * 257" in generated
    assert "batch_gather_" not in generated


def test_embedding_generated_sources_accept_int32_indices_and_reduced_precision_tables():
    spec = _trace_embedding(dtype="bfloat16", index_dtype="int32", table_shape=(8, 5), index_shape=(2, 3))
    lowered, _ = PassManager().run(spec.ir)
    tensor_map = {tensor["name"]: tensor for tensor in lowered["tensors"]}

    cpu_source = render_generated_kernels("cpu", lowered["nodes"], tensor_map)[0]
    cuda_source = render_generated_kernels("cuda", lowered["nodes"], tensor_map)[0]

    assert "const int32_t* DINO_RESTRICT indices" in cpu_source
    assert "const __nv_bfloat16* DINO_RESTRICT table" in cuda_source
    assert "__nv_bfloat16* DINO_RESTRICT y" in cuda_source
    assert "embedding runtime output size mismatch" in cpu_source
    assert "#include <assert.h>" in cuda_source
    assert "assert(selected_index >= 0 && selected_index < 8);" in cuda_source


def test_cpu_artifact_runs_generated_embedding_with_dynamic_batch(tmp_path, monkeypatch):
    batch = dml.Dim("batch", min=1, max=4)
    spec = dml.trace(
        EmbeddingModule(),
        inputs={
            "table": dml.TensorSpec([8, 4], "float32"),
            "indices": dml.TensorSpec([batch, 3], "int64"),
        },
        name="embedding_dynamic_cpu",
    )
    artifact = dml.compile(spec, dml.Target("cpu"), tmp_path / "embedding_dynamic_cpu.dinoml")
    generated = (artifact.path / "debug" / "generated_src" / "module.cpp").read_text(encoding="utf-8")
    assert "embedding_" in generated
    assert "runtime_numel_indices" in generated
    assert "embedding runtime output size mismatch" in generated

    table = _table("float32")
    module = runtime.load(artifact.path)
    session = module.create_session()
    for rows in (2, 4):
        indices = (np.arange(rows * 3, dtype=np.int64).reshape(rows, 3) + 1) % 7
        expected = _reference_embedding(table, indices, dtype="float32")
        actual = session.run_numpy({"table": table, "indices": indices})["out"]
        assert actual.shape == (rows, 3, 4)
        np.testing.assert_array_equal(actual, expected)
    session.close()
    module.close()


def test_embedding_generated_cpu_runtime_rejects_oob_index(tmp_path, monkeypatch):
    spec = _trace_embedding(dtype="float32")
    artifact = dml.compile(spec, dml.Target("cpu"), tmp_path / "embedding_oob_cpu.dinoml")
    module = runtime.load(artifact.path)
    session = module.create_session()
    table = _table("float32")
    indices = _indices().copy()
    indices[0, 0] = 8
    try:
        with pytest.raises(RuntimeError, match="embedding index out of bounds"):
            session.run_numpy({"table": table, "indices": indices})
    finally:
        session.close()
        module.close()


@pytest.mark.skipif(shutil.which("nvcc") is None, reason="nvcc is required")
def test_cuda_artifact_runs_generated_embedding(tmp_path, monkeypatch):
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA device is required")

    spec = _trace_embedding(dtype="float32", index_dtype="int32", table_shape=(8, 5), index_shape=(2, 3))
    artifact = dml.compile(spec, dml.Target("cuda", arch="sm_86"), tmp_path / "embedding_cuda.dinoml")
    generated = (artifact.path / "debug" / "generated_src" / "module.cu").read_text(encoding="utf-8")
    assert "embedding_" in generated
    assert "const int32_t* DINO_RESTRICT indices" in generated
    assert "table[table_idx]" in generated
    assert "runtime_numel_out != runtime_numel_indices * 5" in generated

    table = _table("float32", shape=(8, 5))
    indices = _indices("int32")
    expected = _reference_embedding(table, indices, dtype="float32")

    module = runtime.load(artifact.path)
    session = module.create_session()
    actual = session.run_numpy({"table": table, "indices": indices})["out"]
    session.close()
    module.close()

    np.testing.assert_allclose(actual, expected, atol=1e-6, rtol=1e-6)
