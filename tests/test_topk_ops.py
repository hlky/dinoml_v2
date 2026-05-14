import numpy as np
import pytest

import dinoml as dml
from dinoml.backends.cpu import execute_cpu
from dinoml.ir import array_from_storage, array_to_storage
from dinoml.layout import dense_layout
from dinoml.lowering.ops import render_generated_kernels
from dinoml.passes import PassManager, validate_ir
from dinoml.passes.validation import ValidationError
from dinoml.runtime import load
from dinoml.shapes import Dim


class TopkTupleModule(dml.Module):
    def __init__(self, k=2, dim=-1, largest=True, sorted=True):
        self.k = k
        self.dim = dim
        self.largest = largest
        self.sorted = sorted

    def forward(self, x):
        return dml.ops.topk(x, self.k, dim=self.dim, largest=self.largest, sorted=self.sorted)


class TopkNamedModule(TopkTupleModule):
    def forward(self, x):
        values, indices = dml.ops.topk(x, self.k, dim=self.dim, largest=self.largest, sorted=self.sorted)
        return {"values": values, "indices": indices}


def _trace(dtype="float32", shape=(2, 3, 5), k=2, dim=-1, named=False, largest=True, sorted=True):
    module_cls = TopkNamedModule if named else TopkTupleModule
    return dml.trace(
        module_cls(k=k, dim=dim, largest=largest, sorted=sorted),
        inputs={"x": dml.TensorSpec(shape, dtype)},
        name=f"topk_{dtype}",
    )


def _input(dtype, shape=(2, 3, 5)):
    values = np.array(
        [
            [[1.0, 5.0, 5.0, -1.0, 2.0], [0.0, -2.0, 3.0, 3.0, 1.0], [7.0, 1.0, 7.0, 0.0, 8.0]],
            [[4.0, 4.0, 2.0, 1.0, 0.0], [-1.0, -1.0, -1.0, -2.0, 9.0], [0.0, 9.0, 8.0, 9.0, 7.0]],
        ],
        dtype=np.float32,
    ).reshape(shape)
    if dtype == "bool":
        return (values.astype(np.int64) % 3) == 0
    if dtype in {"float16", "bfloat16"}:
        return array_from_storage(array_to_storage(values, dtype), dtype)
    return values


def _expected_indices(x, k):
    rows = x.reshape((-1, x.shape[-1]))
    out = np.empty((rows.shape[0], k), dtype=np.int64)
    for row_idx, row in enumerate(rows):
        used = np.zeros((row.shape[0],), dtype=np.bool_)
        for out_col in range(k):
            best = None
            for col, value in enumerate(row):
                if used[col]:
                    continue
                if best is None or _is_better(value, row[best]):
                    best = col
            used[best] = True
            out[row_idx, out_col] = best
    return out.reshape((*x.shape[:-1], k))


def _is_better(candidate, current):
    if isinstance(candidate, (bool, np.bool_)) or isinstance(current, (bool, np.bool_)):
        return bool(candidate) > bool(current)
    return float(candidate) > float(current) or (np.isnan(float(candidate)) and not np.isnan(float(current)))


def _expected_values(x, k, dtype):
    indices = _expected_indices(x, k)
    values = np.take_along_axis(x, indices, axis=-1)
    if dtype in {"float16", "bfloat16"}:
        return array_from_storage(array_to_storage(values, dtype), dtype)
    return np.asarray(values, dtype=np.bool_ if dtype == "bool" else np.float32)


def _numpy_result_dtype(dtype):
    if dtype == "float16":
        return np.dtype(np.float16)
    if dtype == "bool":
        return np.dtype(np.bool_)
    return np.dtype(np.float32)


def test_topk_frontend_returns_tuple_emits_two_nodes_and_normalizes_dim():
    spec = _trace("float32", k=3, dim=-1)

    assert [output["name"] for output in spec.ir["outputs"]] == ["output_0", "output_1"]
    assert [output["shape"] for output in spec.ir["outputs"]] == [[2, 3, 3], [2, 3, 3]]
    assert [output["shape_spec"] for output in spec.ir["outputs"]] == [[2, 3, 3], [2, 3, 3]]
    assert [output["dtype"] for output in spec.ir["outputs"]] == ["float32", "int64"]
    assert [node["op"] for node in spec.ir["nodes"]] == ["topk_values", "topk_indices"]
    assert [node["attrs"] for node in spec.ir["nodes"]] == [
        {"k": 3, "dim": 2, "largest": True, "sorted": True},
        {"k": 3, "dim": 2, "largest": True, "sorted": True},
    ]


@pytest.mark.parametrize("dtype", ["float32", "float16", "bfloat16", "bool"])
def test_cpu_reference_topk_supported_dtypes(dtype):
    spec = _trace(dtype, k=2, named=True)
    x = _input(dtype)

    actual = execute_cpu(spec, {"x": x})

    assert actual["indices"].dtype == np.int64
    assert actual["values"].dtype == _numpy_result_dtype(dtype)
    np.testing.assert_array_equal(actual["indices"], _expected_indices(x, 2))
    np.testing.assert_array_equal(actual["values"], _expected_values(x, 2, dtype))


def test_topk_ties_return_lower_indices_first_and_values_descending():
    spec = _trace("float32", shape=(2, 5), k=4, named=True)
    x = np.array([[3.0, 1.0, 3.0, 2.0, 3.0], [5.0, 5.0, 4.0, 5.0, 3.0]], dtype=np.float32)

    actual = execute_cpu(spec, {"x": x})

    np.testing.assert_array_equal(actual["indices"], np.array([[0, 2, 4, 3], [0, 1, 3, 2]], dtype=np.int64))
    np.testing.assert_array_equal(actual["values"], np.array([[3.0, 3.0, 3.0, 2.0], [5.0, 5.0, 5.0, 4.0]], dtype=np.float32))


def test_topk_generated_cpu_source_and_runtime(tmp_path):
    spec = _trace("float32", k=3, named=True)
    lowered, _ = PassManager().run(spec.ir)
    validate_ir(lowered)
    assert [node["op"] for node in lowered["nodes"]] == ["topk_values", "topk_indices"]
    tensor_map = {tensor["name"]: tensor for tensor in lowered["tensors"]}

    cpu_sources = render_generated_kernels("cpu", lowered["nodes"], tensor_map)
    combined = "\n".join(cpu_sources)

    assert "static int topk_values_" in combined
    assert "static int topk_indices_" in combined
    assert "const float* DINO_RESTRICT x" in combined
    assert "float* DINO_RESTRICT y" in combined
    assert "int64_t* DINO_RESTRICT y" in combined
    assert "selected[out_col] = best_index" in combined
    assert "value > best_value" in combined

    artifact = dml.compile(spec, dml.Target("cpu"), tmp_path / "topk_float32_cpu.dinoml")
    session = load(artifact.path).create_session()
    x = _input("float32")
    try:
        actual = session.run_numpy({"x": x})
    finally:
        session.close()

    np.testing.assert_array_equal(actual["indices"], _expected_indices(x, 3))
    np.testing.assert_array_equal(actual["values"], _expected_values(x, 3, "float32"))


def test_topk_generated_cuda_source_supports_reduced_precision_bool_and_int64_indices():
    for dtype, pointer_type in (
        ("float16", "const half* DINO_RESTRICT x"),
        ("bfloat16", "const __nv_bfloat16* DINO_RESTRICT x"),
        ("bool", "const bool* DINO_RESTRICT x"),
    ):
        spec = _trace(dtype, k=2, named=True)
        lowered, _ = PassManager().run(spec.ir)
        tensor_map = {tensor["name"]: tensor for tensor in lowered["tensors"]}

        cuda_source = "\n".join(render_generated_kernels("cuda", lowered["nodes"], tensor_map))

        assert pointer_type in cuda_source
        assert "topk_values_" in cuda_source
        assert "topk_indices_" in cuda_source
        assert "int64_t* DINO_RESTRICT y" in cuda_source
        assert "selected[out_col] = best_index" in cuda_source


def test_topk_frontend_rejects_dynamic_bad_k_non_last_dim_modes_and_bad_dtype():
    class DynamicShapeTopk(dml.Module):
        def forward(self, x):
            return dml.ops.topk(x, 2)

    with pytest.raises(ValueError, match="only static input shapes"):
        dml.trace(DynamicShapeTopk(), inputs={"x": dml.TensorSpec([Dim("n", 1, 4), 3])})
    with pytest.raises(ValueError, match="positive non-bool integer"):
        _trace("float32", k=True)
    with pytest.raises(ValueError, match="must be positive"):
        _trace("float32", k=0)
    with pytest.raises(ValueError, match="exceeds last dimension"):
        _trace("float32", k=6)
    with pytest.raises(ValueError, match="dim must be an integer"):
        _trace("float32", dim=True)
    with pytest.raises(NotImplementedError, match="only the last dimension"):
        _trace("float32", dim=1)
    with pytest.raises(NotImplementedError, match="largest=True"):
        _trace("float32", largest=False)
    with pytest.raises(NotImplementedError, match="sorted=True"):
        _trace("float32", sorted=False)
    with pytest.raises(ValueError, match="does not support dtype int32"):
        _trace("int32")
    with pytest.raises(ValueError, match="does not support dtype int64"):
        _trace("int64")


def test_topk_validation_rejects_dynamic_attrs_shape_and_dtype():
    spec = _trace("float32", k=2)
    spec.ir["inputs"][0]["shape_spec"] = [Dim("n", 1, 2).to_json(), 3, 5]
    input_tensor = next(tensor for tensor in spec.ir["tensors"] if tensor["name"] == "x")
    input_tensor["shape_spec"] = [Dim("n", 1, 2).to_json(), 3, 5]
    with pytest.raises(ValidationError, match="only static shapes"):
        validate_ir(spec.ir)

    spec = _trace("float32", k=2)
    spec.ir["nodes"][0]["attrs"]["k"] = 0
    with pytest.raises(ValidationError, match="must be positive"):
        validate_ir(spec.ir)

    spec = _trace("float32", k=2)
    spec.ir["nodes"][0]["attrs"]["dim"] = 1
    with pytest.raises(ValidationError, match="only the last dimension"):
        validate_ir(spec.ir)

    spec = _trace("float32", k=2)
    spec.ir["outputs"][0]["shape"] = [2, 3, 3]
    spec.ir["outputs"][0]["shape_spec"] = [2, 3, 3]
    output_tensor = next(tensor for tensor in spec.ir["tensors"] if tensor["name"] == spec.ir["outputs"][0]["tensor"])
    output_tensor["shape"] = [2, 3, 3]
    output_tensor["shape_spec"] = [2, 3, 3]
    output_tensor["layout"] = dense_layout([2, 3, 3])
    with pytest.raises(ValidationError, match=r"expected \[2, 3, 2\]"):
        validate_ir(spec.ir)

    spec = _trace("float32", k=2)
    spec.ir["outputs"][1]["dtype"] = "float32"
    output_tensor = next(tensor for tensor in spec.ir["tensors"] if tensor["name"] == spec.ir["outputs"][1]["tensor"])
    output_tensor["dtype"] = "float32"
    with pytest.raises(ValidationError, match="expected int64"):
        validate_ir(spec.ir)

    spec = _trace("float32", k=2)
    input_tensor = next(tensor for tensor in spec.ir["tensors"] if tensor["name"] == "x")
    input_tensor["dtype"] = "int32"
    with pytest.raises(ValidationError, match="topk does not support dtype int32"):
        validate_ir(spec.ir)

    spec = _trace("float32", k=2)
    input_tensor = next(tensor for tensor in spec.ir["tensors"] if tensor["name"] == "x")
    input_tensor["dtype"] = "int64"
    with pytest.raises(ValidationError, match="topk does not support dtype int64"):
        validate_ir(spec.ir)
