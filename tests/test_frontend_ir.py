import pytest

import dinoml as dml
from dinoml.ir import array_from_storage, array_to_storage, canonical_json
from dinoml.shapes import infer_output_shape, validate_runtime_shape


class BadBroadcast(dml.Module):
    def forward(self, x):
        return dml.ops.add(x, dml.Parameter([5], dtype="float32", name="bias"))


def test_ir_serialization_is_stable():
    from tests.models.fused_elementwise import build_spec

    spec_a = build_spec()
    spec_b = build_spec()
    assert canonical_json(spec_a.ir) == canonical_json(spec_b.ir)


def test_shape_errors_are_reported_during_trace():
    with pytest.raises(ValueError, match="not broadcastable"):
        dml.trace(BadBroadcast(), inputs={"x": dml.TensorSpec([1, 4, 3])}, constants={"bias": [1, 2, 3, 4, 5]})


def test_parameters_are_symbolic_and_constants_bind_later():
    parameter = dml.Parameter([2, 3], dtype="float32", name="w")
    assert parameter.value is None
    assert parameter.shape == [2, 3]

    bound = parameter.bind([[1, 2, 3], [4, 5, 6]])
    assert bound.value.shape == (2, 3)


def test_tensor_spec_records_dynamic_shape_metadata():
    batch = dml.Dim("batch", min=1, max=4, typical=2)
    spec = dml.TensorSpec([batch, 16], "fp32")
    assert spec.max_shape == [4, 16]
    assert spec.dynamic
    assert spec.shape_spec[0]["name"] == "batch"


def test_runtime_shape_helpers_validate_dim_constraints():
    height = dml.Dim("height", min=8, max=32, divisible_by=8)
    spec = {"name": "x", "shape": [32, 4], "shape_spec": dml.TensorSpec([height, 4]).shape_spec}
    assert validate_runtime_shape("x", [16, 4], spec) == (16, 4)

    with pytest.raises(ValueError, match=r"x axis 0 \(height\).*divisible by 8"):
        validate_runtime_shape("x", [10, 4], spec)


def test_runtime_shape_helpers_infer_outputs_and_check_named_dims():
    batch = dml.Dim("batch", min=1, max=4)
    x_spec = {"name": "x", "shape": [4, 16], "shape_spec": dml.TensorSpec([batch, 16]).shape_spec}
    z_spec = {"name": "z", "shape": [4, 1], "shape_spec": dml.TensorSpec([batch, 1]).shape_spec}
    y_spec = {"name": "y", "shape": [4, 16], "shape_spec": dml.TensorSpec([batch, 16]).shape_spec}

    assert infer_output_shape(y_spec, [x_spec, z_spec], {"x": [3, 16], "z": [3, 1]}) == (3, 16)
    with pytest.raises(ValueError, match="Dynamic dimension batch has inconsistent values 3 and 2"):
        infer_output_shape(y_spec, [x_spec, z_spec], {"x": [3, 16], "z": [2, 1]})


class Identity(dml.Module):
    def forward(self, x):
        return dml.ops.output(x, "y")


class DynamicRelu(dml.Module):
    def forward(self, x):
        return dml.ops.output(dml.ops.relu(x), "y")


def test_compile_accepts_dynamic_runtime_metadata(tmp_path):
    batch = dml.Dim("batch", min=1, max=4)
    spec = dml.trace(DynamicRelu(), inputs={"x": dml.TensorSpec([batch, 16])}, name="dynamic_relu")
    assert spec.ir["metadata"]["dynamic_shapes"]
    artifact = dml.compile(spec, dml.Target("cpu"), tmp_path / "dynamic_relu.dinoml")
    assert artifact.path.exists()


def test_compile_rejects_unimplemented_runtime_dtype(tmp_path):
    spec = dml.trace(Identity(), inputs={"x": dml.TensorSpec([1, 16], "float16")}, name="half_identity")
    with pytest.raises(NotImplementedError, match="cpu runtime supports dtypes"):
        dml.compile(spec, dml.Target("cpu"), tmp_path / "half_identity.dinoml")


class HalfScalar(dml.Module):
    def forward(self, x):
        return dml.ops.output(x * 0.5, "y")


def test_scalar_literals_follow_tensor_dtype():
    spec = dml.trace(HalfScalar(), inputs={"x": dml.TensorSpec([1, 16], "float16")}, name="half_scalar")
    scalar_constants = [constant for constant in spec.ir["constants"] if constant["shape"] == []]
    assert len(scalar_constants) == 1
    assert scalar_constants[0]["dtype"] == "float16"


def test_bfloat16_storage_roundtrip_uses_uint16_storage():
    values = [1.0, -2.25, 0.333]
    storage = array_to_storage(values, "bfloat16")
    assert storage.dtype.name == "uint16"
    restored = array_from_storage(storage, "bfloat16")
    assert restored.dtype.name == "float32"
    assert restored.shape == (3,)
