import pytest

import dinoml as dml
from dinoml.ir import array_from_storage, array_to_storage, canonical_json


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
