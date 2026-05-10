import numpy as np
import pytest

import dinoml as dml
from dinoml.backends.cpu import execute_cpu
from dinoml.ir import ModelSpec
from dinoml.ops.definitions import OP_REGISTRY
from dinoml.passes import PassManager, validate_ir


class ConcatenateHelperModule(dml.Module):
    def __init__(self, op_name, dim=0):
        self.op_name = op_name
        self.dim = dim

    def forward(self, x, y):
        op = getattr(dml.ops, self.op_name)
        return dml.ops.output(op([x, y], dim=self.dim), "out")


class ExpandHelperModule(dml.Module):
    def __init__(self, op_name, shape):
        self.op_name = op_name
        self.shape = shape

    def forward(self, x):
        op = getattr(dml.ops, self.op_name)
        return dml.ops.output(op(x, self.shape), "out")


def _trace_concat(op_name="concatenate_fast", dim=-1):
    return dml.trace(
        ConcatenateHelperModule(op_name, dim=dim),
        inputs={
            "x": dml.TensorSpec([2, 3, 1], "float32"),
            "y": dml.TensorSpec([2, 3, 2], "float32"),
        },
        name=f"{op_name}_helper",
    )


def _trace_expand(op_name="expand_static_shape", shape=(2, 3)):
    return dml.trace(
        ExpandHelperModule(op_name, shape),
        inputs={"x": dml.TensorSpec([1, 3], "float32")},
        name=f"{op_name}_helper",
    )


def test_concatenate_fast_matches_concatenate_frontend_ir_and_runtime():
    helper = _trace_concat("concatenate_fast")
    base = _trace_concat("concatenate")

    assert helper.ir["outputs"][0]["shape"] == base.ir["outputs"][0]["shape"] == [2, 3, 3]
    assert helper.ir["outputs"][0]["dtype"] == base.ir["outputs"][0]["dtype"] == "float32"
    assert helper.ir["nodes"] == base.ir["nodes"]
    assert helper.ir["nodes"][0]["op"] == "concatenate"
    assert helper.ir["nodes"][0]["attrs"] == {"dim": 2}

    inputs = {
        "x": np.arange(6, dtype=np.float32).reshape(2, 3, 1),
        "y": (10 + np.arange(12, dtype=np.float32)).reshape(2, 3, 2),
    }
    expected = np.concatenate([inputs["x"], inputs["y"]], axis=2)
    np.testing.assert_array_equal(execute_cpu(helper, inputs)["out"], expected)


def test_concatenate_tanh_composes_concatenate_then_elementwise_tanh_and_lowers_to_fused():
    spec = _trace_concat("concatenate_tanh", dim=2)

    assert spec.ir["outputs"][0]["shape"] == [2, 3, 3]
    assert spec.ir["outputs"][0]["dtype"] == "float32"
    assert [node["op"] for node in spec.ir["nodes"]] == ["concatenate", "tanh"]
    assert spec.ir["nodes"][0]["attrs"] == {"dim": 2}
    assert spec.ir["nodes"][1]["inputs"] == spec.ir["nodes"][0]["outputs"]

    lowered, _ = PassManager().run(spec.ir)
    validate_ir(lowered)
    assert [node["op"] for node in lowered["nodes"]] == ["concatenate", "fused_elementwise"]
    fused = lowered["nodes"][1]
    assert [sub_op["op"] for sub_op in fused["attrs"]["sub_ops"]] == ["tanh"]
    assert fused["attrs"]["sub_ops"][0]["inputs"] == lowered["nodes"][0]["outputs"]
    assert {"concatenate_tanh", "concatenate_fast", "expand_static_shape"}.isdisjoint(OP_REGISTRY.frontend_names())
    assert {node["op"] for node in lowered["nodes"]} == {"concatenate", "fused_elementwise"}

    inputs = {
        "x": np.linspace(-1.0, 0.0, 6, dtype=np.float32).reshape(2, 3, 1),
        "y": np.linspace(0.25, 1.25, 6, dtype=np.float32).reshape(2, 3, 1),
    }
    lowered_spec = ModelSpec(spec.name, lowered, spec.constants)
    expected = np.tanh(np.concatenate([inputs["x"], inputs["y"]], axis=2)).astype(np.float32)
    np.testing.assert_allclose(execute_cpu(lowered_spec, inputs)["out"], expected, atol=1e-6, rtol=1e-6)


def test_expand_static_shape_matches_expand_frontend_ir_and_runtime():
    helper = _trace_expand("expand_static_shape", (2, -1))
    base = _trace_expand("expand", (2, -1))

    assert helper.ir["outputs"][0]["shape"] == base.ir["outputs"][0]["shape"] == [2, 3]
    assert helper.ir["outputs"][0]["dtype"] == base.ir["outputs"][0]["dtype"] == "float32"
    assert helper.ir["nodes"] == base.ir["nodes"]
    assert helper.ir["nodes"][0]["op"] == "expand"
    assert helper.ir["nodes"][0]["attrs"] == {"shape": [2, -1]}

    x = np.array([[0.0, 1.0, 2.0]], dtype=np.float32)
    expected = np.broadcast_to(x, [2, 3]).copy()
    np.testing.assert_array_equal(execute_cpu(helper, {"x": x})["out"], expected)


def test_helper_errors_delegate_to_existing_validation():
    with pytest.raises(ValueError, match="non-empty sequence"):
        dml.ops.concatenate_fast([], dim=0)
    with pytest.raises(ValueError, match="positive or -1"):
        _trace_expand("expand_static_shape", (2, 0))

    with pytest.raises(ValueError, match="tanh does not support dtype bool"):
        dml.trace(
            ConcatenateHelperModule("concatenate_tanh", dim=0),
            inputs={"x": dml.TensorSpec([1, 2], "bool"), "y": dml.TensorSpec([1, 2], "bool")},
            name="concatenate_tanh_bool_reject",
        )
