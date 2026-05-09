import numpy as np
import pytest

import dinoml as dml
from dinoml.backends.cpu import execute_cpu
from dinoml.lowering.ops import render_generated_kernels
from dinoml.passes import PassManager, validate_ir
from dinoml.passes.validation import ValidationError
from dinoml.runtime import load


RELATIONAL_CASES = (
    ("eq", np.equal),
    ("ge", np.greater_equal),
    ("gt", np.greater),
    ("le", np.less_equal),
    ("lt", np.less),
    ("ne", np.not_equal),
)


class RelationalModule(dml.Module):
    def __init__(self, op_name: str):
        self.op_name = op_name

    def forward(self, x, y):
        return dml.ops.output(getattr(dml.ops, self.op_name)(x, y), "out")


def _trace_relational(op_name: str, dtype: str = "float32"):
    return dml.trace(
        RelationalModule(op_name),
        inputs={"x": dml.TensorSpec([2, 3], dtype), "y": dml.TensorSpec([1, 3], dtype)},
        name=f"{op_name}_{dtype}_relational",
    )


@pytest.mark.parametrize(("op_name", "_np_op"), RELATIONAL_CASES)
def test_relational_frontend_outputs_bool_with_broadcast_shape(op_name, _np_op):
    assert hasattr(dml.ops, op_name)

    spec = _trace_relational(op_name)

    assert spec.ir["outputs"][0]["shape"] == [2, 3]
    assert spec.ir["outputs"][0]["dtype"] == "bool"
    output_tensor = next(tensor for tensor in spec.ir["tensors"] if tensor["name"] == spec.ir["outputs"][0]["tensor"])
    assert output_tensor["dtype"] == "bool"
    assert output_tensor["nbytes"] == 6


@pytest.mark.parametrize(("op_name", "np_op"), RELATIONAL_CASES)
def test_cpu_reference_relational_ops_return_bool_arrays(op_name, np_op):
    spec = _trace_relational(op_name)
    x = np.array([[0.0, 1.0, 2.0], [3.0, np.nan, 5.0]], dtype=np.float32)
    y = np.array([[0.0, 2.0, 2.0]], dtype=np.float32)

    actual = execute_cpu(spec, {"x": x, "y": y})["out"]

    assert actual.dtype == np.bool_
    np.testing.assert_array_equal(actual, np_op(x, y))


def test_fused_relational_validation_rejects_float_output_dtype():
    spec = _trace_relational("gt")
    spec.ir["tensors"][-1]["dtype"] = "float32"
    spec.ir["outputs"][0]["dtype"] = "float32"

    with pytest.raises(ValidationError, match="expected bool"):
        validate_ir(spec.ir)


def test_relational_fused_cpu_and_cuda_sources_use_float_inputs_and_bool_output(tmp_path):
    spec = _trace_relational("lt")
    lowered, _ = PassManager().run(spec.ir)
    validate_ir(lowered)
    tensor_map = {tensor["name"]: tensor for tensor in lowered["tensors"]}
    fused = next(node for node in lowered["nodes"] if node["op"] == "fused_elementwise")

    cpu_source = render_generated_kernels("cpu", [fused], tensor_map)[0]
    cuda_source = render_generated_kernels("cuda", [fused], tensor_map)[0]

    assert "const float* DINO_RESTRICT ptr_x" in cpu_source
    assert "const float* DINO_RESTRICT ptr_y" in cpu_source
    assert "bool* DINO_RESTRICT ptr_" in cpu_source
    assert "const float* DINO_RESTRICT ptr_x" in cuda_source
    assert "const float* DINO_RESTRICT ptr_y" in cuda_source
    assert "bool* DINO_RESTRICT ptr_" in cuda_source

    artifact = dml.compile(spec, dml.Target("cpu"), tmp_path / "relational_cpu.dinoml")
    session = load(artifact.path).create_session()
    x = np.array([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]], dtype=np.float32)
    y = np.array([[1.0, 1.0, 6.0]], dtype=np.float32)

    actual = session.run_numpy({"x": x, "y": y})["out"]

    assert actual.dtype == np.bool_
    np.testing.assert_array_equal(actual, x < y)
