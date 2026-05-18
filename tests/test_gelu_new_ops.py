import shutil

import numpy as np
import pytest

import dinoml as dml
from dinoml.reference import reference_numpy
from dinoml.ir import ModelSpec
from dinoml.ops.definitions import OP_REGISTRY
from dinoml.passes import PassManager, validate_ir
from dinoml.runtime import load


class GeluModule(dml.Module):
    def __init__(self, op_name: str):
        self.op_name = op_name

    def forward(self, x):
        op = getattr(dml.ops, self.op_name)
        return dml.ops.output(op(x), "out")


def _trace_gelu(op_name: str, *, dtype: str = "float32"):
    return dml.trace(
        GeluModule(op_name),
        inputs={"x": dml.TensorSpec([2, 3, 5], dtype)},
        name=f"{op_name}_{dtype}",
    )


def _reference_gelu_new(x: np.ndarray) -> np.ndarray:
    x32 = x.astype(np.float32, copy=False)
    inner = np.float32(0.7978845608028654) * (x32 + np.float32(0.044715) * x32 * x32 * x32)
    return (np.float32(0.5) * x32 * (np.float32(1.0) + np.tanh(inner))).astype(np.float32, copy=False)


def test_gelu_new_helper_matches_gelu_frontend_ir_and_lowering():
    helper = _trace_gelu("gelu_new")
    base = _trace_gelu("gelu")

    assert helper.ir["outputs"] == base.ir["outputs"]
    assert helper.ir["nodes"] == base.ir["nodes"]
    assert helper.ir["nodes"][0]["op"] == "gelu"
    assert helper.ir["nodes"][0]["attrs"] == {"approximation": "tanh"}

    lowered, _ = PassManager().run(helper.ir)
    validate_ir(lowered)
    assert [node["op"] for node in lowered["nodes"]] == ["fused_elementwise"]
    fused = lowered["nodes"][0]
    assert fused["attrs"]["sub_ops"][0]["op"] == "gelu"

    lowered_spec = ModelSpec(helper.name, lowered, helper.constants)
    x = np.linspace(-3.0, 3.0, 30, dtype=np.float32).reshape(2, 3, 5)
    expected = _reference_gelu_new(x)
    np.testing.assert_allclose(reference_numpy(lowered_spec, {"x": x})["out"], expected, atol=1e-6, rtol=1e-6)


def test_gelu_new_cpu_reference_matches_hf_tanh_formula():
    spec = _trace_gelu("gelu_new")
    x = np.linspace(-4.0, 4.0, 30, dtype=np.float32).reshape(2, 3, 5)

    actual = reference_numpy(spec, {"x": x})["out"]

    np.testing.assert_allclose(actual, _reference_gelu_new(x), atol=1e-6, rtol=1e-6)


def test_gelu_new_helper_stays_out_of_registry_and_delegates_validation():
    assert "gelu_new" not in OP_REGISTRY.frontend_names()

    with pytest.raises(ValueError, match="gelu does not support dtype bool"):
        _trace_gelu("gelu_new", dtype="bool")


def test_cuda_artifact_runs_generated_gelu_new(tmp_path):
    if shutil.which("nvcc") is None:
        pytest.skip("nvcc is required")

    spec = _trace_gelu("gelu_new")
    artifact = dml.compile(spec, dml.Target("cuda", arch="sm_86"), tmp_path / "gelu_new_cuda.dinoml")
    generated = (artifact.path / "debug" / "generated_src" / "module.cu").read_text(encoding="utf-8")
    assert "dinoml::math::gelu" in generated

    x = np.random.default_rng(42).standard_normal((2, 3, 5)).astype(np.float32)
    expected = _reference_gelu_new(x)

    module = load(artifact.path)
    session = module.create_session()
    try:
        actual = session.run_numpy({"x": x})["out"]
    finally:
        session.close()
        module.close()

    np.testing.assert_allclose(actual, expected, atol=1e-5, rtol=1e-5)
