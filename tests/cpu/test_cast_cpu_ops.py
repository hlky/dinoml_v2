import numpy as np

import dinoml as dml
from dinoml.ir import array_from_storage, array_to_storage
from dinoml.lowering.ops import render_generated_kernels
from dinoml.passes import PassManager, validate_ir
from dinoml.runtime import load


class CastModule(dml.Module):
    def __init__(self, dtype: str):
        self.dtype = dtype

    def forward(self, x):
        return dml.ops.output(dml.ops.cast(x, self.dtype), "out")


class CastRoundtripModule(dml.Module):
    def forward(self, x):
        y = dml.ops.cast(x, "float16")
        return dml.ops.output(dml.ops.cast(y, "float32"), "out")


def _trace_cast(input_dtype: str = "float32", output_dtype: str = "float16"):
    return dml.trace(
        CastModule(output_dtype),
        inputs={"x": dml.TensorSpec([2, 3], input_dtype)},
        name=f"cast_{input_dtype}_to_{output_dtype}",
    )


def test_cast_lowers_to_fused_elementwise_with_mixed_pointer_types_and_runs_cpu(tmp_path):
    spec = _trace_cast("float32", "float16")
    lowered, _ = PassManager().run(spec.ir)
    validate_ir(lowered)
    assert [node["op"] for node in lowered["nodes"]] == ["fused_elementwise"]
    fused = lowered["nodes"][0]
    assert fused["attrs"]["sub_ops"][0]["op"] == "cast"
    tensor_map = {tensor["name"]: tensor for tensor in lowered["tensors"]}

    cpu_source = render_generated_kernels("cpu", [fused], tensor_map)[0]

    assert "const float* DINO_RESTRICT ptr_x" in cpu_source
    assert "dinoml::math::float16* DINO_RESTRICT ptr_" in cpu_source
    assert "dinoml::math::cast<" in cpu_source

    artifact = dml.compile(spec, dml.Target("cpu"), tmp_path / "cast_float32_to_float16_cpu.dinoml")
    session = load(artifact.path).create_session()
    x = np.array([[-2.25, -1.0, 0.0], [1.125, 2.5, 3.75]], dtype=np.float32)
    expected = array_from_storage(array_to_storage(x, "float16"), "float16")
    try:
        actual = session.run_numpy({"x": x})["out"]
    finally:
        session.close()

    assert actual.dtype == np.float16
    np.testing.assert_array_equal(actual, expected)


def test_reduced_precision_cast_roundtrip_runs_cpu(tmp_path):
    spec = dml.trace(
        CastRoundtripModule(),
        inputs={"x": dml.TensorSpec([2, 3], "float32")},
        name="cast_float16_roundtrip",
    )
    artifact = dml.compile(spec, dml.Target("cpu"), tmp_path / "cast_float16_roundtrip_cpu.dinoml")
    session = load(artifact.path).create_session()
    x = np.array([[-2.25, -1.0, 0.0], [1.125, 2.5, 3.75]], dtype=np.float32)
    expected = array_from_storage(array_to_storage(x, "float16"), "float16").astype(np.float32)
    try:
        actual = session.run_numpy({"x": x})["out"]
    finally:
        session.close()

    assert actual.dtype == np.float32
    np.testing.assert_array_equal(actual, expected)
