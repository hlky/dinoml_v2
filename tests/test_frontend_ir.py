import sys
import struct
from types import SimpleNamespace

import pytest
import numpy as np

import dinoml as dml
from dinoml.constant_sources import GGUFConstant, MaterializedConstant, gguf_constant
from dinoml.compiler import _write_constants
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


class MaterializingConstant:
    def __init__(self, value, storage=None):
        self.value = np.asarray(value)
        self.storage = storage
        self.materialize_calls = 0

    def materialize(self, dtype, shape):
        self.materialize_calls += 1
        return MaterializedConstant(array_to_storage(self.value, dtype), self.storage)


def _gguf_constant_ir():
    return {
        "name": "gguf_constant",
        "constants": [
            {
                "name": "weight",
                "tensor": "weight",
                "shape": [2, 2],
                "shape_spec": [2, 2],
                "layout": {"kind": "dense", "order": "row_major", "strides": [2, 1], "alignment": 16},
                "dtype": "float32",
                "offset": None,
                "nbytes": 16,
                "storage": {
                    "kind": "gguf",
                    "source": {
                        "path": "weights.gguf",
                        "tensor": "blk.0.ffn.weight",
                        "quantization": "Q4_K_M",
                        "byte_offset": 4096,
                    },
                },
            }
        ],
        "metadata": {},
    }


def _gguf_string(value: str) -> bytes:
    raw = value.encode("utf-8")
    return struct.pack("<Q", len(raw)) + raw


def _write_minimal_gguf_tensor(path, *, name, gguf_shape, qtype_value, payload):
    data = bytearray()
    data += b"GGUF"
    data += struct.pack("<IQQ", 3, 1, 0)
    data += _gguf_string(name)
    data += struct.pack("<I", len(gguf_shape))
    data += struct.pack("<" + "Q" * len(gguf_shape), *gguf_shape)
    data += struct.pack("<IQ", int(qtype_value), 0)
    data += b"\0" * ((32 - len(data) % 32) % 32)
    data += payload
    path.write_bytes(data)


def test_write_constants_materializes_constant_source_and_preserves_gguf_metadata(tmp_path):
    storage = {
        "kind": "gguf",
        "path": "weights.gguf",
        "tensor": "blk.0.ffn.weight",
        "qtype": "Q4_K_M",
        "encoded_nbytes": 4096,
        "logical_dtype": "float32",
        "materialization": "dequantize_full_before_launch",
        "residency": "eager_dense_device",
    }
    value = MaterializingConstant([[1.0, 2.0], [3.0, 4.0]], storage=storage)

    lowered = _write_constants(tmp_path, _gguf_constant_ir(), {"weight": value})

    assert value.materialize_calls == 1
    assert (tmp_path / "constants.bin").read_bytes() == np.asarray(value.value, dtype=np.float32).tobytes(order="C")
    constant = lowered["constants"][0]
    assert constant["offset"] == 0
    assert constant["nbytes"] == 16
    assert constant["storage"] == storage
    assert lowered["metadata"]["constants_nbytes"] == 16


def test_write_constants_clears_source_metadata_for_dense_rebinding(tmp_path):
    lowered = _write_constants(
        tmp_path,
        _gguf_constant_ir(),
        {"weight": np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)},
    )

    assert "storage" not in lowered["constants"][0]


def test_trace_bind_constants_preserves_materialized_source_metadata():
    storage = {"kind": "gguf", "path": "weights.gguf", "tensor": "blk.0.ffn.weight", "logical_dtype": "float32"}

    class ConstantAdd(dml.Module):
        def forward(self, x):
            return dml.ops.output(dml.ops.add(x, dml.Parameter([2, 2], dtype="float32", name="weight")), "y")

    spec = dml.trace(
        ConstantAdd(),
        inputs={"x": dml.TensorSpec([2, 2])},
        constants={"weight": MaterializingConstant([[1.0, 2.0], [3.0, 4.0]], storage=storage)},
        name="gguf_source_bind",
    )

    assert spec.ir["constants"][0]["storage"] == storage
    np.testing.assert_array_equal(spec.constants["weight"], np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32))


def test_write_constants_validates_missing_and_shape_mismatched_constant_sources(tmp_path):
    missing_dir = tmp_path / "missing"
    shape_dir = tmp_path / "shape"
    missing_dir.mkdir()
    shape_dir.mkdir()

    with pytest.raises(ValueError, match="Missing constant value: weight"):
        _write_constants(missing_dir, _gguf_constant_ir(), {})

    with pytest.raises(ValueError, match=r"Constant weight has shape \(3,\), expected \(2, 2\)"):
        _write_constants(shape_dir, _gguf_constant_ir(), {"weight": MaterializingConstant([1.0, 2.0, 3.0])})


def test_gguf_constant_validates_shape_and_logical_dtype_before_importing_libgguf():
    assert isinstance(gguf_constant("weights.gguf", "blk.0.ffn.weight"), GGUFConstant)
    assert isinstance(dml.gguf_constant("weights.gguf", "blk.0.ffn.weight"), GGUFConstant)

    with pytest.raises(ValueError, match="logical dtype float16, expected float32"):
        GGUFConstant("weights.gguf", "blk.0.ffn.weight", logical_dtype="float16").materialize("float32", [2, 2])

    with pytest.raises(ValueError, match=r"has shape \(3, 2\), expected \(2, 2\)"):
        GGUFConstant("weights.gguf", "blk.0.ffn.weight", shape=[3, 2]).materialize("float32", [2, 2])


def test_gguf_constant_materializes_rows_with_explicit_shape_mapping(monkeypatch):
    values = np.arange(6, dtype=np.float32).reshape(3, 2)

    class FakeGGUFFile:
        def __init__(self):
            self.tensor_info = SimpleNamespace(qtype="F32", qtype_value=0, shape=(2, 3), data_offset=128)

        def get_tensor(self, name):
            assert name == "blk.0.ffn.weight"
            return self.tensor_info

        def read_tensor_bytes(self, tensor_info):
            assert tensor_info is self.tensor_info
            return values.tobytes(order="C")

    fake_file = FakeGGUFFile()
    fake_libgguf = SimpleNamespace(open_gguf=lambda path: fake_file)
    monkeypatch.setitem(sys.modules, "libgguf", fake_libgguf)

    materialized = GGUFConstant("weights.gguf", "blk.0.ffn.weight").materialize("float32", [3, 2])

    np.testing.assert_array_equal(materialized.array, values)
    assert materialized.storage["gguf_shape"] == [2, 3]
    assert materialized.storage["n_per_row"] == 2
    assert materialized.storage["encoded_nbytes"] == values.nbytes

    fake_file.tensor_info = SimpleNamespace(qtype="F32", qtype_value=0, shape=(4, 3), data_offset=128)
    with pytest.raises(
        ValueError,
        match=r"has logical shape \(3, 4\), expected \(3, 2\) \(stored GGUF shape \(4, 3\)\)",
    ):
        GGUFConstant("weights.gguf", "blk.0.ffn.weight").materialize("float32", [3, 2])


def test_gguf_constant_validates_observed_descriptor_hints(monkeypatch):
    values = np.arange(6, dtype=np.float32).reshape(3, 2)

    tensor_info = SimpleNamespace(qtype="F32", qtype_value=0, shape=(2, 3), data_offset=128)
    fake_file = SimpleNamespace(
        get_tensor=lambda name: tensor_info,
        read_tensor_bytes=lambda tensor: values.tobytes(order="C"),
    )
    monkeypatch.setitem(sys.modules, "libgguf", SimpleNamespace(open_gguf=lambda path: fake_file))

    with pytest.raises(ValueError, match="expected qtype F16, observed F32"):
        GGUFConstant("weights.gguf", "blk.0.ffn.weight", qtype="F16").materialize("float32", [3, 2])

    with pytest.raises(ValueError, match=f"expected {values.nbytes + 1} encoded bytes, observed {values.nbytes}"):
        GGUFConstant(
            "weights.gguf",
            "blk.0.ffn.weight",
            encoded_nbytes=values.nbytes + 1,
        ).materialize("float32", [3, 2])

    with pytest.raises(ValueError, match="expected n_per_row 4, observed 2"):
        GGUFConstant("weights.gguf", "blk.0.ffn.weight", n_per_row=4).materialize("float32", [3, 2])


def test_gguf_constant_materializes_real_libgguf_quantized_rows(tmp_path):
    libgguf = pytest.importorskip("libgguf")
    rows = np.linspace(-1.0, 1.0, 64, dtype=np.float32).reshape(2, 32)
    qtype = libgguf.GGMLQuantizationType.Q4_0
    encoded = libgguf.quantize_rows(rows, qtype)
    path = tmp_path / "weights.gguf"
    _write_minimal_gguf_tensor(
        path,
        name="blk.0.ffn.weight",
        gguf_shape=(32, 2),
        qtype_value=int(qtype),
        payload=encoded.tobytes(order="C"),
    )

    materialized = GGUFConstant(
        path,
        "blk.0.ffn.weight",
        qtype="Q4_0",
        encoded_nbytes=encoded.nbytes,
        n_per_row=32,
    ).materialize("float32", [2, 32])

    expected = libgguf.dequantize_rows(encoded, qtype, n_per_row=32).reshape(2, 32)
    np.testing.assert_array_equal(materialized.array, expected)
    assert materialized.storage["qtype"] == "Q4_0"
    assert materialized.storage["qtype_value"] == int(qtype)
    assert materialized.storage["encoded_nbytes"] == encoded.nbytes
    assert materialized.storage["gguf_shape"] == [32, 2]
    assert materialized.storage["n_per_row"] == 32


def test_frontend_emits_dense_layout_metadata():
    spec = dml.trace(Identity(), inputs={"x": dml.TensorSpec([2, 3])}, name="layout_identity")

    input_info = spec.ir["inputs"][0]
    output_info = spec.ir["outputs"][0]
    tensor_info = next(tensor for tensor in spec.ir["tensors"] if tensor["name"] == output_info["tensor"])

    assert input_info["layout"]["kind"] == "dense"
    assert input_info["layout"]["strides"] == [3, 1]
    assert output_info["layout"]["strides"] == [3, 1]
    assert tensor_info["layout"]["order"] == "row_major"


def test_tensor_spec_records_dynamic_shape_metadata():
    batch = dml.Dim("batch", min=1, max=4, typical=2)
    spec = dml.TensorSpec([batch, 16], "fp32")
    assert spec.max_shape == [4, 16]
    assert spec.dynamic
    assert spec.rank == 2
    assert spec.numel == 64
    assert spec.shape_spec[0]["name"] == "batch"


def test_shape_object_canonicalizes_dynamic_metadata():
    batch = dml.Dim("batch", min=1, max=4, divisible_by=1, typical=2, buckets=(1, 2, 4))
    shape = dml.Shape([batch, 16])

    assert shape.rank == 2
    assert shape.max_shape == [4, 16]
    assert shape.dynamic
    assert shape.numel == 64
    assert shape.constraints[0]["axis"] == 0
    assert shape.to_json()[0]["buckets"] == [1, 2, 4]
    assert shape.validate_runtime("x", [2, 16]) == (2, 16)


def test_shape_object_is_accepted_by_specs_and_parameters():
    batch = dml.Dim("batch", min=1, max=4)
    shape = dml.Shape([batch, 16])

    spec = dml.TensorSpec(shape)
    parameter = dml.Parameter(shape, name="w")

    assert spec.shape_spec[0]["name"] == "batch"
    assert parameter.shape == [4, 16]
    assert parameter.shape_spec[0]["name"] == "batch"
    assert parameter.value is None


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


class SoftmaxModule(dml.Module):
    def forward(self, x):
        return dml.ops.output(dml.ops.softmax(x, dim=-1), "y")


class ReduceSumModule(dml.Module):
    def forward(self, x):
        return dml.ops.output(dml.ops.reduce_sum(x, dim=-1), "y")


class GemmRRRModule(dml.Module):
    def forward(self, a, b):
        return dml.ops.output(dml.ops.gemm_rrr(a, b), "y")


class GemmRCRModule(dml.Module):
    def forward(self, a, b):
        return dml.ops.output(dml.ops.gemm_rcr(a, b), "y")


class GemmRRRBiasModule(dml.Module):
    def forward(self, a, b, bias):
        return dml.ops.output(dml.ops.gemm_rrr_bias(a, b, bias), "y")


class GemmRCRBiasModule(dml.Module):
    def forward(self, a, b, bias):
        return dml.ops.output(dml.ops.gemm_rcr_bias(a, b, bias), "y")


class GemmRRRBiasReluModule(dml.Module):
    def forward(self, a, b, bias):
        return dml.ops.output(dml.ops.gemm_rrr_bias_relu(a, b, bias), "y")


class GemmRCRBiasReluModule(dml.Module):
    def forward(self, a, b, bias):
        return dml.ops.output(dml.ops.gemm_rcr_bias_relu(a, b, bias), "y")


class GemmBiasOpModule(dml.Module):
    def __init__(self, op_name: str):
        self.op_name = op_name

    def forward(self, a, b, bias):
        op = getattr(dml.ops, self.op_name)
        return dml.ops.output(op(a, b, bias), "y")


class GemmResidualOpModule(dml.Module):
    def __init__(self, op_name: str):
        self.op_name = op_name

    def forward(self, a, b, bias, d0, d1=None):
        op = getattr(dml.ops, self.op_name)
        if d1 is None:
            return dml.ops.output(op(a, b, bias, d0), "y")
        return dml.ops.output(op(a, b, bias, d0, d1), "y")


class DynamicRelu(dml.Module):
    def forward(self, x):
        return dml.ops.output(dml.ops.relu(x), "y")


class ShapeViewOps(dml.Module):
    def forward(self, x, z):
        return {
            "id": dml.ops.identity(x),
            "reshaped": dml.ops.reshape(x, [3, -1]),
            "flat": dml.ops.flatten(x),
            "squeezed": dml.ops.squeeze(z),
            "unsqueezed": dml.ops.unsqueeze(x, 0),
        }


class DynamicReshape(dml.Module):
    def forward(self, x):
        return dml.ops.reshape(x, [4, 4])


class DynamicSimpleViews(dml.Module):
    def forward(self, x):
        return {
            "squeezed": dml.ops.squeeze(x, 1),
            "unsqueezed": dml.ops.unsqueeze(x, -1),
        }


def test_compile_accepts_dynamic_runtime_metadata(tmp_path):
    batch = dml.Dim("batch", min=1, max=4)
    spec = dml.trace(DynamicRelu(), inputs={"x": dml.TensorSpec([batch, 16])}, name="dynamic_relu")
    assert spec.ir["metadata"]["dynamic_shapes"]
    node_output = spec.ir["nodes"][0]["outputs"][0]
    tensor_info = next(tensor for tensor in spec.ir["tensors"] if tensor["name"] == node_output)
    assert tensor_info["shape_spec"][0]["name"] == "batch"
    artifact = dml.compile(spec, dml.Target("cpu"), tmp_path / "dynamic_relu.dinoml")
    assert artifact.path.exists()


def test_direct_input_output_is_normalized_to_shape_view_alias():
    spec = dml.trace(Identity(), inputs={"x": dml.TensorSpec([2, 3])}, name="direct_identity")

    assert spec.ir["nodes"] == []
    assert spec.ir["outputs"][0]["name"] == "y"
    assert spec.ir["outputs"][0]["tensor"] != "x"
    views = spec.ir["metadata"]["views"]["views"]
    assert views == [
        {
            "tensor": spec.ir["outputs"][0]["tensor"],
            "source": "x",
            "kind": "shape_view",
            "transform": "identity",
            "offset_elements": 0,
            "shape": [2, 3],
            "shape_spec": [2, 3],
        }
    ]


def test_shape_view_ops_emit_metadata_without_nodes():
    spec = dml.trace(
        ShapeViewOps(),
        inputs={"x": dml.TensorSpec([2, 3]), "z": dml.TensorSpec([1, 2, 1, 3])},
        name="shape_view_ops",
    )

    assert spec.ir["nodes"] == []
    views = spec.ir["metadata"]["views"]["views"]
    assert [view["transform"] for view in views] == ["identity", "reshape", "flatten", "squeeze", "unsqueeze"]
    assert all(view["source"] in {"x", "z"} for view in views)
    outputs = {output["name"]: output for output in spec.ir["outputs"]}
    assert outputs["id"]["shape"] == [2, 3]
    assert outputs["reshaped"]["shape"] == [3, 2]
    assert outputs["flat"]["shape"] == [6]
    assert outputs["squeezed"]["shape"] == [2, 3]
    assert outputs["unsqueezed"]["shape"] == [1, 2, 3]


def test_reshape_rejects_dynamic_input_shape():
    batch = dml.Dim("batch", min=1, max=4)
    with pytest.raises(NotImplementedError, match="reshape currently supports only static input shapes"):
        dml.trace(DynamicReshape(), inputs={"x": dml.TensorSpec([batch, 4])}, name="dynamic_reshape")


def test_simple_dynamic_shape_views_preserve_shape_spec():
    batch = dml.Dim("batch", min=1, max=4)
    spec = dml.trace(DynamicSimpleViews(), inputs={"x": dml.TensorSpec([batch, 1, 16])}, name="dynamic_shape_views")
    outputs = {output["name"]: output for output in spec.ir["outputs"]}
    assert outputs["squeezed"]["shape"] == [4, 16]
    assert outputs["squeezed"]["shape_spec"][0]["name"] == "batch"
    assert outputs["unsqueezed"]["shape"] == [4, 1, 16, 1]
    assert outputs["unsqueezed"]["shape_spec"][0]["name"] == "batch"


@pytest.mark.parametrize("dtype", ["float16", "bfloat16"])
def test_compile_accepts_reduced_precision_cpu_runtime_dtype(tmp_path, dtype):
    spec = dml.trace(Identity(), inputs={"x": dml.TensorSpec([1, 16], dtype)}, name=f"{dtype}_identity")
    artifact = dml.compile(spec, dml.Target("cpu"), tmp_path / f"{dtype}_identity.dinoml")
    assert artifact.path.exists()


@pytest.mark.parametrize(
    ("module", "message"),
    [
        (SoftmaxModule(), "softmax does not support dtype bfloat16"),
        (ReduceSumModule(), "reduce_sum does not support dtype bfloat16"),
    ],
)
def test_non_elementwise_ops_reject_reduced_precision_frontend(module, message):
    with pytest.raises(ValueError, match=message):
        dml.trace(module, inputs={"x": dml.TensorSpec([2, 16], "bfloat16")})


def test_gemm_frontend_emits_explicit_layout_ops():
    rrr = dml.trace(
        GemmRRRModule(),
        inputs={"a": dml.TensorSpec([4, 8]), "b": dml.TensorSpec([8, 6])},
        name="gemm_rrr_frontend",
    )
    rcr = dml.trace(
        GemmRCRModule(),
        inputs={"a": dml.TensorSpec([4, 8]), "b": dml.TensorSpec([6, 8])},
        name="gemm_rcr_frontend",
    )

    assert rrr.ir["nodes"][0]["op"] == "gemm_rrr"
    assert rrr.ir["outputs"][0]["shape"] == [4, 6]
    assert rcr.ir["nodes"][0]["op"] == "gemm_rcr"
    assert rcr.ir["outputs"][0]["shape"] == [4, 6]
    assert all(node["op"] != "matmul" for node in [*rrr.ir["nodes"], *rcr.ir["nodes"]])


@pytest.mark.parametrize(
    ("module", "b_shape", "op_name"),
    [
        (GemmRRRBiasModule(), [8, 6], "gemm_rrr_bias"),
        (GemmRCRBiasModule(), [6, 8], "gemm_rcr_bias"),
        (GemmRRRBiasReluModule(), [8, 6], "gemm_rrr_bias_relu"),
        (GemmRCRBiasReluModule(), [6, 8], "gemm_rcr_bias_relu"),
    ],
)
def test_gemm_bias_frontend_emits_epilogue_ops(module, b_shape, op_name):
    spec = dml.trace(
        module,
        inputs={"a": dml.TensorSpec([4, 8]), "b": dml.TensorSpec(b_shape), "bias": dml.TensorSpec([6])},
        name=f"{op_name}_frontend",
    )

    assert spec.ir["nodes"][0]["op"] == op_name
    assert spec.ir["outputs"][0]["shape"] == [4, 6]
    assert spec.ir["outputs"][0]["shape_spec"] == [4, 6]


@pytest.mark.parametrize("activation", ["gelu", "fast_gelu", "sigmoid", "tanh", "swish", "hardswish", "elup1"])
@pytest.mark.parametrize(("layout", "b_shape"), [("rcr", [6, 8]), ("rrr", [8, 6])])
def test_gemm_bias_activation_frontend_emits_epilogue_ops(layout, b_shape, activation):
    op_name = f"gemm_{layout}_bias_{activation}"
    spec = dml.trace(
        GemmBiasOpModule(op_name),
        inputs={"a": dml.TensorSpec([4, 8]), "b": dml.TensorSpec(b_shape), "bias": dml.TensorSpec([6])},
        name=f"{op_name}_frontend",
    )

    assert spec.ir["nodes"][0]["op"] == op_name
    assert spec.ir["outputs"][0]["shape"] == [4, 6]
    assert spec.ir["outputs"][0]["shape_spec"] == [4, 6]


@pytest.mark.parametrize(
    ("layout", "b_shape", "suffix", "epilogue_inputs"),
    [
        ("rcr", [6, 8], "add", ("bias", "d0")),
        ("rcr", [6, 8], "add_add", ("bias", "d0", "d1")),
        ("rcr", [6, 8], "add_relu", ("bias", "d0")),
        ("rcr", [6, 8], "add_add_relu", ("bias", "d0", "d1")),
        ("rcr", [6, 8], "mul", ("bias", "d0")),
        ("rcr", [6, 8], "mul_add", ("bias", "d0", "d1")),
        ("rcr", [6, 8], "mul_tanh", ("bias", "d0")),
        ("rcr", [6, 8], "sigmoid_mul", ("bias", "d0")),
        ("rcr", [6, 8], "sigmoid_mul_tanh", ("bias", "d0")),
        ("rrr", [8, 6], "add", ("bias", "d0")),
        ("rrr", [8, 6], "add_add", ("bias", "d0", "d1")),
        ("rrr", [8, 6], "mul", ("bias", "d0")),
        ("rrr", [8, 6], "mul_add", ("bias", "d0", "d1")),
    ],
)
def test_gemm_bias_residual_frontend_emits_epilogue_ops(layout, b_shape, suffix, epilogue_inputs):
    op_name = f"gemm_{layout}_bias_{suffix}"
    inputs = {
        "a": dml.TensorSpec([4, 8]),
        "b": dml.TensorSpec(b_shape),
        "bias": dml.TensorSpec([6]),
        "d0": dml.TensorSpec([4, 6]),
    }
    if "d1" in epilogue_inputs:
        inputs["d1"] = dml.TensorSpec([4, 6])

    spec = dml.trace(GemmResidualOpModule(op_name), inputs=inputs, name=f"{op_name}_frontend")

    assert spec.ir["nodes"][0]["op"] == op_name
    assert spec.ir["nodes"][0]["inputs"] == ["a", "b", *epilogue_inputs]
    assert spec.ir["outputs"][0]["shape"] == [4, 6]
    assert spec.ir["outputs"][0]["shape_spec"] == [4, 6]


@pytest.mark.parametrize("suffix", ["add", "mul", "mul_tanh", "sigmoid_mul", "sigmoid_mul_tanh"])
def test_gemm_rcr_single_residual_frontend_accepts_folded_m(suffix):
    batch = dml.Dim("batch", min=1, max=4)
    heads = dml.Dim("heads", min=1, max=3)
    tokens = dml.Dim("tokens", min=1, max=6)
    op_name = f"gemm_rcr_bias_{suffix}"
    spec = dml.trace(
        GemmResidualOpModule(op_name),
        inputs={
            "a": dml.TensorSpec([batch, heads, 8]),
            "b": dml.TensorSpec([tokens, 8]),
            "bias": dml.TensorSpec([tokens]),
            "d0": dml.TensorSpec([batch, heads, tokens]),
        },
        name=f"{op_name}_folded_m_frontend",
    )

    assert spec.ir["nodes"][0]["op"] == op_name
    assert spec.ir["outputs"][0]["shape"] == [4, 3, 6]
    assert spec.ir["outputs"][0]["shape_spec"][0]["name"] == "batch"
    assert spec.ir["outputs"][0]["shape_spec"][1]["name"] == "heads"
    assert spec.ir["outputs"][0]["shape_spec"][2]["name"] == "tokens"


@pytest.mark.parametrize("suffix", ["add_add", "mul_add", "add_add_relu"])
def test_gemm_rcr_dual_residual_frontend_accepts_folded_m(suffix):
    batch = dml.Dim("batch", min=1, max=4)
    heads = dml.Dim("heads", min=1, max=3)
    tokens = dml.Dim("tokens", min=1, max=6)
    op_name = f"gemm_rcr_bias_{suffix}"
    spec = dml.trace(
        GemmResidualOpModule(op_name),
        inputs={
            "a": dml.TensorSpec([batch, heads, 8]),
            "b": dml.TensorSpec([tokens, 8]),
            "bias": dml.TensorSpec([tokens]),
            "d0": dml.TensorSpec([batch, heads, tokens]),
            "d1": dml.TensorSpec([batch, heads, tokens]),
        },
        name=f"{op_name}_folded_m_frontend",
    )

    assert spec.ir["nodes"][0]["op"] == op_name
    assert spec.ir["outputs"][0]["shape"] == [4, 3, 6]
    assert spec.ir["outputs"][0]["shape_spec"][0]["name"] == "batch"
    assert spec.ir["outputs"][0]["shape_spec"][1]["name"] == "heads"
    assert spec.ir["outputs"][0]["shape_spec"][2]["name"] == "tokens"


@pytest.mark.parametrize("dtype", ["float16", "bfloat16"])
def test_gemm_frontend_accepts_reduced_precision_dtype(dtype):
    spec = dml.trace(
        GemmRRRModule(),
        inputs={"a": dml.TensorSpec([4, 8], dtype), "b": dml.TensorSpec([8, 6], dtype)},
        name=f"gemm_rrr_{dtype}_frontend",
    )

    assert spec.ir["nodes"][0]["op"] == "gemm_rrr"
    assert spec.ir["outputs"][0]["dtype"] == dtype


@pytest.mark.parametrize(
    ("module", "b_spec", "n_axis"),
    [
        (GemmRRRModule(), lambda tokens: [8, tokens], 1),
        (GemmRCRModule(), lambda tokens: [tokens, 8], 0),
    ],
)
def test_gemm_frontend_preserves_dynamic_mn_shape_spec(module, b_spec, n_axis):
    batch = dml.Dim("batch", min=1, max=4)
    tokens = dml.Dim("tokens", min=1, max=16)
    spec = dml.trace(
        module,
        inputs={"a": dml.TensorSpec([batch, 8]), "b": dml.TensorSpec(b_spec(tokens))},
        name="dynamic_gemm_frontend",
    )
    output = spec.ir["outputs"][0]
    assert output["shape"] == [4, 16]
    assert output["shape_spec"][0]["name"] == "batch"
    assert output["shape_spec"][1]["name"] == "tokens"
    b_input = spec.ir["inputs"][1]
    assert b_input["shape_spec"][n_axis]["name"] == "tokens"


@pytest.mark.parametrize(
    ("module", "b_shape", "runtime_b_shape"),
    [
        (GemmRRRModule(), [32, dml.Dim("tokens", min=1, max=16)], [32, 11]),
        (GemmRCRModule(), [dml.Dim("tokens", min=1, max=16), 32], [11, 32]),
    ],
)
def test_gemm_runtime_shape_inference_uses_dynamic_mn_shape_spec(module, b_shape, runtime_b_shape):
    batch = dml.Dim("batch", min=1, max=8)
    spec = dml.trace(
        module,
        inputs={"a": dml.TensorSpec([batch, 32]), "b": dml.TensorSpec(b_shape)},
        name="dynamic_gemm_shape_infer",
    )

    assert infer_output_shape(spec.ir["outputs"][0], spec.ir["inputs"], {"a": [7, 32], "b": runtime_b_shape}) == (7, 11)


@pytest.mark.parametrize(
    ("module", "b_shape", "message"),
    [
        (GemmRRRModule(), [7, 6], "gemm_rrr expected"),
        (GemmRCRModule(), [6, 7], "gemm_rcr expected"),
    ],
)
def test_gemm_frontend_rejects_incompatible_k(module, b_shape, message):
    with pytest.raises(ValueError, match=message):
        dml.trace(module, inputs={"a": dml.TensorSpec([4, 8]), "b": dml.TensorSpec(b_shape)})


@pytest.mark.parametrize(
    ("module", "b_shape", "message"),
    [
        (GemmRRRModule(), [dml.Dim("k8", min=1, max=8), dml.Dim("tokens", min=1, max=16)], "gemm_rrr expected"),
        (GemmRCRModule(), [dml.Dim("tokens", min=1, max=16), dml.Dim("k8", min=1, max=8)], "gemm_rcr expected"),
    ],
)
def test_gemm_frontend_rejects_dynamic_max_k_mismatch(module, b_shape, message):
    batch = dml.Dim("batch", min=1, max=4)
    k16 = dml.Dim("k16", min=1, max=16)
    with pytest.raises(ValueError, match=message):
        dml.trace(module, inputs={"a": dml.TensorSpec([batch, k16]), "b": dml.TensorSpec(b_shape)})


def test_gemm_bias_frontend_rejects_bad_bias_shape():
    with pytest.raises(ValueError, match="gemm_rcr_bias expected bias shape"):
        dml.trace(
            GemmRCRBiasModule(),
            inputs={"a": dml.TensorSpec([4, 8]), "b": dml.TensorSpec([6, 8]), "bias": dml.TensorSpec([5])},
        )


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
