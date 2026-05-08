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
