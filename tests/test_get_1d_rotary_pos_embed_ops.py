import numpy as np
import pytest

import dinoml as dml
from dinoml.backends.cpu import execute_cpu
from dinoml.ir import array_from_storage, array_to_storage
from dinoml.ops.definitions import OP_REGISTRY
from dinoml.passes import PassManager, validate_ir


class RotaryIntModule(dml.Module):
    def __init__(
        self,
        *,
        dim: int,
        pos: int,
        theta: float = 10000.0,
        linear_factor: float = 1.0,
        ntk_factor: float = 1.0,
        repeat_interleave_real: bool = True,
        dtype: str = "float32",
    ):
        self.dim = dim
        self.pos = pos
        self.theta = theta
        self.linear_factor = linear_factor
        self.ntk_factor = ntk_factor
        self.repeat_interleave_real = repeat_interleave_real
        self.dtype = dtype

    def forward(self):
        cos_part, sin_part = dml.ops.get_1d_rotary_pos_embed(
            self.dim,
            self.pos,
            theta=self.theta,
            linear_factor=self.linear_factor,
            ntk_factor=self.ntk_factor,
            repeat_interleave_real=self.repeat_interleave_real,
            dtype=self.dtype,
        )
        return dml.ops.output(cos_part, "cos"), dml.ops.output(sin_part, "sin")


class RotaryTensorModule(dml.Module):
    def __init__(
        self,
        *,
        dim: int,
        theta: float = 10000.0,
        linear_factor: float = 1.0,
        ntk_factor: float = 1.0,
        repeat_interleave_real: bool = True,
        dtype: str = "float32",
    ):
        self.dim = dim
        self.theta = theta
        self.linear_factor = linear_factor
        self.ntk_factor = ntk_factor
        self.repeat_interleave_real = repeat_interleave_real
        self.dtype = dtype

    def forward(self, pos):
        cos_part, sin_part = dml.ops.get_1d_rotary_pos_embed(
            self.dim,
            pos,
            theta=self.theta,
            linear_factor=self.linear_factor,
            ntk_factor=self.ntk_factor,
            repeat_interleave_real=self.repeat_interleave_real,
            dtype=self.dtype,
        )
        return dml.ops.output(cos_part, "cos"), dml.ops.output(sin_part, "sin")


def _trace_rotary_int(
    *,
    dim: int = 8,
    pos: int = 4,
    theta: float = 10000.0,
    linear_factor: float = 1.0,
    ntk_factor: float = 1.0,
    repeat_interleave_real: bool = True,
    dtype: str = "float32",
):
    return dml.trace(
        RotaryIntModule(
            dim=dim,
            pos=pos,
            theta=theta,
            linear_factor=linear_factor,
            ntk_factor=ntk_factor,
            repeat_interleave_real=repeat_interleave_real,
            dtype=dtype,
        ),
        inputs={},
        name=f"get_1d_rotary_pos_embed_int_{dtype}_{dim}",
    )


def _trace_rotary_tensor(
    *,
    dim: int = 8,
    pos_shape=(4,),
    pos_dtype: str = "float32",
    theta: float = 10000.0,
    linear_factor: float = 1.0,
    ntk_factor: float = 1.0,
    repeat_interleave_real: bool = True,
    dtype: str = "float32",
):
    return dml.trace(
        RotaryTensorModule(
            dim=dim,
            theta=theta,
            linear_factor=linear_factor,
            ntk_factor=ntk_factor,
            repeat_interleave_real=repeat_interleave_real,
            dtype=dtype,
        ),
        inputs={"pos": dml.TensorSpec(pos_shape, pos_dtype)},
        name=f"get_1d_rotary_pos_embed_tensor_{dtype}_{dim}",
    )


def _storage_roundtrip(value, dtype: str) -> np.ndarray:
    if dtype == "float32":
        return np.asarray(value, dtype=np.float32)
    return array_from_storage(array_to_storage(np.asarray(value, dtype=np.float32), dtype), dtype)


def _reference_get_1d_rotary_pos_embed(
    *,
    dim: int,
    positions: np.ndarray,
    theta: float,
    linear_factor: float,
    ntk_factor: float,
    repeat_interleave_real: bool,
    dtype: str,
) -> tuple[np.ndarray, np.ndarray]:
    positions_fp32 = positions.astype(np.float32, copy=False)
    theta_scaled = np.float32(theta * ntk_factor)
    exponents = np.arange(0, dim, 2, dtype=np.float32) / np.float32(dim)
    inv_freqs = (1.0 / (np.power(theta_scaled, exponents) * np.float32(linear_factor))).astype(np.float32, copy=False)
    freqs = positions_fp32[:, None] * inv_freqs[None, :]
    cos_base = np.cos(freqs).astype(np.float32, copy=False)
    sin_base = np.sin(freqs).astype(np.float32, copy=False)
    if repeat_interleave_real:
        cos_part = np.repeat(cos_base, 2, axis=1)
        sin_part = np.repeat(sin_base, 2, axis=1)
    else:
        cos_part = np.concatenate([cos_base, cos_base], axis=1)
        sin_part = np.concatenate([sin_base, sin_base], axis=1)
    return _storage_roundtrip(cos_part, dtype), _storage_roundtrip(sin_part, dtype)


def test_get_1d_rotary_pos_embed_helper_stays_out_of_registry_and_composes_existing_ops():
    spec = _trace_rotary_int(
        dim=8,
        pos=5,
        theta=4096.0,
        linear_factor=1.25,
        ntk_factor=1.1,
        repeat_interleave_real=True,
        dtype="float16",
    )

    assert "get_1d_rotary_pos_embed" not in OP_REGISTRY.frontend_names()
    assert [output["shape"] for output in spec.ir["outputs"]] == [[5, 8], [5, 8]]
    assert [output["dtype"] for output in spec.ir["outputs"]] == ["float16", "float16"]
    assert all(node["op"] != "get_1d_rotary_pos_embed" for node in spec.ir["nodes"])
    assert {"arange", "mul", "cos", "sin", "repeat_interleave", "cast"}.issubset(
        {node["op"] for node in spec.ir["nodes"]}
    )

    lowered, _ = PassManager().run(spec.ir)
    validate_ir(lowered)
    lowered_ops = [node["op"] for node in lowered["nodes"]]
    assert "get_1d_rotary_pos_embed" not in lowered_ops
    assert "repeat_interleave" in lowered_ops
    assert lowered_ops.count("fused_elementwise") >= 2


@pytest.mark.parametrize(
    ("dtype", "repeat_interleave_real", "theta", "linear_factor", "ntk_factor", "atol", "rtol"),
    [
        ("float32", True, 10000.0, 1.0, 1.0, 1e-6, 1e-6),
        ("float32", False, 4096.0, 2.0, 1.5, 1e-6, 1e-6),
        ("float16", True, 1000.0, 1.25, 1.1, 2e-3, 2e-3),
        ("bfloat16", False, 256.0, 0.75, 1.25, 2e-2, 2e-2),
    ],
)
def test_cpu_reference_get_1d_rotary_pos_embed_matches_formula_for_static_int_positions(
    dtype,
    repeat_interleave_real,
    theta,
    linear_factor,
    ntk_factor,
    atol,
    rtol,
):
    spec = _trace_rotary_int(
        dim=8,
        pos=4,
        theta=theta,
        linear_factor=linear_factor,
        ntk_factor=ntk_factor,
        repeat_interleave_real=repeat_interleave_real,
        dtype=dtype,
    )
    positions = np.arange(4, dtype=np.float32)
    expected_cos, expected_sin = _reference_get_1d_rotary_pos_embed(
        dim=8,
        positions=positions,
        theta=theta,
        linear_factor=linear_factor,
        ntk_factor=ntk_factor,
        repeat_interleave_real=repeat_interleave_real,
        dtype=dtype,
    )

    actual = execute_cpu(spec, {})

    assert actual["cos"].dtype == expected_cos.dtype
    assert actual["sin"].dtype == expected_sin.dtype
    np.testing.assert_allclose(actual["cos"].astype(np.float32), expected_cos.astype(np.float32), atol=atol, rtol=rtol)
    np.testing.assert_allclose(actual["sin"].astype(np.float32), expected_sin.astype(np.float32), atol=atol, rtol=rtol)


def test_cpu_reference_get_1d_rotary_pos_embed_matches_formula_for_static_tensor_positions():
    spec = _trace_rotary_tensor(
        dim=6,
        pos_shape=(3,),
        pos_dtype="float16",
        theta=64.0,
        linear_factor=1.5,
        ntk_factor=1.25,
        repeat_interleave_real=False,
        dtype="float32",
    )
    positions = np.array([0.0, 1.5, 3.25], dtype=np.float32)
    expected_cos, expected_sin = _reference_get_1d_rotary_pos_embed(
        dim=6,
        positions=positions,
        theta=64.0,
        linear_factor=1.5,
        ntk_factor=1.25,
        repeat_interleave_real=False,
        dtype="float32",
    )

    actual = execute_cpu(spec, {"pos": positions})

    np.testing.assert_allclose(actual["cos"], expected_cos, atol=1e-6, rtol=1e-6)
    np.testing.assert_allclose(actual["sin"], expected_sin, atol=1e-6, rtol=1e-6)


def test_get_1d_rotary_pos_embed_frontend_rejects_unsupported_inputs():
    dynamic_n = dml.Dim("n", min=1, max=4)

    with pytest.raises(ValueError, match="positive integer"):
        _trace_rotary_int(dim=0)
    with pytest.raises(ValueError, match="even dim"):
        _trace_rotary_int(dim=7)
    with pytest.raises(ValueError, match="supports only use_real=True"):
        class UseRealFalseModule(dml.Module):
            def forward(self):
                return dml.ops.get_1d_rotary_pos_embed(8, 4, use_real=False)

        dml.trace(UseRealFalseModule(), inputs={})
    with pytest.raises(ValueError, match="does not support dtype bool"):
        _trace_rotary_int(dtype="bool")
    with pytest.raises(ValueError, match="positive sequence length"):
        _trace_rotary_int(pos=0)
    with pytest.raises(ValueError, match="static pos length"):
        dml.trace(
            RotaryTensorModule(dim=8),
            inputs={"pos": dml.TensorSpec([dynamic_n], "float32")},
            name="get_1d_rotary_pos_embed_dynamic",
        )
    with pytest.raises(ValueError, match="rank-1 pos tensor"):
        _trace_rotary_tensor(pos_shape=(2, 2))
    with pytest.raises(ValueError, match="does not support pos dtype bool"):
        _trace_rotary_tensor(pos_dtype="bool")
    with pytest.raises(ValueError, match="theta must be a positive finite number"):
        _trace_rotary_int(theta=float("inf"))
    with pytest.raises(ValueError, match="linear_factor must be a positive finite number"):
        _trace_rotary_int(linear_factor=0.0)
    with pytest.raises(ValueError, match="ntk_factor must be a positive finite number"):
        _trace_rotary_int(ntk_factor=-1.0)
