from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

from dinoml.frontend import GraphBuilder, Tensor, as_tensor
from dinoml.ir import normalize_dtype
from dinoml.ops.registry import AttrDef, FrontendBinding, KernelBinding, OpDef, OpSchema, op_def


GET_TIMESTEP_EMBEDDING_DTYPES = ("float16", "float32", "bfloat16")
GET_1D_ROTARY_POS_EMBED_DTYPES = ("float16", "float32", "bfloat16")
GET_1D_ROTARY_POS_EMBED_COMPONENT_OPS = (
    "get_1d_rotary_pos_embed_cos",
    "get_1d_rotary_pos_embed_sin",
)


def infer_get_timestep_embedding(shapes: Sequence[Sequence[int]]) -> list[int]:
    raise ValueError("get_timestep_embedding shape inference requires embedding attrs")


def infer_get_timestep_embedding_with_attrs(
    input_shapes: Sequence[Sequence[int]],
    attrs: Mapping[str, Any],
) -> list[int]:
    if len(input_shapes) != 1:
        raise ValueError("get_timestep_embedding expects exactly one input")
    input_shape = list(input_shapes[0])
    if len(input_shape) != 1:
        raise ValueError(f"get_timestep_embedding expects rank-1 timesteps, got rank {len(input_shape)}")
    normalized = normalize_get_timestep_embedding_attrs(
        embedding_dim=attrs.get("embedding_dim"),
        flip_sin_to_cos=attrs.get("flip_sin_to_cos", False),
        downscale_freq_shift=attrs.get("downscale_freq_shift", 1.0),
        scale=attrs.get("scale", 1.0),
        max_period=attrs.get("max_period", 10000.0),
    )
    return [int(input_shape[0]), int(normalized["embedding_dim"])]


def normalize_get_timestep_embedding_attrs(
    *,
    embedding_dim: Any,
    flip_sin_to_cos: Any = False,
    downscale_freq_shift: Any = 1.0,
    scale: Any = 1.0,
    max_period: Any = 10000.0,
) -> dict[str, Any]:
    if not isinstance(embedding_dim, int) or isinstance(embedding_dim, bool) or embedding_dim <= 0:
        raise ValueError(f"get_timestep_embedding embedding_dim must be a positive integer, got {embedding_dim!r}")
    if not isinstance(flip_sin_to_cos, bool):
        raise ValueError(f"get_timestep_embedding flip_sin_to_cos must be bool, got {flip_sin_to_cos!r}")
    if not isinstance(downscale_freq_shift, (int, float)) or isinstance(downscale_freq_shift, bool):
        raise ValueError("get_timestep_embedding downscale_freq_shift must be finite")
    if not isinstance(scale, (int, float)) or isinstance(scale, bool):
        raise ValueError("get_timestep_embedding scale must be finite")
    if not isinstance(max_period, (int, float)) or isinstance(max_period, bool):
        raise ValueError("get_timestep_embedding max_period must be a positive finite number")

    normalized_shift = float(downscale_freq_shift)
    normalized_scale = float(scale)
    normalized_max_period = float(max_period)
    if not math.isfinite(normalized_shift):
        raise ValueError("get_timestep_embedding downscale_freq_shift must be finite")
    if not math.isfinite(normalized_scale):
        raise ValueError("get_timestep_embedding scale must be finite")
    if not math.isfinite(normalized_max_period) or normalized_max_period <= 0.0:
        raise ValueError("get_timestep_embedding max_period must be a positive finite number")

    half_dim = int(embedding_dim) // 2
    if half_dim > 0:
        denominator = float(half_dim) - normalized_shift
        if denominator == 0.0:
            raise ValueError("get_timestep_embedding requires half_dim - downscale_freq_shift to be non-zero")

    return {
        "embedding_dim": int(embedding_dim),
        "flip_sin_to_cos": bool(flip_sin_to_cos),
        "downscale_freq_shift": normalized_shift,
        "scale": normalized_scale,
        "max_period": normalized_max_period,
    }


def infer_get_1d_rotary_pos_embed_component(shapes: Sequence[Sequence[int]]) -> list[int]:
    raise ValueError("get_1d_rotary_pos_embed component shape inference requires rotary attrs")


def infer_get_1d_rotary_pos_embed_component_with_attrs(
    input_shapes: Sequence[Sequence[int]],
    attrs: Mapping[str, Any],
) -> list[int]:
    if len(input_shapes) not in {0, 1}:
        raise ValueError("get_1d_rotary_pos_embed component expects zero or one input")
    normalized = normalize_get_1d_rotary_pos_embed_attrs(
        dim=attrs.get("dim"),
        theta=attrs.get("theta", 10000.0),
        use_real=attrs.get("use_real", True),
        linear_factor=attrs.get("linear_factor", 1.0),
        ntk_factor=attrs.get("ntk_factor", 1.0),
        repeat_interleave_real=attrs.get("repeat_interleave_real", True),
        output_kind=attrs.get("output_kind"),
    )
    if not input_shapes:
        sequence_length = attrs.get("sequence_length", 0)
        if not isinstance(sequence_length, int) or isinstance(sequence_length, bool) or sequence_length <= 0:
            raise ValueError("get_1d_rotary_pos_embed integer pos must be a positive sequence length")
        return [int(sequence_length), rotary_output_cols(normalized)]
    input_shape = list(input_shapes[0])
    if len(input_shape) != 1:
        raise ValueError(f"get_1d_rotary_pos_embed component expects rank-1 pos, got rank {len(input_shape)}")
    return [int(input_shape[0]), rotary_output_cols(normalized)]


def normalize_get_1d_rotary_pos_embed_attrs(
    *,
    dim: Any,
    theta: Any = 10000.0,
    use_real: Any = True,
    linear_factor: Any = 1.0,
    ntk_factor: Any = 1.0,
    repeat_interleave_real: Any = True,
    output_kind: Any,
) -> dict[str, Any]:
    if not isinstance(dim, int) or isinstance(dim, bool) or dim <= 0:
        raise ValueError(f"get_1d_rotary_pos_embed dim must be a positive integer, got {dim!r}")
    if dim % 2 != 0:
        raise ValueError("get_1d_rotary_pos_embed requires an even dim")
    if not isinstance(use_real, bool):
        raise ValueError(f"get_1d_rotary_pos_embed use_real must be bool, got {use_real!r}")
    if not isinstance(repeat_interleave_real, bool):
        raise ValueError(
            f"get_1d_rotary_pos_embed repeat_interleave_real must be bool, got {repeat_interleave_real!r}"
        )
    if not isinstance(theta, (int, float)) or isinstance(theta, bool):
        raise ValueError("get_1d_rotary_pos_embed theta must be a positive finite number")
    if not isinstance(linear_factor, (int, float)) or isinstance(linear_factor, bool):
        raise ValueError("get_1d_rotary_pos_embed linear_factor must be a positive finite number")
    if not isinstance(ntk_factor, (int, float)) or isinstance(ntk_factor, bool):
        raise ValueError("get_1d_rotary_pos_embed ntk_factor must be a positive finite number")
    normalized_theta = float(theta)
    normalized_linear_factor = float(linear_factor)
    normalized_ntk_factor = float(ntk_factor)
    if not math.isfinite(normalized_theta) or normalized_theta <= 0.0:
        raise ValueError("get_1d_rotary_pos_embed theta must be a positive finite number")
    if not math.isfinite(normalized_linear_factor) or normalized_linear_factor <= 0.0:
        raise ValueError("get_1d_rotary_pos_embed linear_factor must be a positive finite number")
    if not math.isfinite(normalized_ntk_factor) or normalized_ntk_factor <= 0.0:
        raise ValueError("get_1d_rotary_pos_embed ntk_factor must be a positive finite number")
    if output_kind not in {"cos", "sin"}:
        raise ValueError(f"get_1d_rotary_pos_embed output_kind must be 'cos' or 'sin', got {output_kind!r}")

    return {
        "dim": int(dim),
        "theta": normalized_theta,
        "use_real": bool(use_real),
        "linear_factor": normalized_linear_factor,
        "ntk_factor": normalized_ntk_factor,
        "repeat_interleave_real": bool(repeat_interleave_real),
        "output_kind": str(output_kind),
    }


def rotary_output_cols(attrs: Mapping[str, Any]) -> int:
    return int(attrs["dim"]) if bool(attrs["use_real"]) else int(attrs["dim"]) // 2


def infer_get_1d_rotary_pos_embed_component_shape_spec(
    input_shape_spec: Sequence[Any] | None,
    attrs: Mapping[str, Any],
) -> list[Any]:
    normalized = normalize_get_1d_rotary_pos_embed_attrs(
        dim=attrs.get("dim"),
        theta=attrs.get("theta", 10000.0),
        use_real=attrs.get("use_real", True),
        linear_factor=attrs.get("linear_factor", 1.0),
        ntk_factor=attrs.get("ntk_factor", 1.0),
        repeat_interleave_real=attrs.get("repeat_interleave_real", True),
        output_kind=attrs.get("output_kind"),
    )
    if input_shape_spec is None:
        return [int(attrs.get("sequence_length", 0)), rotary_output_cols(normalized)]
    return [_copy_shape_dim(input_shape_spec[0]), rotary_output_cols(normalized)]


@op_def
class GetTimestepEmbedding(OpDef):
    name = "get_timestep_embedding"
    schema = OpSchema(
        inputs=("timesteps",),
        attrs=(
            AttrDef("embedding_dim", "int", required=True),
            AttrDef("flip_sin_to_cos", "bool", False),
            AttrDef("downscale_freq_shift", "float", 1.0),
            AttrDef("scale", "float", 1.0),
            AttrDef("max_period", "float", 10000.0),
        ),
    )
    infer_shape = infer_get_timestep_embedding
    infer_shape_with_attrs = infer_get_timestep_embedding_with_attrs
    backend_kernels = {
        "cpu": KernelBinding(
            symbol="generated_get_timestep_embedding",
            library="model",
            source_template="get_timestep_embedding_cpu.cpp.j2",
        ),
        "cuda": KernelBinding(
            symbol="generated_get_timestep_embedding",
            library="model",
            source_template="get_timestep_embedding_cuda.cu.j2",
        ),
    }
    frontend = FrontendBinding("get_timestep_embedding")
    allowed_dtypes = GET_TIMESTEP_EMBEDDING_DTYPES
    description = (
        "Diffusers/v1 sinusoidal timestep embedding for rank-1 dense float timesteps with "
        "generated CPU/CUDA kernels, fp32 internal math, odd-dimension zero padding, and "
        "optional sin/cos half flipping."
    )

    @classmethod
    def forward(
        cls,
        timesteps: Any,
        embedding_dim: int,
        flip_sin_to_cos: bool = False,
        downscale_freq_shift: float = 1.0,
        scale: float = 1.0,
        max_period: float = 10000.0,
    ) -> Tensor:
        timestep_tensor = as_tensor(timesteps, dtype_hint="float32")
        if timestep_tensor.dtype not in GET_TIMESTEP_EMBEDDING_DTYPES:
            raise ValueError(f"get_timestep_embedding does not support dtype {timestep_tensor.dtype}")
        if timestep_tensor.rank != 1:
            raise ValueError(f"get_timestep_embedding expects rank-1 timesteps, got rank {timestep_tensor.rank}")

        attrs = normalize_get_timestep_embedding_attrs(
            embedding_dim=embedding_dim,
            flip_sin_to_cos=flip_sin_to_cos,
            downscale_freq_shift=downscale_freq_shift,
            scale=scale,
            max_period=max_period,
        )
        output_shape = infer_get_timestep_embedding_with_attrs([timestep_tensor.shape], attrs)
        output_shape_spec = [_copy_shape_dim(timestep_tensor.shape_spec[0]), int(attrs["embedding_dim"])]
        return timestep_tensor.builder.emit(
            "get_timestep_embedding",
            [timestep_tensor],
            output_shape,
            timestep_tensor.dtype,
            attrs,
            shape_spec=output_shape_spec,
        )


def _rotary_component_schema(output_kind: str) -> OpSchema:
    return OpSchema(
        inputs=(),
        attrs=(
            AttrDef("dim", "int", required=True),
            AttrDef("theta", "float", 10000.0),
            AttrDef("use_real", "bool", True),
            AttrDef("linear_factor", "float", 1.0),
            AttrDef("ntk_factor", "float", 1.0),
            AttrDef("repeat_interleave_real", "bool", True),
            AttrDef("sequence_length", "int", 0),
            AttrDef("output_kind", "str", output_kind),
        ),
    )


class _Get1dRotaryPosEmbedComponent(OpDef):
    infer_shape = infer_get_1d_rotary_pos_embed_component
    infer_shape_with_attrs = infer_get_1d_rotary_pos_embed_component_with_attrs
    accepted_input_counts = (0, 1)
    backend_kernels = {
        "cpu": KernelBinding(
            symbol="generated_get_1d_rotary_pos_embed",
            library="model",
            source_template="get_1d_rotary_pos_embed_cpu.cpp.j2",
        ),
        "cuda": KernelBinding(
            symbol="generated_get_1d_rotary_pos_embed",
            library="model",
            source_template="get_1d_rotary_pos_embed_cuda.cu.j2",
        ),
    }
    allowed_dtypes = GET_1D_ROTARY_POS_EMBED_DTYPES
    description = (
        "Internal generated 1D rotary table component over rank-1 float32 positions with "
        "fp32 internal math and float16/float32/bfloat16 output storage."
    )


@op_def
class Get1dRotaryPosEmbedCos(_Get1dRotaryPosEmbedComponent):
    name = "get_1d_rotary_pos_embed_cos"
    schema = _rotary_component_schema("cos")


@op_def
class Get1dRotaryPosEmbedSin(_Get1dRotaryPosEmbedComponent):
    name = "get_1d_rotary_pos_embed_sin"
    schema = _rotary_component_schema("sin")


def get_timestep_embedding(
    timesteps: Any,
    embedding_dim: int,
    flip_sin_to_cos: bool = False,
    downscale_freq_shift: float = 1.0,
    scale: float = 1.0,
    max_period: float = 10000.0,
) -> Tensor:
    return GetTimestepEmbedding.forward(
        timesteps,
        embedding_dim,
        flip_sin_to_cos,
        downscale_freq_shift,
        scale,
        max_period,
    )


def get_1d_rotary_pos_embed(
    dim: int,
    pos: Any,
    theta: float = 10000.0,
    use_real: bool = True,
    linear_factor: float = 1.0,
    ntk_factor: float = 1.0,
    repeat_interleave_real: bool = True,
    dtype: str = "float32",
) -> tuple[Tensor, Tensor]:
    output_dtype = normalize_dtype(dtype)
    if output_dtype not in GET_1D_ROTARY_POS_EMBED_DTYPES:
        raise ValueError(f"get_1d_rotary_pos_embed does not support dtype {output_dtype}")
    normalized_attrs = normalize_get_1d_rotary_pos_embed_attrs(
        dim=dim,
        theta=theta,
        use_real=use_real,
        linear_factor=linear_factor,
        ntk_factor=ntk_factor,
        repeat_interleave_real=repeat_interleave_real,
        output_kind="cos",
    )

    sequence_length: int | None = None
    if isinstance(pos, int) and not isinstance(pos, bool):
        sequence_length = int(pos)
        if sequence_length <= 0:
            raise ValueError("get_1d_rotary_pos_embed integer pos must be a positive sequence length")
        pos_tensor = None
    else:
        pos_tensor = as_tensor(pos, dtype_hint="float32")
        if pos_tensor.dtype not in GET_1D_ROTARY_POS_EMBED_DTYPES:
            raise ValueError(f"get_1d_rotary_pos_embed does not support pos dtype {pos_tensor.dtype}")
        if pos_tensor.rank != 1:
            raise ValueError(f"get_1d_rotary_pos_embed expects rank-1 pos tensor, got rank {pos_tensor.rank}")
        if int(pos_tensor.shape[0]) <= 0:
            raise ValueError("get_1d_rotary_pos_embed pos length must be positive")
        if pos_tensor.dtype != "float32":
            from dinoml.ops.cast import cast

            pos_tensor = cast(pos_tensor, "float32")

    cos_out = emit_get_1d_rotary_pos_embed_component(
        pos_tensor,
        dim=int(normalized_attrs["dim"]),
        theta=float(normalized_attrs["theta"]),
        use_real=bool(normalized_attrs["use_real"]),
        linear_factor=float(normalized_attrs["linear_factor"]),
        ntk_factor=float(normalized_attrs["ntk_factor"]),
        repeat_interleave_real=bool(normalized_attrs["repeat_interleave_real"]),
        sequence_length=sequence_length if pos_tensor is None else None,
        output_kind="cos",
        dtype=output_dtype,
    )
    sin_out = emit_get_1d_rotary_pos_embed_component(
        pos_tensor,
        dim=int(normalized_attrs["dim"]),
        theta=float(normalized_attrs["theta"]),
        use_real=bool(normalized_attrs["use_real"]),
        linear_factor=float(normalized_attrs["linear_factor"]),
        ntk_factor=float(normalized_attrs["ntk_factor"]),
        repeat_interleave_real=bool(normalized_attrs["repeat_interleave_real"]),
        sequence_length=sequence_length if pos_tensor is None else None,
        output_kind="sin",
        dtype=output_dtype,
    )
    return cos_out, sin_out


def emit_get_1d_rotary_pos_embed_component(
    pos: Tensor | None,
    *,
    dim: int,
    theta: float = 10000.0,
    use_real: bool = True,
    linear_factor: float = 1.0,
    ntk_factor: float = 1.0,
    repeat_interleave_real: bool = True,
    sequence_length: int | None = None,
    output_kind: str,
    dtype: str = "float32",
) -> Tensor:
    attrs = normalize_get_1d_rotary_pos_embed_attrs(
        dim=dim,
        theta=theta,
        use_real=use_real,
        linear_factor=linear_factor,
        ntk_factor=ntk_factor,
        repeat_interleave_real=repeat_interleave_real,
        output_kind=output_kind,
    )
    if dtype not in GET_1D_ROTARY_POS_EMBED_DTYPES:
        raise ValueError(f"get_1d_rotary_pos_embed does not support dtype {dtype}")
    if pos is None:
        if not isinstance(sequence_length, int) or isinstance(sequence_length, bool) or sequence_length <= 0:
            raise ValueError("get_1d_rotary_pos_embed integer pos must be a positive sequence length")
        attrs["sequence_length"] = int(sequence_length)
        inputs: list[Tensor] = []
        output_shape = infer_get_1d_rotary_pos_embed_component_with_attrs([], attrs)
        output_shape_spec = infer_get_1d_rotary_pos_embed_component_shape_spec(None, attrs)
        builder = GraphBuilder.current()
    else:
        if pos.dtype != "float32":
            raise ValueError("get_1d_rotary_pos_embed component lowering requires float32 pos input")
        if pos.rank != 1:
            raise ValueError(f"get_1d_rotary_pos_embed expects rank-1 pos tensor, got rank {pos.rank}")
        attrs["sequence_length"] = 0
        inputs = [pos]
        output_shape = infer_get_1d_rotary_pos_embed_component_with_attrs([pos.shape], attrs)
        output_shape_spec = infer_get_1d_rotary_pos_embed_component_shape_spec(pos.shape_spec, attrs)
        builder = pos.builder
    op_name = "get_1d_rotary_pos_embed_cos" if output_kind == "cos" else "get_1d_rotary_pos_embed_sin"
    return builder.emit(
        op_name,
        inputs,
        output_shape,
        dtype,
        attrs,
        shape_spec=output_shape_spec,
    )


def _copy_shape_dim(dim: Any) -> Any:
    if isinstance(dim, Mapping):
        return dict(dim)
    return dim
__all__ = [
    "GET_1D_ROTARY_POS_EMBED_COMPONENT_OPS",
    "GET_1D_ROTARY_POS_EMBED_DTYPES",
    "GET_TIMESTEP_EMBEDDING_DTYPES",
    "Get1dRotaryPosEmbedCos",
    "Get1dRotaryPosEmbedSin",
    "GetTimestepEmbedding",
    "emit_get_1d_rotary_pos_embed_component",
    "get_1d_rotary_pos_embed",
    "get_timestep_embedding",
    "infer_get_1d_rotary_pos_embed_component",
    "infer_get_1d_rotary_pos_embed_component_shape_spec",
    "infer_get_1d_rotary_pos_embed_component_with_attrs",
    "infer_get_timestep_embedding",
    "infer_get_timestep_embedding_with_attrs",
    "normalize_get_1d_rotary_pos_embed_attrs",
    "normalize_get_timestep_embedding_attrs",
    "rotary_output_cols",
]
