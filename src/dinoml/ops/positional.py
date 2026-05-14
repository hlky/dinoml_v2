from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

from dinoml.frontend import Tensor, as_tensor
from dinoml.ops.registry import AttrDef, FrontendBinding, KernelBinding, OpDef, OpRegistry, OpSchema


GET_TIMESTEP_EMBEDDING_DTYPES = ("float16", "float32", "bfloat16")


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


def register_positional_ops(registry: OpRegistry) -> None:
    registry.register(
        OpDef(
            name="get_timestep_embedding",
            schema=OpSchema(
                inputs=("timesteps",),
                attrs=(
                    AttrDef("embedding_dim", "int", required=True),
                    AttrDef("flip_sin_to_cos", "bool", False),
                    AttrDef("downscale_freq_shift", "float", 1.0),
                    AttrDef("scale", "float", 1.0),
                    AttrDef("max_period", "float", 10000.0),
                ),
            ),
            infer_shape=infer_get_timestep_embedding,
            infer_shape_with_attrs=infer_get_timestep_embedding_with_attrs,
            backend_kernels={
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
            },
            frontend=FrontendBinding("get_timestep_embedding"),
            allowed_dtypes=GET_TIMESTEP_EMBEDDING_DTYPES,
            description=(
                "Diffusers/v1 sinusoidal timestep embedding for rank-1 dense float timesteps with "
                "generated CPU/CUDA kernels, fp32 internal math, odd-dimension zero padding, and "
                "optional sin/cos half flipping."
            ),
        )
    )


def get_timestep_embedding(
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


def _copy_shape_dim(dim: Any) -> Any:
    if isinstance(dim, Mapping):
        return dict(dim)
    return dim


__all__ = [
    "GET_TIMESTEP_EMBEDDING_DTYPES",
    "get_timestep_embedding",
    "infer_get_timestep_embedding",
    "infer_get_timestep_embedding_with_attrs",
    "normalize_get_timestep_embedding_attrs",
    "register_positional_ops",
]
