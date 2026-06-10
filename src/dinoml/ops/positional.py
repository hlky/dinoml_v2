from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

from dinoml.frontend import GraphBuilder, Tensor, as_tensor
from dinoml.ir import normalize_dtype
from dinoml.ops.registry import AttrDef, FrontendBinding, KernelBinding, OpDef, OpSchema, op_def


GET_TIMESTEP_EMBEDDING_DTYPES = ("float16", "float32", "bfloat16")
GET_1D_ROTARY_POS_EMBED_DTYPES = ("float16", "float32", "bfloat16")
ROTARY_POSITIONAL_FUSION_DTYPES = GET_1D_ROTARY_POS_EMBED_DTYPES
GET_3D_ROTARY_POS_EMBED_ALLEGRO_DTYPES = (*ROTARY_POSITIONAL_FUSION_DTYPES, "int64")
GLM_OCR_ROPE_DTYPES = ("float16", "float32", "bfloat16")
GET_1D_ROTARY_POS_EMBED_COMPONENT_OPS = (
    "get_1d_rotary_pos_embed_cos",
    "get_1d_rotary_pos_embed_sin",
)
ROTARY_POSITIONAL_FUSION_OPS = (
    "get_2d_rotary_pos_embed",
    "get_2d_rotary_pos_embed_lumina",
    "get_3d_rotary_pos_embed",
    "get_3d_rotary_pos_embed_allegro",
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


def infer_get_2d_rotary_pos_embed(shapes: Sequence[Sequence[int]]) -> list[int]:
    raise ValueError("get_2d_rotary_pos_embed shape inference requires rotary attrs")


def infer_get_2d_rotary_pos_embed_with_attrs(
    input_shapes: Sequence[Sequence[int]],
    attrs: Mapping[str, Any],
) -> list[int]:
    if input_shapes:
        raise ValueError("get_2d_rotary_pos_embed expects no tensor inputs")
    normalized = normalize_get_2d_rotary_pos_embed_attrs(
        embed_dim=attrs.get("embed_dim"),
        crop_start_h=attrs.get("crop_start_h"),
        crop_start_w=attrs.get("crop_start_w"),
        crop_stop_h=attrs.get("crop_stop_h"),
        crop_stop_w=attrs.get("crop_stop_w"),
        grid_h=attrs.get("grid_h"),
        grid_w=attrs.get("grid_w"),
        theta=attrs.get("theta", 10000.0),
        use_real=attrs.get("use_real", True),
    )
    return [int(normalized["grid_h"]) * int(normalized["grid_w"]), int(normalized["embed_dim"])]


def normalize_get_2d_rotary_pos_embed_attrs(
    *,
    embed_dim: Any,
    crop_start_h: Any,
    crop_start_w: Any,
    crop_stop_h: Any,
    crop_stop_w: Any,
    grid_h: Any,
    grid_w: Any,
    theta: Any = 10000.0,
    use_real: Any = True,
) -> dict[str, Any]:
    normalized_embed_dim = _positive_int_attr(embed_dim, "get_2d_rotary_pos_embed embed_dim")
    if normalized_embed_dim % 4 != 0:
        raise ValueError("get_2d_rotary_pos_embed embed_dim must be divisible by 4")
    normalized_theta = _positive_finite_float_attr(theta, "get_2d_rotary_pos_embed theta")
    normalized_use_real = _required_true_bool_attr(use_real, "get_2d_rotary_pos_embed use_real")
    return {
        "embed_dim": normalized_embed_dim,
        "crop_start_h": _finite_float_attr(crop_start_h, "get_2d_rotary_pos_embed crop_start_h"),
        "crop_start_w": _finite_float_attr(crop_start_w, "get_2d_rotary_pos_embed crop_start_w"),
        "crop_stop_h": _finite_float_attr(crop_stop_h, "get_2d_rotary_pos_embed crop_stop_h"),
        "crop_stop_w": _finite_float_attr(crop_stop_w, "get_2d_rotary_pos_embed crop_stop_w"),
        "grid_h": _positive_int_attr(grid_h, "get_2d_rotary_pos_embed grid_h"),
        "grid_w": _positive_int_attr(grid_w, "get_2d_rotary_pos_embed grid_w"),
        "theta": normalized_theta,
        "use_real": normalized_use_real,
    }


def infer_get_2d_rotary_pos_embed_lumina(shapes: Sequence[Sequence[int]]) -> list[int]:
    raise ValueError("get_2d_rotary_pos_embed_lumina shape inference requires rotary attrs")


def infer_get_2d_rotary_pos_embed_lumina_with_attrs(
    input_shapes: Sequence[Sequence[int]],
    attrs: Mapping[str, Any],
) -> list[int]:
    if input_shapes:
        raise ValueError("get_2d_rotary_pos_embed_lumina expects no tensor inputs")
    normalized = normalize_get_2d_rotary_pos_embed_lumina_attrs(
        embed_dim=attrs.get("embed_dim"),
        len_h=attrs.get("len_h"),
        len_w=attrs.get("len_w"),
        linear_factor=attrs.get("linear_factor", 1.0),
        ntk_factor=attrs.get("ntk_factor", 1.0),
    )
    return [int(normalized["len_h"]), int(normalized["len_w"]), int(normalized["embed_dim"]) // 2]


def normalize_get_2d_rotary_pos_embed_lumina_attrs(
    *,
    embed_dim: Any,
    len_h: Any,
    len_w: Any,
    linear_factor: Any = 1.0,
    ntk_factor: Any = 1.0,
) -> dict[str, Any]:
    normalized_embed_dim = _positive_int_attr(embed_dim, "get_2d_rotary_pos_embed_lumina embed_dim")
    if normalized_embed_dim % 4 != 0:
        raise ValueError("get_2d_rotary_pos_embed_lumina embed_dim must be divisible by 4")
    return {
        "embed_dim": normalized_embed_dim,
        "len_h": _positive_int_attr(len_h, "get_2d_rotary_pos_embed_lumina len_h"),
        "len_w": _positive_int_attr(len_w, "get_2d_rotary_pos_embed_lumina len_w"),
        "linear_factor": _positive_finite_float_attr(
            linear_factor,
            "get_2d_rotary_pos_embed_lumina linear_factor",
        ),
        "ntk_factor": _positive_finite_float_attr(
            ntk_factor,
            "get_2d_rotary_pos_embed_lumina ntk_factor",
        ),
    }


def infer_get_3d_rotary_pos_embed(shapes: Sequence[Sequence[int]]) -> list[int]:
    raise ValueError("get_3d_rotary_pos_embed shape inference requires rotary attrs")


def infer_get_3d_rotary_pos_embed_with_attrs(
    input_shapes: Sequence[Sequence[int]],
    attrs: Mapping[str, Any],
) -> list[int]:
    if input_shapes:
        raise ValueError("get_3d_rotary_pos_embed expects no tensor inputs")
    normalized = normalize_get_3d_rotary_pos_embed_attrs(
        embed_dim=attrs.get("embed_dim"),
        crop_start_h=attrs.get("crop_start_h"),
        crop_start_w=attrs.get("crop_start_w"),
        crop_stop_h=attrs.get("crop_stop_h"),
        crop_stop_w=attrs.get("crop_stop_w"),
        grid_h=attrs.get("grid_h"),
        grid_w=attrs.get("grid_w"),
        temporal_size=attrs.get("temporal_size"),
        theta=attrs.get("theta", 10000.0),
        use_real=attrs.get("use_real", True),
        grid_type=attrs.get("grid_type", "linspace"),
        max_h=attrs.get("max_h", 0),
        max_w=attrs.get("max_w", 0),
    )
    return [
        int(normalized["temporal_size"]) * int(normalized["grid_h"]) * int(normalized["grid_w"]),
        int(normalized["embed_dim"]),
    ]


def normalize_get_3d_rotary_pos_embed_attrs(
    *,
    embed_dim: Any,
    crop_start_h: Any,
    crop_start_w: Any,
    crop_stop_h: Any,
    crop_stop_w: Any,
    grid_h: Any,
    grid_w: Any,
    temporal_size: Any,
    theta: Any = 10000.0,
    use_real: Any = True,
    grid_type: Any = "linspace",
    max_h: Any = 0,
    max_w: Any = 0,
) -> dict[str, Any]:
    normalized_embed_dim = _positive_int_attr(embed_dim, "get_3d_rotary_pos_embed embed_dim")
    if normalized_embed_dim % 16 != 0:
        raise ValueError("get_3d_rotary_pos_embed embed_dim must be divisible by 16")
    normalized_grid_h = _positive_int_attr(grid_h, "get_3d_rotary_pos_embed grid_h")
    normalized_grid_w = _positive_int_attr(grid_w, "get_3d_rotary_pos_embed grid_w")
    normalized_temporal = _positive_int_attr(temporal_size, "get_3d_rotary_pos_embed temporal_size")
    normalized_theta = _positive_finite_float_attr(theta, "get_3d_rotary_pos_embed theta")
    normalized_use_real = _required_true_bool_attr(use_real, "get_3d_rotary_pos_embed use_real")
    if not isinstance(grid_type, str):
        raise ValueError(f"get_3d_rotary_pos_embed grid_type must be a string, got {grid_type!r}")
    normalized_grid_type = grid_type.lower()
    if normalized_grid_type not in {"linspace", "slice"}:
        raise ValueError("get_3d_rotary_pos_embed grid_type must be 'linspace' or 'slice'")
    normalized_max_h = 0
    normalized_max_w = 0
    if normalized_grid_type == "slice":
        normalized_max_h = _positive_int_attr(max_h, "get_3d_rotary_pos_embed max_h")
        normalized_max_w = _positive_int_attr(max_w, "get_3d_rotary_pos_embed max_w")
        if normalized_grid_h > normalized_max_h or normalized_grid_w > normalized_max_w:
            raise ValueError("get_3d_rotary_pos_embed grid size must not exceed max_size in slice mode")
    return {
        "embed_dim": normalized_embed_dim,
        "crop_start_h": _finite_float_attr(crop_start_h, "get_3d_rotary_pos_embed crop_start_h"),
        "crop_start_w": _finite_float_attr(crop_start_w, "get_3d_rotary_pos_embed crop_start_w"),
        "crop_stop_h": _finite_float_attr(crop_stop_h, "get_3d_rotary_pos_embed crop_stop_h"),
        "crop_stop_w": _finite_float_attr(crop_stop_w, "get_3d_rotary_pos_embed crop_stop_w"),
        "grid_h": normalized_grid_h,
        "grid_w": normalized_grid_w,
        "temporal_size": normalized_temporal,
        "theta": normalized_theta,
        "use_real": normalized_use_real,
        "grid_type": normalized_grid_type,
        "max_h": normalized_max_h,
        "max_w": normalized_max_w,
    }


def infer_get_3d_rotary_pos_embed_allegro(shapes: Sequence[Sequence[int]]) -> list[int]:
    raise ValueError("get_3d_rotary_pos_embed_allegro shape inference requires rotary attrs")


def infer_get_3d_rotary_pos_embed_allegro_with_attrs(
    input_shapes: Sequence[Sequence[int]],
    attrs: Mapping[str, Any],
) -> list[int]:
    if input_shapes:
        raise ValueError("get_3d_rotary_pos_embed_allegro expects no tensor inputs")
    normalized = normalize_get_3d_rotary_pos_embed_allegro_attrs(
        height=attrs.get("height"),
        width=attrs.get("width"),
        num_frames=attrs.get("num_frames"),
        vae_scale_factor_spatial=attrs.get("vae_scale_factor_spatial", 8),
        patch_size=attrs.get("patch_size", 2),
        interpolation_scale_h=attrs.get("interpolation_scale_h", 2.0),
        interpolation_scale_t=attrs.get("interpolation_scale_t", 2.2),
        interpolation_scale_w=attrs.get("interpolation_scale_w", 2.0),
        attention_head_dim=attrs.get("attention_head_dim", 96),
    )
    return [int(normalized["num_frames"]), int(normalized["attention_head_dim"]) // 3]


def normalize_get_3d_rotary_pos_embed_allegro_attrs(
    *,
    height: Any,
    width: Any,
    num_frames: Any,
    vae_scale_factor_spatial: Any = 8,
    patch_size: Any = 2,
    interpolation_scale_h: Any = 2.0,
    interpolation_scale_t: Any = 2.2,
    interpolation_scale_w: Any = 2.0,
    attention_head_dim: Any = 96,
) -> dict[str, Any]:
    normalized_height = _positive_int_attr(height, "get_3d_rotary_pos_embed_allegro height")
    normalized_width = _positive_int_attr(width, "get_3d_rotary_pos_embed_allegro width")
    normalized_num_frames = _positive_int_attr(num_frames, "get_3d_rotary_pos_embed_allegro num_frames")
    normalized_vae = _positive_int_attr(
        vae_scale_factor_spatial,
        "get_3d_rotary_pos_embed_allegro vae_scale_factor_spatial",
    )
    normalized_patch = _positive_int_attr(patch_size, "get_3d_rotary_pos_embed_allegro patch_size")
    normalized_attention_head_dim = _positive_int_attr(
        attention_head_dim,
        "get_3d_rotary_pos_embed_allegro attention_head_dim",
    )
    if normalized_attention_head_dim % 3 != 0:
        raise ValueError("get_3d_rotary_pos_embed_allegro attention_head_dim must be divisible by 3")
    if (normalized_attention_head_dim // 3) % 2 != 0:
        raise ValueError("get_3d_rotary_pos_embed_allegro attention_head_dim / 3 must be even")
    grid_h = normalized_height // (normalized_vae * normalized_patch)
    grid_w = normalized_width // (normalized_vae * normalized_patch)
    if grid_h <= 0 or grid_w <= 0:
        raise ValueError("get_3d_rotary_pos_embed_allegro derived grid size must be positive")
    return {
        "height": normalized_height,
        "width": normalized_width,
        "num_frames": normalized_num_frames,
        "vae_scale_factor_spatial": normalized_vae,
        "patch_size": normalized_patch,
        "interpolation_scale_h": _positive_finite_float_attr(
            interpolation_scale_h,
            "get_3d_rotary_pos_embed_allegro interpolation_scale_h",
        ),
        "interpolation_scale_t": _positive_finite_float_attr(
            interpolation_scale_t,
            "get_3d_rotary_pos_embed_allegro interpolation_scale_t",
        ),
        "interpolation_scale_w": _positive_finite_float_attr(
            interpolation_scale_w,
            "get_3d_rotary_pos_embed_allegro interpolation_scale_w",
        ),
        "attention_head_dim": normalized_attention_head_dim,
        "grid_h": grid_h,
        "grid_w": grid_w,
    }


def normalize_glm_ocr_text_rope_attrs(*, rotary_dim: Any) -> dict[str, int]:
    if not isinstance(rotary_dim, int) or isinstance(rotary_dim, bool) or rotary_dim <= 0:
        raise ValueError(f"glm_ocr_text_rope rotary_dim must be a positive integer, got {rotary_dim!r}")
    if rotary_dim % 2 != 0:
        raise ValueError("glm_ocr_text_rope rotary_dim must be even")
    return {"rotary_dim": int(rotary_dim)}


def _validate_glm_ocr_rope_common_inputs(
    op_name: str,
    q: Tensor,
    k: Tensor,
    cos: Tensor,
    sin: Tensor,
) -> None:
    if any(tensor.builder is not q.builder for tensor in (k, cos, sin)):
        raise ValueError("Cannot combine tensors from different DinoML traces")
    if q.dtype not in GLM_OCR_ROPE_DTYPES:
        raise ValueError(f"{op_name} does not support dtype {q.dtype}")
    if k.dtype != q.dtype:
        raise ValueError(f"{op_name} q/k dtype mismatch: {q.dtype} vs {k.dtype}")
    if cos.dtype != sin.dtype:
        raise ValueError(f"{op_name} cos/sin dtype mismatch: {cos.dtype} vs {sin.dtype}")
    if cos.dtype not in GLM_OCR_ROPE_DTYPES:
        raise ValueError(f"{op_name} does not support cos/sin dtype {cos.dtype}")
    if list(cos.shape) != list(sin.shape):
        raise ValueError(f"{op_name} cos/sin shape mismatch: {cos.shape} vs {sin.shape}")


def infer_glm_ocr_text_rope_q_shape(shapes: Sequence[Sequence[int]]) -> list[int]:
    if len(shapes) != 4:
        raise ValueError(f"glm_ocr_text_rope expects 4 inputs, got {len(shapes)}")
    _validate_glm_ocr_text_rope_shapes(shapes, {"rotary_dim": list(shapes[0])[-1]})
    return list(shapes[0])


def infer_glm_ocr_text_rope_q_shape_with_attrs(
    input_shapes: Sequence[Sequence[int]],
    attrs: Mapping[str, Any],
) -> list[int]:
    _validate_glm_ocr_text_rope_shapes(input_shapes, attrs)
    return list(input_shapes[0])


def infer_glm_ocr_vision_rope_q_shape(shapes: Sequence[Sequence[int]]) -> list[int]:
    if len(shapes) != 4:
        raise ValueError(f"glm_ocr_vision_rope expects 4 inputs, got {len(shapes)}")
    _validate_glm_ocr_vision_rope_shapes(shapes)
    return list(shapes[0])


def _validate_glm_ocr_text_rope_shapes(input_shapes: Sequence[Sequence[int]], attrs: Mapping[str, Any]) -> None:
    if len(input_shapes) != 4:
        raise ValueError(f"glm_ocr_text_rope expects 4 inputs, got {len(input_shapes)}")
    q_shape, k_shape, cos_shape, sin_shape = [list(shape) for shape in input_shapes]
    if len(q_shape) != 4 or len(k_shape) != 4:
        raise ValueError("glm_ocr_text_rope expects q/k with shape [batch, seq, heads, head_dim]")
    if len(cos_shape) != 3 or len(sin_shape) != 3:
        raise ValueError("glm_ocr_text_rope expects cos/sin with shape [batch, seq, rotary_cols]")
    if q_shape[0] != k_shape[0] or q_shape[1] != k_shape[1] or q_shape[3] != k_shape[3]:
        raise ValueError("glm_ocr_text_rope q/k batch, sequence, and head_dim must match")
    if cos_shape != sin_shape:
        raise ValueError("glm_ocr_text_rope cos/sin shape mismatch")
    if cos_shape[0] != q_shape[0] or cos_shape[1] != q_shape[1]:
        raise ValueError("glm_ocr_text_rope cos/sin batch and sequence must match q/k")
    normalized = normalize_glm_ocr_text_rope_attrs(rotary_dim=attrs.get("rotary_dim"))
    rotary_dim = int(normalized["rotary_dim"])
    if rotary_dim > int(q_shape[3]):
        raise ValueError("glm_ocr_text_rope rotary_dim must not exceed head_dim")
    if int(cos_shape[2]) < rotary_dim // 2:
        raise ValueError("glm_ocr_text_rope cos/sin last dimension is smaller than rotary_dim / 2")


def _validate_glm_ocr_vision_rope_shapes(input_shapes: Sequence[Sequence[int]]) -> None:
    if len(input_shapes) != 4:
        raise ValueError(f"glm_ocr_vision_rope expects 4 inputs, got {len(input_shapes)}")
    q_shape, k_shape, cos_shape, sin_shape = [list(shape) for shape in input_shapes]
    if len(q_shape) != 3 or len(k_shape) != 3:
        raise ValueError("glm_ocr_vision_rope expects q/k with shape [seq, heads, head_dim]")
    if q_shape != k_shape:
        raise ValueError("glm_ocr_vision_rope q/k shapes must match")
    if len(cos_shape) != 2 or len(sin_shape) != 2:
        raise ValueError("glm_ocr_vision_rope expects cos/sin with shape [seq, head_dim]")
    if cos_shape != sin_shape:
        raise ValueError("glm_ocr_vision_rope cos/sin shape mismatch")
    if cos_shape[0] != q_shape[0] or cos_shape[1] != q_shape[2]:
        raise ValueError("glm_ocr_vision_rope cos/sin shape must match q/k sequence and head_dim")
    if int(q_shape[2]) % 2 != 0:
        raise ValueError("glm_ocr_vision_rope head_dim must be even")


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
            source_template="get_timestep_embedding_gpu.j2",
        ),
        "rocm": KernelBinding(
            symbol="generated_get_timestep_embedding",
            library="model",
            source_template="get_timestep_embedding_gpu.j2",
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
            source_template="get_1d_rotary_pos_embed_gpu.j2",
        ),
        "rocm": KernelBinding(
            symbol="generated_get_1d_rotary_pos_embed",
            library="model",
            source_template="get_1d_rotary_pos_embed_gpu.j2",
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


@op_def
class Get2dRotaryPosEmbed(OpDef):
    name = "get_2d_rotary_pos_embed"
    schema = OpSchema(
        inputs=(),
        attrs=(
            AttrDef("embed_dim", "int", required=True),
            AttrDef("crop_start_h", "float", required=True),
            AttrDef("crop_start_w", "float", required=True),
            AttrDef("crop_stop_h", "float", required=True),
            AttrDef("crop_stop_w", "float", required=True),
            AttrDef("grid_h", "int", required=True),
            AttrDef("grid_w", "int", required=True),
            AttrDef("theta", "float", 10000.0),
            AttrDef("use_real", "bool", True),
            AttrDef("dtype", "dtype", "float32"),
        ),
    )
    infer_shape = infer_get_2d_rotary_pos_embed
    infer_shape_with_attrs = infer_get_2d_rotary_pos_embed_with_attrs
    backend_kernels = {
        "cpu": KernelBinding("generated_get_2d_rotary_pos_embed", "model", source_template="get_2d_rotary_pos_embed_cpu.cpp.j2"),
        "cuda": KernelBinding("generated_get_2d_rotary_pos_embed", "model", source_template="get_2d_rotary_pos_embed_gpu.j2"),
        "rocm": KernelBinding("generated_get_2d_rotary_pos_embed", "model", source_template="get_2d_rotary_pos_embed_gpu.j2"),
    }
    frontend = FrontendBinding("get_2d_rotary_pos_embed")
    allowed_dtypes = ROTARY_POSITIONAL_FUSION_DTYPES
    description = "Fused 2D rotary cos/sin table generation for image grids."


@op_def
class Get2dRotaryPosEmbedLumina(OpDef):
    name = "get_2d_rotary_pos_embed_lumina"
    schema = OpSchema(
        inputs=(),
        attrs=(
            AttrDef("embed_dim", "int", required=True),
            AttrDef("len_h", "int", required=True),
            AttrDef("len_w", "int", required=True),
            AttrDef("linear_factor", "float", 1.0),
            AttrDef("ntk_factor", "float", 1.0),
            AttrDef("dtype", "dtype", "float32"),
        ),
    )
    infer_shape = infer_get_2d_rotary_pos_embed_lumina
    infer_shape_with_attrs = infer_get_2d_rotary_pos_embed_lumina_with_attrs
    backend_kernels = {
        "cpu": KernelBinding(
            "generated_get_2d_rotary_pos_embed_lumina",
            "model",
            source_template="get_2d_rotary_pos_embed_lumina_cpu.cpp.j2",
        ),
        "cuda": KernelBinding(
            "generated_get_2d_rotary_pos_embed_lumina",
            "model",
            source_template="get_2d_rotary_pos_embed_lumina_gpu.j2",
        ),
        "rocm": KernelBinding(
            "generated_get_2d_rotary_pos_embed_lumina",
            "model",
            source_template="get_2d_rotary_pos_embed_lumina_gpu.j2",
        ),
    }
    frontend = FrontendBinding("get_2d_rotary_pos_embed_lumina")
    allowed_dtypes = ROTARY_POSITIONAL_FUSION_DTYPES
    description = "Fused Lumina 2D rotary real/imag table generation."


@op_def
class Get3dRotaryPosEmbed(OpDef):
    name = "get_3d_rotary_pos_embed"
    schema = OpSchema(
        inputs=(),
        attrs=(
            AttrDef("embed_dim", "int", required=True),
            AttrDef("crop_start_h", "float", required=True),
            AttrDef("crop_start_w", "float", required=True),
            AttrDef("crop_stop_h", "float", required=True),
            AttrDef("crop_stop_w", "float", required=True),
            AttrDef("grid_h", "int", required=True),
            AttrDef("grid_w", "int", required=True),
            AttrDef("temporal_size", "int", required=True),
            AttrDef("theta", "float", 10000.0),
            AttrDef("use_real", "bool", True),
            AttrDef("grid_type", "str", "linspace"),
            AttrDef("max_h", "int", 0),
            AttrDef("max_w", "int", 0),
            AttrDef("dtype", "dtype", "float32"),
        ),
    )
    infer_shape = infer_get_3d_rotary_pos_embed
    infer_shape_with_attrs = infer_get_3d_rotary_pos_embed_with_attrs
    backend_kernels = {
        "cpu": KernelBinding("generated_get_3d_rotary_pos_embed", "model", source_template="get_3d_rotary_pos_embed_cpu.cpp.j2"),
        "cuda": KernelBinding("generated_get_3d_rotary_pos_embed", "model", source_template="get_3d_rotary_pos_embed_gpu.j2"),
        "rocm": KernelBinding("generated_get_3d_rotary_pos_embed", "model", source_template="get_3d_rotary_pos_embed_gpu.j2"),
    }
    frontend = FrontendBinding("get_3d_rotary_pos_embed")
    allowed_dtypes = ROTARY_POSITIONAL_FUSION_DTYPES
    description = "Fused 3D rotary cos/sin table generation for video grids."


@op_def
class Get3dRotaryPosEmbedAllegro(OpDef):
    name = "get_3d_rotary_pos_embed_allegro"
    schema = OpSchema(
        inputs=(),
        attrs=(
            AttrDef("height", "int", required=True),
            AttrDef("width", "int", required=True),
            AttrDef("num_frames", "int", required=True),
            AttrDef("vae_scale_factor_spatial", "int", 8),
            AttrDef("patch_size", "int", 2),
            AttrDef("interpolation_scale_h", "float", 2.0),
            AttrDef("interpolation_scale_t", "float", 2.2),
            AttrDef("interpolation_scale_w", "float", 2.0),
            AttrDef("attention_head_dim", "int", 96),
            AttrDef("dtype", "dtype", "float32"),
        ),
    )
    infer_shape = infer_get_3d_rotary_pos_embed_allegro
    infer_shape_with_attrs = infer_get_3d_rotary_pos_embed_allegro_with_attrs
    backend_kernels = {
        "cpu": KernelBinding(
            "generated_get_3d_rotary_pos_embed_allegro",
            "model",
            source_template="get_3d_rotary_pos_embed_allegro_cpu.cpp.j2",
        ),
        "cuda": KernelBinding(
            "generated_get_3d_rotary_pos_embed_allegro",
            "model",
            source_template="get_3d_rotary_pos_embed_allegro_gpu.j2",
        ),
        "rocm": KernelBinding(
            "generated_get_3d_rotary_pos_embed_allegro",
            "model",
            source_template="get_3d_rotary_pos_embed_allegro_gpu.j2",
        ),
    }
    frontend = FrontendBinding("get_3d_rotary_pos_embed_allegro")
    allowed_dtypes = GET_3D_ROTARY_POS_EMBED_ALLEGRO_DTYPES
    description = "Fused Allegro-style 3D rotary tables and integer cartesian grids."


@op_def
class GlmOcrTextRope(OpDef):
    name = "glm_ocr_text_rope"
    schema = OpSchema(
        inputs=("q", "k", "cos", "sin"),
        attrs=(AttrDef("rotary_dim", "int", required=True),),
    )
    infer_shape = infer_glm_ocr_text_rope_q_shape
    infer_shape_with_attrs = infer_glm_ocr_text_rope_q_shape_with_attrs
    backend_kernels = {
        "cpu": KernelBinding("generated_glm_ocr_text_rope", "model", source_template="glm_ocr_text_rope_cpu"),
        "cuda": KernelBinding("generated_glm_ocr_text_rope", "model", source_template="glm_ocr_text_rope_gpu"),
        "rocm": KernelBinding("generated_glm_ocr_text_rope", "model", source_template="glm_ocr_text_rope_gpu"),
    }
    frontend = FrontendBinding("glm_ocr_text_rope")
    allowed_dtypes = GLM_OCR_ROPE_DTYPES
    description = "Fused GLM-OCR text RoPE for Q/K tensors using even/odd rotation."


@op_def
class GlmOcrVisionRope(OpDef):
    name = "glm_ocr_vision_rope"
    schema = OpSchema(inputs=("q", "k", "cos", "sin"))
    infer_shape = infer_glm_ocr_vision_rope_q_shape
    backend_kernels = {
        "cpu": KernelBinding("generated_glm_ocr_vision_rope", "model", source_template="glm_ocr_vision_rope_cpu"),
        "cuda": KernelBinding("generated_glm_ocr_vision_rope", "model", source_template="glm_ocr_vision_rope_gpu"),
        "rocm": KernelBinding("generated_glm_ocr_vision_rope", "model", source_template="glm_ocr_vision_rope_gpu"),
    }
    frontend = FrontendBinding("glm_ocr_vision_rope")
    allowed_dtypes = GLM_OCR_ROPE_DTYPES
    description = "Fused GLM-OCR vision RoPE for Q/K tensors using half rotation and fp32 internal math."


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


def get_2d_rotary_pos_embed(
    embed_dim: int,
    crops_coords: Any,
    grid_size: Any,
    use_real: bool = True,
    theta: float = 10000.0,
    dtype: str = "float32",
) -> tuple[Tensor, Tensor]:
    output_dtype = normalize_dtype(dtype)
    if output_dtype not in ROTARY_POSITIONAL_FUSION_DTYPES:
        raise ValueError(f"get_2d_rotary_pos_embed does not support dtype {output_dtype}")
    crop_start, crop_stop = _normalize_rotary_crop_coords(crops_coords, "get_2d_rotary_pos_embed")
    grid_h, grid_w = _normalize_rotary_grid_size(grid_size, "get_2d_rotary_pos_embed")
    attrs = normalize_get_2d_rotary_pos_embed_attrs(
        embed_dim=embed_dim,
        crop_start_h=crop_start[0],
        crop_start_w=crop_start[1],
        crop_stop_h=crop_stop[0],
        crop_stop_w=crop_stop[1],
        grid_h=grid_h,
        grid_w=grid_w,
        theta=theta,
        use_real=use_real,
    )
    attrs["dtype"] = output_dtype
    output_shape = infer_get_2d_rotary_pos_embed_with_attrs([], attrs)
    cos_out, sin_out = GraphBuilder.current().emit_multi(
        "get_2d_rotary_pos_embed",
        [],
        (
            (output_shape, output_dtype, output_shape),
            (output_shape, output_dtype, output_shape),
        ),
        attrs,
    )
    return cos_out, sin_out


def get_2d_rotary_pos_embed_lumina(
    embed_dim: int,
    len_h: int,
    len_w: int,
    linear_factor: float = 1.0,
    ntk_factor: float = 1.0,
    dtype: str = "float32",
) -> tuple[Tensor, Tensor]:
    output_dtype = normalize_dtype(dtype)
    if output_dtype not in ROTARY_POSITIONAL_FUSION_DTYPES:
        raise ValueError(f"get_2d_rotary_pos_embed_lumina does not support dtype {output_dtype}")
    attrs = normalize_get_2d_rotary_pos_embed_lumina_attrs(
        embed_dim=embed_dim,
        len_h=len_h,
        len_w=len_w,
        linear_factor=linear_factor,
        ntk_factor=ntk_factor,
    )
    attrs["dtype"] = output_dtype
    output_shape = infer_get_2d_rotary_pos_embed_lumina_with_attrs([], attrs)
    real_out, imag_out = GraphBuilder.current().emit_multi(
        "get_2d_rotary_pos_embed_lumina",
        [],
        (
            (output_shape, output_dtype, output_shape),
            (output_shape, output_dtype, output_shape),
        ),
        attrs,
    )
    return real_out, imag_out


def get_3d_rotary_pos_embed(
    embed_dim: int,
    crops_coords: Any,
    grid_size: Any,
    temporal_size: int,
    theta: float = 10000.0,
    use_real: bool = True,
    grid_type: str = "linspace",
    max_size: Any | None = None,
    dtype: str = "float32",
) -> tuple[Tensor, Tensor]:
    output_dtype = normalize_dtype(dtype)
    if output_dtype not in ROTARY_POSITIONAL_FUSION_DTYPES:
        raise ValueError(f"get_3d_rotary_pos_embed does not support dtype {output_dtype}")
    crop_start, crop_stop = _normalize_rotary_crop_coords(crops_coords, "get_3d_rotary_pos_embed")
    grid_h, grid_w = _normalize_rotary_grid_size(grid_size, "get_3d_rotary_pos_embed")
    max_h, max_w = _normalize_optional_rotary_grid_size(max_size, "get_3d_rotary_pos_embed")
    attrs = normalize_get_3d_rotary_pos_embed_attrs(
        embed_dim=embed_dim,
        crop_start_h=crop_start[0],
        crop_start_w=crop_start[1],
        crop_stop_h=crop_stop[0],
        crop_stop_w=crop_stop[1],
        grid_h=grid_h,
        grid_w=grid_w,
        temporal_size=temporal_size,
        theta=theta,
        use_real=use_real,
        grid_type=grid_type,
        max_h=max_h,
        max_w=max_w,
    )
    attrs["dtype"] = output_dtype
    output_shape = infer_get_3d_rotary_pos_embed_with_attrs([], attrs)
    cos_out, sin_out = GraphBuilder.current().emit_multi(
        "get_3d_rotary_pos_embed",
        [],
        (
            (output_shape, output_dtype, output_shape),
            (output_shape, output_dtype, output_shape),
        ),
        attrs,
    )
    return cos_out, sin_out


def get_3d_rotary_pos_embed_allegro(
    height: int,
    width: int,
    num_frames: int,
    vae_scale_factor_spatial: int = 8,
    patch_size: int = 2,
    interpolation_scale_h: float = 2.0,
    interpolation_scale_t: float = 2.2,
    interpolation_scale_w: float = 2.0,
    attention_head_dim: int = 96,
    dtype: str = "float32",
) -> tuple[tuple[tuple[Tensor, Tensor], tuple[Tensor, Tensor], tuple[Tensor, Tensor]], tuple[Tensor, Tensor, Tensor]]:
    output_dtype = normalize_dtype(dtype)
    if output_dtype not in ROTARY_POSITIONAL_FUSION_DTYPES:
        raise ValueError(f"get_3d_rotary_pos_embed_allegro does not support dtype {output_dtype}")
    attrs = normalize_get_3d_rotary_pos_embed_allegro_attrs(
        height=height,
        width=width,
        num_frames=num_frames,
        vae_scale_factor_spatial=vae_scale_factor_spatial,
        patch_size=patch_size,
        interpolation_scale_h=interpolation_scale_h,
        interpolation_scale_t=interpolation_scale_t,
        interpolation_scale_w=interpolation_scale_w,
        attention_head_dim=attention_head_dim,
    )
    attrs["dtype"] = output_dtype
    dim_axis = int(attrs["attention_head_dim"]) // 3
    t_shape = [int(attrs["num_frames"]), dim_axis]
    h_shape = [int(attrs["grid_h"]), dim_axis]
    w_shape = [int(attrs["grid_w"]), dim_axis]
    grid_shape = [1, int(attrs["num_frames"]) * int(attrs["grid_h"]) * int(attrs["grid_w"])]
    outputs = GraphBuilder.current().emit_multi(
        "get_3d_rotary_pos_embed_allegro",
        [],
        (
            (t_shape, output_dtype, t_shape),
            (t_shape, output_dtype, t_shape),
            (h_shape, output_dtype, h_shape),
            (h_shape, output_dtype, h_shape),
            (w_shape, output_dtype, w_shape),
            (w_shape, output_dtype, w_shape),
            (grid_shape, "int64", grid_shape),
            (grid_shape, "int64", grid_shape),
            (grid_shape, "int64", grid_shape),
        ),
        attrs,
    )
    t_cos, t_sin, h_cos, h_sin, w_cos, w_sin, grid_t, grid_h, grid_w = outputs
    return ((t_cos, t_sin), (h_cos, h_sin), (w_cos, w_sin)), (grid_t, grid_h, grid_w)


def glm_ocr_text_rope(q: Any, k: Any, cos: Any, sin: Any, rotary_dim: int) -> tuple[Tensor, Tensor]:
    q_tensor = as_tensor(q, dtype_hint="float32")
    k_tensor = as_tensor(k, dtype_hint=q_tensor.dtype)
    cos_tensor = as_tensor(cos, dtype_hint=q_tensor.dtype)
    sin_tensor = as_tensor(sin, dtype_hint=cos_tensor.dtype)
    _validate_glm_ocr_rope_common_inputs("glm_ocr_text_rope", q_tensor, k_tensor, cos_tensor, sin_tensor)
    attrs = normalize_glm_ocr_text_rope_attrs(rotary_dim=rotary_dim)
    _validate_glm_ocr_text_rope_shapes(
        [q_tensor.shape, k_tensor.shape, cos_tensor.shape, sin_tensor.shape],
        attrs,
    )
    q_out, k_out = q_tensor.builder.emit_multi(
        "glm_ocr_text_rope",
        [q_tensor, k_tensor, cos_tensor, sin_tensor],
        (
            (q_tensor.shape, q_tensor.dtype, q_tensor.shape_spec),
            (k_tensor.shape, k_tensor.dtype, k_tensor.shape_spec),
        ),
        attrs,
    )
    return q_out, k_out


def glm_ocr_vision_rope(q: Any, k: Any, cos: Any, sin: Any) -> tuple[Tensor, Tensor]:
    q_tensor = as_tensor(q, dtype_hint="float32")
    k_tensor = as_tensor(k, dtype_hint=q_tensor.dtype)
    cos_tensor = as_tensor(cos, dtype_hint="float32")
    sin_tensor = as_tensor(sin, dtype_hint=cos_tensor.dtype)
    _validate_glm_ocr_rope_common_inputs("glm_ocr_vision_rope", q_tensor, k_tensor, cos_tensor, sin_tensor)
    _validate_glm_ocr_vision_rope_shapes([q_tensor.shape, k_tensor.shape, cos_tensor.shape, sin_tensor.shape])
    q_out, k_out = q_tensor.builder.emit_multi(
        "glm_ocr_vision_rope",
        [q_tensor, k_tensor, cos_tensor, sin_tensor],
        (
            (q_tensor.shape, q_tensor.dtype, q_tensor.shape_spec),
            (k_tensor.shape, k_tensor.dtype, k_tensor.shape_spec),
        ),
        {},
    )
    return q_out, k_out


def _positive_int_attr(value: Any, name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or int(value) <= 0:
        raise ValueError(f"{name} must be a positive integer, got {value!r}")
    return int(value)


def _finite_float_attr(value: Any, name: str) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ValueError(f"{name} must be a finite number, got {value!r}")
    normalized = float(value)
    if not math.isfinite(normalized):
        raise ValueError(f"{name} must be a finite number, got {value!r}")
    return normalized


def _positive_finite_float_attr(value: Any, name: str) -> float:
    normalized = _finite_float_attr(value, name)
    if normalized <= 0.0:
        raise ValueError(f"{name} must be a positive finite number, got {value!r}")
    return normalized


def _required_true_bool_attr(value: Any, name: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{name} must be bool, got {value!r}")
    if value is not True:
        raise ValueError(f"{name}=False is not supported")
    return True


def _normalize_rotary_crop_coords(value: Any, op_name: str) -> tuple[tuple[float, float], tuple[float, float]]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)) or len(value) != 2:
        raise ValueError(f"{op_name} crops_coords must be ((start_h, start_w), (stop_h, stop_w))")
    start, stop = value
    if (
        not isinstance(start, Sequence)
        or isinstance(start, (str, bytes))
        or len(start) != 2
        or not isinstance(stop, Sequence)
        or isinstance(stop, (str, bytes))
        or len(stop) != 2
    ):
        raise ValueError(f"{op_name} crops_coords must be ((start_h, start_w), (stop_h, stop_w))")
    return (
        (_finite_float_attr(start[0], f"{op_name} crop start h"), _finite_float_attr(start[1], f"{op_name} crop start w")),
        (_finite_float_attr(stop[0], f"{op_name} crop stop h"), _finite_float_attr(stop[1], f"{op_name} crop stop w")),
    )


def _normalize_rotary_grid_size(value: Any, op_name: str) -> tuple[int, int]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)) or len(value) != 2:
        raise ValueError(f"{op_name} grid_size must be a (height, width) pair")
    return (
        _positive_int_attr(value[0], f"{op_name} grid height"),
        _positive_int_attr(value[1], f"{op_name} grid width"),
    )


def _normalize_optional_rotary_grid_size(value: Any, op_name: str) -> tuple[int, int]:
    if value is None:
        return 0, 0
    return _normalize_rotary_grid_size(value, f"{op_name} max_size")


def _copy_shape_dim(dim: Any) -> Any:
    if isinstance(dim, Mapping):
        return dict(dim)
    return dim
__all__ = [
    "GET_1D_ROTARY_POS_EMBED_COMPONENT_OPS",
    "GET_1D_ROTARY_POS_EMBED_DTYPES",
    "GET_3D_ROTARY_POS_EMBED_ALLEGRO_DTYPES",
    "GET_TIMESTEP_EMBEDDING_DTYPES",
    "GLM_OCR_ROPE_DTYPES",
    "ROTARY_POSITIONAL_FUSION_DTYPES",
    "ROTARY_POSITIONAL_FUSION_OPS",
    "Get1dRotaryPosEmbedCos",
    "Get1dRotaryPosEmbedSin",
    "Get2dRotaryPosEmbed",
    "Get2dRotaryPosEmbedLumina",
    "Get3dRotaryPosEmbed",
    "Get3dRotaryPosEmbedAllegro",
    "GlmOcrTextRope",
    "GlmOcrVisionRope",
    "GetTimestepEmbedding",
    "emit_get_1d_rotary_pos_embed_component",
    "get_2d_rotary_pos_embed",
    "get_2d_rotary_pos_embed_lumina",
    "get_3d_rotary_pos_embed",
    "get_3d_rotary_pos_embed_allegro",
    "get_1d_rotary_pos_embed",
    "get_timestep_embedding",
    "glm_ocr_text_rope",
    "glm_ocr_vision_rope",
    "infer_get_2d_rotary_pos_embed",
    "infer_get_2d_rotary_pos_embed_lumina",
    "infer_get_2d_rotary_pos_embed_lumina_with_attrs",
    "infer_get_2d_rotary_pos_embed_with_attrs",
    "infer_get_3d_rotary_pos_embed",
    "infer_get_3d_rotary_pos_embed_allegro",
    "infer_get_3d_rotary_pos_embed_allegro_with_attrs",
    "infer_get_3d_rotary_pos_embed_with_attrs",
    "infer_get_1d_rotary_pos_embed_component",
    "infer_get_1d_rotary_pos_embed_component_shape_spec",
    "infer_get_1d_rotary_pos_embed_component_with_attrs",
    "infer_glm_ocr_text_rope_q_shape",
    "infer_glm_ocr_text_rope_q_shape_with_attrs",
    "infer_glm_ocr_vision_rope_q_shape",
    "infer_get_timestep_embedding",
    "infer_get_timestep_embedding_with_attrs",
    "normalize_get_2d_rotary_pos_embed_attrs",
    "normalize_get_2d_rotary_pos_embed_lumina_attrs",
    "normalize_get_3d_rotary_pos_embed_allegro_attrs",
    "normalize_get_3d_rotary_pos_embed_attrs",
    "normalize_glm_ocr_text_rope_attrs",
    "normalize_get_1d_rotary_pos_embed_attrs",
    "normalize_get_timestep_embedding_attrs",
    "rotary_output_cols",
]
