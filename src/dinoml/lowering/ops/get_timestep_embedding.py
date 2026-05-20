from __future__ import annotations

import hashlib
import math
import re
from pathlib import Path
from typing import Any, Mapping

from jinja2 import Environment, FileSystemLoader, StrictUndefined

from dinoml.lowering.target_specs import storage_type as target_storage_type
from dinoml.lowering.ops.base import OpLowering
from dinoml.lowering.ops.template_rendering import supported_target_spec
from dinoml.ops.positional import GET_TIMESTEP_EMBEDDING_DTYPES, normalize_get_timestep_embedding_attrs
from dinoml.lowering.shape_buffers import c_ident as _c_ident


def render_generated_kernel(target: str, node: Mapping[str, Any], tensor_map: Mapping[str, Mapping[str, Any]]) -> str:
    spec = supported_target_spec(target, "get_timestep_embedding")
    context = _context(target, node, tensor_map)
    if not spec.is_gpu:
        return _render_template("get_timestep_embedding_cpu.cpp.j2", context)
    context.update(spec.gpu_template_context())
    return _render_template("get_timestep_embedding_gpu.j2", context)


def render_launch(
    target: str,
    node: Mapping[str, Any],
    tensor_map: Mapping[str, Mapping[str, Any]],
    kernel_manifest: Mapping[str, Any] | None = None,
) -> str:
    del kernel_manifest
    spec = supported_target_spec(target, "get_timestep_embedding")
    func = _function_name(node, tensor_map)
    timesteps = _c_ident(node["inputs"][0])
    out = _c_ident(node["outputs"][0])
    args = f"ptr_{timesteps}, ptr_{out}, runtime_numel_{timesteps}, runtime_numel_{out}"
    if not spec.is_gpu:
        return f"if (int err = {func}({args})) return err;"
    return f"if (int err = {func}({args}, {spec.stream_expr})) return err;"


def source_key(target: str, node: Mapping[str, Any], tensor_map: Mapping[str, Mapping[str, Any]]) -> str:
    supported_target_spec(target, "get_timestep_embedding")
    return f"{target}:{_function_name(node, tensor_map)}"


def generated_function_name(target: str, node: Mapping[str, Any], tensor_map: Mapping[str, Mapping[str, Any]]) -> str:
    del target
    return _function_name(node, tensor_map)


def _context(target: str, node: Mapping[str, Any], tensor_map: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    input_tensor = tensor_map[node["inputs"][0]]
    output_tensor = tensor_map[node["outputs"][0]]
    normalized_attrs = _validate_node_contract(node, input_tensor, output_tensor)
    dtype = str(input_tensor["dtype"])
    storage_type = target_storage_type(dtype, target)
    embedding_dim = int(normalized_attrs["embedding_dim"])
    half_dim = embedding_dim // 2
    denominator = 1.0 if half_dim == 0 else (float(half_dim) - float(normalized_attrs["downscale_freq_shift"]))
    return {
        "func": _function_name(node, tensor_map),
        "kernel": f"{_function_name(node, tensor_map)}_kernel",
        "storage_type": storage_type,
        "embedding_dim": embedding_dim,
        "half_dim": half_dim,
        "odd_embedding_dim": embedding_dim % 2 == 1,
        "flip_sin_to_cos": bool(normalized_attrs["flip_sin_to_cos"]),
        "denominator_literal": _float_literal(denominator),
        "neg_log_max_period_literal": _float_literal(-math.log(float(normalized_attrs["max_period"]))),
        "scale_literal": _float_literal(float(normalized_attrs["scale"])),
        "zero_literal": _float_literal(0.0),
        "block_size": 256,
    }


def _validate_node_contract(
    node: Mapping[str, Any],
    input_tensor: Mapping[str, Any],
    output_tensor: Mapping[str, Any],
) -> Mapping[str, Any]:
    if str(node["op"]) != "get_timestep_embedding":
        raise ValueError(f"Unsupported positional op: {node['op']}")
    if len(node.get("inputs", ())) != 1 or len(node.get("outputs", ())) != 1:
        raise ValueError("get_timestep_embedding expects exactly one input and one output")
    dtype = str(input_tensor["dtype"])
    if dtype != str(output_tensor["dtype"]):
        raise NotImplementedError("get_timestep_embedding lowering requires matching input/output dtypes")
    if dtype not in GET_TIMESTEP_EMBEDDING_DTYPES:
        raise NotImplementedError(
            "get_timestep_embedding lowering supports float16, float32, and bfloat16 tensors only"
        )
    if len(input_tensor["shape"]) != 1:
        raise ValueError("get_timestep_embedding lowering requires rank-1 timesteps")
    if len(output_tensor["shape"]) != 2:
        raise ValueError("get_timestep_embedding lowering requires rank-2 output")
    normalized_attrs = normalize_get_timestep_embedding_attrs(
        embedding_dim=node.get("attrs", {}).get("embedding_dim"),
        flip_sin_to_cos=node.get("attrs", {}).get("flip_sin_to_cos", False),
        downscale_freq_shift=node.get("attrs", {}).get("downscale_freq_shift", 1.0),
        scale=node.get("attrs", {}).get("scale", 1.0),
        max_period=node.get("attrs", {}).get("max_period", 10000.0),
    )
    if int(output_tensor["shape"][0]) != int(input_tensor["shape"][0]):
        raise ValueError("get_timestep_embedding output batch must match the timesteps length")
    if int(output_tensor["shape"][1]) != int(normalized_attrs["embedding_dim"]):
        raise ValueError("get_timestep_embedding output embedding dim must match attrs")
    return normalized_attrs


def _function_name(node: Mapping[str, Any], tensor_map: Mapping[str, Mapping[str, Any]]) -> str:
    input_tensor = tensor_map[node["inputs"][0]]
    normalized_attrs = normalize_get_timestep_embedding_attrs(
        embedding_dim=node.get("attrs", {}).get("embedding_dim"),
        flip_sin_to_cos=node.get("attrs", {}).get("flip_sin_to_cos", False),
        downscale_freq_shift=node.get("attrs", {}).get("downscale_freq_shift", 1.0),
        scale=node.get("attrs", {}).get("scale", 1.0),
        max_period=node.get("attrs", {}).get("max_period", 10000.0),
    )
    signature = {
        "op": "get_timestep_embedding",
        "dtype": str(input_tensor["dtype"]),
        "embedding_dim": int(normalized_attrs["embedding_dim"]),
        "flip_sin_to_cos": bool(normalized_attrs["flip_sin_to_cos"]),
        "downscale_freq_shift": float(normalized_attrs["downscale_freq_shift"]),
        "scale": float(normalized_attrs["scale"]),
        "max_period": float(normalized_attrs["max_period"]),
    }
    digest = hashlib.sha256(repr(signature).encode("utf-8")).hexdigest()[:12]
    return f"get_timestep_embedding_{digest}"


def _float_literal(value: float) -> str:
    if math.isnan(value) or math.isinf(value):
        raise ValueError("get_timestep_embedding lowering supports only finite attrs")
    literal = f"{value:.9g}"
    if "." not in literal and "e" not in literal and "E" not in literal:
        literal = f"{literal}.0"
    return f"{literal}f"


def _render_template(name: str, context: Mapping[str, Any]) -> str:
    env = Environment(
        loader=FileSystemLoader(str(Path(__file__).resolve().parent / "templates")),
        undefined=StrictUndefined,
        trim_blocks=True,
        lstrip_blocks=True,
        keep_trailing_newline=True,
    )
    return env.get_template(name).render(**context)


GET_TIMESTEP_EMBEDDING_LOWERING = OpLowering(
    op_name="get_timestep_embedding",
    render_generated_kernel=render_generated_kernel,
    render_launch=render_launch,
    source_key=source_key,
    generated_function_name=generated_function_name,
)
