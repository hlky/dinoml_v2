from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Any, Mapping

from jinja2 import Environment, FileSystemLoader, StrictUndefined

from dinoml.lowering.cpp_types import cpu_storage_type, cuda_storage_type
from dinoml.lowering.ops.base import OpLowering
from dinoml.ops.normalization import T5_LAYER_NORM_DTYPES
from dinoml.lowering.shape_buffers import c_ident as _c_ident


def render_generated_kernel(target: str, node: Mapping[str, Any], tensor_map: Mapping[str, Mapping[str, Any]]) -> str:
    context = _context(node, tensor_map)
    if target == "cpu":
        return _render_template("t5_layer_norm_cpu.cpp.j2", context)
    if target == "cuda":
        return _render_template("t5_layer_norm_cuda.cu.j2", context)
    raise ValueError(f"Unsupported t5_layer_norm target: {target}")


def render_launch(
    target: str,
    node: Mapping[str, Any],
    tensor_map: Mapping[str, Mapping[str, Any]],
    kernel_manifest: Mapping[str, Any] | None = None,
) -> str:
    del kernel_manifest
    func = _function_name(node, tensor_map)
    x = _c_ident(node["inputs"][0])
    weight = _c_ident(node["inputs"][1])
    out = _c_ident(node["outputs"][0])
    args = f"ptr_{x}, ptr_{weight}, ptr_{out}, runtime_numel_{out}"
    if target == "cpu":
        return f"if (int err = {func}({args})) return err;"
    if target == "cuda":
        return f"if (int err = {func}({args}, session->stream)) return err;"
    raise ValueError(f"Unsupported t5_layer_norm target: {target}")


def source_key(target: str, node: Mapping[str, Any], tensor_map: Mapping[str, Mapping[str, Any]]) -> str:
    if target not in {"cpu", "cuda"}:
        raise ValueError(f"Unsupported t5_layer_norm target: {target}")
    return f"{target}:{_function_name(node, tensor_map)}"


def generated_function_name(target: str, node: Mapping[str, Any], tensor_map: Mapping[str, Mapping[str, Any]]) -> str:
    del target
    return _function_name(node, tensor_map)


def _context(node: Mapping[str, Any], tensor_map: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    x_tensor = tensor_map[node["inputs"][0]]
    weight_tensor = tensor_map[node["inputs"][1]]
    output_tensor = tensor_map[node["outputs"][0]]
    _validate_node_contract(node, x_tensor, weight_tensor, output_tensor)
    cols = int(x_tensor["shape"][-1])
    dtype = str(x_tensor["dtype"])
    return {
        "func": _function_name(node, tensor_map),
        "kernel": f"{_function_name(node, tensor_map)}_kernel",
        "warp_kernel": f"{_function_name(node, tensor_map)}_warp_kernel",
        "cpu_storage_type": cpu_storage_type(dtype),
        "cuda_storage_type": cuda_storage_type(dtype),
        "cols": cols,
        "eps": float(node.get("attrs", {}).get("eps", 1e-6)),
        "block_size": _cuda_block_size(cols),
        "cols_per_thread": (cols + 31) // 32,
        "rows_per_block": _cuda_rows_per_block(cols),
        "use_warp_kernel": cols <= 1024,
    }


def _validate_node_contract(
    node: Mapping[str, Any],
    x_tensor: Mapping[str, Any],
    weight_tensor: Mapping[str, Any],
    output_tensor: Mapping[str, Any],
) -> None:
    if str(node["op"]) != "t5_layer_norm":
        raise ValueError(f"Unsupported normalization op: {node['op']}")
    x_dtype = str(x_tensor["dtype"])
    weight_dtype = str(weight_tensor["dtype"])
    output_dtype = str(output_tensor["dtype"])
    if x_dtype != weight_dtype or x_dtype != output_dtype:
        raise NotImplementedError("t5_layer_norm lowering currently requires matching input/output dtypes")
    if x_dtype not in T5_LAYER_NORM_DTYPES:
        raise NotImplementedError("t5_layer_norm lowering supports float16, float32, and bfloat16 tensors only")
    if list(x_tensor["shape"]) != list(output_tensor["shape"]):
        raise ValueError("t5_layer_norm input and output shapes must match")
    if not x_tensor["shape"]:
        raise ValueError("t5_layer_norm requires rank >= 1 input")
    if len(weight_tensor["shape"]) != 1:
        raise ValueError("t5_layer_norm requires rank-1 weight")
    x_shape_spec = x_tensor.get("shape_spec", x_tensor["shape"])
    weight_shape_spec = weight_tensor.get("shape_spec", weight_tensor["shape"])
    cols = x_tensor["shape"][-1]
    if not isinstance(x_shape_spec[-1], int) or not isinstance(cols, int) or int(cols) <= 0:
        raise ValueError("t5_layer_norm lowering requires a positive static last dimension")
    if not isinstance(weight_shape_spec[0], int) or int(weight_tensor["shape"][0]) != int(cols):
        raise ValueError("t5_layer_norm lowering requires weight shape [hidden] matching the input hidden size")


def _cuda_block_size(cols: int) -> int:
    block = 1
    while block < cols and block < 256:
        block *= 2
    return max(32, block)


def _cuda_rows_per_block(cols: int) -> int:
    if cols <= 256:
        return 8
    if cols <= 512:
        return 4
    return 2


def _function_name(node: Mapping[str, Any], tensor_map: Mapping[str, Mapping[str, Any]]) -> str:
    x_tensor = tensor_map[node["inputs"][0]]
    signature = {
        "op": "t5_layer_norm",
        "shape": list(x_tensor["shape"]),
        "dtype": str(x_tensor["dtype"]),
        "eps": float(node.get("attrs", {}).get("eps", 1e-6)),
    }
    digest = hashlib.sha256(repr(signature).encode("utf-8")).hexdigest()[:12]
    return f"t5_layer_norm_{digest}"


def _render_template(name: str, context: Mapping[str, Any]) -> str:
    env = Environment(
        loader=FileSystemLoader(str(Path(__file__).resolve().parent / "templates")),
        undefined=StrictUndefined,
        trim_blocks=True,
        lstrip_blocks=True,
        keep_trailing_newline=True,
    )
    return env.get_template(name).render(**context)


T5_LAYER_NORM_LOWERING = OpLowering(
    op_name="t5_layer_norm",
    render_generated_kernel=render_generated_kernel,
    render_launch=render_launch,
    source_key=source_key,
    generated_function_name=generated_function_name,
)
