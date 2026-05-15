from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Any, Mapping

from jinja2 import Environment, FileSystemLoader, StrictUndefined

from dinoml.lowering.ops.base import OpLowering
from dinoml.lowering.cpp_types import cpu_storage_type, cuda_storage_type


SOFTMAX_DTYPES = {"float16", "float32", "bfloat16"}
from dinoml.lowering.shape_buffers import c_ident as _c_ident


def render_generated_kernel(target: str, node: Mapping[str, Any], tensor_map: Mapping[str, Mapping[str, Any]]) -> str:
    context = _context(node, tensor_map)
    if target == "cpu":
        return _render_template("softmax_cpu.cpp.j2", context)
    if target == "cuda":
        return _render_template("softmax_cuda.cu.j2", context)
    raise ValueError(f"Unsupported softmax target: {target}")


def render_launch(
    target: str,
    node: Mapping[str, Any],
    tensor_map: Mapping[str, Mapping[str, Any]],
    kernel_manifest: Mapping[str, Any] | None = None,
) -> str:
    del kernel_manifest
    func = _function_name(node, tensor_map)
    inp = _c_ident(node["inputs"][0])
    out = _c_ident(node["outputs"][0])
    args = f"ptr_{inp}, ptr_{out}, runtime_numel_{out}"
    if target == "cpu":
        return f"if (int err = {func}({args})) return err;"
    if target == "cuda":
        return f"if (int err = {func}({args}, session->stream)) return err;"
    raise ValueError(f"Unsupported softmax target: {target}")


def source_key(target: str, node: Mapping[str, Any], tensor_map: Mapping[str, Mapping[str, Any]]) -> str:
    if target not in {"cpu", "cuda"}:
        raise ValueError(f"Unsupported softmax target: {target}")
    return f"{target}:{_function_name(node, tensor_map)}"


def generated_function_name(target: str, node: Mapping[str, Any], tensor_map: Mapping[str, Mapping[str, Any]]) -> str:
    del target
    return _function_name(node, tensor_map)


def _context(node: Mapping[str, Any], tensor_map: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    input_name = node["inputs"][0]
    output_name = node["outputs"][0]
    input_tensor = tensor_map[input_name]
    output_tensor = tensor_map[output_name]
    _validate_node_contract(node, input_tensor, output_tensor)
    cols = int(input_tensor["shape"][-1])
    dtype = str(input_tensor["dtype"])
    pack_width = _cuda_pack_width(cols) if dtype == "float32" else 1
    use_packed_kernel = pack_width > 1
    use_warp_kernel = not use_packed_kernel and cols <= 2048
    return {
        "func": _function_name(node, tensor_map),
        "kernel": f"{_function_name(node, tensor_map)}_kernel",
        "packed_kernel": f"{_function_name(node, tensor_map)}_packed_kernel",
        "warp_kernel": f"{_function_name(node, tensor_map)}_warp_kernel",
        "cpu_storage_type": cpu_storage_type(dtype),
        "cuda_storage_type": cuda_storage_type(dtype),
        "cols": cols,
        "num_packs": cols // pack_width,
        "pack_type": "float4" if pack_width == 4 else "float2",
        "pack_width": pack_width,
        "pack_values_per_thread": ((cols // pack_width + 31) // 32) * pack_width,
        "packs_per_thread": (cols // pack_width + 31) // 32,
        "cols_per_thread": (cols + 31) // 32,
        "rows_per_block": _cuda_rows_per_block(cols),
        "use_packed_kernel": use_packed_kernel,
        "use_warp_kernel": use_warp_kernel,
        "block_size": _cuda_block_size(cols),
    }


def _validate_node_contract(
    node: Mapping[str, Any],
    input_tensor: Mapping[str, Any],
    output_tensor: Mapping[str, Any],
) -> None:
    input_dtype = str(input_tensor["dtype"])
    output_dtype = str(output_tensor["dtype"])
    if input_dtype != output_dtype:
        raise NotImplementedError("softmax lowering currently requires matching input/output dtypes")
    if input_dtype not in SOFTMAX_DTYPES:
        raise NotImplementedError("softmax lowering supports float16, float32, and bfloat16 tensors only")
    if list(input_tensor["shape"]) != list(output_tensor["shape"]):
        raise ValueError("softmax input and output shapes must match")
    if not input_tensor["shape"]:
        raise ValueError("softmax requires a ranked tensor")
    dim = int(node.get("attrs", {}).get("dim", -1))
    if dim < 0:
        dim += len(input_tensor["shape"])
    if dim != len(input_tensor["shape"]) - 1:
        raise NotImplementedError("softmax lowering currently supports only the last dimension")
    shape_spec = input_tensor.get("shape_spec", input_tensor["shape"])
    cols = input_tensor["shape"][-1]
    if not isinstance(shape_spec[-1], int) or not isinstance(cols, int) or int(cols) <= 0:
        raise ValueError("softmax lowering requires a positive static last dimension")


def _cuda_block_size(cols: int) -> int:
    block = 1
    while block < cols and block < 256:
        block *= 2
    return max(32, block)


def _cuda_rows_per_block(cols: int) -> int:
    if cols <= 1024:
        return 4
    return 2


def _cuda_pack_width(cols: int) -> int:
    if 32 < cols <= 3840 and cols % 4 == 0:
        return 4
    if 32 < cols <= 1152 and cols % 2 == 0:
        return 2
    return 1


def _function_name(node: Mapping[str, Any], tensor_map: Mapping[str, Mapping[str, Any]]) -> str:
    input_tensor = tensor_map[node["inputs"][0]]
    dim = int(node.get("attrs", {}).get("dim", -1))
    if dim < 0:
        dim += len(input_tensor["shape"])
    signature = {
        "op": "softmax",
        "shape": list(input_tensor["shape"]),
        "dtype": str(input_tensor["dtype"]),
        "dim": dim,
    }
    digest = hashlib.sha256(repr(signature).encode("utf-8")).hexdigest()[:12]
    return f"softmax_{digest}"


def _render_template(name: str, context: Mapping[str, Any]) -> str:
    env = Environment(
        loader=FileSystemLoader(str(Path(__file__).resolve().parent / "templates")),
        undefined=StrictUndefined,
        trim_blocks=True,
        lstrip_blocks=True,
        keep_trailing_newline=True,
    )
    return env.get_template(name).render(**context)


SOFTMAX_LOWERING = OpLowering(
    op_name="softmax",
    render_generated_kernel=render_generated_kernel,
    render_launch=render_launch,
    source_key=source_key,
    generated_function_name=generated_function_name,
)
