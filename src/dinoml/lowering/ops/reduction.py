from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Any, Mapping

from jinja2 import Environment, FileSystemLoader, StrictUndefined

from dinoml.lowering.ops.base import OpLowering
from dinoml.lowering.ops.template_rendering import supported_target_spec
from dinoml.lowering.target_specs import storage_type as target_storage_type
from dinoml.ops.reductions import BASIC_REDUCTION_OPS, REDUCTION_DTYPES, REDUCTION_OPS
from dinoml.lowering.shape_buffers import c_ident as _c_ident


def render_generated_kernel(target: str, node: Mapping[str, Any], tensor_map: Mapping[str, Mapping[str, Any]]) -> str:
    spec = supported_target_spec(target, "reduction")
    context = _context(target, node, tensor_map)
    if not spec.is_gpu:
        return _render_template("reduction_cpu.cpp.j2", context)
    context.update(spec.gpu_template_context())
    return _render_template("reduction_gpu.j2", context)


def render_launch(
    target: str,
    node: Mapping[str, Any],
    tensor_map: Mapping[str, Mapping[str, Any]],
    kernel_manifest: Mapping[str, Any] | None = None,
) -> str:
    del kernel_manifest
    spec = supported_target_spec(target, "reduction")
    func = _function_name(node, tensor_map)
    inp = _c_ident(node["inputs"][0])
    out = _c_ident(node["outputs"][0])
    args = f"ptr_{inp}, ptr_{out}, runtime_numel_{out}"
    if not spec.is_gpu:
        return f"if (int err = {func}({args})) return err;"
    return f"if (int err = {func}({args}, {spec.stream_expr})) return err;"


def source_key(target: str, node: Mapping[str, Any], tensor_map: Mapping[str, Mapping[str, Any]]) -> str:
    supported_target_spec(target, "reduction")
    return f"{target}:{_function_name(node, tensor_map)}"


def generated_function_name(target: str, node: Mapping[str, Any], tensor_map: Mapping[str, Mapping[str, Any]]) -> str:
    del target
    return _function_name(node, tensor_map)


def _context(target: str, node: Mapping[str, Any], tensor_map: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    input_tensor = tensor_map[node["inputs"][0]]
    output_tensor = tensor_map[node["outputs"][0]]
    _validate_node_contract(node, input_tensor, output_tensor)
    cols = int(input_tensor["shape"][-1])
    dtype = str(input_tensor["dtype"])
    context = {
        "func": _function_name(node, tensor_map),
        "kernel": f"{_function_name(node, tensor_map)}_kernel",
        "warp_kernel": f"{_function_name(node, tensor_map)}_warp_kernel",
        "op": node["op"],
        "cpu_storage_type": target_storage_type(dtype, "cpu"),
        "storage_type": target_storage_type(dtype, target),
        "cols": cols,
        "block_size": _cuda_block_size(cols),
        "cols_per_thread": (cols + 31) // 32,
        "rows_per_block": _gpu_rows_per_block(target, cols),
        "use_warp_kernel": cols <= 1024,
        "two_accumulators": node["op"] == "var",
        "initial_value": _initial_value(node["op"]),
        "combine_expr": _combine_expr(node["op"]),
        "partial_combine_expr": _partial_combine_expr(node["op"]),
        "unbiased": bool(node.get("attrs", {}).get("unbiased", False)),
    }
    context["final_expr"] = _final_expr(node["op"], cols, context["unbiased"])
    return context


def _validate_node_contract(
    node: Mapping[str, Any],
    input_tensor: Mapping[str, Any],
    output_tensor: Mapping[str, Any],
) -> None:
    if node["op"] not in REDUCTION_OPS:
        raise ValueError(f"Unsupported reduction op: {node['op']}")
    input_dtype = str(input_tensor["dtype"])
    output_dtype = str(output_tensor["dtype"])
    if input_dtype != output_dtype:
        raise NotImplementedError("reduction lowering currently requires matching input/output dtypes")
    if node["op"] in BASIC_REDUCTION_OPS:
        if input_dtype not in REDUCTION_DTYPES:
            raise NotImplementedError("basic reduction lowering supports float16, float32, and bfloat16 tensors only")
    elif input_dtype != "float32":
        raise NotImplementedError(f"{node['op']} lowering currently supports float32 tensors only")
    if not input_tensor["shape"]:
        raise ValueError("reduction requires a ranked tensor")
    dim = int(node.get("attrs", {}).get("dim", -1))
    if dim < 0:
        dim += len(input_tensor["shape"])
    if dim != len(input_tensor["shape"]) - 1:
        raise NotImplementedError("reduction lowering currently supports only the last dimension")
    shape_spec = input_tensor.get("shape_spec", input_tensor["shape"])
    cols = input_tensor["shape"][-1]
    if not isinstance(shape_spec[-1], int) or not isinstance(cols, int) or int(cols) <= 0:
        raise ValueError("reduction lowering requires a positive static last dimension")
    if node["op"] == "vector_norm" and float(node.get("attrs", {}).get("ord", 2.0)) != 2.0:
        raise NotImplementedError("vector_norm lowering currently supports only ord=2")
    if node["op"] == "var" and bool(node.get("attrs", {}).get("unbiased", False)) and int(cols) <= 1:
        raise ValueError("unbiased var requires a reduction dimension larger than one")
    expected = list(input_tensor["shape"])
    if bool(node.get("attrs", {}).get("keepdim", False)):
        expected[-1] = 1
    else:
        expected = expected[:-1] or [1]
    if list(output_tensor["shape"]) != expected:
        raise ValueError("reduction output shape does not match reduction attrs")


def _initial_value(op: str) -> str:
    if op in {"reduce_sum", "reduce_mean", "var", "vector_norm"}:
        return "0.0f"
    if op == "reduce_max":
        return "-3.4028234663852886e38f"
    if op == "reduce_min":
        return "3.4028234663852886e38f"
    raise ValueError(op)


def _combine_expr(op: str) -> str:
    if op in {"reduce_sum", "reduce_mean"}:
        return "acc + value"
    if op == "vector_norm":
        return "acc + value * value"
    if op == "var":
        return "acc + value"
    if op == "reduce_max":
        return "fmaxf(acc, value)"
    if op == "reduce_min":
        return "fminf(acc, value)"
    raise ValueError(op)


def _partial_combine_expr(op: str) -> str:
    if op in {"reduce_sum", "reduce_mean", "var", "vector_norm"}:
        return "acc + value"
    if op == "reduce_max":
        return "fmaxf(acc, value)"
    if op == "reduce_min":
        return "fminf(acc, value)"
    raise ValueError(op)


def _final_expr(op: str, cols: int, unbiased: bool = False) -> str:
    if op == "reduce_mean":
        return f"acc / {float(cols):.8f}f"
    if op == "vector_norm":
        return "sqrtf(acc)"
    if op == "var":
        denominator = cols - 1 if unbiased else cols
        return f"(sum_sq_acc - (acc * acc) / {float(cols):.8f}f) / {float(denominator):.8f}f"
    return "acc"


def _cuda_block_size(cols: int) -> int:
    block = 1
    while block < cols and block < 256:
        block *= 2
    return max(32, block)


def _cuda_rows_per_block(cols: int) -> int:
    if cols <= 128:
        return 8
    return 4


def _gpu_rows_per_block(target: str, cols: int) -> int:
    if target == "rocm" and cols <= 1024:
        return 8
    return _cuda_rows_per_block(cols)


def _function_name(node: Mapping[str, Any], tensor_map: Mapping[str, Mapping[str, Any]]) -> str:
    input_tensor = tensor_map[node["inputs"][0]]
    dim = int(node.get("attrs", {}).get("dim", -1))
    if dim < 0:
        dim += len(input_tensor["shape"])
    signature = {
        "op": node["op"],
        "shape": list(input_tensor["shape"]),
        "dtype": str(input_tensor["dtype"]),
        "dim": dim,
        "keepdim": bool(node.get("attrs", {}).get("keepdim", False)),
    }
    if node["op"] == "var":
        signature["unbiased"] = bool(node.get("attrs", {}).get("unbiased", False))
    elif node["op"] == "vector_norm":
        signature["ord"] = float(node.get("attrs", {}).get("ord", 2.0))
    digest = hashlib.sha256(repr(signature).encode("utf-8")).hexdigest()[:12]
    return f"{node['op']}_{digest}"


def _render_template(name: str, context: Mapping[str, Any]) -> str:
    env = Environment(
        loader=FileSystemLoader(str(Path(__file__).resolve().parent / "templates")),
        undefined=StrictUndefined,
        trim_blocks=True,
        lstrip_blocks=True,
        keep_trailing_newline=True,
    )
    return env.get_template(name).render(**context)


REDUCTION_LOWERINGS = {
    op_name: OpLowering(
        op_name=op_name,
        render_generated_kernel=render_generated_kernel,
        render_launch=render_launch,
        source_key=source_key,
        generated_function_name=generated_function_name,
    )
    for op_name in REDUCTION_OPS
}
