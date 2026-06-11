from __future__ import annotations

import hashlib
from typing import Any, Mapping, Sequence

from dinoml.lowering.ops.base import OpLowering
from dinoml.lowering.ops.template_rendering import render_op_template, supported_target_spec
from dinoml.lowering.shape_buffers import c_ident as _c_ident, shape_spec_dim_expr
from dinoml.lowering.target_specs import storage_type as target_storage_type
from dinoml.ops.collections import COLLECTION_DTYPES, broadcast_shape_spec, resolve_masked_select_shape
from dinoml.shapes import normalize_symbolic_int, symbolic_int_expr, symbolic_int_interval


def render_generated_kernel(target: str, node: Mapping[str, Any], tensor_map: Mapping[str, Mapping[str, Any]]) -> str:
    spec = supported_target_spec(target, "masked_select")
    context = _context(target, node, tensor_map)
    if not spec.is_gpu:
        return render_op_template("masked_select_cpu.cpp.j2", context)
    context.update(spec.gpu_template_context())
    return render_op_template("masked_select_gpu.j2", context)


def render_launch(
    target: str,
    node: Mapping[str, Any],
    tensor_map: Mapping[str, Mapping[str, Any]],
    kernel_manifest: Mapping[str, Any] | None = None,
) -> str:
    del kernel_manifest
    spec = supported_target_spec(target, "masked_select")
    func = _function_name(node, tensor_map)
    x = _c_ident(node["inputs"][0])
    mask = _c_ident(node["inputs"][1])
    out = _c_ident(node["outputs"][0])
    shape_arg = f"session->shape_{out}.data()" if not spec.is_gpu else f"session->shape_{out}"
    args = [f"ptr_{x}", f"ptr_{mask}", f"ptr_{out}", f"runtime_numel_{out}", shape_arg, *_launch_args(node, tensor_map)]
    if spec.is_gpu:
        args.extend(("session->masked_select_scratch", "session->masked_select_scratch_nbytes", str(spec.stream_expr)))
    joined = ", ".join(args)
    return f"if (int err = {func}({joined})) return err;"


def source_key(target: str, node: Mapping[str, Any], tensor_map: Mapping[str, Mapping[str, Any]]) -> str:
    supported_target_spec(target, "masked_select")
    return f"{target}:{_function_name(node, tensor_map)}"


def generated_function_name(target: str, node: Mapping[str, Any], tensor_map: Mapping[str, Mapping[str, Any]]) -> str:
    del target
    return _function_name(node, tensor_map)


def _context(target: str, node: Mapping[str, Any], tensor_map: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    input_tensor = tensor_map[node["inputs"][0]]
    mask_tensor = tensor_map[node["inputs"][1]]
    output_tensor = tensor_map[node["outputs"][0]]
    broadcast_shape = _validate_node_contract(node, input_tensor, mask_tensor, output_tensor)
    dynamic_dims = _dynamic_dim_sources(input_tensor, mask_tensor, output_tensor)
    output_extent_exprs = [shape_spec_dim_expr(dim, dynamic_dims) for dim in broadcast_shape]
    x_stride_exprs = _aligned_stride_exprs(input_tensor.get("shape_spec", input_tensor["shape"]), broadcast_shape, dynamic_dims)
    mask_stride_exprs = _aligned_stride_exprs(mask_tensor.get("shape_spec", mask_tensor["shape"]), broadcast_shape, dynamic_dims)
    return {
        "func": _function_name(node, tensor_map),
        "kernel": f"{_function_name(node, tensor_map)}_kernel",
        "storage_type": target_storage_type(str(output_tensor["dtype"]), target),
        "mask_storage_type": target_storage_type("bool", target),
        "rank": len(broadcast_shape),
        "output_extent_exprs": output_extent_exprs,
        "x_stride_exprs": x_stride_exprs,
        "mask_stride_exprs": mask_stride_exprs,
        "copy_body": _copy_body(len(broadcast_shape)),
        "need_expand_input": _needs_expansion(input_tensor.get("shape_spec", input_tensor["shape"]), broadcast_shape),
        "need_expand_mask": _needs_expansion(mask_tensor.get("shape_spec", mask_tensor["shape"]), broadcast_shape),
        "block_size": 256,
    }


def masked_select_scratch_nbytes_for_node(
    target: str,
    node: Mapping[str, Any],
    tensor_map: Mapping[str, Mapping[str, Any]],
) -> int:
    spec = supported_target_spec(target, "masked_select")
    if not spec.is_gpu:
        return 0
    context = _context(target, node, tensor_map)
    output_tensor = tensor_map[node["outputs"][0]]
    output_capacity = int(output_tensor["shape"][0])
    expansion_nbytes = 0
    if context["need_expand_input"]:
        expansion_nbytes += _align_nbytes(output_capacity * _dtype_nbytes(str(output_tensor["dtype"])), 16)
    if context["need_expand_mask"]:
        expansion_nbytes += _align_nbytes(output_capacity, 16)
    # The exact DeviceSelect temp requirement is queried again at runtime. This compile-time
    # allocation is a conservative session scratch budget, not the final authority.
    temp_storage_nbytes = _align_nbytes(output_capacity * 16 + 4096, 16)
    return expansion_nbytes + temp_storage_nbytes


def _validate_node_contract(
    node: Mapping[str, Any],
    input_tensor: Mapping[str, Any],
    mask_tensor: Mapping[str, Any],
    output_tensor: Mapping[str, Any],
) -> list[Any]:
    if node["op"] != "masked_select":
        raise ValueError(f"Unsupported collection op: {node['op']}")
    if len(node.get("inputs", [])) != 2:
        raise ValueError("masked_select expects two tensor inputs")
    if len(node.get("outputs", [])) != 1:
        raise ValueError("masked_select expects exactly one output")
    dtype = str(output_tensor["dtype"])
    if dtype not in COLLECTION_DTYPES:
        raise NotImplementedError(f"masked_select lowering does not support dtype {output_tensor['dtype']}")
    if str(input_tensor["dtype"]) != dtype:
        raise ValueError("masked_select input and output dtype must match")
    if str(mask_tensor["dtype"]) != "bool":
        raise ValueError("masked_select mask must have dtype bool")
    expected_shape = resolve_masked_select_shape(input_tensor["shape"], mask_tensor["shape"])
    if list(expected_shape) != list(output_tensor["shape"]):
        raise ValueError("masked_select output shape does not match broadcast capacity")
    broadcast_shape = broadcast_shape_spec(
        input_tensor.get("shape_spec", input_tensor["shape"]),
        mask_tensor.get("shape_spec", mask_tensor["shape"]),
    )
    expected_shape_spec = [_symbolic_numel(broadcast_shape)]
    if list(output_tensor.get("shape_spec", output_tensor["shape"])) != expected_shape_spec:
        raise ValueError("masked_select output shape_spec does not match broadcast capacity expression")
    return broadcast_shape


def _launch_args(node: Mapping[str, Any], tensor_map: Mapping[str, Mapping[str, Any]]) -> list[str]:
    input_tensor = tensor_map[node["inputs"][0]]
    mask_tensor = tensor_map[node["inputs"][1]]
    output_tensor = tensor_map[node["outputs"][0]]
    broadcast_shape = _validate_node_contract(node, input_tensor, mask_tensor, output_tensor)
    dynamic_dims = _dynamic_dim_sources(input_tensor, mask_tensor, output_tensor)
    output_extent_exprs = [shape_spec_dim_expr(dim, dynamic_dims) for dim in broadcast_shape]
    x_stride_exprs = _aligned_stride_exprs(input_tensor.get("shape_spec", input_tensor["shape"]), broadcast_shape, dynamic_dims)
    mask_stride_exprs = _aligned_stride_exprs(mask_tensor.get("shape_spec", mask_tensor["shape"]), broadcast_shape, dynamic_dims)
    args: list[str] = []
    for axis in range(len(broadcast_shape)):
        args.extend((output_extent_exprs[axis], x_stride_exprs[axis], mask_stride_exprs[axis]))
    return args


def _aligned_stride_exprs(
    input_shape_spec: Sequence[Any],
    broadcast_shape: Sequence[Any],
    dynamic_dims: Mapping[str, str],
) -> list[str]:
    rank = len(broadcast_shape)
    prefix = rank - len(input_shape_spec)
    aligned = [1] * prefix + [dict(dim) if isinstance(dim, Mapping) else dim for dim in input_shape_spec]
    strides = ["0"] * rank
    running = "1"
    for axis in range(rank - 1, -1, -1):
        dim = aligned[axis]
        strides[axis] = "0" if _dim_is_known_one(dim) else running
        dim_expr = shape_spec_dim_expr(dim, dynamic_dims)
        running = f"({dim_expr})" if running == "1" else f"({dim_expr}) * ({running})"
    return strides


def _copy_body(rank: int) -> str:
    lines = [
        "    int64_t count = 0;",
        "    for (int64_t idx = 0; idx < output_capacity; ++idx) {",
        "      int64_t remaining = idx;",
        "      int64_t x_index = 0;",
        "      int64_t mask_index = 0;",
        "      int64_t coord = 0;",
    ]
    for axis in range(rank - 1, -1, -1):
        lines.append(f"      coord = remaining % out_extent_{axis};")
        lines.append(f"      remaining = remaining / out_extent_{axis};")
        lines.append(f"      x_index += coord * x_stride_{axis};")
        lines.append(f"      mask_index += coord * mask_stride_{axis};")
    lines.extend(
        [
            "      if (mask[mask_index]) {",
            "        y[count++] = x[x_index];",
            "      }",
            "    }",
            "    out_shape[0] = count;",
        ]
    )
    return "\n".join(lines)


def _needs_expansion(input_shape_spec: Sequence[Any], broadcast_shape: Sequence[Any]) -> bool:
    rank = len(broadcast_shape)
    prefix = rank - len(input_shape_spec)
    aligned = [1] * prefix + [dict(dim) if isinstance(dim, Mapping) else dim for dim in input_shape_spec]
    for dim, out_dim in zip(aligned, broadcast_shape, strict=True):
        if dim == out_dim or (_dim_is_known_one(dim) and _dim_is_known_one(out_dim)):
            continue
        if _dim_is_known_one(dim):
            return True
    return False


def _dtype_nbytes(dtype: str) -> int:
    if dtype in {"float16", "bfloat16"}:
        return 2
    if dtype in {"float32", "int32"}:
        return 4
    if dtype == "int64":
        return 8
    if dtype == "bool":
        return 1
    raise NotImplementedError(f"masked_select scratch sizing does not support dtype {dtype!r}")


def _align_nbytes(value: int, alignment: int) -> int:
    return ((int(value) + alignment - 1) // alignment) * alignment


def _dynamic_dim_sources(
    input_tensor: Mapping[str, Any],
    mask_tensor: Mapping[str, Any],
    output_tensor: Mapping[str, Any],
) -> dict[str, str]:
    sources: dict[str, str] = {}
    _record_tensor_dim_sources(sources, input_tensor, f"shape_{_c_ident(str(input_tensor['name']))}")
    _record_tensor_dim_sources(sources, mask_tensor, f"shape_{_c_ident(str(mask_tensor['name']))}")
    _record_tensor_dim_sources(sources, output_tensor, f"shape_{_c_ident(str(output_tensor['name']))}")
    return sources


def _record_tensor_dim_sources(
    sources: dict[str, str],
    tensor: Mapping[str, Any],
    prefix: str,
) -> None:
    for axis, dim in enumerate(tensor.get("shape_spec", tensor["shape"])):
        _record_dim_sources(sources, dim, f"{prefix}_{axis}")


def _record_dim_sources(sources: dict[str, str], dim: Any, expr: str) -> None:
    if isinstance(dim, int):
        return
    kind = dim.get("kind")
    if kind == "dim":
        sources.setdefault(str(dim["name"]), expr)
        return
    if kind == "int_expr":
        _record_dim_sources(sources, dim["lhs"], expr)
        _record_dim_sources(sources, dim["rhs"], expr)
        return
    raise ValueError(f"Unsupported shape dimension kind: {kind!r}")


def _dim_is_known_one(dim: Any) -> bool:
    if isinstance(dim, int):
        return int(dim) == 1
    min_dim, max_dim = symbolic_int_interval(normalize_symbolic_int(dim))
    return min_dim == 1 and max_dim == 1


def _symbolic_numel(shape_spec: Sequence[Any]) -> int | dict[str, Any]:
    total: int | dict[str, Any] = 1
    for dim in shape_spec:
        total = symbolic_int_expr("mul", total, normalize_symbolic_int(dim))
    return total


def _function_name(node: Mapping[str, Any], tensor_map: Mapping[str, Mapping[str, Any]]) -> str:
    input_tensor = tensor_map[node["inputs"][0]]
    mask_tensor = tensor_map[node["inputs"][1]]
    output_tensor = tensor_map[node["outputs"][0]]
    signature = {
        "op": "masked_select",
        "input_shape": list(input_tensor["shape"]),
        "input_shape_spec": list(input_tensor.get("shape_spec", input_tensor["shape"])),
        "mask_shape": list(mask_tensor["shape"]),
        "mask_shape_spec": list(mask_tensor.get("shape_spec", mask_tensor["shape"])),
        "output_shape": list(output_tensor["shape"]),
        "output_shape_spec": list(output_tensor.get("shape_spec", output_tensor["shape"])),
        "dtype": str(output_tensor["dtype"]),
    }
    digest = hashlib.sha256(repr(signature).encode("utf-8")).hexdigest()[:12]
    return f"masked_select_{digest}"


MASKED_SELECT_LOWERING = OpLowering(
    op_name="masked_select",
    render_generated_kernel=render_generated_kernel,
    render_launch=render_launch,
    source_key=source_key,
    generated_function_name=generated_function_name,
)
