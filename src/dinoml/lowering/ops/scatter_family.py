from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Mapping

from jinja2 import Environment, FileSystemLoader, StrictUndefined

from dinoml.lowering.ops.base import OpLowering
from dinoml.lowering.ops.gather import _index_storage_type
from dinoml.lowering.ops.template_rendering import supported_target_spec
from dinoml.lowering.shape_buffers import c_ident as _c_ident
from dinoml.lowering.target_specs import storage_type as target_storage_type
from dinoml.ops.collections import (
    GATHER_INDEX_DTYPES,
    SCATTER_DTYPES,
    SCATTER_REDUCE_DTYPES,
    normalize_scatter_attrs,
    normalize_scatter_reduce_include_self,
    normalize_scatter_reduce_name,
    resolve_scatter_add_shape,
    resolve_scatter_reduce_shape,
    resolve_scatter_shape,
)


def render_generated_kernel(target: str, node: Mapping[str, Any], tensor_map: Mapping[str, Mapping[str, Any]]) -> str:
    spec = supported_target_spec(target, str(node["op"]))
    context = _context(target, node, tensor_map)
    if not spec.is_gpu:
        return _render_template("scatter_family_cpu.cpp.j2", context)
    context.update(spec.gpu_template_context())
    return _render_template("scatter_family_gpu.j2", context)


def render_launch(
    target: str,
    node: Mapping[str, Any],
    tensor_map: Mapping[str, Mapping[str, Any]],
    kernel_manifest: Mapping[str, Any] | None = None,
) -> str:
    del kernel_manifest
    spec = supported_target_spec(target, str(node["op"]))
    func = _function_name(node, tensor_map)
    x = _c_ident(node["inputs"][0])
    index = _c_ident(node["inputs"][1])
    source = _c_ident(node["inputs"][2])
    out = _c_ident(node["outputs"][0])
    rank = len(tensor_map[node["outputs"][0]]["shape"])
    args = [
        f"ptr_{x}",
        f"ptr_{index}",
        f"ptr_{source}",
        f"ptr_{out}",
        f"runtime_numel_{out}",
        f"runtime_numel_{source}",
        *[f"static_cast<int64_t>(shape_{out}_{axis})" for axis in range(rank)],
        *[f"static_cast<int64_t>(shape_{source}_{axis})" for axis in range(rank)],
    ]
    joined = ", ".join(args)
    if not spec.is_gpu:
        return f"if (int err = {func}({joined})) return err;"
    return f"if (int err = {func}({joined}, {spec.stream_expr})) return err;"


def source_key(target: str, node: Mapping[str, Any], tensor_map: Mapping[str, Mapping[str, Any]]) -> str:
    supported_target_spec(target, str(node["op"]))
    return f"{target}:{_function_name(node, tensor_map)}"


def generated_function_name(target: str, node: Mapping[str, Any], tensor_map: Mapping[str, Mapping[str, Any]]) -> str:
    del target
    return _function_name(node, tensor_map)


def _context(target: str, node: Mapping[str, Any], tensor_map: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    input_tensor = tensor_map[node["inputs"][0]]
    index_tensor = tensor_map[node["inputs"][1]]
    source_tensor = tensor_map[node["inputs"][2]]
    output_tensor = tensor_map[node["outputs"][0]]
    op_name = str(node["op"])
    dim, reduction = _validate_node_contract(node, input_tensor, index_tensor, source_tensor, output_tensor)
    dtype = str(output_tensor["dtype"])
    rank = len(output_tensor["shape"])
    return {
        "func": _function_name(node, tensor_map),
        "kernel": f"{_function_name(node, tensor_map)}_kernel",
        "storage_type": target_storage_type(dtype, target),
        "index_storage_type": _index_storage_type(str(index_tensor["dtype"])),
        "op_name": op_name,
        "operation": "assign" if op_name == "scatter" else ("add" if op_name == "scatter_add" else "reduce"),
        "reduction": reduction or "",
        "rank": rank,
        "dim": dim,
        "update_body": _update_body(op_name, reduction, target_storage_type(dtype, target)),
    }


def _validate_node_contract(
    node: Mapping[str, Any],
    input_tensor: Mapping[str, Any],
    index_tensor: Mapping[str, Any],
    source_tensor: Mapping[str, Any],
    output_tensor: Mapping[str, Any],
) -> tuple[int, str | None]:
    op_name = str(node["op"])
    if op_name not in {"scatter", "scatter_add", "scatter_reduce"}:
        raise ValueError(f"Unsupported scatter-family op: {op_name}")
    if len(node.get("inputs", [])) != 3:
        raise ValueError(f"{op_name} expects three tensor inputs")
    if len(node.get("outputs", [])) != 1:
        raise ValueError(f"{op_name} expects exactly one output")
    dtype = str(output_tensor["dtype"])
    allowed_dtypes = SCATTER_DTYPES if op_name == "scatter" else SCATTER_REDUCE_DTYPES
    if dtype not in allowed_dtypes:
        raise NotImplementedError(f"{op_name} lowering does not support dtype {output_tensor['dtype']}")
    if str(input_tensor["dtype"]) != dtype or str(source_tensor["dtype"]) != dtype:
        raise ValueError(f"{op_name} input, source, and output dtype must match")
    if str(index_tensor["dtype"]) not in GATHER_INDEX_DTYPES:
        raise ValueError(f"{op_name} index must have dtype int64 or int32, got {index_tensor['dtype']}")
    attrs = node.get("attrs", {})
    dim = normalize_scatter_attrs(
        attrs.get("dim", 0),
        input_tensor["shape"],
        index_tensor["shape"],
        source_tensor["shape"],
        op_name=op_name,
    )
    expected_shape = _resolve_shape(op_name, input_tensor["shape"], index_tensor["shape"], source_tensor["shape"], attrs)
    if list(expected_shape) != list(output_tensor["shape"]):
        raise ValueError(f"{op_name} output shape must match input shape")
    expected_shape_spec = list(input_tensor.get("shape_spec", input_tensor["shape"]))
    if list(output_tensor.get("shape_spec", output_tensor["shape"])) != expected_shape_spec:
        raise ValueError(f"{op_name} output shape_spec must match input shape_spec")
    reduction: str | None = None
    if op_name == "scatter_reduce":
        reduction = normalize_scatter_reduce_name(attrs.get("reduce", "sum"))
        normalize_scatter_reduce_include_self(attrs.get("include_self", True))
    return dim, reduction


def _resolve_shape(
    op_name: str,
    input_shape: list[int],
    index_shape: list[int],
    source_shape: list[int],
    attrs: Mapping[str, Any],
) -> list[int]:
    if op_name == "scatter":
        return resolve_scatter_shape(input_shape, index_shape, source_shape, attrs.get("dim", 0))
    if op_name == "scatter_add":
        return resolve_scatter_add_shape(input_shape, index_shape, source_shape, attrs.get("dim", 0))
    return resolve_scatter_reduce_shape(
        input_shape,
        index_shape,
        source_shape,
        attrs.get("dim", 0),
        attrs.get("reduce", "sum"),
        attrs.get("include_self", True),
    )


def _update_body(op_name: str, reduction: str | None, storage_type: str) -> str:
    if op_name == "scatter":
        return "    y[out_idx] = source[src_idx];"
    if op_name == "scatter_add":
        return (
            f"    y[out_idx] = dinoml::math::cast<{storage_type}>("
            "dinoml::math::cast<float>(y[out_idx]) + dinoml::math::cast<float>(source[src_idx]));"
        )
    assert reduction is not None
    if reduction == "sum":
        return (
            f"    y[out_idx] = dinoml::math::cast<{storage_type}>("
            "dinoml::math::cast<float>(y[out_idx]) + dinoml::math::cast<float>(source[src_idx]));"
        )
    if reduction == "prod":
        return (
            f"    y[out_idx] = dinoml::math::cast<{storage_type}>("
            "dinoml::math::cast<float>(y[out_idx]) * dinoml::math::cast<float>(source[src_idx]));"
        )
    if reduction == "amax":
        return (
            f"    y[out_idx] = dinoml::math::cast<{storage_type}>(dinoml::math::cast<float>(y[out_idx]) > "
            "dinoml::math::cast<float>(source[src_idx]) ? dinoml::math::cast<float>(y[out_idx]) : "
            "dinoml::math::cast<float>(source[src_idx]));"
        )
    if reduction == "amin":
        return (
            f"    y[out_idx] = dinoml::math::cast<{storage_type}>(dinoml::math::cast<float>(y[out_idx]) < "
            "dinoml::math::cast<float>(source[src_idx]) ? dinoml::math::cast<float>(y[out_idx]) : "
            "dinoml::math::cast<float>(source[src_idx]));"
        )
    raise ValueError(f"Unsupported scatter_reduce reduction {reduction!r}")


def _function_name(node: Mapping[str, Any], tensor_map: Mapping[str, Mapping[str, Any]]) -> str:
    input_tensor = tensor_map[node["inputs"][0]]
    index_tensor = tensor_map[node["inputs"][1]]
    source_tensor = tensor_map[node["inputs"][2]]
    output_tensor = tensor_map[node["outputs"][0]]
    attrs = node.get("attrs", {})
    signature = {
        "op": str(node["op"]),
        "input_shape": list(input_tensor["shape"]),
        "index_shape": list(index_tensor["shape"]),
        "source_shape": list(source_tensor["shape"]),
        "output_shape": list(output_tensor["shape"]),
        "dim": int(attrs.get("dim", 0)),
        "reduce": attrs.get("reduce"),
        "include_self": attrs.get("include_self", True),
        "dtype": str(output_tensor["dtype"]),
        "index_dtype": str(index_tensor["dtype"]),
        "rank": len(output_tensor["shape"]),
    }
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


SCATTER_LOWERING = OpLowering(
    op_name="scatter",
    render_generated_kernel=render_generated_kernel,
    render_launch=render_launch,
    source_key=source_key,
    generated_function_name=generated_function_name,
)

SCATTER_ADD_LOWERING = OpLowering(
    op_name="scatter_add",
    render_generated_kernel=render_generated_kernel,
    render_launch=render_launch,
    source_key=source_key,
    generated_function_name=generated_function_name,
)

SCATTER_REDUCE_LOWERING = OpLowering(
    op_name="scatter_reduce",
    render_generated_kernel=render_generated_kernel,
    render_launch=render_launch,
    source_key=source_key,
    generated_function_name=generated_function_name,
)
