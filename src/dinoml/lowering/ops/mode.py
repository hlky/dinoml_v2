from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Mapping

from jinja2 import Environment, FileSystemLoader, StrictUndefined

from dinoml.lowering.ops.base import OpLowering
from dinoml.lowering.ops.template_rendering import supported_target_spec
from dinoml.lowering.shape_buffers import c_ident as _c_ident
from dinoml.lowering.target_specs import storage_type as target_storage_type
from dinoml.ops.reductions import MODE_DTYPES, normalize_mode_dim, resolve_mode_shape


def render_generated_kernel(target: str, node: Mapping[str, Any], tensor_map: Mapping[str, Mapping[str, Any]]) -> str | None:
    spec = supported_target_spec(target, "mode")
    if _is_paired_indices_node(node):
        return None
    context = _context(target, node, tensor_map)
    if not spec.is_gpu:
        return _render_template("mode_cpu.cpp.j2", context)
    context.update(spec.gpu_template_context())
    return _render_template("mode_gpu.j2", context)


def render_launch(
    target: str,
    node: Mapping[str, Any],
    tensor_map: Mapping[str, Mapping[str, Any]],
    kernel_manifest: Mapping[str, Any] | None = None,
) -> str:
    del kernel_manifest
    spec = supported_target_spec(target, "mode")
    if _is_paired_indices_node(node):
        return "/* paired mode_indices is produced by the mode_values launch */"
    func = _function_name(node, tensor_map)
    inp = _c_ident(node["inputs"][0])
    out = _c_ident(node["outputs"][0])
    if _is_paired_values_node(node):
        indices_out = _c_ident(str(node.get("attrs", {})["paired_indices_output"]))
        args = f"ptr_{inp}, ptr_{out}, ptr_{indices_out}, runtime_numel_{out}"
    else:
        args = f"ptr_{inp}, ptr_{out}, runtime_numel_{out}"
    if not spec.is_gpu:
        return f"if (int err = {func}({args})) return err;"
    return f"if (int err = {func}({args}, {spec.stream_expr})) return err;"


def source_key(target: str, node: Mapping[str, Any], tensor_map: Mapping[str, Mapping[str, Any]]) -> str | None:
    supported_target_spec(target, "mode")
    if _is_paired_indices_node(node):
        return None
    return f"{target}:{_function_name(node, tensor_map)}"


def generated_function_name(target: str, node: Mapping[str, Any], tensor_map: Mapping[str, Mapping[str, Any]]) -> str:
    del target
    return _function_name(node, tensor_map)


def _context(target: str, node: Mapping[str, Any], tensor_map: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    input_tensor = tensor_map[node["inputs"][0]]
    output_tensor = tensor_map[node["outputs"][0]]
    _validate_node_contract(node, input_tensor, output_tensor)
    input_dtype = str(input_tensor["dtype"])
    output_kind = "indices" if node["op"] == "mode_indices" else "values"
    if _is_paired_values_node(node):
        output_kind = "both"
    return {
        "func": _function_name(node, tensor_map),
        "kernel": f"{_function_name(node, tensor_map)}_kernel",
        "input_storage_type": target_storage_type(input_dtype, target),
        "value_output_storage_type": target_storage_type(input_dtype, target),
        "index_output_storage_type": "int64_t",
        "input_dtype": input_dtype,
        "input_is_bool": input_dtype == "bool",
        "input_is_float": input_dtype in {"float16", "float32", "bfloat16"},
        "output_kind": output_kind,
        "cols": int(input_tensor["shape"][-1]),
        "block_size": 256,
    }


def _validate_node_contract(
    node: Mapping[str, Any],
    input_tensor: Mapping[str, Any],
    output_tensor: Mapping[str, Any],
) -> None:
    if node["op"] not in {"mode_values", "mode_indices"}:
        raise ValueError(f"Unsupported mode op: {node['op']}")
    if len(node.get("inputs", [])) != 1:
        raise ValueError("mode expects exactly one input")
    if len(node.get("outputs", [])) != 1:
        raise ValueError("mode expects exactly one output")
    input_dtype = str(input_tensor["dtype"])
    if input_dtype not in MODE_DTYPES:
        raise NotImplementedError(f"mode lowering does not support dtype {input_dtype}")
    expected_dtype = input_dtype if node["op"] == "mode_values" else "int64"
    if str(output_tensor["dtype"]) != expected_dtype:
        raise ValueError(f"{node['op']} output dtype must be {expected_dtype}, got {output_tensor['dtype']}")
    if not input_tensor["shape"]:
        raise ValueError("mode requires a ranked tensor")
    attrs = node.get("attrs", {})
    dim = normalize_mode_dim(attrs.get("dim", -1), len(input_tensor["shape"]))
    if dim != len(input_tensor["shape"]) - 1:
        raise NotImplementedError("mode lowering currently supports only the last dimension")
    shape_spec = input_tensor.get("shape_spec", input_tensor["shape"])
    cols = input_tensor["shape"][-1]
    if not isinstance(shape_spec[-1], int) or not isinstance(cols, int) or int(cols) <= 0:
        raise ValueError("mode lowering requires a positive static last dimension")
    expected = resolve_mode_shape(input_tensor["shape"], dim, bool(attrs.get("keepdim", False)))
    if list(output_tensor["shape"]) != expected:
        raise ValueError("mode output shape does not match mode attrs")


def _is_paired_values_node(node: Mapping[str, Any]) -> bool:
    return node["op"] == "mode_values" and "paired_indices_output" in node.get("attrs", {})


def _is_paired_indices_node(node: Mapping[str, Any]) -> bool:
    return node["op"] == "mode_indices" and "paired_values_output" in node.get("attrs", {})


def _function_name(node: Mapping[str, Any], tensor_map: Mapping[str, Mapping[str, Any]]) -> str:
    input_tensor = tensor_map[node["inputs"][0]]
    attrs = node.get("attrs", {})
    dim = normalize_mode_dim(attrs.get("dim", -1), len(input_tensor["shape"]))
    signature = {
        "op": str(node["op"]),
        "shape": list(input_tensor["shape"]),
        "dtype": str(input_tensor["dtype"]),
        "dim": dim,
        "keepdim": bool(attrs.get("keepdim", False)),
        "paired_indices": bool(node["op"] == "mode_values" and "paired_indices_output" in attrs),
        "paired_values": bool(node["op"] == "mode_indices" and "paired_values_output" in attrs),
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


MODE_VALUES_LOWERING = OpLowering(
    op_name="mode_values",
    render_generated_kernel=render_generated_kernel,
    render_launch=render_launch,
    source_key=source_key,
    generated_function_name=generated_function_name,
)
MODE_INDICES_LOWERING = OpLowering(
    op_name="mode_indices",
    render_generated_kernel=render_generated_kernel,
    render_launch=render_launch,
    source_key=source_key,
    generated_function_name=generated_function_name,
)
MODE_LOWERINGS = {
    MODE_VALUES_LOWERING.op_name: MODE_VALUES_LOWERING,
    MODE_INDICES_LOWERING.op_name: MODE_INDICES_LOWERING,
}
