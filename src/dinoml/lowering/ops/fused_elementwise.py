from __future__ import annotations

import re
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from jinja2 import Environment, FileSystemLoader, StrictUndefined

from dinoml.ir import canonical_json, dtype_nbytes
from dinoml.lowering.cpp_types import cpu_storage_type, cuda_storage_type
from dinoml.lowering.ops.base import OpLowering
from dinoml.ops.elementwise import ELEMENTWISE_BY_NAME


@dataclass(frozen=True)
class _VectorPlan:
    width: int = 1
    bytes: int = 0
    cpp_type: str = ""


def render_generated_kernel(target: str, node: Mapping[str, Any], tensor_map: Mapping[str, Mapping[str, Any]]) -> str:
    if target == "cpu":
        return _render_template("fused_elementwise_cpu.cpp.j2", _context(node, tensor_map, target=target))
    if target == "cuda":
        return _render_template("fused_elementwise_cuda.cu.j2", _context(node, tensor_map, target=target))
    raise ValueError(f"Unsupported fused_elementwise target: {target}")


def render_launch(
    target: str,
    node: Mapping[str, Any],
    tensor_map: Mapping[str, Mapping[str, Any]],
    kernel_manifest: Mapping[str, Any] | None = None,
) -> str:
    del kernel_manifest
    func = _function_name(node)
    output_shape = tensor_map[node["outputs"][0]]["shape"]
    output_shape_ident = f"shape_{_c_ident(node['outputs'][0])}"
    inputs = _inputs(node, tensor_map, output_shape, target=target)
    outputs = [_output_info(name, tensor_map, target=target) for name in node["outputs"]]
    args = _kernel_arg_names(inputs, outputs, output_shape=output_shape, output_shape_ident=output_shape_ident).split(", ")
    args.append(_runtime_total_expr(node, tensor_map))
    if target == "cpu":
        return f"if (int err = {func}({', '.join(args)})) return err;"
    if target == "cuda":
        return f"if (int err = {func}({', '.join(args)}, session->stream)) return err;"
    raise ValueError(f"Unsupported fused_elementwise target: {target}")


def source_key(target: str, node: Mapping[str, Any], tensor_map: Mapping[str, Mapping[str, Any]]) -> str:
    del tensor_map
    if target not in {"cpu", "cuda"}:
        raise ValueError(f"Unsupported fused_elementwise target: {target}")
    return f"{target}:{_function_name(node)}"


def generated_function_name(target: str, node: Mapping[str, Any], tensor_map: Mapping[str, Mapping[str, Any]]) -> str:
    del target, tensor_map
    return _function_name(node)


def _context(node: Mapping[str, Any], tensor_map: Mapping[str, Mapping[str, Any]], *, target: str) -> dict[str, Any]:
    output_shape = tensor_map[node["outputs"][0]]["shape"]
    output_dtype = str(tensor_map[node["outputs"][0]]["dtype"])
    compute_dtype = _fused_compute_dtype(node, tensor_map, output_dtype)
    func = _function_name(node)
    storage_type = _storage_type(output_dtype, target=target)
    compute_type = _compute_type(compute_dtype, node, target=target)
    storage_alias = f"{func}_storage_t"
    compute_alias = f"{func}_compute_t"
    inputs = _inputs(node, tensor_map, output_shape, target=target)
    outputs = [_output_info(name, tensor_map, target=target) for name in node["outputs"]]
    output_shape_ident = f"shape_{outputs[0]['ident']}"
    vector_plan = _cuda_vector_plan(inputs, outputs, output_shape, output_dtype) if target == "cuda" else _VectorPlan()
    suffix_inner_extent = None if vector_plan.width > 1 else _suffix_inner_extent(inputs, output_shape)
    return {
        "func": func,
        "kernel": f"{func}_kernel",
        "total": _numel(output_shape),
        "suffix_inner_extent": suffix_inner_extent,
        "broadcast_helpers": _broadcast_helpers(inputs, output_shape, device=target == "cuda"),
        "accessor_decls": _accessor_decls(inputs, outputs),
        "kernel_params": _kernel_params(
            inputs,
            outputs,
            output_shape=output_shape,
            storage_type=storage_alias,
            output_shape_ident=output_shape_ident,
        ),
        "call_params": f"{_kernel_params(inputs, outputs, output_shape=output_shape, storage_type=storage_alias, output_shape_ident=output_shape_ident)}, int64_t runtime_total",
        "kernel_arg_names": _kernel_arg_names(inputs, outputs, output_shape=output_shape, output_shape_ident=output_shape_ident),
        "null_check": _null_check(inputs, outputs),
        "storage_type": storage_type,
        "compute_type": compute_type,
        "storage_alias": storage_alias,
        "compute_alias": compute_alias,
        "use_fp32_acc": compute_type == "float" and storage_type != "float",
        "scalar_body": _scalar_body(
            inputs,
            outputs,
            node["attrs"]["sub_ops"],
            compute_type=compute_alias,
            output_shape=output_shape,
            output_shape_ident=output_shape_ident,
        ),
        "inner_scalar_body": _scalar_body(
            inputs,
            outputs,
            node["attrs"]["sub_ops"],
            compute_type=compute_alias,
            output_shape=output_shape,
            output_shape_ident=output_shape_ident,
            suffix_index_var="inner_idx",
            suffix_extent=suffix_inner_extent,
        )
        if suffix_inner_extent
        else "",
        "vector_width": vector_plan.width,
        "vector_type": vector_plan.cpp_type,
        "vector_bytes": vector_plan.bytes,
        "vector_body": _vector_body(
            inputs,
            outputs,
            node["attrs"]["sub_ops"],
            storage_type=storage_alias,
            compute_type=compute_alias,
            vector_width=vector_plan.width,
            vector_type=vector_plan.cpp_type,
            output_shape_ident=output_shape_ident,
        )
        if vector_plan.width > 1
        else "",
        "elements_per_thread": _cuda_elements_per_thread(node, output_shape) if target == "cuda" else 1,
        "block_size": _cuda_block_size() if target == "cuda" else 1,
    }


def _inputs(
    node: Mapping[str, Any],
    tensor_map: Mapping[str, Mapping[str, Any]],
    output_shape: Sequence[int],
    *,
    target: str,
) -> list[dict[str, Any]]:
    result = []
    for name in node["inputs"]:
        shape = tensor_map[name]["shape"]
        dtype = str(tensor_map[name]["dtype"])
        index_kind, index_expr = _broadcast_index_expr(shape, output_shape)
        input_numel = _numel(shape)
        result.append(
            {
                "name": name,
                "ident": _c_ident(name),
                "shape": shape,
                "dtype": dtype,
                "storage_type": _storage_type(dtype, target=target),
                "numel": input_numel,
                "index_kind": index_kind,
                "broadcast_func": _broadcast_function_name(node, name),
                "broadcast_needed": shape != output_shape,
                "broadcast_helper_needed": index_expr is None,
                "index_expr": index_expr,
            }
        )
    return result


def _output_info(name: str, tensor_map: Mapping[str, Mapping[str, Any]], *, target: str) -> dict[str, Any]:
    dtype = str(tensor_map[name]["dtype"])
    return {"name": name, "ident": _c_ident(name), "dtype": dtype, "storage_type": _storage_type(dtype, target=target)}


def _broadcast_helpers(inputs: Sequence[Mapping[str, Any]], output_shape: Sequence[int], *, device: bool = False) -> list[str]:
    helpers = []
    prefix = "__device__ " if device else ""
    for input_info in inputs:
        if not input_info["broadcast_helper_needed"]:
            continue
        helpers.append(
            _broadcast_helper(
                prefix=prefix,
                name=input_info["broadcast_func"],
                input_shape=input_info["shape"],
                output_shape=output_shape,
            )
        )
    return helpers


def _accessor_decls(inputs: Sequence[Mapping[str, Any]], outputs: Sequence[Mapping[str, Any]]) -> str:
    lines = []
    for input_info in inputs:
        pattern = _accessor_pattern(input_info["index_kind"])
        if pattern is None:
            continue
        lines.append(
            f"  const dinoml::access::TensorAccessor acc_{input_info['ident']}{{"
            f"dinoml::access::Pattern::{pattern}, 0, {int(input_info['numel'])}, -1, -1}};"
        )
    for output in outputs:
        lines.append(
            f"  const dinoml::access::TensorAccessor acc_{output['ident']}{{"
            "dinoml::access::Pattern::kDense, 0, 1, -1, -1};"
        )
    return "\n".join(lines)


def _accessor_pattern(index_kind: str) -> str | None:
    if index_kind == "full":
        return "kDense"
    if index_kind == "scalar":
        return "kScalar"
    if index_kind == "suffix":
        return "kSuffix"
    return None


def _broadcast_helper(*, prefix: str, name: str, input_shape: Sequence[int], output_shape: Sequence[int]) -> str:
    aligned_input = [1] * (len(output_shape) - len(input_shape)) + list(input_shape)
    input_axis = [axis - (len(output_shape) - len(input_shape)) for axis in range(len(output_shape))]
    lines = [
        f"{prefix}static inline int64_t {name}(int64_t idx, const int64_t* input_shape, const int64_t* output_shape) {{",
        "  int64_t remaining = idx;",
        "  int64_t input_idx = 0;",
        "  int64_t input_stride = 1;",
        "  int64_t coord = 0;",
        "  int64_t input_dim = 1;",
    ]
    for axis, input_dim in reversed(list(enumerate(aligned_input))):
        source_axis = input_axis[axis]
        lines.append(f"  coord = remaining % output_shape[{axis}];")
        lines.append(f"  remaining /= output_shape[{axis}];")
        if source_axis >= 0:
            lines.append(f"  input_dim = input_shape[{source_axis}];")
        else:
            lines.append("  input_dim = 1;")
        if int(input_dim) != 1:
            lines.append("  input_idx += coord * input_stride;")
        lines.append("  input_stride *= input_dim;")
    lines.extend(["  return input_idx;", "}"])
    return "\n".join(lines)


def _scalar_body(
    inputs: Sequence[Mapping[str, Any]],
    outputs: Sequence[Mapping[str, Any]],
    sub_ops: Sequence[Mapping[str, Any]],
    *,
    compute_type: str,
    output_shape: Sequence[int],
    output_shape_ident: str,
    suffix_index_var: str | None = None,
    suffix_extent: int | None = None,
) -> str:
    exprs: dict[str, str] = {}
    lines = []
    for input_info in inputs:
        index_expr = _input_index_expr(
            input_info,
            output_shape=output_shape,
            output_shape_ident=output_shape_ident,
            suffix_index_var=suffix_index_var,
            suffix_extent=suffix_extent,
        )
        value = f"v_{input_info['ident']}"
        value_type = "bool" if input_info["dtype"] == "bool" else compute_type
        lines.append(f"    {value_type} {value} = dinoml::math::cast<{value_type}>(ptr_{input_info['ident']}[{index_expr}]);")
        exprs[input_info["name"]] = value

    for sub_op in sub_ops:
        op = sub_op["op"]
        elementwise_spec = ELEMENTWISE_BY_NAME.get(op)
        if elementwise_spec is None:
            raise ValueError(f"Unsupported fused elementwise sub-op: {op}")
        args = [exprs[name] for name in sub_op["inputs"]]
        output = sub_op["outputs"][0]
        ident = f"v_{_c_ident(output)}"
        result_type = _sub_op_result_type(sub_op, elementwise_spec.output_dtype, compute_type)
        if op == "cast":
            lines.append(f"    {result_type} {ident} = dinoml::math::cast<{result_type}>({args[0]});")
        else:
            args.extend(_attr_args(sub_op, elementwise_spec.attr_defaults))
            lines.append(f"    {result_type} {ident} = dinoml::math::{elementwise_spec.math_func}({', '.join(args)});")
        exprs[output] = ident

    for output in outputs:
        output_name = output["name"]
        if output_name not in exprs:
            raise ValueError(f"Fused output {output_name} was not produced by sub_ops")
        lines.append(
            f"    ptr_{output['ident']}[acc_{output['ident']}.index(idx)] = "
            f"dinoml::math::cast<{output['storage_type']}>({exprs[output_name]});"
        )
    return "\n".join(lines)


def _vector_body(
    inputs: Sequence[Mapping[str, Any]],
    outputs: Sequence[Mapping[str, Any]],
    sub_ops: Sequence[Mapping[str, Any]],
    *,
    storage_type: str,
    compute_type: str,
    vector_width: int,
    vector_type: str,
    output_shape_ident: str,
) -> str:
    lines = []
    for input_info in inputs:
        if input_info["index_kind"] == "full":
            lines.append(
                f"  const {vector_type}* vec_ptr_{input_info['ident']} = "
                f"dinoml::access::strided_address<const {input_info['storage_type']}, const {vector_type}, true>(ptr_{input_info['ident']}, vec_idx, 0, 0, 0);"
            )
            lines.append(f"  {vector_type} raw_{input_info['ident']} = *vec_ptr_{input_info['ident']};")
            lines.append(
                f"  const {input_info['storage_type']}* lane_{input_info['ident']} = "
                f"reinterpret_cast<const {input_info['storage_type']}*>(&raw_{input_info['ident']});"
            )
    for output in outputs:
        lines.append(f"  {vector_type} raw_{output['ident']};")
        lines.append(f"  {storage_type}* lane_{output['ident']} = reinterpret_cast<{storage_type}*>(&raw_{output['ident']});")
    for lane in range(vector_width):
        exprs: dict[str, str] = {}
        for input_info in inputs:
            value = f"v_{input_info['ident']}_{lane}"
            if input_info["index_kind"] == "full":
                source = f"lane_{input_info['ident']}[{lane}]"
            elif input_info["index_kind"] in {"scalar", "suffix"}:
                source = f"ptr_{input_info['ident']}[acc_{input_info['ident']}.index(vec_idx * {vector_width} + {lane})]"
            else:
                raise ValueError(f"Input {input_info['name']} is not vectorizable")
            value_type = "bool" if input_info["dtype"] == "bool" else compute_type
            lines.append(f"  {value_type} {value} = dinoml::math::cast<{value_type}>({source});")
            exprs[input_info["name"]] = value
        for sub_op in sub_ops:
            op = sub_op["op"]
            elementwise_spec = ELEMENTWISE_BY_NAME.get(op)
            if elementwise_spec is None:
                raise ValueError(f"Unsupported fused elementwise sub-op: {op}")
            args = [exprs[name] for name in sub_op["inputs"]]
            output_name = sub_op["outputs"][0]
            ident = f"v_{_c_ident(output_name)}_{lane}"
            result_type = _sub_op_result_type(sub_op, elementwise_spec.output_dtype, compute_type)
            if op == "cast":
                lines.append(f"  {result_type} {ident} = dinoml::math::cast<{result_type}>({args[0]});")
            else:
                args.extend(_attr_args(sub_op, elementwise_spec.attr_defaults))
                lines.append(f"  {result_type} {ident} = dinoml::math::{elementwise_spec.math_func}({', '.join(args)});")
            exprs[output_name] = ident
        for output in outputs:
            output_name = output["name"]
            if output_name not in exprs:
                raise ValueError(f"Fused output {output_name} was not produced by sub_ops")
            lines.append(f"  lane_{output['ident']}[{lane}] = dinoml::math::cast<{storage_type}>({exprs[output_name]});")
    for output in outputs:
        lines.append(
            f"  {vector_type}* vec_ptr_{output['ident']} = "
            f"dinoml::access::strided_address<{storage_type}, {vector_type}, true>(ptr_{output['ident']}, vec_idx, 0, 0, 0);"
        )
        lines.append(f"  *vec_ptr_{output['ident']} = raw_{output['ident']};")
    return "\n".join(lines)


def _kernel_params(
    inputs: Sequence[Mapping[str, Any]],
    outputs: Sequence[Mapping[str, Any]],
    *,
    output_shape: Sequence[int],
    storage_type: str,
    output_shape_ident: str,
) -> str:
    del storage_type
    input_params = [f"const {item['storage_type']}* DINO_RESTRICT ptr_{item['ident']}" for item in inputs]
    output_params = [f"{item['storage_type']}* DINO_RESTRICT ptr_{item['ident']}" for item in outputs]
    shape_params = [f"const int64_t* shape_{item['ident']}" for item in inputs if item["index_kind"] == "generic"]
    if shape_params:
        shape_params.append(f"const int64_t* {output_shape_ident}")
    return ", ".join([*input_params, *output_params, *shape_params])


def _kernel_arg_names(
    inputs: Sequence[Mapping[str, Any]],
    outputs: Sequence[Mapping[str, Any]],
    *,
    output_shape: Sequence[int],
    output_shape_ident: str,
) -> str:
    names = [f"ptr_{item['ident']}" for item in [*inputs, *outputs]]
    names.extend(f"shape_{item['ident']}" for item in inputs if item["index_kind"] == "generic")
    if any(item["index_kind"] == "generic" for item in inputs):
        names.append(output_shape_ident)
    return ", ".join(names)


def _runtime_total_expr(node: Mapping[str, Any], tensor_map: Mapping[str, Mapping[str, Any]]) -> str:
    for name in node["outputs"]:
        if tensor_map[name].get("kind") in {"output", "temporary"}:
            return f"runtime_numel_{_c_ident(name)}"
    for name in node["inputs"]:
        if tensor_map[name].get("kind") == "input":
            return f"runtime_numel_{_c_ident(name)}"
    return str(_numel(tensor_map[node["outputs"][0]]["shape"]))


def _null_check(inputs: Sequence[Mapping[str, Any]], outputs: Sequence[Mapping[str, Any]]) -> str:
    names = [f"ptr_{item['ident']}" for item in [*inputs, *outputs]]
    return " || ".join(f"{name} == nullptr" for name in names)


def _input_index_expr(
    input_info: Mapping[str, Any],
    *,
    output_shape: Sequence[int],
    output_shape_ident: str,
    suffix_index_var: str | None = None,
    suffix_extent: int | None = None,
) -> str:
    if input_info["index_kind"] in {"full", "scalar", "suffix"}:
        if input_info["index_kind"] == "suffix" and suffix_index_var is not None:
            return f"acc_{input_info['ident']}.index(idx, {suffix_index_var})"
        return f"acc_{input_info['ident']}.index(idx)"
    if input_info["index_kind"] == "suffix" and suffix_index_var is not None:
        input_numel = int(input_info["numel"])
        if suffix_extent == input_numel:
            return suffix_index_var
        return f"{suffix_index_var} % {input_numel}"
    index_expr = input_info["index_expr"]
    if index_expr is None:
        return f"{input_info['broadcast_func']}(idx, shape_{input_info['ident']}, {output_shape_ident})"
    return str(index_expr)


def _broadcast_index_expr(input_shape: Sequence[int], output_shape: Sequence[int]) -> tuple[str, str | None]:
    if list(input_shape) == list(output_shape):
        return "full", "idx"
    input_numel = _numel(input_shape)
    if input_numel == 1:
        return "scalar", "0"
    aligned_input = [1] * (len(output_shape) - len(input_shape)) + list(input_shape)
    prefix_len = len(output_shape) - len(input_shape)
    if all(dim == 1 for dim in aligned_input[:prefix_len]) and list(input_shape) == list(output_shape[-len(input_shape) :]):
        return "suffix", f"idx % {input_numel}"
    return "generic", None


def _suffix_inner_extent(inputs: Sequence[Mapping[str, Any]], output_shape: Sequence[int]) -> int | None:
    suffix_inputs = [item for item in inputs if item["index_kind"] == "suffix"]
    if not suffix_inputs:
        return None
    if any(item["index_kind"] == "generic" for item in inputs):
        return None
    inner_extent = max(int(item["numel"]) for item in suffix_inputs)
    total = _numel(output_shape)
    if inner_extent <= 1 or total % inner_extent != 0:
        return None
    return inner_extent


def _cuda_elements_per_thread(node: Mapping[str, Any], output_shape: Sequence[int]) -> int:
    override = os.environ.get("DINOML_CUDA_ELEMENTWISE_EPT")
    if override:
        value = int(override)
        if value <= 0:
            raise ValueError("DINOML_CUDA_ELEMENTWISE_EPT must be positive")
        return value
    del node, output_shape
    return 1


def _cuda_block_size() -> int:
    value = int(os.environ.get("DINOML_CUDA_ELEMENTWISE_BLOCK", "256"))
    if value <= 0:
        raise ValueError("DINOML_CUDA_ELEMENTWISE_BLOCK must be positive")
    return value


def _cuda_vector_plan(
    inputs: Sequence[Mapping[str, Any]],
    outputs: Sequence[Mapping[str, Any]],
    output_shape: Sequence[int],
    output_dtype: str,
) -> _VectorPlan:
    width_override = os.environ.get("DINOML_CUDA_ELEMENTWISE_VECTOR_WIDTH", "auto").lower()
    if width_override in {"0", "1", "off", "false", "none"}:
        return _VectorPlan()
    supported_dtype = output_dtype in {"float16", "float32", "bfloat16"}
    can_vectorize_graph = (
        supported_dtype
        and all(item["dtype"] == output_dtype for item in inputs)
        and all(item["index_kind"] in {"full", "scalar", "suffix"} for item in inputs)
    )
    if not can_vectorize_graph:
        if width_override == "auto":
            return _VectorPlan()
        raise ValueError("Requested fused_elementwise vectorization for a non-vectorizable graph")
    total = _numel(output_shape)
    element_nbytes = dtype_nbytes(output_dtype)
    if width_override != "auto":
        width = int(width_override)
        if total % width != 0:
            raise ValueError(f"Requested vector width {width}, but element count {total} is not divisible by it")
        vector_bytes = width * element_nbytes
        return _VectorPlan(width=width, bytes=vector_bytes, cpp_type=_cuda_vector_type(output_dtype, vector_bytes))

    max_vector_bytes = int(os.environ.get("DINOML_CUDA_ELEMENTWISE_VECTOR_BYTES", "16"))
    if max_vector_bytes <= 0:
        return _VectorPlan()
    for vector_bytes in (16, 8, 4):
        if vector_bytes > max_vector_bytes or vector_bytes % element_nbytes != 0:
            continue
        width = vector_bytes // element_nbytes
        if width > 1 and total % width == 0:
            return _VectorPlan(width=width, bytes=vector_bytes, cpp_type=_cuda_vector_type(output_dtype, vector_bytes))
    return _VectorPlan()


def _cuda_vector_type(dtype: str, vector_bytes: int) -> str:
    if dtype == "float32":
        if vector_bytes == 16:
            return "float4"
        if vector_bytes == 8:
            return "float2"
    if vector_bytes == 16:
        return "uint4"
    if vector_bytes == 8:
        return "uint2"
    if vector_bytes == 4:
        return "uint32_t"
    raise ValueError(f"Unsupported CUDA raw vector byte width: {vector_bytes}")


def _compute_type(dtype: str, node: Mapping[str, Any], *, target: str) -> str:
    policy = str(node.get("attrs", {}).get("accumulation", os.environ.get("DINOML_ELEMENTWISE_ACCUM", "auto"))).lower()
    storage_type = _storage_type(dtype, target=target)
    if target == "cpu" and dtype in {"float16", "bfloat16"}:
        return "float"
    if policy in {"storage", "native"}:
        return storage_type
    if policy in {"fp32", "float32"}:
        return "float"
    if policy != "auto":
        raise ValueError(f"Unsupported fused_elementwise accumulation policy: {policy!r}")
    if dtype in {"float16", "bfloat16"}:
        return "float"
    return storage_type


def _fused_compute_dtype(
    node: Mapping[str, Any],
    tensor_map: Mapping[str, Mapping[str, Any]],
    output_dtype: str,
) -> str:
    for name in node.get("inputs", []):
        dtype = str(tensor_map[name]["dtype"])
        if dtype != "bool":
            return dtype
    return output_dtype


def _sub_op_result_type(sub_op: Mapping[str, Any], static_output_dtype: str | None, compute_type: str) -> str:
    if static_output_dtype == "bool":
        return "bool"
    if str(sub_op.get("op")) == "cast" and str(sub_op.get("attrs", {}).get("dtype")) == "bool":
        return "bool"
    return compute_type


def _storage_type(dtype: str, *, target: str) -> str:
    if target == "cpu":
        return cpu_storage_type(dtype)
    if target == "cuda":
        return cuda_storage_type(dtype)
    raise ValueError(f"Unsupported fused_elementwise target: {target}")


def _render_template(name: str, context: Mapping[str, Any]) -> str:
    env = Environment(
        loader=FileSystemLoader(str(Path(__file__).resolve().parent / "templates")),
        undefined=StrictUndefined,
        trim_blocks=True,
        lstrip_blocks=True,
        keep_trailing_newline=True,
    )
    return env.get_template(name).render(**context)


def _function_name(node: Mapping[str, Any]) -> str:
    return f"fused_elementwise_{_short_signature_hash(_function_signature(node))}"


def _broadcast_function_name(node: Mapping[str, Any], tensor_name: str) -> str:
    return f"{_function_name(node)}_idx_{_c_ident(tensor_name)}"


def _function_signature(node: Mapping[str, Any]) -> dict[str, Any]:
    """Return the per-node codegen signature used for stable function names.

    Tensor names stay in this signature because generated function parameter and
    local variable names currently include tensor identifiers. A future module
    source dedup pass can use a normalized signature that rewrites operands to
    positional names before rendering only one copy of identical kernels.
    """

    attrs = node.get("attrs", {})
    return {
        "op": "fused_elementwise",
        "inputs": list(node.get("inputs", ())),
        "outputs": list(node.get("outputs", ())),
        "sub_ops": _canonical_sub_ops(attrs.get("sub_ops", ())),
        "accumulation": str(attrs.get("accumulation", "auto")),
    }


def _canonical_sub_ops(sub_ops: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    canonical = []
    for sub_op in sub_ops:
        canonical.append(
            {
                "op": str(sub_op["op"]),
                "inputs": list(sub_op.get("inputs", ())),
                "outputs": list(sub_op.get("outputs", ())),
                "attrs": dict(sorted(sub_op.get("attrs", {}).items())),
            }
        )
    return canonical


def _short_signature_hash(signature: Mapping[str, Any]) -> str:
    import hashlib

    return hashlib.sha256(canonical_json(signature).encode("utf-8")).hexdigest()[:12]


def _numel(shape: Sequence[int]) -> int:
    return int(np.prod(list(shape), dtype=np.int64))


def _attr_args(sub_op: Mapping[str, Any], defaults: Sequence[tuple[str, Any]]) -> list[str]:
    attrs = sub_op.get("attrs", {})
    values = []
    for name, default in defaults:
        if name == "approximation":
            continue
        values.append(_literal(attrs.get(name, default)))
    return values


def _literal(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return f"{value:.9g}f"
    raise ValueError(f"Unsupported fused elementwise scalar attr literal: {value!r}")


def _c_ident(name: str) -> str:
    ident = re.sub(r"[^0-9A-Za-z_]", "_", name)
    if not ident or ident[0].isdigit():
        ident = f"_{ident}"
    ident = re.sub(r"_(\d+)$", r"__\1", ident)
    return ident


FUSED_ELEMENTWISE_LOWERING = OpLowering(
    op_name="fused_elementwise",
    render_generated_kernel=render_generated_kernel,
    render_launch=render_launch,
    source_key=source_key,
    generated_function_name=generated_function_name,
)
