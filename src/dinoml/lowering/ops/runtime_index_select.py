from __future__ import annotations

import hashlib
from typing import Any, Mapping

from dinoml.lowering.ops.base import OpLowering
from dinoml.lowering.ops.template_rendering import supported_target_spec
from dinoml.lowering.shape_buffers import c_ident as _c_ident
from dinoml.lowering.target_specs import storage_type as target_storage_type
from dinoml.ops.collections import (
    COLLECTION_DTYPES,
    GATHER_INDEX_DTYPES,
    normalize_index_select_dim,
    resolve_runtime_index_select_shape,
)


def render_generated_kernel(target: str, node: Mapping[str, Any], tensor_map: Mapping[str, Mapping[str, Any]]) -> str:
    spec = supported_target_spec(target, "runtime_index_select")
    context = _context(target, node, tensor_map)
    if not spec.is_gpu:
        return _render_cpu(context)
    context.update(spec.gpu_template_context())
    return _render_gpu(context)


def render_launch(
    target: str,
    node: Mapping[str, Any],
    tensor_map: Mapping[str, Mapping[str, Any]],
    kernel_manifest: Mapping[str, Any] | None = None,
) -> str:
    del kernel_manifest
    spec = supported_target_spec(target, "runtime_index_select")
    func = _function_name(node, tensor_map)
    x = _c_ident(node["inputs"][0])
    indices = _c_ident(node["inputs"][1])
    out = _c_ident(node["outputs"][0])
    args = (
        f"ptr_{x}, ptr_{indices}, ptr_{out}, runtime_numel_{out}, "
        f"static_cast<int64_t>(shape_{out}_0), "
        f"static_cast<int64_t>(runtime_numel_{out} / shape_{out}_0), "
        f"static_cast<int64_t>(shape_{x}_0)"
    )
    if not spec.is_gpu:
        return f"if (int err = {func}({args})) return err;"
    return f"if (int err = {func}({args}, {spec.stream_expr})) return err;"


def source_key(target: str, node: Mapping[str, Any], tensor_map: Mapping[str, Mapping[str, Any]]) -> str:
    supported_target_spec(target, "runtime_index_select")
    return f"{target}:{_function_name(node, tensor_map)}"


def generated_function_name(target: str, node: Mapping[str, Any], tensor_map: Mapping[str, Mapping[str, Any]]) -> str:
    del target
    return _function_name(node, tensor_map)


def _context(target: str, node: Mapping[str, Any], tensor_map: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    input_tensor = tensor_map[node["inputs"][0]]
    index_tensor = tensor_map[node["inputs"][1]]
    output_tensor = tensor_map[node["outputs"][0]]
    _validate_node_contract(node, input_tensor, index_tensor, output_tensor)
    dtype = str(output_tensor["dtype"])
    index_dtype = str(index_tensor["dtype"])
    return {
        "func": _function_name(node, tensor_map),
        "kernel": f"{_function_name(node, tensor_map)}_kernel",
        "storage_type": target_storage_type(dtype, target),
        "index_storage_type": target_storage_type(index_dtype, target),
        "block_size": 256,
    }


def _validate_node_contract(
    node: Mapping[str, Any],
    input_tensor: Mapping[str, Any],
    index_tensor: Mapping[str, Any],
    output_tensor: Mapping[str, Any],
) -> None:
    if node["op"] != "runtime_index_select":
        raise ValueError(f"Unsupported collection op: {node['op']}")
    if len(node.get("inputs", [])) != 2:
        raise ValueError("runtime_index_select expects two tensor inputs")
    if len(node.get("outputs", [])) != 1:
        raise ValueError("runtime_index_select expects exactly one output")
    dtype = str(output_tensor["dtype"])
    if dtype not in COLLECTION_DTYPES:
        raise NotImplementedError(f"runtime_index_select lowering does not support dtype {output_tensor['dtype']}")
    if str(input_tensor["dtype"]) != dtype:
        raise ValueError("runtime_index_select input and output dtype must match")
    if str(index_tensor["dtype"]) not in GATHER_INDEX_DTYPES:
        raise ValueError("runtime_index_select indices must be int32 or int64")
    dim = normalize_index_select_dim(node.get("attrs", {}).get("dim", 0), len(input_tensor["shape"]))
    if dim != 0:
        raise ValueError("runtime_index_select lowering currently supports dim=0 only")
    expected_shape = resolve_runtime_index_select_shape(input_tensor["shape"], index_tensor["shape"], dim)
    if list(expected_shape) != list(output_tensor["shape"]):
        raise ValueError("runtime_index_select output shape does not match selected indices")


def _render_cpu(context: Mapping[str, Any]) -> str:
    return f"""static int {context["func"]}(
    const {context["storage_type"]}* DINO_RESTRICT x,
    const {context["index_storage_type"]}* DINO_RESTRICT indices,
    {context["storage_type"]}* DINO_RESTRICT y,
    int64_t runtime_numel,
    int64_t select_count,
    int64_t inner_size,
    int64_t input_dim0) {{
  if (x == nullptr || indices == nullptr || y == nullptr) {{
    return dino_runtime_fail("{context["func"]} received null pointer");
  }}
  if (select_count <= 0 || inner_size <= 0 || input_dim0 <= 0) {{
    return dino_runtime_fail("{context["func"]} received invalid runtime shape");
  }}
  for (int64_t idx = 0; idx < runtime_numel; ++idx) {{
    const int64_t row = idx / inner_size;
    const int64_t inner = idx - row * inner_size;
    const int64_t selected = static_cast<int64_t>(indices[row]);
    if (selected < 0 || selected >= input_dim0) {{
      return dino_runtime_fail("{context["func"]} index out of bounds");
    }}
    y[idx] = x[selected * inner_size + inner];
  }}
  return 0;
}}
"""


def _render_gpu(context: Mapping[str, Any]) -> str:
    return f"""__global__ void {context["kernel"]}(
    const {context["storage_type"]}* DINO_RESTRICT x,
    const {context["index_storage_type"]}* DINO_RESTRICT indices,
    {context["storage_type"]}* DINO_RESTRICT y,
    int64_t runtime_numel,
    int64_t inner_size) {{
  const int64_t stride = static_cast<int64_t>(blockDim.x) * static_cast<int64_t>(gridDim.x);
  for (int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x; idx < runtime_numel; idx += stride) {{
    const int64_t row = idx / inner_size;
    const int64_t inner = idx - row * inner_size;
    const int64_t selected = static_cast<int64_t>(indices[row]);
    y[idx] = x[selected * inner_size + inner];
  }}
}}

static int {context["func"]}(
    const {context["storage_type"]}* DINO_RESTRICT x,
    const {context["index_storage_type"]}* DINO_RESTRICT indices,
    {context["storage_type"]}* DINO_RESTRICT y,
    int64_t runtime_numel,
    int64_t select_count,
    int64_t inner_size,
    int64_t input_dim0,
    {context["gpu_stream_type"]} stream) {{
  if (x == nullptr || indices == nullptr || y == nullptr) {{
    return dino_runtime_fail("{context["func"]} received null pointer");
  }}
  if (select_count <= 0 || inner_size <= 0 || input_dim0 <= 0) {{
    return dino_runtime_fail("{context["func"]} received invalid runtime shape");
  }}
  const int block = {context["block_size"]};
  const int64_t grid64 = (runtime_numel + block - 1) / block;
  const int grid = static_cast<int>(grid64 > 65535 ? 65535 : grid64);
  if (grid > 0) {{
    {context["kernel"]}<<<grid, block, 0, stream>>>(x, indices, y, runtime_numel, inner_size);
    {context["gpu_check_macro"]}({context["gpu_last_error_call"]});
  }}
  return 0;
}}
"""


def _function_name(node: Mapping[str, Any], tensor_map: Mapping[str, Mapping[str, Any]]) -> str:
    input_tensor = tensor_map[node["inputs"][0]]
    index_tensor = tensor_map[node["inputs"][1]]
    output_tensor = tensor_map[node["outputs"][0]]
    signature = {
        "op": "runtime_index_select",
        "input_shape": list(input_tensor["shape"]),
        "index_shape": list(index_tensor["shape"]),
        "output_shape": list(output_tensor["shape"]),
        "dim": int(node.get("attrs", {}).get("dim", 0)),
        "dtype": str(output_tensor["dtype"]),
        "index_dtype": str(index_tensor["dtype"]),
    }
    digest = hashlib.sha256(repr(signature).encode("utf-8")).hexdigest()[:12]
    return f"runtime_index_select_{digest}"


RUNTIME_INDEX_SELECT_LOWERING = OpLowering(
    op_name="runtime_index_select",
    render_generated_kernel=render_generated_kernel,
    render_launch=render_launch,
    source_key=source_key,
    generated_function_name=generated_function_name,
)
