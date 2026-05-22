from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Any, Mapping

from jinja2 import Environment, FileSystemLoader, StrictUndefined

from dinoml.lowering.target_specs import storage_type as target_storage_type
from dinoml.lowering.ops.base import OpLowering
from dinoml.lowering.ops.template_rendering import supported_target_spec
from dinoml.ops.reductions import TOPK_DTYPES, normalize_topk_dim, resolve_topk_shape
from dinoml.lowering.shape_buffers import c_ident as _c_ident


def render_generated_kernel(target: str, node: Mapping[str, Any], tensor_map: Mapping[str, Mapping[str, Any]]) -> str | None:
    spec = supported_target_spec(target, "topk")
    if spec.is_gpu and _is_paired_indices_node(node):
        return None
    context = _context(target, node, tensor_map)
    if not spec.is_gpu:
        return _render_template("topk_cpu.cpp.j2", context)
    context.update(spec.gpu_template_context())
    return _render_template("topk_gpu.j2", context)


def render_launch(
    target: str,
    node: Mapping[str, Any],
    tensor_map: Mapping[str, Mapping[str, Any]],
    kernel_manifest: Mapping[str, Any] | None = None,
) -> str:
    del kernel_manifest
    spec = supported_target_spec(target, "topk")
    if spec.is_gpu and _is_paired_indices_node(node):
        return "/* paired topk_indices is produced by the topk_values launch */"
    func = _function_name(node, tensor_map)
    inp = _c_ident(node["inputs"][0])
    out = _c_ident(node["outputs"][0])
    context = _context(target, node, tensor_map) if spec.is_gpu else {}
    if spec.is_gpu and _is_paired_values_node(node):
        indices_out = _c_ident(str(node.get("attrs", {})["paired_indices_output"]))
        args = f"ptr_{inp}, ptr_{out}, ptr_{indices_out}, runtime_numel_{out}"
        if context.get("use_two_pass_keys") or context.get("use_radix_prefilter"):
            args = f"{args}, session->topk_scratch, session->topk_scratch_nbytes"
    else:
        args = f"ptr_{inp}, ptr_{out}, runtime_numel_{out}"
    if not spec.is_gpu:
        return f"if (int err = {func}({args})) return err;"
    return f"if (int err = {func}({args}, {spec.stream_expr})) return err;"


def source_key(target: str, node: Mapping[str, Any], tensor_map: Mapping[str, Mapping[str, Any]]) -> str | None:
    spec = supported_target_spec(target, "topk")
    if spec.is_gpu and _is_paired_indices_node(node):
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
    input_storage_type = target_storage_type(input_dtype, target)
    output_storage_type = _output_storage_type(target, node["op"], input_dtype)
    output_kind = "indices" if node["op"] == "topk_indices" else "values"
    if supported_target_spec(target, "topk").is_gpu and _is_paired_values_node(node):
        output_kind = "both"
        paired_output = str(node.get("attrs", {})["paired_indices_output"])
        paired_tensor = tensor_map.get(paired_output)
        if paired_tensor is None:
            raise ValueError(f"paired topk indices output {paired_output!r} is not a known tensor")
        if str(paired_tensor["dtype"]) != "int64":
            raise ValueError("paired topk indices output must have dtype int64")
        if list(paired_tensor["shape"]) != list(output_tensor["shape"]):
            raise ValueError("paired topk indices output shape must match topk values output shape")
    cols = int(input_tensor["shape"][-1])
    rows = 1
    for dim_size in input_tensor["shape"][:-1]:
        rows *= int(dim_size)
    k = int(node.get("attrs", {})["k"])
    sort_width = 1 << (cols - 1).bit_length()
    use_shared_sort = k >= 32 and sort_width <= 4096
    two_pass_key_config = _two_pass_key_config(
        input_dtype=input_dtype,
        output_kind=output_kind,
        rows=rows,
        cols=cols,
        k=k,
    )
    use_two_pass_keys = two_pass_key_config is not None
    radix_prefilter_config = _radix_prefilter_config(
        input_dtype=input_dtype,
        output_kind=output_kind,
        rows=rows,
        cols=cols,
        k=k,
        use_two_pass_keys=use_two_pass_keys,
    )
    use_radix_prefilter = radix_prefilter_config is not None
    use_value_radix_repair = input_dtype == "float32" and 32768 <= cols <= 131072 and k in {
        16,
        32,
        64,
        128,
        256,
    } and not use_two_pass_keys and not use_radix_prefilter
    use_radix_filter = input_dtype == "float32" and cols <= 131072 and (
        use_value_radix_repair
        or (cols == 4096 and (128 <= k <= 256 or (k == 64 and rows >= 32)))
        or (k == 300 and cols >= 4096)
        or (64 <= k <= 256 and 32768 <= cols <= 131072)
    ) and not use_two_pass_keys
    use_block_merge_rows = 2 <= k <= 8 and cols >= 2048 and not use_two_pass_keys
    use_block_argmax_rows = k == 1 and cols >= 2048 and not use_two_pass_keys
    if use_block_argmax_rows:
        block_size = 512
    elif use_block_merge_rows:
        block_size = 64 if k <= 4 else 128
    elif use_radix_filter:
        if k in {16, 32} and cols >= 32768:
            block_size = 1024 if rows <= 64 else (512 if rows <= 128 else 256)
        else:
            block_size = 1024 if k == 300 or (k == 64 and rows <= 64) or (k == 256 and (rows <= 8 or cols >= 32768)) else 512
    else:
        block_size = 256
    use_subwarp_rows = 2 <= k <= 8 and cols <= 256 and not use_two_pass_keys
    if k <= 2 and cols <= 16:
        subwarp_group_lanes = 4
    elif k <= 4 and cols <= 16:
        subwarp_group_lanes = 8
    else:
        subwarp_group_lanes = 16
    subwarp_warps_per_block = 8 if k <= 2 else 4
    subwarp_rows_per_block = subwarp_warps_per_block * (32 // subwarp_group_lanes)
    warp_rows_per_block = 16 if k == 16 and cols >= 2048 and not use_shared_sort else (2 if k >= 128 else 8)
    sort_block_size = 512 if sort_width <= 1024 or k <= 64 else 1024
    return {
        "func": _function_name(node, tensor_map),
        "kernel": f"{_function_name(node, tensor_map)}_kernel",
        "input_storage_type": input_storage_type,
        "output_storage_type": output_storage_type,
        "index_storage_type": "int" if cols <= 2_147_483_647 else "int64_t",
        "input_dtype": input_dtype,
        "output_kind": output_kind,
        "cols": cols,
        "k": k,
        "sort_width": sort_width,
        "radix_candidate_width": 1 << (k - 1).bit_length(),
        "block_size": block_size,
        "sort_block_size": sort_block_size,
        "use_two_pass_keys": use_two_pass_keys,
        "two_pass_tile_cols": two_pass_key_config["tile_cols"] if two_pass_key_config is not None else 0,
        "two_pass_first_block": two_pass_key_config["first_block"] if two_pass_key_config is not None else 0,
        "use_radix_prefilter": use_radix_prefilter,
        "radix_prefilter_tile_cols": radix_prefilter_config["tile_cols"] if radix_prefilter_config is not None else 0,
        "radix_prefilter_first_block": radix_prefilter_config["first_block"] if radix_prefilter_config is not None else 0,
        "radix_prefilter_passes": radix_prefilter_config["passes"] if radix_prefilter_config is not None else 0,
        "use_shared_sort": use_shared_sort,
        "use_radix_filter": use_radix_filter,
        "use_value_radix_repair": use_value_radix_repair,
        "use_block_merge_rows": use_block_merge_rows,
        "use_subwarp_rows": use_subwarp_rows,
        "subwarp_group_lanes": subwarp_group_lanes,
        "subwarp_warps_per_block": subwarp_warps_per_block,
        "subwarp_rows_per_block": subwarp_rows_per_block,
        "warp_rows_per_block": warp_rows_per_block,
        "use_warp_argmax": k == 1 and not use_block_argmax_rows,
        "unroll_warp_topk": k <= 32,
        "use_warp_worst_guard": k <= 16,
        "use_warp_rows": (
            k <= 256
            and not use_radix_filter
            and not use_shared_sort
            and not use_block_merge_rows
            and not use_block_argmax_rows
            and not use_subwarp_rows
        ),
        "use_parallel_rows": k <= 2048 and not use_radix_filter and not use_block_merge_rows,
    }


def topk_scratch_nbytes_for_node(
    target: str,
    node: Mapping[str, Any],
    tensor_map: Mapping[str, Mapping[str, Any]],
) -> int:
    spec = supported_target_spec(target, "topk")
    if not spec.is_gpu or _is_paired_indices_node(node):
        return 0
    context = _context(target, node, tensor_map)
    if not context["use_two_pass_keys"] and not context["use_radix_prefilter"]:
        return 0
    input_tensor = tensor_map[node["inputs"][0]]
    rows = 1
    for dim_size in input_tensor["shape"][:-1]:
        rows *= int(dim_size)
    cols = int(context["cols"])
    k = int(context["k"])
    if context["use_radix_prefilter"]:
        tile_cols = int(context["radix_prefilter_tile_cols"])
        tiles_per_row = (cols + tile_cols - 1) // tile_cols
        header_nbytes = _align_nbytes(rows * 2 * 4, 8)
        counts_nbytes = rows * tiles_per_row * 256 * 4
        temp_keys_nbytes = rows * tiles_per_row * k * 8
        return header_nbytes + max(counts_nbytes, temp_keys_nbytes)
    tile_cols = int(context["two_pass_tile_cols"])
    tiles_per_row = (cols + tile_cols - 1) // tile_cols
    return rows * tiles_per_row * k * 8


def _align_nbytes(value: int, alignment: int) -> int:
    return ((value + alignment - 1) // alignment) * alignment


def _two_pass_key_config(
    *,
    input_dtype: str,
    output_kind: str,
    rows: int,
    cols: int,
    k: int,
) -> dict[str, int] | None:
    if input_dtype != "float32" or output_kind != "both":
        return None
    if k not in {4, 8, 16} or cols < 32768 or cols % 4 != 0:
        return None
    if cols < 65536 and rows < 64:
        return None
    if k == 4:
        if rows <= 1:
            return {"tile_cols": 2048, "first_block": 256}
        if rows <= 8:
            return {"tile_cols": 2048, "first_block": 64}
        if rows <= 64:
            return {"tile_cols": 8192, "first_block": 256}
        return {"tile_cols": 4096, "first_block": 256}
    if k == 8:
        if rows <= 1:
            return {"tile_cols": 2048, "first_block": 256}
        if rows <= 8:
            return {"tile_cols": 8192, "first_block": 512}
        if rows <= 64:
            return {"tile_cols": 4096, "first_block": 128}
        return {"tile_cols": 8192, "first_block": 64}
    if cols < 65536 or rows > 32:
        return None
    return {"tile_cols": 8192, "first_block": 256}


def _radix_prefilter_config(
    *,
    input_dtype: str,
    output_kind: str,
    rows: int,
    cols: int,
    k: int,
    use_two_pass_keys: bool,
) -> dict[str, int] | None:
    if use_two_pass_keys or input_dtype != "float32" or output_kind != "both":
        return None
    if k != 32 or rows > 32 or cols < 65536 or cols > 131072 or cols % 4 != 0:
        return None
    if cols <= 65536:
        return None
    if rows <= 8:
        return {"tile_cols": 16384, "first_block": 256, "passes": 3}
    return {"tile_cols": 16384, "first_block": 128, "passes": 3}


def _validate_node_contract(
    node: Mapping[str, Any],
    input_tensor: Mapping[str, Any],
    output_tensor: Mapping[str, Any],
) -> None:
    if node["op"] not in {"topk_values", "topk_indices"}:
        raise ValueError(f"Unsupported topk op: {node['op']}")
    if len(node.get("inputs", [])) != 1:
        raise ValueError("topk expects exactly one input")
    if len(node.get("outputs", [])) != 1:
        raise ValueError("topk expects exactly one output")
    input_dtype = str(input_tensor["dtype"])
    if input_dtype not in TOPK_DTYPES:
        raise NotImplementedError(f"topk lowering does not support dtype {input_dtype}")
    expected_dtype = input_dtype if node["op"] == "topk_values" else "int64"
    if str(output_tensor["dtype"]) != expected_dtype:
        raise ValueError(f"{node['op']} output dtype must be {expected_dtype}, got {output_tensor['dtype']}")
    if not input_tensor["shape"]:
        raise ValueError("topk requires a ranked tensor")
    attrs = node.get("attrs", {})
    dim = normalize_topk_dim(attrs.get("dim", -1), len(input_tensor["shape"]))
    if dim != len(input_tensor["shape"]) - 1:
        raise NotImplementedError("topk lowering currently supports only the last dimension")
    if not bool(attrs.get("largest", True)):
        raise NotImplementedError("topk lowering currently supports only largest=True")
    if not bool(attrs.get("sorted", True)):
        raise NotImplementedError("topk lowering currently supports only sorted=True")
    shape_spec = input_tensor.get("shape_spec", input_tensor["shape"])
    cols = input_tensor["shape"][-1]
    if not isinstance(shape_spec[-1], int) or not isinstance(cols, int) or int(cols) <= 0:
        raise ValueError("topk lowering requires a positive static last dimension")
    expected = resolve_topk_shape(input_tensor["shape"], attrs.get("k"), dim, True, True)
    if list(output_tensor["shape"]) != expected:
        raise ValueError("topk output shape does not match topk attrs")


def _is_paired_values_node(node: Mapping[str, Any]) -> bool:
    return node["op"] == "topk_values" and "paired_indices_output" in node.get("attrs", {})


def _is_paired_indices_node(node: Mapping[str, Any]) -> bool:
    return node["op"] == "topk_indices" and "paired_values_output" in node.get("attrs", {})


def _output_storage_type(target: str, op_name: str, input_dtype: str) -> str:
    if op_name == "topk_indices":
        return "int64_t"
    return target_storage_type(input_dtype, target)


def _function_name(node: Mapping[str, Any], tensor_map: Mapping[str, Mapping[str, Any]]) -> str:
    input_tensor = tensor_map[node["inputs"][0]]
    attrs = node.get("attrs", {})
    dim = normalize_topk_dim(attrs.get("dim", -1), len(input_tensor["shape"]))
    signature = {
        "op": str(node["op"]),
        "shape": list(input_tensor["shape"]),
        "dtype": str(input_tensor["dtype"]),
        "k": int(attrs["k"]),
        "dim": dim,
        "largest": bool(attrs.get("largest", True)),
        "sorted": bool(attrs.get("sorted", True)),
        "paired_indices": bool(node["op"] == "topk_values" and "paired_indices_output" in attrs),
        "paired_values": bool(node["op"] == "topk_indices" and "paired_values_output" in attrs),
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


TOPK_VALUES_LOWERING = OpLowering(
    op_name="topk_values",
    render_generated_kernel=render_generated_kernel,
    render_launch=render_launch,
    source_key=source_key,
    generated_function_name=generated_function_name,
)
TOPK_INDICES_LOWERING = OpLowering(
    op_name="topk_indices",
    render_generated_kernel=render_generated_kernel,
    render_launch=render_launch,
    source_key=source_key,
    generated_function_name=generated_function_name,
)
TOPK_LOWERINGS = {
    TOPK_VALUES_LOWERING.op_name: TOPK_VALUES_LOWERING,
    TOPK_INDICES_LOWERING.op_name: TOPK_INDICES_LOWERING,
}
