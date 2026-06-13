from __future__ import annotations

from typing import Any, Dict, Mapping, Sequence, Set

import numpy as np

from dinoml.ir import VIEW_METADATA_VERSION, dtype_nbytes
from dinoml.layout import dense_layout
from dinoml.ops.collections import normalize_one_hot_num_classes
from dinoml.ops.definitions import get_op_def
from dinoml.ops.elementwise import FUSABLE_ELEMENTWISE_OPS, elementwise_output_dtype
from dinoml.ops.positional import (
    GET_1D_ROTARY_POS_EMBED_COMPONENT_OPS,
    ROTARY_POSITIONAL_FUSION_OPS,
    infer_get_1d_rotary_pos_embed_component_shape_spec,
    normalize_get_2d_rotary_pos_embed_attrs,
    normalize_get_2d_rotary_pos_embed_lumina_attrs,
    normalize_get_3d_rotary_pos_embed_allegro_attrs,
    normalize_get_3d_rotary_pos_embed_attrs,
)
from dinoml.ops.reductions import MODE_INTERNAL_OPS, REDUCTION_OPS, TOPK_INTERNAL_OPS, infer_reduction_with_attrs
from dinoml.ops.vision import (
    infer_batched_nms_shape_with_attrs,
    infer_efficient_nms_output_shapes,
    normalize_batched_nms_attrs,
    normalize_efficient_nms_attrs,
)
from dinoml.passes.memory_planning import plan_temporary_memory
from dinoml.passes.utils import tensor_map
from dinoml.passes.validation import ValidationError

ELEMENTWISE_SHAPE_SPEC_OPS = frozenset((*FUSABLE_ELEMENTWISE_OPS, "fused_elementwise"))


def canonicalize(ir: Dict[str, Any]) -> Dict[str, Any]:
    ir["metadata"] = dict(sorted(ir.get("metadata", {}).items()))
    for node in ir["nodes"]:
        node.setdefault("attrs", {})
    return ir


def shape_type_infer(ir: Dict[str, Any]) -> Dict[str, Any]:
    tensors = tensor_map(ir)
    for node in ir["nodes"]:
        inputs = [tensors[name] for name in node["inputs"]]
        op_def = get_op_def(node["op"])
        if node["op"] in {"group_layernorm", "group_layernorm_sigmoid_mul"}:
            group_count = int(node.get("attrs", {}).get("group_count", 0))
            if group_count <= 0:
                raise ValidationError(f"{node['op']} requires a positive group_count")
            for output_name, input_tensor in zip(node["outputs"], inputs[:group_count]):
                _assign_tensor_shape_type(
                    tensors[output_name],
                    input_tensor["shape"],
                    input_tensor.get("shape_spec", input_tensor["shape"]),
                    str(input_tensor["dtype"]),
                )
            continue
        if node["op"] in {"glm_ocr_text_rope", "glm_ocr_vision_rope"}:
            _assign_tensor_shape_type(tensors[node["outputs"][0]], inputs[0]["shape"], inputs[0].get("shape_spec", inputs[0]["shape"]), inputs[0]["dtype"])
            _assign_tensor_shape_type(tensors[node["outputs"][1]], inputs[1]["shape"], inputs[1].get("shape_spec", inputs[1]["shape"]), inputs[1]["dtype"])
            continue
        if node["op"] == "batched_nms":
            expected_shape = infer_batched_nms_shape_with_attrs([inputs[0]["shape"]], normalize_batched_nms_attrs(**node.get("attrs", {})))
            _assign_tensor_shape_type(tensors[node["outputs"][0]], expected_shape, expected_shape, "int64")
            continue
        if node["op"] == "efficient_nms":
            num_shape, boxes_shape, scores_shape, classes_shape = infer_efficient_nms_output_shapes(
                [inputs[0]["shape"], inputs[1]["shape"]],
                normalize_efficient_nms_attrs(**node.get("attrs", {})),
            )
            _assign_tensor_shape_type(tensors[node["outputs"][0]], num_shape, num_shape, "int64")
            _assign_tensor_shape_type(
                tensors[node["outputs"][1]],
                boxes_shape,
                boxes_shape,
                str(inputs[0]["dtype"]),
            )
            _assign_tensor_shape_type(
                tensors[node["outputs"][2]],
                scores_shape,
                scores_shape,
                str(inputs[0]["dtype"]),
            )
            _assign_tensor_shape_type(tensors[node["outputs"][3]], classes_shape, classes_shape, "int64")
            continue
        if node["op"] in ROTARY_POSITIONAL_FUSION_OPS:
            _assign_rotary_positional_fusion_shapes(node, tensors)
            continue
        if node["op"] in REDUCTION_OPS:
            expected_shape = infer_reduction_with_attrs(
                inputs[0]["shape"],
                bool(node.get("attrs", {}).get("keepdim", False)),
            )
        else:
            expected_shape = op_def.infer_shape_for([input_info["shape"] for input_info in inputs], node.get("attrs", {}))
        expected_shape_spec = _infer_node_shape_spec(node, inputs, expected_shape)
        expected_dtype = inputs[0]["dtype"] if inputs else str(node.get("attrs", {}).get("dtype", tensors[node["outputs"][0]]["dtype"]))
        if node["op"] == "where" and len(inputs) == 3:
            expected_dtype = str(inputs[1]["dtype"])
        elif node["op"] in {"glm_ocr_stitch_image_features", "qwen2_5_vl_stitch_image_features"}:
            expected_dtype = str(inputs[1]["dtype"])
        elif node["op"] == "one_hot":
            expected_dtype = "int64"
        elif node["op"] == "argmax":
            expected_dtype = "int64"
        elif node["op"] == "topk_indices":
            expected_dtype = "int64"
        elif node["op"] == "mode_indices":
            expected_dtype = "int64"
        elif node["op"] in GET_1D_ROTARY_POS_EMBED_COMPONENT_OPS:
            expected_dtype = str(tensors[node["outputs"][0]]["dtype"])
        elif node["op"] in FUSABLE_ELEMENTWISE_OPS and inputs:
            expected_dtype = elementwise_output_dtype(str(node["op"]), str(inputs[0]["dtype"]), node.get("attrs", {}))
        elif node["op"] == "fused_elementwise":
            expected_dtype = _fused_output_dtype(node, tensors)
        for output_name in node["outputs"]:
            out = tensors[output_name]
            out["shape"] = expected_shape
            if expected_shape_spec is not None:
                out["shape_spec"] = _copy_shape_spec(expected_shape_spec)
            out["layout"] = dense_layout(expected_shape)
            out["dtype"] = expected_dtype
            out["nbytes"] = int(np.prod(expected_shape, dtype=np.int64) * dtype_nbytes(expected_dtype))
    ir["tensors"] = list(tensors.values())
    for output in ir["outputs"]:
        tensor = tensors[output["tensor"]]
        output["shape"] = tensor["shape"]
        output["shape_spec"] = _copy_shape_spec(tensor.get("shape_spec", tensor["shape"]))
        if "layout" in tensor:
            output["layout"] = dict(tensor["layout"])
        output["dtype"] = tensor["dtype"]
    return ir

def _assign_tensor_shape_type(
    tensor: Dict[str, Any],
    shape: Sequence[int],
    shape_spec: Sequence[Any],
    dtype: str,
) -> None:
    tensor["shape"] = list(shape)
    tensor["shape_spec"] = _copy_shape_spec(shape_spec)
    tensor["layout"] = dense_layout(shape)
    tensor["dtype"] = str(dtype)
    tensor["nbytes"] = int(np.prod(shape, dtype=np.int64) * dtype_nbytes(str(dtype)))


def _assign_rotary_positional_fusion_shapes(
    node: Mapping[str, Any],
    tensors: Mapping[str, Dict[str, Any]],
) -> None:
    op_name = str(node["op"])
    attrs = node.get("attrs", {})
    output_dtype = str(attrs.get("dtype", tensors[node["outputs"][0]]["dtype"]))
    if op_name == "get_2d_rotary_pos_embed":
        normalized = normalize_get_2d_rotary_pos_embed_attrs(
            embed_dim=attrs.get("embed_dim"),
            crop_start_h=attrs.get("crop_start_h"),
            crop_start_w=attrs.get("crop_start_w"),
            crop_stop_h=attrs.get("crop_stop_h"),
            crop_stop_w=attrs.get("crop_stop_w"),
            grid_h=attrs.get("grid_h"),
            grid_w=attrs.get("grid_w"),
            theta=attrs.get("theta", 10000.0),
            use_real=attrs.get("use_real", True),
        )
        output_specs = [([int(normalized["grid_h"]) * int(normalized["grid_w"]), int(normalized["embed_dim"])], output_dtype)] * 2
    elif op_name == "get_2d_rotary_pos_embed_lumina":
        normalized = normalize_get_2d_rotary_pos_embed_lumina_attrs(
            embed_dim=attrs.get("embed_dim"),
            len_h=attrs.get("len_h"),
            len_w=attrs.get("len_w"),
            linear_factor=attrs.get("linear_factor", 1.0),
            ntk_factor=attrs.get("ntk_factor", 1.0),
        )
        output_specs = [([int(normalized["len_h"]), int(normalized["len_w"]), int(normalized["embed_dim"]) // 2], output_dtype)] * 2
    elif op_name == "get_3d_rotary_pos_embed":
        normalized = normalize_get_3d_rotary_pos_embed_attrs(
            embed_dim=attrs.get("embed_dim"),
            crop_start_h=attrs.get("crop_start_h"),
            crop_start_w=attrs.get("crop_start_w"),
            crop_stop_h=attrs.get("crop_stop_h"),
            crop_stop_w=attrs.get("crop_stop_w"),
            grid_h=attrs.get("grid_h"),
            grid_w=attrs.get("grid_w"),
            temporal_size=attrs.get("temporal_size"),
            theta=attrs.get("theta", 10000.0),
            use_real=attrs.get("use_real", True),
            grid_type=attrs.get("grid_type", "linspace"),
            max_h=attrs.get("max_h", 0),
            max_w=attrs.get("max_w", 0),
        )
        output_specs = [
            (
                [
                    int(normalized["temporal_size"]) * int(normalized["grid_h"]) * int(normalized["grid_w"]),
                    int(normalized["embed_dim"]),
                ],
                output_dtype,
            )
        ] * 2
    elif op_name == "get_3d_rotary_pos_embed_allegro":
        normalized = normalize_get_3d_rotary_pos_embed_allegro_attrs(
            height=attrs.get("height"),
            width=attrs.get("width"),
            num_frames=attrs.get("num_frames"),
            vae_scale_factor_spatial=attrs.get("vae_scale_factor_spatial", 8),
            patch_size=attrs.get("patch_size", 2),
            interpolation_scale_h=attrs.get("interpolation_scale_h", 2.0),
            interpolation_scale_t=attrs.get("interpolation_scale_t", 2.2),
            interpolation_scale_w=attrs.get("interpolation_scale_w", 2.0),
            attention_head_dim=attrs.get("attention_head_dim", 96),
        )
        dim_axis = int(normalized["attention_head_dim"]) // 3
        grid_shape = [1, int(normalized["num_frames"]) * int(normalized["grid_h"]) * int(normalized["grid_w"])]
        output_specs = [
            ([int(normalized["num_frames"]), dim_axis], output_dtype),
            ([int(normalized["num_frames"]), dim_axis], output_dtype),
            ([int(normalized["grid_h"]), dim_axis], output_dtype),
            ([int(normalized["grid_h"]), dim_axis], output_dtype),
            ([int(normalized["grid_w"]), dim_axis], output_dtype),
            ([int(normalized["grid_w"]), dim_axis], output_dtype),
            (grid_shape, "int64"),
            (grid_shape, "int64"),
            (grid_shape, "int64"),
        ]
    else:
        raise ValidationError(f"Unsupported rotary positional fusion op: {op_name}")
    for output_name, (shape, dtype) in zip(node["outputs"], output_specs):
        _assign_tensor_shape_type(tensors[output_name], shape, shape, dtype)


def _infer_node_shape_spec(
    node: Mapping[str, Any],
    inputs: Sequence[Mapping[str, Any]],
    expected_shape: Sequence[int],
) -> list[Any] | None:
    if node["op"] in ELEMENTWISE_SHAPE_SPEC_OPS:
        return _infer_broadcast_shape_spec(
            [input_info.get("shape_spec", input_info["shape"]) for input_info in inputs],
            expected_shape,
        )
    if node["op"] in REDUCTION_OPS:
        keepdim = bool(node.get("attrs", {}).get("keepdim", False))
        return _infer_reduction_shape_spec(inputs[0].get("shape_spec", inputs[0]["shape"]), keepdim)
    if node["op"] == "argmax":
        keepdim = bool(node.get("attrs", {}).get("keepdim", False))
        return _infer_reduction_shape_spec(inputs[0].get("shape_spec", inputs[0]["shape"]), keepdim)
    if node["op"] in TOPK_INTERNAL_OPS:
        shape_spec = _copy_shape_spec(inputs[0].get("shape_spec", inputs[0]["shape"]))
        shape_spec[-1] = int(node.get("attrs", {})["k"])
        return shape_spec
    if node["op"] in MODE_INTERNAL_OPS:
        keepdim = bool(node.get("attrs", {}).get("keepdim", False))
        return _infer_reduction_shape_spec(inputs[0].get("shape_spec", inputs[0]["shape"]), keepdim)
    if node["op"] == "one_hot":
        num_classes = normalize_one_hot_num_classes(node.get("attrs", {}).get("num_classes"))
        return [*_copy_shape_spec(inputs[0].get("shape_spec", inputs[0]["shape"])), num_classes]
    if node["op"] in GET_1D_ROTARY_POS_EMBED_COMPONENT_OPS:
        return infer_get_1d_rotary_pos_embed_component_shape_spec(
            None if not inputs else inputs[0].get("shape_spec", inputs[0]["shape"]),
            node.get("attrs", {}),
        )
    return None


def _fused_output_dtype(node: Mapping[str, Any], tensors: Mapping[str, Mapping[str, Any]]) -> str:
    dtype_env = {name: str(tensor["dtype"]) for name, tensor in tensors.items()}
    for sub_op in node.get("attrs", {}).get("sub_ops", []):
        input_names = list(sub_op.get("inputs", []))
        if str(sub_op["op"]) == "where" and len(input_names) == 3:
            input_dtype = dtype_env[input_names[1]]
        elif input_names:
            input_dtype = dtype_env[input_names[0]]
        else:
            input_dtype = str(tensors[node["outputs"][0]]["dtype"])
        dtype_env[sub_op["outputs"][0]] = elementwise_output_dtype(str(sub_op["op"]), input_dtype, sub_op.get("attrs", {}))
    return dtype_env.get(node["outputs"][0], str(tensors[node["outputs"][0]]["dtype"]))


def _infer_broadcast_shape_spec(shape_specs: Sequence[Sequence[Any]], expected_shape: Sequence[int]) -> list[Any]:
    if not shape_specs:
        return list(expected_shape)
    result: list[Any] = []
    max_rank = max(len(shape_spec) for shape_spec in shape_specs)
    aligned = [[1] * (max_rank - len(shape_spec)) + list(shape_spec) for shape_spec in shape_specs]
    for dims in zip(*aligned):
        chosen = dims[0]
        for dim in dims[1:]:
            if _dim_is_one(chosen):
                chosen = dim
            elif _dim_is_one(dim):
                continue
            elif dim == chosen:
                continue
            else:
                return list(expected_shape)
        result.append(chosen)
    return result


def _infer_reduction_shape_spec(shape_spec: Sequence[Any], keepdim: bool) -> list[Any]:
    if keepdim:
        output_shape_spec = list(shape_spec)
        output_shape_spec[-1] = 1
        return output_shape_spec
    return list(shape_spec[:-1]) or [1]


def _dim_is_one(dim: Any) -> bool:
    return isinstance(dim, int) and int(dim) == 1


def _copy_shape_spec(shape_spec: Sequence[Any]) -> list[Any]:
    return [dict(dim) if isinstance(dim, Mapping) else dim for dim in shape_spec]


def constant_bind(ir: Dict[str, Any]) -> Dict[str, Any]:
    constants = {constant["tensor"] for constant in ir["constants"]}
    tensors = tensor_map(ir)
    for name in constants:
        if name not in tensors:
            raise ValidationError(f"Constant {name} is not present in tensor table")
        tensors[name]["kind"] = "constant"
    ir["tensors"] = list(tensors.values())
    return ir


def dead_code_eliminate(ir: Dict[str, Any]) -> Dict[str, Any]:
    view_sources = {
        str(view["tensor"]): str(view["source"])
        for view in ir.get("metadata", {}).get("views", {}).get("views", [])
    }
    required_tensors: Set[str] = {output["tensor"] for output in ir["outputs"]}
    _include_view_sources(required_tensors, view_sources)
    kept_nodes_reversed = []
    for node in reversed(ir["nodes"]):
        if any(output in required_tensors for output in node["outputs"]):
            kept_nodes_reversed.append(node)
            required_tensors.update(node["outputs"])
            required_tensors.update(node["inputs"])
            _include_view_sources(required_tensors, view_sources)
    required_tensors.update(input_info["tensor"] for input_info in ir["inputs"])
    required_tensors.update(state["tensor"] for state in ir.get("states", []))
    required_tensors.update(constant["tensor"] for constant in ir["constants"])
    ir["nodes"] = list(reversed(kept_nodes_reversed))
    ir["tensors"] = [tensor for tensor in ir["tensors"] if tensor["name"] in required_tensors]
    return ir


def dynamic_slice_view_eliminate(ir: Dict[str, Any]) -> Dict[str, Any]:
    tensors = tensor_map(ir)
    views = [dict(view) for view in ir.get("metadata", {}).get("views", {}).get("views", [])]
    views_by_source: dict[str, list[dict[str, Any]]] = {}
    for view in views:
        views_by_source.setdefault(str(view["source"]), []).append(view)

    removed_outputs: set[str] = set()
    replacement_views: dict[str, dict[str, Any]] = {}
    new_nodes = []
    for node in ir["nodes"]:
        if node["op"] != "dynamic_slice" or len(node.get("inputs", [])) != 1 or len(node.get("outputs", [])) != 1:
            new_nodes.append(node)
            continue
        input_name = str(node["inputs"][0])
        output_name = str(node["outputs"][0])
        view = _dynamic_slice_static_contiguous_view(node, tensors[input_name], tensors[output_name])
        if view is None:
            new_nodes.append(node)
            continue
        source_name, offset = view
        removed_outputs.add(output_name)
        replacement_views[output_name] = {
            "tensor": output_name,
            "source": source_name,
            "kind": "shape_view",
            "transform": "dynamic_slice",
            "offset_elements": offset,
            "shape": list(tensors[output_name]["shape"]),
            "shape_spec": list(tensors[output_name].get("shape_spec", tensors[output_name]["shape"])),
        }

    rewritten_views = []
    for view in views:
        source = str(view["source"])
        if source not in replacement_views:
            rewritten_views.append(view)
            continue
        replacement = replacement_views[source]
        rewritten = dict(view)
        rewritten["source"] = replacement["source"]
        rewritten["offset_elements"] = int(replacement["offset_elements"]) + int(view.get("offset_elements", 0))
        rewritten_views.append(rewritten)

    direct_uses = {
        str(input_name)
        for node in new_nodes
        for input_name in node.get("inputs", [])
    }
    direct_uses.update(str(output["tensor"]) for output in ir.get("outputs", []))
    for output_name, view in replacement_views.items():
        if output_name not in views_by_source or output_name in direct_uses:
            rewritten_views.append(view)

    if replacement_views:
        ir["nodes"] = new_nodes
        metadata = ir.setdefault("metadata", {})
        metadata["views"] = {"version": VIEW_METADATA_VERSION, "views": rewritten_views}
        metadata.pop("memory_plan", None)
        view_tensor_names = {str(view["tensor"]) for view in rewritten_views}
        ir["tensors"] = [
            tensor
            for tensor in ir["tensors"]
            if str(tensor["name"]) not in removed_outputs or str(tensor["name"]) in view_tensor_names
        ]
    return ir


def flatten_views(ir: Dict[str, Any]) -> Dict[str, Any]:
    metadata = ir.setdefault("metadata", {})
    view_metadata = metadata.get("views")
    if not view_metadata:
        return ir
    views = [dict(view) for view in view_metadata.get("views", [])]
    if not views:
        return ir

    view_by_tensor = {str(view["tensor"]): view for view in views}
    flattened = [_flatten_view(view, view_by_tensor) for view in views]
    if flattened == views:
        return ir
    metadata["views"] = {"version": VIEW_METADATA_VERSION, "views": flattened}
    metadata.pop("memory_plan", None)
    return ir


def _flatten_view(view: Mapping[str, Any], view_by_tensor: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    flattened = dict(view)
    offset = int(flattened.get("offset_elements", 0))
    source = str(flattened["source"])
    seen = {str(flattened["tensor"])}
    while source in view_by_tensor:
        if source in seen:
            raise ValidationError(f"View metadata contains a cycle involving {source}")
        seen.add(source)
        parent = view_by_tensor[source]
        offset += int(parent.get("offset_elements", 0))
        source = str(parent["source"])
    flattened["source"] = source
    flattened["offset_elements"] = offset
    return flattened


def _dynamic_slice_static_contiguous_view(
    node: Mapping[str, Any],
    input_tensor: Mapping[str, Any],
    output_tensor: Mapping[str, Any],
) -> tuple[str, int] | None:
    attrs = node.get("attrs", {})
    raw_starts = list(attrs.get("start_indices", []))
    raw_sizes = list(attrs.get("slice_sizes", []))
    if any(not isinstance(value, int) or isinstance(value, bool) for value in [*raw_starts, *raw_sizes]):
        return None
    starts = [int(value) for value in raw_starts]
    sizes = [int(value) for value in raw_sizes]
    input_shape = [int(dim) for dim in input_tensor["shape"]]
    output_shape = [int(dim) for dim in output_tensor["shape"]]
    if len(starts) != len(input_shape) or len(sizes) != len(input_shape) or sizes != output_shape:
        return None
    strides = _dense_strides(input_shape)
    offset = sum(start * stride for start, stride in zip(starts, strides))
    if int(np.prod(output_shape, dtype=np.int64)) == 0:
        return None
    first_varying_axis = next((axis for axis, size in enumerate(sizes) if size > 1), None)
    if first_varying_axis is None:
        return str(node["inputs"][0]), int(offset)
    if any(
        starts[later] != 0 or sizes[later] != input_shape[later]
        for later in range(first_varying_axis + 1, len(input_shape))
    ):
        return None
    return str(node["inputs"][0]), int(offset)


def _dense_strides(shape: Sequence[int]) -> list[int]:
    strides = [1] * len(shape)
    running = 1
    for axis in range(len(shape) - 1, -1, -1):
        strides[axis] = running
        running *= int(shape[axis])
    return strides


def _include_view_sources(required_tensors: Set[str], view_sources: Dict[str, str]) -> None:
    changed = True
    while changed:
        changed = False
        for tensor, source in view_sources.items():
            if tensor in required_tensors and source not in required_tensors:
                required_tensors.add(source)
                changed = True


def memory_plan(ir: Dict[str, Any]) -> Dict[str, Any]:
    ir.setdefault("metadata", {})["memory_plan"] = plan_temporary_memory(ir)
    return ir


def backend_lower(ir: Dict[str, Any]) -> Dict[str, Any]:
    ir.setdefault("metadata", {})["lowering"] = {
        "backend": "runtime_target",
        "kernel_style": "generated_static_float32",
    }
    return ir
