from __future__ import annotations

from typing import Any, Dict, Mapping, Sequence, Set

import numpy as np

from dinoml.ir import VIEW_METADATA_VERSION, dtype_nbytes
from dinoml.layout import dense_layout
from dinoml.ops.definitions import get_op_def
from dinoml.ops.elementwise import FUSABLE_ELEMENTWISE_OPS
from dinoml.ops.reductions import REDUCTION_OPS, infer_reduction_with_attrs
from dinoml.passes.utils import tensor_map
from dinoml.passes.validation import ValidationError, validate_view_metadata

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
        if node["op"] in REDUCTION_OPS:
            expected_shape = infer_reduction_with_attrs(
                inputs[0]["shape"],
                bool(node.get("attrs", {}).get("keepdim", False)),
            )
        else:
            expected_shape = op_def.infer_shape([input_info["shape"] for input_info in inputs])
        expected_shape_spec = _infer_node_shape_spec(node, inputs, expected_shape)
        expected_dtype = inputs[0]["dtype"] if inputs else tensors[node["outputs"][0]]["dtype"]
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
    return None


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
            required_tensors.update(node["inputs"])
            _include_view_sources(required_tensors, view_sources)
    required_tensors.update(input_info["tensor"] for input_info in ir["inputs"])
    required_tensors.update(constant["tensor"] for constant in ir["constants"])
    ir["nodes"] = list(reversed(kept_nodes_reversed))
    ir["tensors"] = [tensor for tensor in ir["tensors"] if tensor["name"] in required_tensors]
    return ir


def _include_view_sources(required_tensors: Set[str], view_sources: Dict[str, str]) -> None:
    changed = True
    while changed:
        changed = False
        for tensor, source in view_sources.items():
            if tensor in required_tensors and source not in required_tensors:
                required_tensors.add(source)
                changed = True


def memory_plan(ir: Dict[str, Any]) -> Dict[str, Any]:
    output_tensors = {output["tensor"] for output in ir["outputs"]}
    input_tensors = {input_info["tensor"] for input_info in ir["inputs"]}
    constant_tensors = {constant["tensor"] for constant in ir["constants"]}
    tensors = tensor_map(ir)
    views = validate_view_metadata(ir.get("metadata", {}).get("views"), tensors)
    view_tensors = {view["tensor"] for view in views}
    temporaries = []
    for tensor in ir["tensors"]:
        name = tensor["name"]
        if name in output_tensors or name in input_tensors or name in constant_tensors or name in view_tensors:
            continue
        temporaries.append({"tensor": name, "nbytes": tensor["nbytes"]})
    ir.setdefault("metadata", {})["memory_plan"] = {
        "allocation": "per_session_static_temporaries",
        "temporaries": temporaries,
        "views": {"version": VIEW_METADATA_VERSION, "views": views},
        "workspace_nbytes": sum(item["nbytes"] for item in temporaries),
    }
    return ir


def backend_lower(ir: Dict[str, Any]) -> Dict[str, Any]:
    ir.setdefault("metadata", {})["lowering"] = {
        "backend": "runtime_target",
        "kernel_style": "generated_static_float32",
    }
    return ir
