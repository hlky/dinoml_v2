from __future__ import annotations

from typing import Any, Dict, Set

import numpy as np

from dinoml.ir import VIEW_METADATA_VERSION, dtype_nbytes
from dinoml.ops.definitions import get_op_def
from dinoml.passes.utils import tensor_map
from dinoml.passes.validation import ValidationError, validate_view_metadata


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
        expected_shape = op_def.infer_shape([input_info["shape"] for input_info in inputs])
        expected_dtype = inputs[0]["dtype"] if inputs else tensors[node["outputs"][0]]["dtype"]
        for output_name in node["outputs"]:
            out = tensors[output_name]
            out["shape"] = expected_shape
            out["dtype"] = expected_dtype
            out["nbytes"] = int(np.prod(expected_shape, dtype=np.int64) * dtype_nbytes(expected_dtype))
    ir["tensors"] = list(tensors.values())
    for output in ir["outputs"]:
        tensor = tensors[output["tensor"]]
        output["shape"] = tensor["shape"]
        output["dtype"] = tensor["dtype"]
    return ir


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
