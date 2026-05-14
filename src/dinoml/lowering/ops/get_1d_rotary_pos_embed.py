from __future__ import annotations

import hashlib
import math
import re
from pathlib import Path
from typing import Any, Mapping

from jinja2 import Environment, FileSystemLoader, StrictUndefined

from dinoml.lowering.cpp_types import cpu_storage_type, cuda_storage_type
from dinoml.lowering.ops.base import OpLowering
from dinoml.ops.positional import (
    GET_1D_ROTARY_POS_EMBED_COMPONENT_OPS,
    GET_1D_ROTARY_POS_EMBED_DTYPES,
    normalize_get_1d_rotary_pos_embed_attrs,
    rotary_output_cols,
)


def render_generated_kernel(target: str, node: Mapping[str, Any], tensor_map: Mapping[str, Mapping[str, Any]]) -> str:
    context = _context(target, node, tensor_map)
    if target == "cpu":
        return _render_template("get_1d_rotary_pos_embed_cpu.cpp.j2", context)
    if target == "cuda":
        return _render_template("get_1d_rotary_pos_embed_cuda.cu.j2", context)
    raise ValueError(f"Unsupported get_1d_rotary_pos_embed target: {target}")


def render_launch(
    target: str,
    node: Mapping[str, Any],
    tensor_map: Mapping[str, Mapping[str, Any]],
    kernel_manifest: Mapping[str, Any] | None = None,
) -> str:
    del kernel_manifest
    func = _function_name(node, tensor_map)
    out = _c_ident(node["outputs"][0])
    if node["inputs"]:
        pos = _c_ident(node["inputs"][0])
        runtime_pos = f"runtime_numel_{pos}"
        pos_ptr = f"ptr_{pos}"
    else:
        runtime_pos = str(int(node.get("attrs", {}).get("sequence_length", 0)))
        pos_ptr = "nullptr"
    args = f"{pos_ptr}, ptr_{out}, {runtime_pos}, runtime_numel_{out}"
    if target == "cpu":
        return f"if (int err = {func}({args})) return err;"
    if target == "cuda":
        return f"if (int err = {func}({args}, session->stream)) return err;"
    raise ValueError(f"Unsupported get_1d_rotary_pos_embed target: {target}")


def source_key(target: str, node: Mapping[str, Any], tensor_map: Mapping[str, Mapping[str, Any]]) -> str:
    if target not in {"cpu", "cuda"}:
        raise ValueError(f"Unsupported get_1d_rotary_pos_embed target: {target}")
    return f"{target}:{_function_name(node, tensor_map)}"


def generated_function_name(target: str, node: Mapping[str, Any], tensor_map: Mapping[str, Mapping[str, Any]]) -> str:
    del target
    return _function_name(node, tensor_map)


def _context(target: str, node: Mapping[str, Any], tensor_map: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    input_tensor = tensor_map[node["inputs"][0]] if node["inputs"] else None
    output_tensor = tensor_map[node["outputs"][0]]
    normalized_attrs = _validate_node_contract(node, input_tensor, output_tensor)
    output_dtype = str(output_tensor["dtype"])
    input_storage_type = cpu_storage_type("float32") if target == "cpu" else cuda_storage_type("float32")
    output_storage_type = cpu_storage_type(output_dtype) if target == "cpu" else cuda_storage_type(output_dtype)
    rotary_dim = int(normalized_attrs["dim"]) // 2
    scaled_theta = float(normalized_attrs["theta"]) * float(normalized_attrs["ntk_factor"])
    return {
        "func": _function_name(node, tensor_map),
        "kernel": f"{_function_name(node, tensor_map)}_kernel",
        "input_storage_type": input_storage_type,
        "output_storage_type": output_storage_type,
        "has_input": input_tensor is not None,
        "output_cols": rotary_output_cols(normalized_attrs),
        "rotary_dim": rotary_dim,
        "use_real": bool(normalized_attrs["use_real"]),
        "repeat_interleave_real": bool(normalized_attrs["repeat_interleave_real"]),
        "write_cos": str(normalized_attrs["output_kind"]) == "cos",
        "neg_log_scaled_theta_literal": _float_literal(-math.log(scaled_theta)),
        "inv_linear_factor_literal": _float_literal(1.0 / float(normalized_attrs["linear_factor"])),
        "block_size": 256,
    }


def _validate_node_contract(
    node: Mapping[str, Any],
    input_tensor: Mapping[str, Any] | None,
    output_tensor: Mapping[str, Any],
) -> Mapping[str, Any]:
    if str(node["op"]) not in GET_1D_ROTARY_POS_EMBED_COMPONENT_OPS:
        raise ValueError(f"Unsupported rotary positional op: {node['op']}")
    if len(node.get("inputs", ())) not in {0, 1} or len(node.get("outputs", ())) != 1:
        raise ValueError("get_1d_rotary_pos_embed component expects zero or one input and exactly one output")
    input_dtype = None if input_tensor is None else str(input_tensor["dtype"])
    output_dtype = str(output_tensor["dtype"])
    if input_dtype is not None and input_dtype != "float32":
        raise NotImplementedError("get_1d_rotary_pos_embed lowering requires float32 pos input")
    if output_dtype not in GET_1D_ROTARY_POS_EMBED_DTYPES:
        raise NotImplementedError(
            "get_1d_rotary_pos_embed lowering supports float16, float32, and bfloat16 output tensors only"
        )
    if input_tensor is not None and len(input_tensor["shape"]) != 1:
        raise ValueError("get_1d_rotary_pos_embed lowering requires rank-1 pos input")
    if len(output_tensor["shape"]) != 2:
        raise ValueError("get_1d_rotary_pos_embed lowering requires rank-2 output")
    normalized_attrs = normalize_get_1d_rotary_pos_embed_attrs(
        dim=node.get("attrs", {}).get("dim"),
        theta=node.get("attrs", {}).get("theta", 10000.0),
        use_real=node.get("attrs", {}).get("use_real", True),
        linear_factor=node.get("attrs", {}).get("linear_factor", 1.0),
        ntk_factor=node.get("attrs", {}).get("ntk_factor", 1.0),
        repeat_interleave_real=node.get("attrs", {}).get("repeat_interleave_real", True),
        output_kind=node.get("attrs", {}).get("output_kind"),
    )
    expected_kind = "cos" if str(node["op"]).endswith("_cos") else "sin"
    if str(normalized_attrs["output_kind"]) != expected_kind:
        raise ValueError(f"get_1d_rotary_pos_embed node {node['op']} must use output_kind={expected_kind}")
    expected_rows = int(input_tensor["shape"][0]) if input_tensor is not None else int(node.get("attrs", {}).get("sequence_length", 0))
    if int(output_tensor["shape"][0]) != expected_rows:
        raise ValueError("get_1d_rotary_pos_embed output sequence length must match the input/attr length")
    if int(output_tensor["shape"][1]) != rotary_output_cols(normalized_attrs):
        raise ValueError("get_1d_rotary_pos_embed output width must match attrs")
    return normalized_attrs


def _function_name(node: Mapping[str, Any], tensor_map: Mapping[str, Mapping[str, Any]]) -> str:
    input_tensor = tensor_map[node["inputs"][0]] if node["inputs"] else None
    output_tensor = tensor_map[node["outputs"][0]]
    normalized_attrs = normalize_get_1d_rotary_pos_embed_attrs(
        dim=node.get("attrs", {}).get("dim"),
        theta=node.get("attrs", {}).get("theta", 10000.0),
        use_real=node.get("attrs", {}).get("use_real", True),
        linear_factor=node.get("attrs", {}).get("linear_factor", 1.0),
        ntk_factor=node.get("attrs", {}).get("ntk_factor", 1.0),
        repeat_interleave_real=node.get("attrs", {}).get("repeat_interleave_real", True),
        output_kind=node.get("attrs", {}).get("output_kind"),
    )
    signature = {
        "op": str(node["op"]),
        "input_dtype": None if input_tensor is None else str(input_tensor["dtype"]),
        "output_dtype": str(output_tensor["dtype"]),
        "sequence_length": int(node.get("attrs", {}).get("sequence_length", 0)),
        "dim": int(normalized_attrs["dim"]),
        "theta": float(normalized_attrs["theta"]),
        "use_real": bool(normalized_attrs["use_real"]),
        "linear_factor": float(normalized_attrs["linear_factor"]),
        "ntk_factor": float(normalized_attrs["ntk_factor"]),
        "repeat_interleave_real": bool(normalized_attrs["repeat_interleave_real"]),
        "output_kind": str(normalized_attrs["output_kind"]),
    }
    digest = hashlib.sha256(repr(signature).encode("utf-8")).hexdigest()[:12]
    return f"{node['op']}_{digest}"


def _float_literal(value: float) -> str:
    if math.isnan(value) or math.isinf(value):
        raise ValueError("get_1d_rotary_pos_embed lowering supports only finite attrs")
    literal = f"{value:.9g}"
    if "." not in literal and "e" not in literal and "E" not in literal:
        literal = f"{literal}.0"
    return f"{literal}f"


def _render_template(name: str, context: Mapping[str, Any]) -> str:
    env = Environment(
        loader=FileSystemLoader(str(Path(__file__).resolve().parent / "templates")),
        undefined=StrictUndefined,
        trim_blocks=True,
        lstrip_blocks=True,
        keep_trailing_newline=True,
    )
    return env.get_template(name).render(**context)


def _c_ident(name: str) -> str:
    ident = re.sub(r"[^0-9A-Za-z_]", "_", name)
    if not ident or ident[0].isdigit():
        ident = f"_{ident}"
    return ident


GET_1D_ROTARY_POS_EMBED_LOWERINGS = {
    op_name: OpLowering(
        op_name=op_name,
        render_generated_kernel=render_generated_kernel,
        render_launch=render_launch,
        source_key=source_key,
        generated_function_name=generated_function_name,
    )
    for op_name in GET_1D_ROTARY_POS_EMBED_COMPONENT_OPS
}
