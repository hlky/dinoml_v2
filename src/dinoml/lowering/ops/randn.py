from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Any, Mapping

from jinja2 import Environment, FileSystemLoader, StrictUndefined

from dinoml.lowering.target_specs import storage_type as target_storage_type
from dinoml.lowering.ops.base import OpLowering
from dinoml.lowering.ops.template_rendering import supported_target_spec
from dinoml.ops.creation import RANDN_DTYPES, RANDN_RNGS, infer_randn_shape_with_attrs
from dinoml.lowering.shape_buffers import c_ident as _c_ident


def render_generated_kernel(target: str, node: Mapping[str, Any], tensor_map: Mapping[str, Mapping[str, Any]]) -> str:
    spec = supported_target_spec(target, "randn")
    context = _context(target, node, tensor_map)
    if not spec.is_gpu:
        return _render_template("randn_cpu.cpp.j2", context)
    context.update(spec.gpu_template_context())
    return _render_template("randn_gpu.j2", context)


def render_launch(
    target: str,
    node: Mapping[str, Any],
    tensor_map: Mapping[str, Mapping[str, Any]],
    kernel_manifest: Mapping[str, Any] | None = None,
) -> str:
    del kernel_manifest
    spec = supported_target_spec(target, "randn")
    func = _function_name(node, tensor_map)
    out = _c_ident(node["outputs"][0])
    args = f"ptr_{out}, runtime_numel_{out}"
    if not spec.is_gpu:
        return f"if (int err = {func}({args})) return err;"
    return f"if (int err = {func}({args}, {spec.stream_expr})) return err;"


def source_key(target: str, node: Mapping[str, Any], tensor_map: Mapping[str, Mapping[str, Any]]) -> str:
    supported_target_spec(target, "randn")
    return f"{target}:{_function_name(node, tensor_map)}"


def generated_function_name(target: str, node: Mapping[str, Any], tensor_map: Mapping[str, Mapping[str, Any]]) -> str:
    del target
    return _function_name(node, tensor_map)


def _context(target: str, node: Mapping[str, Any], tensor_map: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    output_tensor = tensor_map[node["outputs"][0]]
    _validate_node_contract(node, output_tensor)
    dtype = str(output_tensor["dtype"])
    storage_type = target_storage_type(dtype, target)
    rng = _randn_rng(node)
    return {
        "func": _function_name(node, tensor_map),
        "kernel": f"{_function_name(node, tensor_map)}_kernel",
        "storage_type": storage_type,
        "seed_literal": f"{int(node.get('attrs', {}).get('seed', 0))}ull",
        "rng": rng,
        "use_torch_rng": rng == "torch",
        "rand_state_type": "hiprandStatePhilox4_32_10_t" if target == "rocm" else "curandStatePhilox4_32_10_t",
        "rand_init": "hiprand_init" if target == "rocm" else "curand_init",
        "rand_normal4": "hiprand_normal4" if target == "rocm" else "curand_normal4",
        "rand_kernel_header": "hiprand/hiprand_kernel.h" if target == "rocm" else "curand_kernel.h",
        "torch_uniform_mask": (1 << _torch_uniform_digits(dtype)) - 1,
        "torch_uniform_divisor": f"{1.0 / float(1 << _torch_uniform_digits(dtype)):.30g}f",
        "block_size": 256,
    }


def _validate_node_contract(node: Mapping[str, Any], output_tensor: Mapping[str, Any]) -> None:
    if node["op"] != "randn":
        raise ValueError(f"Unsupported creation op: {node['op']}")
    if node.get("inputs"):
        raise ValueError("randn expects no tensor inputs")
    if len(node.get("outputs", [])) != 1:
        raise ValueError("randn expects exactly one output")
    if str(output_tensor["dtype"]) not in RANDN_DTYPES:
        raise NotImplementedError(f"randn lowering does not support dtype {output_tensor['dtype']}")
    _randn_rng(node)
    expected_shape = infer_randn_shape_with_attrs([], node.get("attrs", {}))
    if list(expected_shape) != list(output_tensor["shape"]):
        raise ValueError("randn output shape does not match shape attr")


def _function_name(node: Mapping[str, Any], tensor_map: Mapping[str, Mapping[str, Any]]) -> str:
    output_tensor = tensor_map[node["outputs"][0]]
    signature = {
        "op": "randn",
        "shape": list(output_tensor["shape"]),
        "dtype": str(output_tensor["dtype"]),
        "seed": int(node.get("attrs", {}).get("seed", 0)),
        "rng": _randn_rng(node),
    }
    digest = hashlib.sha256(repr(signature).encode("utf-8")).hexdigest()[:12]
    return f"randn_{digest}"


def _randn_rng(node: Mapping[str, Any]) -> str:
    rng = str(node.get("attrs", {}).get("rng", "dinoml")).lower()
    if rng not in RANDN_RNGS:
        supported = ", ".join(RANDN_RNGS)
        raise ValueError(f"randn rng must be one of: {supported}")
    return rng


def _torch_uniform_digits(dtype: str) -> int:
    if dtype == "float32":
        return 24
    if dtype == "float16":
        return 11
    if dtype == "bfloat16":
        return 8
    raise NotImplementedError(f"torch randn uniform digit width is not defined for dtype {dtype}")


def _render_template(name: str, context: Mapping[str, Any]) -> str:
    env = Environment(
        loader=FileSystemLoader(str(Path(__file__).resolve().parent / "templates")),
        undefined=StrictUndefined,
        trim_blocks=True,
        lstrip_blocks=True,
        keep_trailing_newline=True,
    )
    return env.get_template(name).render(**context)


RANDN_LOWERING = OpLowering(
    op_name="randn",
    render_generated_kernel=render_generated_kernel,
    render_launch=render_launch,
    source_key=source_key,
    generated_function_name=generated_function_name,
)
