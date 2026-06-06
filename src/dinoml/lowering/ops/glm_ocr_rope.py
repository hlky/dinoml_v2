from __future__ import annotations

import hashlib
from typing import Any, Mapping

from dinoml.lowering.ops.base import OpLowering
from dinoml.lowering.ops.template_rendering import render_op_template, supported_target_spec
from dinoml.lowering.shape_buffers import c_ident as _c_ident
from dinoml.lowering.target_specs import storage_type as target_storage_type


def render_generated_kernel(target: str, node: Mapping[str, Any], tensor_map: Mapping[str, Mapping[str, Any]]) -> str:
    op_name = str(node["op"])
    spec = supported_target_spec(target, op_name)
    context = _context(target, node, tensor_map)
    if spec.is_gpu:
        context.update(spec.gpu_template_context())
    template_name = f"{op_name}_{'gpu' if spec.is_gpu else 'cpu.cpp'}.j2"
    return render_op_template(template_name, context)


def render_launch(
    target: str,
    node: Mapping[str, Any],
    tensor_map: Mapping[str, Mapping[str, Any]],
    kernel_manifest: Mapping[str, Any] | None = None,
) -> str:
    del kernel_manifest
    op_name = str(node["op"])
    spec = supported_target_spec(target, op_name)
    q_name, k_name, cos_name, sin_name = (str(name) for name in node["inputs"])
    q_out_name, k_out_name = (str(name) for name in node["outputs"])
    func = _function_name(node, tensor_map)
    args = (
        f"ptr_{_c_ident(q_name)}, ptr_{_c_ident(k_name)}, "
        f"ptr_{_c_ident(cos_name)}, ptr_{_c_ident(sin_name)}, "
        f"ptr_{_c_ident(q_out_name)}, ptr_{_c_ident(k_out_name)}, "
        f"runtime_numel_{_c_ident(q_out_name)}, runtime_numel_{_c_ident(k_out_name)}"
    )
    if op_name == "glm_ocr_text_rope":
        args = (
            f"{args}, "
            f"shape_{_c_ident(q_name)}_1, shape_{_c_ident(k_name)}_1"
        )
    if not spec.is_gpu:
        return f"if (int err = {func}({args})) return err;"
    return f"if (int err = {func}({args}, {spec.stream_expr})) return err;"


def source_key(target: str, node: Mapping[str, Any], tensor_map: Mapping[str, Mapping[str, Any]]) -> str:
    supported_target_spec(target, str(node["op"]))
    return f"{target}:{_function_name(node, tensor_map)}"


def generated_function_name(target: str, node: Mapping[str, Any], tensor_map: Mapping[str, Mapping[str, Any]]) -> str:
    del target
    return _function_name(node, tensor_map)


def _context(target: str, node: Mapping[str, Any], tensor_map: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    op_name = str(node["op"])
    q_tensor = tensor_map[str(node["inputs"][0])]
    k_tensor = tensor_map[str(node["inputs"][1])]
    cos_tensor = tensor_map[str(node["inputs"][2])]
    output_tensors = [tensor_map[str(name)] for name in node["outputs"]]
    _validate_node_contract(node, q_tensor, k_tensor, cos_tensor, output_tensors)
    q_shape = list(q_tensor["shape"])
    k_shape = list(k_tensor["shape"])
    cos_shape = list(cos_tensor["shape"])
    q_dtype = str(q_tensor["dtype"])
    trig_dtype = str(cos_tensor["dtype"])
    context = {
        "func": _function_name(node, tensor_map),
        "kernel": f"{_function_name(node, tensor_map)}_kernel",
        "storage_type": target_storage_type(q_dtype, target),
        "cpu_storage_type": target_storage_type(q_dtype, "cpu"),
        "trig_storage_type": target_storage_type(trig_dtype, target),
        "cpu_trig_storage_type": target_storage_type(trig_dtype, "cpu"),
        "head_dim": int(q_shape[-1]),
    }
    if op_name == "glm_ocr_text_rope":
        context.update(
            {
                "q_heads": int(q_shape[2]),
                "k_heads": int(k_shape[2]),
                "trig_dim": int(cos_shape[-1]),
                "rotary_dim": int(node.get("attrs", {})["rotary_dim"]),
            }
        )
    elif op_name == "glm_ocr_vision_rope":
        context.update(
            {
                "heads": int(q_shape[1]),
                "half_dim": int(q_shape[2]) // 2,
                "trig_dim": int(cos_shape[-1]),
            }
        )
    else:
        raise ValueError(f"Unsupported GLM-OCR RoPE op: {op_name}")
    return context


def _validate_node_contract(
    node: Mapping[str, Any],
    q_tensor: Mapping[str, Any],
    k_tensor: Mapping[str, Any],
    cos_tensor: Mapping[str, Any],
    output_tensors: list[Mapping[str, Any]],
) -> None:
    op_name = str(node["op"])
    if op_name not in {"glm_ocr_text_rope", "glm_ocr_vision_rope"}:
        raise ValueError(f"Unsupported GLM-OCR RoPE op: {op_name}")
    if len(output_tensors) != 2:
        raise ValueError(f"{op_name} expects exactly two outputs")
    q_dtype = str(q_tensor["dtype"])
    k_dtype = str(k_tensor["dtype"])
    trig_dtype = str(cos_tensor["dtype"])
    if q_dtype not in {"float16", "float32", "bfloat16"}:
        raise NotImplementedError(f"{op_name} does not support dtype {q_dtype}")
    if k_dtype != q_dtype:
        raise ValueError(f"{op_name} q/k dtype mismatch")
    if trig_dtype not in {"float16", "float32", "bfloat16"}:
        raise NotImplementedError(f"{op_name} does not support trig dtype {trig_dtype}")
    if list(output_tensors[0]["shape"]) != list(q_tensor["shape"]):
        raise ValueError(f"{op_name} q output shape must match q input")
    if list(output_tensors[1]["shape"]) != list(k_tensor["shape"]):
        raise ValueError(f"{op_name} k output shape must match k input")
    if str(output_tensors[0]["dtype"]) != q_dtype or str(output_tensors[1]["dtype"]) != q_dtype:
        raise ValueError(f"{op_name} output dtype must match q/k dtype")


def _function_name(node: Mapping[str, Any], tensor_map: Mapping[str, Mapping[str, Any]]) -> str:
    q_tensor = tensor_map[str(node["inputs"][0])]
    k_tensor = tensor_map[str(node["inputs"][1])]
    cos_tensor = tensor_map[str(node["inputs"][2])]
    signature = {
        "op": str(node["op"]),
        "q_shape": list(q_tensor["shape"]),
        "k_shape": list(k_tensor["shape"]),
        "cos_shape": list(cos_tensor["shape"]),
        "q_dtype": str(q_tensor["dtype"]),
        "trig_dtype": str(cos_tensor["dtype"]),
        "attrs": dict(node.get("attrs", {})),
    }
    digest = hashlib.sha256(repr(signature).encode("utf-8")).hexdigest()[:12]
    return f"{str(node['op'])}_{digest}"


GLM_OCR_TEXT_ROPE_LOWERING = OpLowering(
    op_name="glm_ocr_text_rope",
    render_generated_kernel=render_generated_kernel,
    render_launch=render_launch,
    source_key=source_key,
    generated_function_name=generated_function_name,
)

GLM_OCR_VISION_ROPE_LOWERING = OpLowering(
    op_name="glm_ocr_vision_rope",
    render_generated_kernel=render_generated_kernel,
    render_launch=render_launch,
    source_key=source_key,
    generated_function_name=generated_function_name,
)
