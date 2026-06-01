from __future__ import annotations

import re
import json
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
from jinja2 import Environment, FileSystemLoader, StrictUndefined

from dinoml.ir import dtype_nbytes, dtype_runtime_enum
from dinoml.kernels.providers.cutlass.conv import cutlass_conv_wrapper_stages
from dinoml.kernels.providers.cutlass.gemm import cutlass_gemm_split_k_supported
from dinoml.lowering.cpp_types import cuda_storage_type
from dinoml.lowering.ops import render_generated_kernels, render_launch
from dinoml.lowering.shape_buffers import (
    c_ident as _c_ident,
    constant_expression_axis_checks,
    dynamic_dim_sources,
    expression_axis_checks,
    named_dim_leaves,
    numel_expr,
    shape_buffer_context,
    shape_dim_expr,
    shape_dim_range,
    shape_literal,
    shape_vars_literal,
    validate_symbolic_int_sources,
)
from dinoml.lowering.target_specs import LoweringTargetSpec, lowering_target_spec, storage_type


def render_gpu_module(
    target_name: str,
    ir: Mapping[str, Any],
    *,
    generated_kernels: Iterable[str] | None = None,
    kernel_manifest: Mapping[str, Any] | None = None,
    **template_kwargs: Mapping[str, Any],
) -> str:
    spec = lowering_target_spec(target_name)
    if not spec.is_gpu or not spec.generated_module_admitted:
        raise ValueError(f"Unsupported GPU module target: {target_name}")
    config = _gpu_target_config(target_name, spec)
    tensor_map = {tensor["name"]: tensor for tensor in ir["tensors"]}
    input_map = {item["tensor"]: idx for idx, item in enumerate(ir["inputs"])}
    output_map = {item["tensor"]: idx for idx, item in enumerate(ir["outputs"])}
    state_tensors = {item["tensor"]: item for item in ir.get("states", [])}
    constant_tensors = {item["tensor"]: item for item in ir["constants"]}
    temporaries = ir.get("metadata", {}).get("memory_plan", {}).get("temporaries", [])
    views = _view_contexts(ir, output_map=output_map, tensor_map=tensor_map, target_name=target_name)
    dynamic_dims = dynamic_dim_sources(input_map=input_map, output_map=output_map, tensor_map=tensor_map)
    validate_symbolic_int_sources(items=ir["inputs"], dynamic_dims=dynamic_dims, context="input")
    validate_symbolic_int_sources(items=ir["outputs"], dynamic_dims=dynamic_dims, context="output")
    validate_symbolic_int_sources(items=ir["constants"], dynamic_dims=dynamic_dims, context="constant")
    validate_symbolic_int_sources(items=tensor_map.values(), dynamic_dims=dynamic_dims, context="tensor")
    lowered_runtime_dequant_constants = (
        _lowered_gguf_runtime_dequant_constant_names(kernel_manifest) if target_name == "cuda" else set()
    )
    generated_kernel_sources = list(generated_kernels) if generated_kernels is not None else render_generated_kernels(target_name, ir["nodes"], tensor_map)
    launches = [render_launch(target_name, node, tensor_map, kernel_manifest=kernel_manifest) for node in ir["nodes"]]
    launch_contexts = _launch_contexts(ir["nodes"], launches)
    output_shape_reports = _output_shape_report_contexts(ir, tensor_map=tensor_map)
    shape_buffer_idents = _required_shape_buffer_idents(
        tensor_map=tensor_map,
        launches=launches,
        output_shape_reports=output_shape_reports,
    )
    shape_buffer_run_update_idents = _shape_buffer_run_update_idents(
        tensor_map=tensor_map,
        shape_buffer_idents=shape_buffer_idents,
    )
    return render_template(
        "gpu_module.cu.j2",
        {
            **config,
            "input_count": len(ir["inputs"]),
            "output_count": len(ir["outputs"]),
            "inputs": [
                _io_context(idx, item["name"], item["shape"], item["dtype"], item.get("shape_spec", item["shape"]))
                for idx, item in enumerate(ir["inputs"])
            ],
            "outputs": [
                _io_context(idx, item["name"], item["shape"], item["dtype"], item.get("shape_spec", item["shape"]))
                for idx, item in enumerate(ir["outputs"])
            ],
            "constants": [_constant_context(item, lowered_runtime_dequant_constants) for item in ir["constants"]],
            "states": [_state_context(item) for item in ir.get("states", [])],
            "temporaries": [_temporary_context(item) for item in temporaries],
            "shape_buffers": [
                shape_buffer_context(item)
                for item in ir["tensors"]
                if _c_ident(str(item["name"])) in shape_buffer_idents
            ],
            "shape_equal_checks": _shape_equal_checks(ir["inputs"], ir["outputs"], ir["constants"]),
            "generated_kernels": generated_kernel_sources,
            "gguf_dequant_scratch": _gguf_dequant_scratch_context(kernel_manifest) if target_name == "cuda" else None,
            "topk_scratch": _topk_scratch_context(target_name, ir, tensor_map),
            "flash_attention_static_kv_cache_scratch": _flash_attention_static_kv_cache_scratch_context(
                target_name, ir, tensor_map
            ),
            "pointer_decls": list(
                _pointer_decls(
                    target_name=target_name,
                    input_map=input_map,
                    output_map=output_map,
                    state_tensors=state_tensors,
                    constant_tensors=constant_tensors,
                    temporaries=temporaries,
                    views=views,
                    tensor_map=tensor_map,
                    shape_buffer_idents=shape_buffer_idents,
                    shape_buffer_run_update_idents=shape_buffer_run_update_idents,
                )
            ),
            "launches": launches,
            "launch_contexts": launch_contexts,
            "profile_ops": _profile_op_contexts(launch_contexts),
            "output_materializations": _output_materializations(views, target_name=target_name),
            "output_shape_reports": output_shape_reports,
            "cutlass_conv_temporaries": [],
            "external_kernel_declarations": [],
            "cutlass_workspace": None,
            **template_kwargs,
        },
    )


def _gpu_target_config(target_name: str, spec: LoweringTargetSpec) -> dict[str, str]:
    if target_name == "cuda":
        return {
            "kernel_header": "dinoml/cuda_kernels.h",
            "runtime_header": "dinoml/runtime.h",
            "stream_type": str(spec.stream_type),
            "check_macro": str(spec.check_macro),
            "runtime_check_function": "dino_runtime_cuda_check",
            "device_error_type": "cudaError_t",
            "success_constant": "cudaSuccess",
            "device_malloc": "cudaMalloc",
            "device_free": "cudaFree",
            "device_synchronize": "cudaDeviceSynchronize",
            "stream_create_with_flags": "cudaStreamCreateWithFlags",
            "stream_destroy": "cudaStreamDestroy",
            "stream_non_blocking": "cudaStreamNonBlocking",
            "event_type": "cudaEvent_t",
            "event_create": "cudaEventCreateWithFlags",
            "event_flags": "cudaEventDefault",
            "event_destroy": "cudaEventDestroy",
            "event_record": "cudaEventRecord",
            "event_synchronize": "cudaEventSynchronize",
            "event_elapsed_time": "cudaEventElapsedTime",
            "graph_type": "cudaGraph_t",
            "graph_exec_type": "cudaGraphExec_t",
            "graph_capture_mode": "cudaStreamCaptureModeThreadLocal",
            "stream_begin_capture": "cudaStreamBeginCapture",
            "stream_end_capture": "cudaStreamEndCapture",
            "graph_instantiate": "cudaGraphInstantiate",
            "graph_launch": "cudaGraphLaunch",
            "graph_destroy": "cudaGraphDestroy",
            "graph_exec_destroy": "cudaGraphExecDestroy",
            "memcpy": "cudaMemcpy",
            "memset": "cudaMemset",
            "memcpy_async": "cudaMemcpyAsync",
            "stream_synchronize": "cudaStreamSynchronize",
            "memcpy_host_to_device": "cudaMemcpyHostToDevice",
            "memcpy_device_to_device": "cudaMemcpyDeviceToDevice",
            "memcpy_device_to_host": "cudaMemcpyDeviceToHost",
        }
    if target_name == "rocm":
        return {
            "kernel_header": "dinoml/rocm_kernels.h",
            "runtime_header": "dinoml/runtime_rocm.h",
            "stream_type": str(spec.stream_type),
            "check_macro": str(spec.check_macro),
            "runtime_check_function": "dino_runtime_rocm_check",
            "device_error_type": "hipError_t",
            "success_constant": "hipSuccess",
            "device_malloc": "hipMalloc",
            "device_free": "hipFree",
            "device_synchronize": "hipDeviceSynchronize",
            "stream_create_with_flags": "hipStreamCreateWithFlags",
            "stream_destroy": "hipStreamDestroy",
            "stream_non_blocking": "hipStreamNonBlocking",
            "event_type": "hipEvent_t",
            "event_create": "hipEventCreateWithFlags",
            "event_flags": "hipEventDisableSystemFence",
            "event_destroy": "hipEventDestroy",
            "event_record": "hipEventRecord",
            "event_synchronize": "hipEventSynchronize",
            "event_elapsed_time": "hipEventElapsedTime",
            "graph_type": "hipGraph_t",
            "graph_exec_type": "hipGraphExec_t",
            "graph_capture_mode": "hipStreamCaptureModeThreadLocal",
            "stream_begin_capture": "hipStreamBeginCapture",
            "stream_end_capture": "hipStreamEndCapture",
            "graph_instantiate": "hipGraphInstantiate",
            "graph_launch": "hipGraphLaunch",
            "graph_destroy": "hipGraphDestroy",
            "graph_exec_destroy": "hipGraphExecDestroy",
            "memcpy": "hipMemcpy",
            "memset": "hipMemset",
            "memcpy_async": "hipMemcpyAsync",
            "stream_synchronize": "hipStreamSynchronize",
            "memcpy_host_to_device": "hipMemcpyHostToDevice",
            "memcpy_device_to_device": "hipMemcpyDeviceToDevice",
            "memcpy_device_to_host": "hipMemcpyDeviceToHost",
        }
    raise ValueError(f"Unsupported GPU module target: {target_name}")


def _launch_contexts(nodes: Iterable[Mapping[str, Any]], launches: Iterable[str]) -> list[dict[str, Any]]:
    return [
        {
            "index": idx,
            "op": str(node.get("op", "unknown")),
            "label": _profile_launch_label(idx, node),
            "profile_op_index": 0,
            "launch": launch,
        }
        for idx, (node, launch) in enumerate(zip(nodes, launches))
    ]


def _profile_launch_label(index: int, node: Mapping[str, Any]) -> str:
    inputs = ",".join(str(name) for name in node.get("inputs", ()))
    outputs = ",".join(str(name) for name in node.get("outputs", ()))
    return json.dumps(f"{index}:{node.get('op', 'unknown')} in={inputs} out={outputs}")


def _profile_op_contexts(launch_contexts: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    op_to_index: dict[str, int] = {}
    contexts: list[dict[str, Any]] = []
    for launch in launch_contexts:
        op = str(launch["op"])
        if op not in op_to_index:
            op_to_index[op] = len(contexts)
            contexts.append({"index": op_to_index[op], "name": op})
        launch["profile_op_index"] = op_to_index[op]
    return contexts


def _view_contexts(
    ir: Mapping[str, Any],
    *,
    output_map: Mapping[str, int],
    tensor_map: Mapping[str, Mapping[str, Any]],
    target_name: str,
) -> list[dict[str, Any]]:
    raw_views = ir.get("metadata", {}).get("memory_plan", {}).get("views", {}).get("views", [])
    view_tensors = {view["tensor"] for view in raw_views}
    view_sources = {view["source"] for view in raw_views}
    view_of_view = sorted(view_tensors & view_sources)
    if view_of_view:
        raise NotImplementedError(
            "View-of-view aliases are not supported by CUDA lowering; "
            f"view tensors used as view sources: {view_of_view}"
        )
    node_view_outputs = sorted(
        output_name
        for node in ir["nodes"]
        for output_name in node["outputs"]
        if output_name in view_tensors
    )
    if node_view_outputs:
        raise NotImplementedError(f"View alias tensors cannot be kernel outputs: {node_view_outputs}")
    contexts = []
    for view in raw_views:
        tensor_name = str(view["tensor"])
        cpp_type = storage_type(str(tensor_map[tensor_name]["dtype"]), target_name)
        contexts.append(
            {
                "tensor": tensor_name,
                "ident": _c_ident(tensor_name),
                "source": str(view["source"]),
                "source_ident": _c_ident(str(view["source"])),
                "offset_elements": int(view.get("offset_elements", 0)),
                "output_index": output_map.get(tensor_name),
                "nbytes_expr": f"runtime_numel_{_c_ident(tensor_name)} * sizeof({cpp_type})",
            }
        )
    return contexts


def render_template(name: str, context: Mapping[str, Any]) -> str:
    env = Environment(
        loader=FileSystemLoader(str(Path(__file__).resolve().parents[1] / "templates")),
        undefined=StrictUndefined,
        trim_blocks=True,
        lstrip_blocks=True,
        keep_trailing_newline=True,
    )
    return env.get_template(name).render(**context)


def _io_context(index: int, name: str, shape: Iterable[int], dtype: str, shape_spec: Iterable[Any]) -> dict[str, Any]:
    dims = _dim_ranges(shape_spec)
    return {
        "index": index,
        "name": name,
        "shape_literal": shape_literal(shape),
        "min_shape_literal": shape_literal(dim["min"] for dim in dims),
        "max_shape_literal": shape_literal(dim["max"] for dim in dims),
        "divisible_by_literal": shape_literal(dim["divisible_by"] for dim in dims),
        "dtype": dtype,
        "dtype_enum": dtype_runtime_enum(dtype),
    }


def _constant_context(item: Mapping[str, Any], lowered_runtime_dequant_constants: set[str]) -> dict[str, Any]:
    dims = _dim_ranges(item.get("shape_spec", item["shape"]))
    storage = item.get("storage")
    storage_kind = storage.get("kind") if isinstance(storage, Mapping) else None
    materialization = storage.get("materialization") if isinstance(storage, Mapping) else None
    residency = storage.get("residency") if isinstance(storage, Mapping) else None
    encoded_runtime_dequant = (
        storage_kind == "gguf"
        and materialization == "dequantize_on_gpu_before_launch"
        and str(item["name"]) in lowered_runtime_dequant_constants
    )
    autoload_from_constants_bin = not (
        storage_kind == "gguf" and str(residency or "eager_dense_device") == "manual_runtime_load"
    )
    return {
        "name": item["name"],
        "ident": _c_ident(item["tensor"]),
        "offset": int(item["offset"]),
        "nbytes": int(item["nbytes"]),
        "shape_literal": shape_literal(item["shape"]),
        "min_shape_literal": shape_literal(dim["min"] for dim in dims),
        "max_shape_literal": shape_literal(dim["max"] for dim in dims),
        "divisible_by_literal": shape_literal(dim["divisible_by"] for dim in dims),
        "dtype": item["dtype"],
        "dtype_enum": dtype_runtime_enum(item["dtype"]),
        "dtype_nbytes": dtype_nbytes(item["dtype"]),
        "encoded_runtime_dequant": encoded_runtime_dequant,
        "autoload_from_constants_bin": autoload_from_constants_bin,
    }


def _lowered_gguf_runtime_dequant_constant_names(kernel_manifest: Mapping[str, Any] | None) -> set[str]:
    if kernel_manifest is None:
        return set()
    return {
        str(plan.get("constant"))
        for item in kernel_manifest.get("required_kernels", [])
        if isinstance(item, Mapping)
        for plan in [item.get("gguf_runtime_dequant")]
        if isinstance(plan, Mapping) and str(plan.get("status", "")) == "lowered_runtime_dequant_scratch"
    }


def _temporary_context(item: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "ident": _c_ident(item["tensor"]),
        "nbytes": int(item["nbytes"]),
    }


def _state_context(item: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "name": item["name"],
        "ident": _c_ident(item["tensor"]),
        "nbytes": int(item["nbytes"]),
    }


def _pointer_decls(
    *,
    target_name: str,
    input_map: Mapping[str, int],
    output_map: Mapping[str, int],
    state_tensors: Mapping[str, Mapping[str, Any]],
    constant_tensors: Mapping[str, Mapping[str, Any]],
    temporaries: Iterable[Mapping[str, Any]],
    views: Iterable[Mapping[str, Any]],
    tensor_map: Mapping[str, Mapping[str, Any]],
    shape_buffer_idents: set[str],
    shape_buffer_run_update_idents: set[str],
) -> Iterable[str]:
    dynamic_dims = dynamic_dim_sources(input_map=input_map, output_map=output_map, tensor_map=tensor_map)
    view_by_tensor = {str(view["tensor"]): view for view in views}
    config = _gpu_target_config(target_name, lowering_target_spec(target_name))
    check_macro = config["check_macro"]
    memcpy = config["memcpy"]
    memcpy_h2d = config["memcpy_host_to_device"]
    for tensor_name, idx in input_map.items():
        ident = _c_ident(tensor_name)
        cpp_type = storage_type(str(tensor_map[tensor_name]["dtype"]), target_name)
        yield f"const DinoTensor* abi_{ident} = &inputs[{idx}];"
        yield f"const {cpp_type}* ptr_{ident} = static_cast<const {cpp_type}*>(dinoml::module::tensor_data(inputs[{idx}]));"
        for axis in range(len(tensor_map[tensor_name]["shape"])):
            yield f"const int64_t shape_{ident}_{axis} = inputs[{idx}].shape[{axis}];"
        if ident in shape_buffer_idents and ident in shape_buffer_run_update_idents:
            yield f"{check_macro}({memcpy}(session->shape_{ident}, inputs[{idx}].shape, sizeof(int64_t) * {len(tensor_map[tensor_name]['shape'])}, {memcpy_h2d}));"
        if ident in shape_buffer_idents:
            yield f"const int64_t* shape_{ident} = session->shape_{ident};"
        yield f"const int64_t runtime_numel_{ident} = dinoml::module::tensor_numel(inputs[{idx}]);"
    for tensor_name in state_tensors:
        ident = _c_ident(tensor_name)
        cpp_type = storage_type(str(tensor_map[tensor_name]["dtype"]), target_name)
        rank = len(tensor_map[tensor_name]["shape"])
        yield f"const DinoTensor* abi_{ident} = nullptr;"
        for axis in range(rank):
            yield f"const int64_t shape_{ident}_{axis} = {int(tensor_map[tensor_name]['shape'][axis])};"
        if ident in shape_buffer_idents and ident in shape_buffer_run_update_idents:
            yield f"const int64_t host_shape_{ident}[] = {{ {shape_vars_literal(ident, rank)} }};"
            yield f"{check_macro}({memcpy}(session->shape_{ident}, host_shape_{ident}, sizeof(int64_t) * {rank}, {memcpy_h2d}));"
        if ident in shape_buffer_idents:
            yield f"const int64_t* shape_{ident} = session->shape_{ident};"
        yield f"const int64_t runtime_numel_{ident} = {numel_expr(ident, rank)};"
        yield f"{cpp_type}* ptr_{ident} = static_cast<{cpp_type}*>(session->state_{ident});"
    for tensor_name in constant_tensors:
        ident = _c_ident(tensor_name)
        cpp_type = storage_type(str(tensor_map[tensor_name]["dtype"]), target_name)
        rank = len(tensor_map[tensor_name]["shape"])
        yield f"const DinoTensor* abi_{ident} = nullptr;"
        for axis in range(rank):
            yield f"const int64_t shape_{ident}_{axis} = module->const_shape_{ident}[{axis}];"
        if ident in shape_buffer_idents and ident in shape_buffer_run_update_idents:
            yield f"{check_macro}({memcpy}(session->shape_{ident}, module->const_shape_{ident}.data(), sizeof(int64_t) * {rank}, {memcpy_h2d}));"
        if ident in shape_buffer_idents:
            yield f"const int64_t* shape_{ident} = session->shape_{ident};"
        yield f"const int64_t runtime_numel_{ident} = {numel_expr(ident, rank)};"
        yield (
            f"const {cpp_type}* ptr_{ident} = "
            f"static_cast<const {cpp_type}*>(module->const_{ident});"
        )
    for item in temporaries:
        tensor_name = item["tensor"]
        ident = _c_ident(tensor_name)
        cpp_type = storage_type(str(tensor_map[tensor_name]["dtype"]), target_name)
        yield f"const DinoTensor* abi_{ident} = nullptr;"
        for axis in range(len(tensor_map[tensor_name]["shape"])):
            yield f"const int64_t shape_{ident}_{axis} = {shape_dim_expr(tensor_map[tensor_name], axis, dynamic_dims)};"
        if ident in shape_buffer_idents and ident in shape_buffer_run_update_idents:
            yield f"const int64_t host_shape_{ident}[] = {{ {shape_vars_literal(ident, len(tensor_map[tensor_name]['shape']))} }};"
            yield f"{check_macro}({memcpy}(session->shape_{ident}, host_shape_{ident}, sizeof(int64_t) * {len(tensor_map[tensor_name]['shape'])}, {memcpy_h2d}));"
        if ident in shape_buffer_idents:
            yield f"const int64_t* shape_{ident} = session->shape_{ident};"
        yield f"const int64_t runtime_numel_{ident} = {numel_expr(ident, len(tensor_map[tensor_name]['shape']))};"
        yield f"{cpp_type}* ptr_{ident} = static_cast<{cpp_type}*>(session->tmp_{ident});"
    for tensor_name, idx in output_map.items():
        if tensor_name in view_by_tensor:
            continue
        ident = _c_ident(tensor_name)
        cpp_type = storage_type(str(tensor_map[tensor_name]["dtype"]), target_name)
        yield f"const DinoTensor* abi_{ident} = &outputs[{idx}];"
        yield f"{cpp_type}* ptr_{ident} = static_cast<{cpp_type}*>(dinoml::module::tensor_data(outputs[{idx}]));"
        for axis in range(len(tensor_map[tensor_name]["shape"])):
            yield f"const int64_t shape_{ident}_{axis} = outputs[{idx}].shape[{axis}];"
        if ident in shape_buffer_idents and ident in shape_buffer_run_update_idents:
            yield f"{check_macro}({memcpy}(session->shape_{ident}, outputs[{idx}].shape, sizeof(int64_t) * {len(tensor_map[tensor_name]['shape'])}, {memcpy_h2d}));"
        if ident in shape_buffer_idents:
            yield f"const int64_t* shape_{ident} = session->shape_{ident};"
        yield f"const int64_t runtime_numel_{ident} = dinoml::module::tensor_numel(outputs[{idx}]);"
    for view in views:
        tensor_name = str(view["tensor"])
        source_name = str(view["source"])
        ident = _c_ident(tensor_name)
        source_ident = _c_ident(source_name)
        cpp_type = storage_type(str(tensor_map[tensor_name]["dtype"]), target_name)
        output_idx = view.get("output_index")
        yield f"const DinoTensor* abi_{ident} = abi_{source_ident};"
        if output_idx is None:
            for axis in range(len(tensor_map[tensor_name]["shape"])):
                yield f"const int64_t shape_{ident}_{axis} = {shape_dim_expr(tensor_map[tensor_name], axis, dynamic_dims)};"
            if ident in shape_buffer_idents and ident in shape_buffer_run_update_idents:
                yield f"const int64_t host_shape_{ident}[] = {{ {shape_vars_literal(ident, len(tensor_map[tensor_name]['shape']))} }};"
                yield f"{check_macro}({memcpy}(session->shape_{ident}, host_shape_{ident}, sizeof(int64_t) * {len(tensor_map[tensor_name]['shape'])}, {memcpy_h2d}));"
        else:
            for axis in range(len(tensor_map[tensor_name]["shape"])):
                yield f"const int64_t shape_{ident}_{axis} = outputs[{int(output_idx)}].shape[{axis}];"
            if ident in shape_buffer_idents and ident in shape_buffer_run_update_idents:
                yield f"{check_macro}({memcpy}(session->shape_{ident}, outputs[{int(output_idx)}].shape, sizeof(int64_t) * {len(tensor_map[tensor_name]['shape'])}, {memcpy_h2d}));"
        if ident in shape_buffer_idents:
            yield f"const int64_t* shape_{ident} = session->shape_{ident};"
        yield f"const int64_t runtime_numel_{ident} = {numel_expr(ident, len(tensor_map[tensor_name]['shape']))};"
        yield f"const {cpp_type}* ptr_{ident} = ptr_{source_ident} + {int(view.get('offset_elements', 0))};"


def _gguf_dequant_scratch_context(kernel_manifest: Mapping[str, Any] | None) -> dict[str, int] | None:
    if kernel_manifest is None:
        return None
    for resource in kernel_manifest.get("session_resources", ()):
        if not isinstance(resource, Mapping):
            continue
        if str(resource.get("kind", "")) != "gguf_runtime_dequant_scratch":
            continue
        nbytes = int(resource.get("nbytes", 0) or 0)
        if nbytes <= 0:
            continue
        return {"nbytes": nbytes}
    max_scratch = 0
    for item in kernel_manifest.get("required_kernels", []):
        if item.get("kernel_library") != "cutlass_gemm":
            continue
        plan = item.get("gguf_runtime_dequant")
        if not isinstance(plan, Mapping):
            continue
        if str(plan.get("status")) != "lowered_runtime_dequant_scratch":
            continue
        max_scratch = max(max_scratch, int(plan.get("scratch_nbytes", 0) or 0))
    if max_scratch <= 0:
        return None
    return {"nbytes": max_scratch}


def _topk_scratch_context(
    target_name: str,
    ir: Mapping[str, Any],
    tensor_map: Mapping[str, Mapping[str, Any]],
) -> dict[str, int] | None:
    from dinoml.lowering.ops.topk import topk_scratch_nbytes_for_node

    max_scratch = 0
    for node in ir.get("nodes", []):
        if str(node.get("op", "")) not in {"topk_values", "topk_indices"}:
            continue
        max_scratch = max(max_scratch, topk_scratch_nbytes_for_node(target_name, node, tensor_map))
    if max_scratch <= 0:
        return None
    return {"nbytes": max_scratch}


def _flash_attention_static_kv_cache_scratch_context(
    target_name: str,
    ir: Mapping[str, Any],
    tensor_map: Mapping[str, Mapping[str, Any]],
) -> dict[str, int] | None:
    if target_name != "rocm":
        return None
    from dinoml.lowering.ops.flash_attention import flash_attention_static_kv_cache_scratch_nbytes_for_node

    max_scratch = 0
    for node in ir["nodes"]:
        max_scratch = max(
            max_scratch,
            flash_attention_static_kv_cache_scratch_nbytes_for_node(target_name, node, tensor_map),
        )
    if max_scratch <= 0:
        return None
    return {"nbytes": max_scratch}



def _output_materializations(views: Iterable[Mapping[str, Any]], *, target_name: str) -> list[str]:
    config = _gpu_target_config(target_name, lowering_target_spec(target_name))
    materializations = []
    for view in views:
        output_idx = view.get("output_index")
        if output_idx is None:
            continue
        materializations.append(
            f"{config['check_macro']}({config['memcpy_async']}("
            f"dinoml::module::tensor_data(outputs[{int(output_idx)}]), ptr_{view['ident']}, {view['nbytes_expr']}, "
            f"{config['memcpy_device_to_device']}, session->stream));"
        )
    return materializations


def _output_shape_report_contexts(
    ir: Mapping[str, Any],
    *,
    tensor_map: Mapping[str, Mapping[str, Any]],
) -> list[dict[str, Any]]:
    shape_buffer_outputs = {
        str(report["output"])
        for report in ir.get("metadata", {}).get("output_shape_reports", {}).get("reports", [])
        if isinstance(report, Mapping) and report.get("kind") == "shape_buffer"
    }
    reports = []
    for idx, output in enumerate(ir["outputs"]):
        tensor_name = str(output["tensor"])
        reports.append(
            {
                "index": idx,
                "name": str(output["name"]),
                "ident": _c_ident(tensor_name),
                "rank": len(tensor_map[tensor_name]["shape"]),
                "source": "shape_buffer" if str(output["name"]) in shape_buffer_outputs else "caller",
            }
        )
    return reports


def _required_shape_buffer_idents(
    *,
    tensor_map: Mapping[str, Mapping[str, Any]],
    launches: Iterable[str],
    output_shape_reports: Iterable[Mapping[str, Any]],
) -> set[str]:
    idents = {_c_ident(str(name)) for name in tensor_map}
    required: set[str] = set()
    launch_text = "\n".join(launches)
    for ident in idents:
        if re.search(rf"(?<![A-Za-z0-9_])shape_{re.escape(ident)}(?![A-Za-z0-9_])", launch_text):
            required.add(ident)
        if re.search(rf"session->shape_{re.escape(ident)}(?![A-Za-z0-9_])", launch_text):
            required.add(ident)
    for report in output_shape_reports:
        if report.get("source") == "shape_buffer":
            required.add(str(report["ident"]))
    return required


def _shape_buffer_run_update_idents(
    *,
    tensor_map: Mapping[str, Mapping[str, Any]],
    shape_buffer_idents: set[str],
) -> set[str]:
    by_ident = {_c_ident(str(name)): tensor for name, tensor in tensor_map.items()}
    return {
        ident
        for ident in shape_buffer_idents
        if not _shape_spec_is_static(by_ident[ident].get("shape_spec", by_ident[ident]["shape"]))
    }


def _shape_spec_is_static(shape_spec: Iterable[Any]) -> bool:
    return all(isinstance(dim, int) for dim in shape_spec)


def _dim_ranges(shape_spec: Iterable[Any]) -> list[dict[str, int]]:
    return [shape_dim_range(dim) for dim in shape_spec]


def _shape_equal_checks(
    inputs: Iterable[Mapping[str, Any]],
    outputs: Iterable[Mapping[str, Any]],
    constants: Iterable[Mapping[str, Any]],
) -> list[str]:
    dim_sources: dict[str, tuple[str, int, str]] = {}
    checks: list[str] = []
    indexed_inputs = list(enumerate(list(inputs)))
    indexed_outputs = list(enumerate(list(outputs)))
    for array_name, indexed in (("inputs", indexed_inputs), ("outputs", indexed_outputs)):
        for tensor_idx, item in indexed:
            for axis, dim in enumerate(item.get("shape_spec", item["shape"])):
                if isinstance(dim, Mapping) and dim.get("kind") == "dim":
                    _append_named_dim_check(
                        dim_sources,
                        checks,
                        name=str(dim["name"]),
                        expr=f"{array_name}[{tensor_idx}].shape[{axis}]",
                        item_name=str(item["name"]),
                    )
                elif isinstance(dim, Mapping) and dim.get("kind") == "int_expr":
                    for leaf in named_dim_leaves(dim):
                        if str(leaf["name"]) in dim_sources:
                            continue
    for item in constants:
        ident = _c_ident(str(item["tensor"]))
        for axis, dim in enumerate(item.get("shape_spec", item["shape"])):
            if isinstance(dim, Mapping) and dim.get("kind") == "dim":
                _append_named_dim_check(
                    dim_sources,
                    checks,
                    name=str(dim["name"]),
                    expr=f"session->module->const_shape_{ident}[{axis}]",
                    item_name=str(item["name"]),
                )
            elif isinstance(dim, Mapping) and dim.get("kind") == "int_expr":
                for leaf in named_dim_leaves(dim):
                    if str(leaf["name"]) in dim_sources:
                        continue
    dynamic_dims = {name: source[0] for name, source in dim_sources.items()}
    checks.extend(expression_axis_checks(items=inputs, array_name="inputs", dynamic_dims=dynamic_dims))
    checks.extend(expression_axis_checks(items=outputs, array_name="outputs", dynamic_dims=dynamic_dims))
    checks.extend(
        constant_expression_axis_checks(constants=constants, dynamic_dims=dynamic_dims, c_ident_fn=_c_ident)
    )
    return checks


def _append_named_dim_check(
    dim_sources: dict[str, tuple[str, int, str]],
    checks: list[str],
    *,
    name: str,
    expr: str,
    item_name: str,
) -> None:
    if name not in dim_sources:
        dim_sources[name] = (expr, -1, item_name)
        return
    source_expr, _source_idx, source_name = dim_sources[name]
    checks.append(
        f'  if ({expr} != {source_expr}) return dinoml::module::fail("Dynamic dimension {name} mismatch between {source_name} and {item_name}");'
    )


def _numel(shape: Iterable[int]) -> int:
    return int(np.prod(list(shape), dtype=np.int64))



def _shape_args(stage: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    args = stage.get("shape_args")
    if not isinstance(args, (list, tuple)):
        return []
    return [arg for arg in args if isinstance(arg, Mapping)]


def _selected_candidate(item: Mapping[str, Any]) -> Mapping[str, Any]:
    selected_id = item.get("selected_candidate_id")
    candidates = [candidate for candidate in item.get("candidates", []) if isinstance(candidate, Mapping)]
    for candidate in candidates:
        if candidate.get("candidate_id") == selected_id:
            return candidate
    if candidates:
        return candidates[0]
    return {}


def _candidate_by_id(item: Mapping[str, Any], candidate_id: str) -> Mapping[str, Any]:
    for candidate in item.get("candidates", []):
        if isinstance(candidate, Mapping) and str(candidate.get("candidate_id")) == candidate_id:
            return candidate
    return {}
