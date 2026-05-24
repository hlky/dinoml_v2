from __future__ import annotations

import hashlib
from copy import deepcopy
from typing import Any, Mapping, Sequence

from dinoml.constant_sources import (
    GGUF_MATERIALIZATION_DEQUANTIZE_ON_GPU_BEFORE_LAUNCH,
    GGUF_RESIDENCY_MANUAL_RUNTIME_LOAD,
)
from dinoml.ir import canonical_json, dtype_nbytes
from dinoml.kernels.external import external_kernel_families
from dinoml.kernels.bmm import bmm_op_spec, bmm_problem
from dinoml.kernels.providers.cutlass.bmm import cutlass_bmm_candidate_set, cutlass_bmm_candidates
from dinoml.kernels.providers.ck.bmm import (
    ck_bmm_candidate_set,
    ck_bmm_candidates,
    ck_bmm_static_library_name,
)
from dinoml.kernels.providers.ck.conv import (
    ck_conv_candidate_set,
    ck_conv_candidates,
    ck_conv_static_library_name,
)
from dinoml.kernels.providers.cutlass.conv import (
    cutlass_conv_candidate_set,
    cutlass_conv_candidate_compatible_with_plan,
    cutlass_conv_candidates,
    cutlass_conv_layout_plan,
    validate_cutlass_conv_plan,
)
from dinoml.kernels.providers.cutlass.gemm import (
    cutlass_gemm_candidate_set,
    cutlass_gemm_candidates,
    cutlass_gemm_static_library_name,
)
from dinoml.kernels.providers.ck.gemm import (
    ck_gemm_candidate_set,
    ck_gemm_candidates,
    ck_gemm_static_library_name,
)
from dinoml.kernels.providers.cutlass.alignment import (
    alignment_context_candidate_filter,
    cutlass_bmm_static_alignment_context,
    cutlass_candidate_alignment,
    cutlass_candidate_epilogue_alignment,
    cutlass_gemm_static_alignment_context,
    merge_cutlass_alignment_contexts,
    filter_candidates_by_alignment,
)
from dinoml.kernels.gemm import gemm_op_spec, gemm_problem
from dinoml.lowering.ops import generated_source_provenance
from dinoml.ops.definitions import get_op_def


KERNEL_MANIFEST_SCHEMA_VERSION = 4
KERNEL_ABI_VERSION = 1
PROFILE_CACHE_SCHEMA_VERSION = 7


def build_kernel_manifest(ir: Mapping[str, Any], target: Mapping[str, Any]) -> dict[str, Any]:
    target_name = target["name"]
    required = []
    seen = set()
    tensor_map = {tensor["name"]: tensor for tensor in ir["tensors"]}
    constant_map = {constant["tensor"]: constant for constant in ir.get("constants", [])}
    cutlass_alignment_contexts = _cutlass_gemm_alignment_contexts(ir, target_name, tensor_map)
    cutlass_bmm_alignment_contexts = _cutlass_bmm_alignment_contexts(ir, target_name, tensor_map)
    for node in ir["nodes"]:
        op_def = get_op_def(node["op"])
        binding = op_def.backend_kernels[target_name]
        output_name = str(node["outputs"][0])
        dtype = str(tensor_map[output_name]["dtype"])
        resolved = binding.resolve(dtype)
        kernel_symbol = resolved.symbol
        profiler_symbol = resolved.profiler_symbol
        candidates = [dict(candidate) for candidate in resolved.candidates]
        candidate_set = dict(resolved.candidate_set) if resolved.candidate_set else None
        selected_candidate_id = candidates[0]["candidate_id"] if candidates else None
        if resolved.library == "cutlass_gemm":
            candidates = [dict(candidate) for candidate in cutlass_gemm_candidates(str(node["op"]), dtype, target=target)]
            candidate_set = cutlass_gemm_candidate_set(str(node["op"]), dtype, target=target)
            alignment_context = cutlass_alignment_contexts.get((str(node["op"]), dtype))
            selected_candidate = _select_cutlass_manifest_candidate(
                str(node["op"]),
                dtype,
                candidates,
                alignment_context,
            )
            kernel_symbol = str(selected_candidate["kernel_symbol"])
            profiler_symbol = str(selected_candidate["profiler_symbol"])
            selected_candidate_id = str(selected_candidate["candidate_id"])
            gguf_runtime_dequant = _gguf_runtime_dequant_gemm_rhs_plan(
                node,
                tensor_map=tensor_map,
                constant_map=constant_map,
                dtype=dtype,
            )
            cutlass_conv_plan = None
        elif resolved.library == "ck_gemm":
            candidates = [dict(candidate) for candidate in ck_gemm_candidates(str(node["op"]), dtype, target=target)]
            candidate_set = ck_gemm_candidate_set(str(node["op"]), dtype, target=target)
            if not candidates:
                raise ValueError(f"CK GEMM manifest candidate selection requires at least one candidate for {node['op']} {dtype}")
            selected_candidate = _select_ck_gemm_manifest_candidate(node, tensor_map, candidates)
            kernel_symbol = str(selected_candidate["kernel_symbol"])
            profiler_symbol = str(selected_candidate["profiler_symbol"])
            selected_candidate_id = str(selected_candidate["candidate_id"])
            gguf_runtime_dequant = None
            cutlass_conv_plan = None
        elif resolved.library == "cutlass_bmm":
            candidates = [dict(candidate) for candidate in cutlass_bmm_candidates(str(node["op"]), dtype, target=target)]
            candidate_set = cutlass_bmm_candidate_set(str(node["op"]), dtype, target=target)
            alignment_context = cutlass_bmm_alignment_contexts.get((str(node["op"]), dtype))
            selected_candidate = _select_cutlass_manifest_candidate(
                str(node["op"]),
                dtype,
                candidates,
                alignment_context,
            )
            kernel_symbol = str(selected_candidate["kernel_symbol"])
            profiler_symbol = str(selected_candidate["profiler_symbol"])
            selected_candidate_id = str(selected_candidate["candidate_id"])
            gguf_runtime_dequant = None
            cutlass_conv_plan = None
        elif resolved.library == "ck_bmm":
            candidates = [dict(candidate) for candidate in ck_bmm_candidates(str(node["op"]), dtype, target=target)]
            candidate_set = ck_bmm_candidate_set(str(node["op"]), dtype, target=target)
            if not candidates:
                raise ValueError(f"CK BMM manifest candidate selection requires at least one candidate for {node['op']} {dtype}")
            selected_candidate = _select_ck_bmm_manifest_candidate(node, tensor_map, candidates)
            kernel_symbol = str(selected_candidate["kernel_symbol"])
            profiler_symbol = str(selected_candidate["profiler_symbol"])
            selected_candidate_id = str(selected_candidate["candidate_id"])
            gguf_runtime_dequant = None
            cutlass_conv_plan = None
        elif resolved.library == "ck_conv":
            candidates = [dict(candidate) for candidate in ck_conv_candidates(str(node["op"]), dtype, target=target)]
            candidate_set = ck_conv_candidate_set(str(node["op"]), dtype, target=target)
            if not candidates:
                raise ValueError(f"CK Conv manifest candidate selection requires at least one candidate for {node['op']} {dtype}")
            selected_candidate = _select_ck_conv_manifest_candidate(node, tensor_map, candidates)
            kernel_symbol = str(selected_candidate["kernel_symbol"])
            profiler_symbol = str(selected_candidate["profiler_symbol"])
            selected_candidate_id = str(selected_candidate["candidate_id"])
            gguf_runtime_dequant = None
            cutlass_conv_plan = None
        elif resolved.library == "cutlass_conv":
            candidates = [dict(candidate) for candidate in cutlass_conv_candidates(str(node["op"]), dtype, target=target)]
            candidate_set = cutlass_conv_candidate_set(str(node["op"]), dtype, target=target)
            cutlass_conv_plan = cutlass_conv_layout_plan(node, tensor_map=tensor_map)
            selected_candidate = _select_cutlass_conv_manifest_candidate(cutlass_conv_plan, candidates)
            cutlass_conv_plan = {
                **cutlass_conv_plan,
                "selected_candidate": {
                    "candidate_id": str(selected_candidate["candidate_id"]),
                    "symbol_id": str(selected_candidate.get("symbol_id", "")),
                    "kernel_symbol": str(selected_candidate["kernel_symbol"]),
                    "opclass": str(selected_candidate.get("cutlass", {}).get("opclass", "")),
                    "iterator_algorithm": str(selected_candidate.get("cutlass", {}).get("iterator_algorithm", "")),
                    "selection_predicate": dict(selected_candidate.get("selection_predicate", {})),
                },
            }
            kernel_symbol = str(selected_candidate["kernel_symbol"])
            profiler_symbol = str(selected_candidate["profiler_symbol"])
            selected_candidate_id = str(selected_candidate["candidate_id"])
            gguf_runtime_dequant = None
        else:
            gguf_runtime_dequant = None
            cutlass_conv_plan = None
        model_generated_source = (
            generated_source_provenance(target_name, node, tensor_map) if resolved.library == "model" else None
        )
        key = (
            node["op"],
            kernel_symbol,
            canonical_json(model_generated_source) if model_generated_source is not None else "",
            canonical_json(gguf_runtime_dequant) if gguf_runtime_dequant is not None else "",
            canonical_json(cutlass_conv_plan) if cutlass_conv_plan is not None else "",
        )
        if key in seen:
            continue
        seen.add(key)
        item = {
            "op": node["op"],
            "dtype": dtype,
            "kernel_symbol": kernel_symbol,
            "kernel_library": resolved.library,
            "profiler_symbol": profiler_symbol,
            "has_profiler": op_def.profiler,
        }
        if resolved.library == "cutlass_gemm":
            item["support_archive"] = cutlass_gemm_static_library_name(str(node["op"]), dtype)
        if resolved.library == "ck_gemm":
            item["support_archive"] = ck_gemm_static_library_name(str(node["op"]), dtype)
        if resolved.library == "ck_bmm":
            item["support_archive"] = ck_bmm_static_library_name(str(node["op"]), dtype)
        if resolved.library == "ck_conv":
            item["support_archive"] = ck_conv_static_library_name(str(node["op"]), dtype)
        if model_generated_source is not None:
            item["generated_source"] = dict(model_generated_source)
        if candidates:
            item["selected_candidate_id"] = selected_candidate_id
            item["candidates"] = candidates
        if candidate_set:
            item["candidate_set_id"] = candidate_set["candidate_set_id"]
            item["candidate_set_key"] = candidate_set["candidate_set_key"]
            item["candidate_set"] = candidate_set
        if gguf_runtime_dequant is not None:
            item["gguf_runtime_dequant"] = gguf_runtime_dequant
        if cutlass_conv_plan is not None:
            item["cutlass_conv_plan"] = cutlass_conv_plan
            if cutlass_conv_plan.get("source_op") is not None:
                item["source_op"] = str(cutlass_conv_plan["source_op"])
                item["bias_mode"] = str(cutlass_conv_plan["bias_mode"])
        if resolved.library in {"cutlass_gemm", "cutlass_bmm"}:
            alignment_context = (
                cutlass_alignment_contexts.get((str(node["op"]), dtype))
                if resolved.library == "cutlass_gemm"
                else cutlass_bmm_alignment_contexts.get((str(node["op"]), dtype))
            )
            if alignment_context is not None:
                item["cutlass_alignment"] = alignment_context
                item["cutlass_alignment_cap"] = alignment_context_candidate_filter(alignment_context)[
                    "max_operand_alignment"
                ]
            _attach_cutlass_alignment_fallbacks(item)
        required.append(item)
    manifest = {
        "schema_version": KERNEL_MANIFEST_SCHEMA_VERSION,
        "kernel_abi_version": KERNEL_ABI_VERSION,
        "profile_cache_schema_version": PROFILE_CACHE_SCHEMA_VERSION,
        "target": dict(target),
        "required_kernels": required,
        "codegen_strategy": "prebuilt_static_library",
        "profiler_strategy": "manifest_declared_not_yet_generated",
    }
    session_resources = _session_resource_plan(required)
    if session_resources:
        manifest["session_resources"] = session_resources
    return _with_kernel_manifest_cache_keys(manifest)


def _select_ck_gemm_manifest_candidate(
    node: Mapping[str, Any],
    tensor_map: Mapping[str, Mapping[str, Any]],
    candidates: Sequence[Mapping[str, Any]],
) -> Mapping[str, Any]:
    input_names = [str(name) for name in node.get("inputs", ())]
    if len(input_names) < 2:
        return candidates[0]
    shapes = [_static_tensor_shape(tensor_map.get(name, {})) for name in input_names]
    if any(shape is None for shape in shapes):
        return candidates[0]
    try:
        m, n, k, _ = gemm_problem(str(node["op"]), [shape for shape in shapes if shape is not None])
    except ValueError:
        return candidates[0]
    spec = gemm_op_spec(str(node["op"]))
    problem = {
        "m": m,
        "n": n,
        "k": k,
        "a_k": k,
        "b_k": k,
        "b_n": n,
        "output_n": n,
        "base_layout": spec.base_layout,
    }
    return _select_ck_candidate(candidates, problem)


def _select_ck_bmm_manifest_candidate(
    node: Mapping[str, Any],
    tensor_map: Mapping[str, Mapping[str, Any]],
    candidates: Sequence[Mapping[str, Any]],
) -> Mapping[str, Any]:
    input_names = [str(name) for name in node.get("inputs", ())]
    if len(input_names) < 2:
        return candidates[0]
    shapes = [_static_tensor_shape(tensor_map.get(name, {})) for name in input_names]
    if any(shape is None for shape in shapes):
        return candidates[0]
    try:
        batch, m, n, k, _ = bmm_problem(str(node["op"]), [shape for shape in shapes if shape is not None])
    except ValueError:
        return candidates[0]
    spec = bmm_op_spec(str(node["op"]))
    problem = {
        "batch": batch,
        "m": m,
        "n": n,
        "k": k,
        "a_m": m,
        "a_k": k,
        "b_n": n,
        "b_k": k,
        "output_n": n,
        "output_layout": spec.c_layout,
        "base_layout": spec.base_layout,
    }
    return _select_ck_candidate(candidates, problem)


def _select_ck_conv_manifest_candidate(
    node: Mapping[str, Any],
    tensor_map: Mapping[str, Mapping[str, Any]],
    candidates: Sequence[Mapping[str, Any]],
) -> Mapping[str, Any]:
    input_names = [str(name) for name in node.get("inputs", ())]
    output_names = [str(name) for name in node.get("outputs", ())]
    if len(input_names) < 2 or not output_names:
        return candidates[0]
    x_shape = _static_tensor_shape(tensor_map.get(input_names[0], {}))
    weight_shape = _static_tensor_shape(tensor_map.get(input_names[1], {}))
    out_shape = _static_tensor_shape(tensor_map.get(output_names[0], {}))
    if x_shape is None or weight_shape is None or out_shape is None:
        return candidates[0]
    if len(x_shape) != 4 or len(weight_shape) != 4 or len(out_shape) != 4:
        return candidates[0]
    batch, in_channels, _, _ = x_shape
    out_channels, _, kernel_h, kernel_w = weight_shape
    out_h, out_w = out_shape[2], out_shape[3]
    problem = {
        "batch": batch,
        "in_channels": in_channels,
        "out_channels": out_channels,
        "kernel_h": kernel_h,
        "kernel_w": kernel_w,
        "out_h": out_h,
        "out_w": out_w,
        "gemm_m": batch * out_h * out_w,
        "gemm_n": out_channels,
        "gemm_k": in_channels * kernel_h * kernel_w,
    }
    return _select_ck_candidate(candidates, problem)


def _select_ck_candidate(
    candidates: Sequence[Mapping[str, Any]],
    problem: Mapping[str, int | str],
) -> Mapping[str, Any]:
    compatible = [candidate for candidate in candidates if _ck_candidate_compatible(candidate, problem)]
    if not compatible:
        return candidates[0]
    return max(
        compatible,
        key=lambda candidate: int(candidate.get("selection_predicate", {}).get("priority", 0)),
    )


def _ck_candidate_compatible(candidate: Mapping[str, Any], problem: Mapping[str, int | str]) -> bool:
    predicate = candidate.get("selection_predicate")
    if not isinstance(predicate, Mapping):
        return True
    required_output_layout = predicate.get("requires_output_layout")
    if required_output_layout is not None and problem.get("output_layout") != required_output_layout:
        return False
    min_problem = predicate.get("min_problem", {})
    if isinstance(min_problem, Mapping):
        for key, minimum in min_problem.items():
            value = problem.get(str(key))
            if not isinstance(value, int) or value < int(minimum):
                return False
    alignment = predicate.get("alignment", {})
    if isinstance(alignment, Mapping):
        for key, divisor in alignment.items():
            width = int(divisor)
            value = problem.get(str(key))
            if width > 1 and (not isinstance(value, int) or value % width != 0):
                return False
    return True


def _static_tensor_shape(tensor: Mapping[str, Any]) -> list[int] | None:
    shape = tensor.get("shape")
    if not isinstance(shape, Sequence):
        return None
    result = []
    for dim in shape:
        if not isinstance(dim, int):
            return None
        result.append(int(dim))
    return result


def _gguf_runtime_dequant_gemm_rhs_plan(
    node: Mapping[str, Any],
    *,
    tensor_map: Mapping[str, Mapping[str, Any]],
    constant_map: Mapping[str, Mapping[str, Any]],
    dtype: str,
) -> dict[str, Any] | None:
    inputs = list(node.get("inputs", ()))
    if len(inputs) < 2:
        return None
    b_name = str(inputs[1])
    constant = constant_map.get(b_name)
    if constant is None:
        return None
    storage = constant.get("storage")
    if not isinstance(storage, Mapping) or storage.get("kind") != "gguf":
        return None
    materialization = str(storage.get("materialization", "dequantize_full_before_launch"))
    if materialization != GGUF_MATERIALIZATION_DEQUANTIZE_ON_GPU_BEFORE_LAUNCH:
        return None
    residency = str(storage.get("residency", "eager_dense_device"))
    shape = [int(dim) for dim in tensor_map[b_name].get("shape", constant.get("shape", []))]
    logical_numel = 1
    for dim in shape:
        logical_numel *= int(dim)
    residency_supported = residency == GGUF_RESIDENCY_MANUAL_RUNTIME_LOAD
    lowered = residency_supported and str(node.get("op", "")) in {
        "gemm_rrr",
        "gemm_rcr",
        "gemm_rrr_bias",
        "gemm_rcr_bias",
    } and dtype in {"float32", "float16"}
    status = "lowered_runtime_dequant_scratch" if lowered else "planned_not_lowered"
    blocked_reason = None
    if not lowered:
        if not residency_supported:
            blocked_reason = f"unsupported_gguf_runtime_dequant_residency:{residency}"
        else:
            blocked_reason = "unsupported_gguf_runtime_dequant_gemm_slice"
    plan = {
        "schema_version": 1,
        "kind": "gguf_runtime_dequant_before_cutlass_gemm",
        "status": status,
        "node_id": str(node.get("id", "")),
        "op": str(node.get("op", "")),
        "operand": "b",
        "constant": b_name,
        "storage_kind": "gguf",
        "materialization": materialization,
        "residency": residency,
        "supported_residency": GGUF_RESIDENCY_MANUAL_RUNTIME_LOAD,
        "qtype": storage.get("qtype"),
        "qtype_value": storage.get("qtype_value"),
        "encoded_nbytes": int(storage.get("encoded_nbytes", 0) or 0),
        "n_per_row": storage.get("n_per_row"),
        "logical_dtype": dtype,
        "logical_shape": shape,
        "scratch_nbytes": logical_numel * dtype_nbytes(dtype),
        "dequant_scratch": "session_temporary_dense_rhs",
        "dense_launcher": "existing_cutlass_gemm",
    }
    if blocked_reason is not None:
        plan["blocked_reason"] = blocked_reason
    return plan


def _select_cutlass_conv_manifest_candidate(
    cutlass_conv_plan: Mapping[str, Any],
    candidates: Sequence[Mapping[str, Any]],
) -> Mapping[str, Any]:
    compatible = [
        candidate
        for candidate in candidates
        if cutlass_conv_candidate_compatible_with_plan(candidate, cutlass_conv_plan)
    ]
    for predicate_kind in ("semantic_input_channels", "natural_alignment"):
        for candidate in compatible:
            predicate = candidate.get("selection_predicate", {})
            if isinstance(predicate, Mapping) and predicate.get("kind") == predicate_kind:
                return candidate
    for candidate in candidates:
        predicate = candidate.get("selection_predicate", {})
        if (
            isinstance(predicate, Mapping)
            and predicate.get("kind") == "fallback"
            and cutlass_conv_candidate_compatible_with_plan(candidate, cutlass_conv_plan)
        ):
            return candidate
    if not candidates:
        raise ValueError("CUTLASS Conv manifest candidate selection requires at least one candidate")
    raise ValueError("CUTLASS Conv manifest candidate selection found no candidate compatible with the transform plan")


def _session_resource_plan(required_kernels: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    gguf_plans = []
    max_scratch_nbytes = 0
    for item in required_kernels:
        if item.get("kernel_library") != "cutlass_gemm":
            continue
        plan = item.get("gguf_runtime_dequant")
        if not isinstance(plan, Mapping):
            continue
        if str(plan.get("status")) != "lowered_runtime_dequant_scratch":
            continue
        scratch_nbytes = int(plan.get("scratch_nbytes", 0) or 0)
        if scratch_nbytes <= 0:
            continue
        max_scratch_nbytes = max(max_scratch_nbytes, scratch_nbytes)
        gguf_plans.append(
            {
                "node_id": str(plan.get("node_id", "")),
                "op": str(plan.get("op", "")),
                "constant": str(plan.get("constant", "")),
                "scratch_nbytes": scratch_nbytes,
            }
        )
    if not gguf_plans:
        return []
    return [
        {
            "schema_version": 1,
            "kind": "gguf_runtime_dequant_scratch",
            "name": "gguf_runtime_dequant_dense_rhs",
            "allocation": "per_session",
            "residency": "cuda_device",
            "reuse": "shared_max_sized",
            "nbytes": max_scratch_nbytes,
            "source_plans": gguf_plans,
        }
    ]


def _cutlass_gemm_alignment_contexts(
    ir: Mapping[str, Any],
    target_name: str,
    tensor_map: Mapping[str, Mapping[str, Any]],
) -> dict[tuple[str, str], dict[str, Any]]:
    contexts: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for node in ir["nodes"]:
        op_def = get_op_def(node["op"])
        binding = op_def.backend_kernels[target_name]
        output_name = str(node["outputs"][0])
        dtype = str(tensor_map[output_name]["dtype"])
        resolved = binding.resolve(dtype)
        if resolved.library != "cutlass_gemm":
            continue
        spec = gemm_op_spec(str(node["op"]))
        a_name, b_name = (str(name) for name in node["inputs"][:2])
        epilogue_names = tuple(
            str(node["inputs"][input_offset])
            for input_offset, _input_name in enumerate(spec.epilogue.inputs, start=2)
        )
        context = cutlass_gemm_static_alignment_context(
            str(node["op"]),
            dtype,
            tensor_map,
            a_name=a_name,
            b_name=b_name,
            c_name=str(node["outputs"][0]),
            epilogue_names=epilogue_names,
        )
        context["node_id"] = str(node["id"])
        key = (str(node["op"]), dtype)
        contexts.setdefault(key, []).append(context)
    return {key: merge_cutlass_alignment_contexts(values) for key, values in contexts.items()}


def _cutlass_bmm_alignment_contexts(
    ir: Mapping[str, Any],
    target_name: str,
    tensor_map: Mapping[str, Mapping[str, Any]],
) -> dict[tuple[str, str], dict[str, Any]]:
    contexts: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for node in ir["nodes"]:
        op_def = get_op_def(node["op"])
        binding = op_def.backend_kernels.get(target_name)
        if binding is None:
            continue
        output_name = str(node["outputs"][0])
        dtype = str(tensor_map[output_name]["dtype"])
        resolved = binding.resolve(dtype)
        if resolved.library != "cutlass_bmm":
            continue
        spec = bmm_op_spec(str(node["op"]))
        a_name, b_name = (str(name) for name in node["inputs"][:2])
        epilogue_names = tuple(
            str(node["inputs"][input_offset])
            for input_offset, _input_name in enumerate(spec.inputs, start=2)
        )
        context = cutlass_bmm_static_alignment_context(
            str(node["op"]),
            dtype,
            tensor_map,
            a_name=a_name,
            b_name=b_name,
            c_name=str(node["outputs"][0]),
            epilogue_names=epilogue_names,
        )
        context["node_id"] = str(node["id"])
        key = (str(node["op"]), dtype)
        contexts.setdefault(key, []).append(context)
    return {key: merge_cutlass_alignment_contexts(values) for key, values in contexts.items()}


def _select_cutlass_manifest_candidate(
    op_name: str,
    dtype: str,
    candidates: Sequence[Mapping[str, Any]],
    alignment_context: Mapping[str, Any] | None,
) -> Mapping[str, Any]:
    candidate_filter = alignment_context_candidate_filter(alignment_context)
    filtered = filter_candidates_by_alignment(
        candidates,
        candidate_filter["max_operand_alignment"],
        candidate_filter["max_epilogue_alignment"],
    )
    if not filtered:
        raise ValueError(
            "CUTLASS GEMM manifest alignment filter removed all candidates "
            f"for {op_name} {dtype} with filter {candidate_filter}"
        )
    return filtered[0]


def apply_execution_plan(
    kernel_manifest: Mapping[str, Any],
    execution_plan: Mapping[str, Any],
    *,
    strict: bool = False,
) -> dict[str, Any]:
    _validate_execution_plan_selection_uniqueness(execution_plan, strict=strict)
    manifest = deepcopy(dict(kernel_manifest))
    selections = _execution_plan_static_selections(execution_plan)
    guarded_selections = _execution_plan_guarded_selections(execution_plan)
    conflict_keys = _execution_plan_conflict_keys(execution_plan)
    if not selections and not guarded_selections:
        return _with_kernel_manifest_cache_keys(manifest)
    applied_keys: set[tuple[str, str, str]] = set()
    guarded_keys = {key for key in guarded_selections if key in conflict_keys or key not in selections}
    applied_guarded_keys: set[tuple[str, str, str]] = set()
    manifest_kernel_libraries: dict[tuple[str, str, str], str] = {}
    for item in manifest.get("required_kernels", []):
        kernel_library = item.get("kernel_library")
        if kernel_library not in {"cutlass_gemm", "cutlass_bmm", "cutlass_conv"}:
            continue
        candidate_set = item.get("candidate_set", {})
        dtype = str(candidate_set.get("dtype", "")) if isinstance(candidate_set, Mapping) else ""
        key = (str(item.get("op", "")), dtype, str(item.get("candidate_set_key", "")))
        manifest_kernel_libraries[key] = str(kernel_library)
        selection = selections.get(key)
        if selection is not None:
            selected_candidate = _execution_plan_candidate(item, key, selection, strict=strict, check_alignment_cap=True)
            if selected_candidate is not None and _execution_plan_selection_supported(
                str(kernel_library),
                key,
                selection,
                strict=strict,
            ):
                if kernel_library == "cutlass_conv" and not _apply_cutlass_conv_static_selection(
                    item,
                    selection,
                    selected_candidate,
                    strict=strict,
                ):
                    continue
                applied_keys.add(key)
                selected_id = str(selection.get("selected_candidate_id", ""))
                item["selected_candidate_id"] = selected_id
                item["kernel_symbol"] = str(selection.get("kernel_symbol") or selected_candidate["kernel_symbol"])
                item["profiler_symbol"] = str(selection.get("profiler_symbol") or selected_candidate["profiler_symbol"])
                item["execution_plan_selection"] = _execution_plan_selection_payload(
                    execution_plan,
                    key,
                    selection,
                    selected_candidate,
                )
        dispatch_group = guarded_selections.get(key, ())
        if dispatch_group and kernel_library == "cutlass_conv" and (key in conflict_keys or key not in selections):
            if strict:
                raise ValueError(
                    "CUTLASS Conv execution plans only support static selections; "
                    f"guarded selection was provided for {key[0]} {key[1]}"
                )
            continue
        if dispatch_group and (key in conflict_keys or key not in selections):
            dispatch_entries = []
            for guarded in dispatch_group:
                if not _execution_plan_guarded_node_supported(item, key, guarded, strict=strict):
                    continue
                if not _execution_plan_guarded_shape_supported(str(kernel_library), key, guarded, strict=strict):
                    continue
                selected_candidate = _execution_plan_candidate(item, key, guarded, strict=strict, check_alignment_cap=False)
                if selected_candidate is None or not _execution_plan_selection_supported(
                    str(kernel_library),
                    key,
                    guarded,
                    strict=strict,
                ):
                    continue
                dispatch_entries.append(_execution_plan_selection_payload(execution_plan, key, guarded, selected_candidate))
            if dispatch_entries:
                item["execution_plan_dispatch"] = dispatch_entries
                applied_guarded_keys.add(key)
        _attach_cutlass_alignment_fallbacks(item)
    if strict:
        missing = sorted(set(selections) - applied_keys)
        if missing:
            missing_text = ", ".join(f"{op}/{dtype}/{candidate_set_key}" for op, dtype, candidate_set_key in missing)
            raise ValueError(f"Execution plan selections did not match the kernel manifest: {missing_text}")
        missing_guarded = sorted(
            key
            for key in guarded_keys - applied_guarded_keys
            if manifest_kernel_libraries.get(key, "cutlass_gemm") in {"cutlass_gemm", "cutlass_bmm", "cutlass_conv"}
        )
        if missing_guarded:
            missing_text = ", ".join(f"{op}/{dtype}/{candidate_set_key}" for op, dtype, candidate_set_key in missing_guarded)
            raise ValueError(f"Execution plan guarded selections did not match the kernel manifest: {missing_text}")
    return _with_kernel_manifest_cache_keys(manifest)


def _validate_execution_plan_selection_uniqueness(
    execution_plan: Mapping[str, Any],
    *,
    strict: bool,
) -> None:
    if not strict:
        return
    _validate_unique_execution_plan_entries(execution_plan.get("static_selections", ()), kind="static")
    _validate_unique_execution_plan_entries(execution_plan.get("selections", ()), kind="guarded")


def _validate_unique_execution_plan_entries(entries: Any, *, kind: str) -> None:
    if not isinstance(entries, Sequence) or isinstance(entries, (str, bytes)):
        return
    seen: dict[tuple[str, ...], int] = {}
    for index, selection in enumerate(entries):
        key = _execution_plan_selection_uniqueness_key(selection, kind=kind)
        if key is None:
            continue
        previous = seen.get(key)
        if previous is not None:
            if kind == "static":
                _, op, dtype, candidate_set_key = key
                raise ValueError(
                    "Execution plan contains duplicate static selections "
                    f"for {op} {dtype} candidate set {candidate_set_key} "
                    f"(entries {previous} and {index})"
                )
            _, op, dtype, candidate_set_key, node_id, shape_key = key
            raise ValueError(
                "Execution plan contains duplicate guarded selections "
                f"for {op} {dtype} candidate set {candidate_set_key} "
                f"node_id={node_id!r} shape={shape_key} "
                f"(entries {previous} and {index})"
            )
        seen[key] = index


def _execution_plan_selection_uniqueness_key(
    selection: Any,
    *,
    kind: str,
) -> tuple[str, ...] | None:
    if not isinstance(selection, Mapping):
        return None
    op = str(selection.get("op", ""))
    dtype = str(selection.get("dtype", ""))
    candidate_set_key = str(selection.get("candidate_set_key", ""))
    if not (op and dtype and candidate_set_key):
        return None
    if kind == "static":
        return (kind, op, dtype, candidate_set_key)
    node_id = str(selection.get("node_id", ""))
    shape = selection.get("shape")
    shape_key = canonical_json(shape) if isinstance(shape, Mapping) else "<missing>"
    return (kind, op, dtype, candidate_set_key, node_id, shape_key)


def _execution_plan_selection_supported(
    kernel_library: str,
    key: tuple[str, str, str],
    selection: Mapping[str, Any],
    *,
    strict: bool,
) -> bool:
    confidence = selection.get("confidence")
    if isinstance(confidence, Mapping) and confidence.get("confident") is False:
        if strict:
            raise ValueError(
                "Execution plan selected low-confidence CUTLASS candidate "
                f"for {key[0]} {key[1]}; low-confidence selections are audit-only"
            )
        return False
    split_k = _execution_plan_int_field(selection, "split_k", 1, minimum=1, key=key, strict=strict)
    workspace_nbytes = _execution_plan_int_field(selection, "workspace_nbytes", 0, minimum=0, key=key, strict=strict)
    if split_k is None or workspace_nbytes is None:
        return False
    if kernel_library != "cutlass_bmm":
        if kernel_library == "cutlass_conv" and (split_k != 1 or workspace_nbytes != 0):
            if strict:
                raise ValueError(
                    "CUTLASS Conv execution plan selections only support split_k=1 "
                    f"and workspace_nbytes=0 for {key[0]} {key[1]}"
                )
            return False
        return True
    if split_k == 1 and workspace_nbytes == 0:
        return True
    if strict:
        raise ValueError(
            "CUTLASS BMM execution plan selections only support split_k=1 "
            f"and workspace_nbytes=0 for {key[0]} {key[1]}"
        )
    return False


def _execution_plan_candidate(
    item: Mapping[str, Any],
    key: tuple[str, str, str],
    selection: Mapping[str, Any],
    *,
    strict: bool,
    check_alignment_cap: bool,
) -> Mapping[str, Any] | None:
    selected_id = str(selection.get("selected_candidate_id", ""))
    selected_candidate = _candidate_by_id(item, selected_id)
    if selected_candidate is None:
        if strict:
            raise ValueError(
                "Execution plan selected unknown CUTLASS candidate "
                f"{selected_id!r} for {key[0]} {key[1]} candidate set {key[2]}"
            )
        return None
    selected_config_key = selected_candidate.get("candidate_config_key")
    plan_config_key = selection.get("candidate_config_key")
    if selected_config_key is not None and plan_config_key is not None and str(selected_config_key) != str(plan_config_key):
        if strict:
            raise ValueError(
                "Execution plan candidate_config_key mismatch for CUTLASS candidate "
                f"{selected_id!r} on {key[0]} {key[1]}"
            )
        return None
    for symbol_field in ("kernel_symbol", "profiler_symbol"):
        plan_symbol = selection.get(symbol_field)
        candidate_symbol = selected_candidate.get(symbol_field)
        if plan_symbol is not None and candidate_symbol is not None and str(plan_symbol) != str(candidate_symbol):
            if strict:
                raise ValueError(
                    f"Execution plan {symbol_field} mismatch for CUTLASS candidate "
                    f"{selected_id!r} on {key[0]} {key[1]}"
                )
            return None
    alignment_cap = item.get("cutlass_alignment_cap")
    if check_alignment_cap and alignment_cap is not None and cutlass_candidate_alignment(selected_candidate) > int(alignment_cap):
        if strict:
            raise ValueError(
                "Execution plan selected CUTLASS candidate "
                f"{selected_id!r} for {key[0]} {key[1]} exceeds alignment cap {int(alignment_cap)}"
            )
        return None
    return selected_candidate


def _execution_plan_guarded_shape_supported(
    kernel_library: str,
    key: tuple[str, str, str],
    selection: Mapping[str, Any],
    *,
    strict: bool,
) -> bool:
    shape = selection.get("shape")
    if not isinstance(shape, Mapping):
        if strict:
            raise ValueError(f"Execution plan guarded selection for {key[0]} {key[1]} is missing shape metadata")
        return False
    required_fields = ("m", "n", "k", "batch_count") if kernel_library == "cutlass_bmm" else ("m", "n", "k")
    for field in required_fields:
        if _execution_plan_int_field(shape, field, None, minimum=1, key=key, strict=strict, context="guarded shape") is None:
            return False
    return True


def _execution_plan_guarded_node_supported(
    item: Mapping[str, Any],
    key: tuple[str, str, str],
    selection: Mapping[str, Any],
    *,
    strict: bool,
) -> bool:
    node_id = selection.get("node_id")
    if not node_id:
        if strict:
            raise ValueError(f"Execution plan guarded selection for {key[0]} {key[1]} is missing node_id")
        return False
    manifest_node_ids = _cutlass_manifest_node_ids(item)
    if manifest_node_ids and str(node_id) not in manifest_node_ids:
        if strict:
            raise ValueError(
                "Execution plan guarded selection node_id "
                f"{str(node_id)!r} does not match the kernel manifest for {key[0]} {key[1]}"
            )
        return False
    return True


def _cutlass_manifest_node_ids(item: Mapping[str, Any]) -> set[str]:
    alignment_context = item.get("cutlass_alignment")
    if not isinstance(alignment_context, Mapping):
        return set()
    nodes = alignment_context.get("nodes")
    if not isinstance(nodes, Sequence) or isinstance(nodes, (str, bytes)):
        return set()
    node_ids = set()
    for node in nodes:
        if isinstance(node, Mapping) and node.get("node_id"):
            node_ids.add(str(node["node_id"]))
    return node_ids


def _execution_plan_int_field(
    payload: Mapping[str, Any],
    field: str,
    default: int | None,
    *,
    minimum: int,
    key: tuple[str, str, str],
    strict: bool,
    context: str = "selection",
) -> int | None:
    raw_value = payload.get(field, default)
    if raw_value is None:
        if strict:
            raise ValueError(f"Execution plan {context} for {key[0]} {key[1]} is missing integer field {field!r}")
        return None
    if type(raw_value) is not int:
        if strict:
            raise ValueError(
                f"Execution plan {context} for {key[0]} {key[1]} has malformed integer field {field!r}: {raw_value!r}"
            )
        return None
    value = raw_value
    if value < minimum:
        if strict:
            raise ValueError(
                f"Execution plan {context} for {key[0]} {key[1]} has invalid integer field "
                f"{field!r}: {value} < {minimum}"
            )
        return None
    return value


def _execution_plan_selection_payload(
    execution_plan: Mapping[str, Any],
    key: tuple[str, str, str],
    selection: Mapping[str, Any],
    selected_candidate: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": int(execution_plan.get("schema_version", 1)),
        "selection_key": selection.get("selection_key"),
        "node_id": selection.get("node_id"),
        "candidate_set_key": key[2],
        "kernel_library": selection.get("kernel_library"),
        "selected_candidate_id": selection.get("selected_candidate_id"),
        "candidate_config_key": selection.get("candidate_config_key") or selected_candidate.get("candidate_config_key"),
        "kernel_symbol": str(selection.get("kernel_symbol") or selected_candidate["kernel_symbol"]),
        "profiler_symbol": str(selection.get("profiler_symbol") or selected_candidate["profiler_symbol"]),
        "shape": dict(selection.get("shape", {})) if isinstance(selection.get("shape"), Mapping) else {},
        "avg_ms": selection.get("avg_ms"),
        "confidence": dict(selection.get("confidence", {})) if isinstance(selection.get("confidence"), Mapping) else {},
        "split_k": int(selection.get("split_k", 1) or 1),
        "workspace_nbytes": int(selection.get("workspace_nbytes", 0) or 0),
        **(
            {"source_op": selection.get("source_op"), "bias_mode": selection.get("bias_mode")}
            if selection.get("source_op") is not None
            else {}
        ),
    }


def _apply_cutlass_conv_static_selection(
    item: dict[str, Any],
    selection: Mapping[str, Any],
    selected_candidate: Mapping[str, Any],
    *,
    strict: bool,
) -> bool:
    conv_plan = validate_cutlass_conv_plan(
        item.get("cutlass_conv_plan"),
        node_id=str(item.get("node_id", "")) if item.get("node_id") is not None else None,
    )
    if not _cutlass_conv_selection_bridge_metadata_matches(selection, conv_plan, strict=strict):
        return False
    if not cutlass_conv_candidate_compatible_with_plan(selected_candidate, conv_plan):
        if strict:
            raise ValueError(
                "Execution plan selected CUTLASS Conv candidate "
                f"{selected_candidate.get('candidate_id')!r} that is incompatible with the manifest transform plan"
            )
        return False
    item["cutlass_conv_plan"] = {
        **conv_plan,
        "selected_candidate": {
            "candidate_id": str(selected_candidate["candidate_id"]),
            "symbol_id": str(selected_candidate.get("symbol_id", "")),
            "kernel_symbol": str(selected_candidate["kernel_symbol"]),
            "profiler_symbol": str(selected_candidate.get("profiler_symbol", "")),
            "opclass": str(selected_candidate.get("cutlass", {}).get("opclass", "")),
            "iterator_algorithm": str(selected_candidate.get("cutlass", {}).get("iterator_algorithm", "")),
            "selection_predicate": dict(selected_candidate.get("selection_predicate", {})),
            "candidate_config_key": str(selected_candidate.get("candidate_config_key", "")),
        },
    }
    return True


def _cutlass_conv_selection_bridge_metadata_matches(
    selection: Mapping[str, Any],
    conv_plan: Mapping[str, Any],
    *,
    strict: bool,
) -> bool:
    expected_source_op = _optional_str(conv_plan.get("source_op"))
    expected_bias_mode = _optional_str(conv_plan.get("bias_mode"))
    selected_source_op = _optional_str(selection.get("source_op"))
    selected_bias_mode = _optional_str(selection.get("bias_mode"))
    if (expected_source_op, expected_bias_mode) == (selected_source_op, selected_bias_mode):
        return True
    if strict:
        raise ValueError(
            "CUTLASS Conv execution plan bridge metadata mismatch: "
            f"manifest source_op={expected_source_op!r}, bias_mode={expected_bias_mode!r}; "
            f"selection source_op={selected_source_op!r}, bias_mode={selected_bias_mode!r}"
        )
    return False


def _optional_str(value: Any) -> str | None:
    if value is None:
        return None
    return str(value)


def _execution_plan_static_selections(execution_plan: Mapping[str, Any]) -> dict[tuple[str, str, str], Mapping[str, Any]]:
    selections: dict[tuple[str, str, str], Mapping[str, Any]] = {}
    for selection in execution_plan.get("static_selections", ()):
        if not isinstance(selection, Mapping):
            continue
        key = (
            str(selection.get("op", "")),
            str(selection.get("dtype", "")),
            str(selection.get("candidate_set_key", "")),
        )
        if all(key):
            selections[key] = selection
    return selections


def _execution_plan_guarded_selections(execution_plan: Mapping[str, Any]) -> dict[tuple[str, str, str], list[Mapping[str, Any]]]:
    selections: dict[tuple[str, str, str], list[Mapping[str, Any]]] = {}
    for selection in execution_plan.get("selections", ()):
        if not isinstance(selection, Mapping):
            continue
        key = (
            str(selection.get("op", "")),
            str(selection.get("dtype", "")),
            str(selection.get("candidate_set_key", "")),
        )
        if all(key):
            selections.setdefault(key, []).append(selection)
    return selections


def _execution_plan_conflict_keys(execution_plan: Mapping[str, Any]) -> set[tuple[str, str, str]]:
    keys = set()
    for conflict in execution_plan.get("conflicts", ()):
        if not isinstance(conflict, Mapping):
            continue
        key = (
            str(conflict.get("op", "")),
            str(conflict.get("dtype", "")),
            str(conflict.get("candidate_set_key", "")),
        )
        if all(key):
            keys.add(key)
    return keys


def _candidate_by_id(item: Mapping[str, Any], candidate_id: str) -> Mapping[str, Any] | None:
    for candidate in item.get("candidates", []):
        if isinstance(candidate, Mapping) and str(candidate.get("candidate_id")) == candidate_id:
            return candidate
    return None


def _attach_cutlass_alignment_fallbacks(item: dict[str, Any]) -> None:
    selected_candidate = _candidate_by_id(item, str(item.get("selected_candidate_id", "")))
    if selected_candidate is None:
        item.pop("alignment_fallbacks", None)
        return
    selected_alignment = cutlass_candidate_alignment(selected_candidate)
    if selected_alignment <= 1:
        item.pop("alignment_fallbacks", None)
        return
    candidate_filter = alignment_context_candidate_filter(item.get("cutlass_alignment"))
    fallback_by_alignment: dict[int, Mapping[str, Any]] = {}
    for candidate in item.get("candidates", []):
        if not isinstance(candidate, Mapping):
            continue
        alignment = cutlass_candidate_alignment(candidate)
        if alignment >= selected_alignment:
            continue
        max_operand_alignment = candidate_filter["max_operand_alignment"]
        if max_operand_alignment is not None and alignment > max_operand_alignment:
            continue
        max_epilogue_alignment = candidate_filter["max_epilogue_alignment"]
        if (
            max_epilogue_alignment is not None
            and cutlass_candidate_epilogue_alignment(candidate) > max_epilogue_alignment
        ):
            continue
        fallback_by_alignment.setdefault(alignment, candidate)
    fallbacks = [
        _alignment_fallback_payload(fallback_by_alignment[alignment])
        for alignment in sorted(fallback_by_alignment, reverse=True)
    ]
    if fallbacks:
        item["alignment_fallbacks"] = fallbacks
    else:
        item.pop("alignment_fallbacks", None)


def _alignment_fallback_payload(candidate: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "candidate_id": str(candidate["candidate_id"]),
        "candidate_config_key": candidate.get("candidate_config_key"),
        "kernel_symbol": str(candidate["kernel_symbol"]),
        "profiler_symbol": str(candidate["profiler_symbol"]),
        "cutlass_alignment": cutlass_candidate_alignment(candidate),
        "split_k": 1,
        "workspace_nbytes": 0,
        "source": "runtime_pointer_alignment_fallback",
    }


def _with_kernel_manifest_cache_keys(manifest: Mapping[str, Any]) -> dict[str, Any]:
    result = deepcopy(dict(manifest))
    result.pop("cache_key", None)
    result.pop("support_cache_key", None)
    result["cache_key"] = hashlib.sha256(canonical_json(result).encode("utf-8")).hexdigest()
    support_manifest = dict(result)
    support_manifest.pop("cache_key", None)
    support_manifest.pop("session_resources", None)
    support_manifest["required_kernels"] = [
        item for item in result.get("required_kernels", []) if item["kernel_library"] != "model"
    ]
    result["support_cache_key"] = hashlib.sha256(canonical_json(support_manifest).encode("utf-8")).hexdigest()
    return result


def build_support_manifest(
    *,
    target: Mapping[str, Any],
    libraries: Mapping[str, Any],
    required_kernel_cache_key: str | None = None,
) -> dict[str, Any]:
    manifest = {
        "schema_version": KERNEL_MANIFEST_SCHEMA_VERSION,
        "kernel_abi_version": KERNEL_ABI_VERSION,
        "profile_cache_schema_version": PROFILE_CACHE_SCHEMA_VERSION,
        "target": dict(target),
        "libraries": dict(libraries),
        "required_kernel_cache_key": required_kernel_cache_key,
        "codegen_strategy": "shared_support_library_cache",
    }
    manifest["cache_key"] = hashlib.sha256(canonical_json(manifest).encode("utf-8")).hexdigest()
    return manifest


def build_external_kernel_plan(target: Mapping[str, Any]) -> dict[str, Any]:
    target_name = target["name"]
    families = [family.to_json() for family in external_kernel_families(backend=target_name)]
    plan = {
        "schema_version": KERNEL_MANIFEST_SCHEMA_VERSION,
        "kernel_abi_version": KERNEL_ABI_VERSION,
        "profile_cache_schema_version": PROFILE_CACHE_SCHEMA_VERSION,
        "target": dict(target),
        "families": families,
        "codegen_strategy": "external_library_candidates",
        "profiler_strategy": "generate_used_candidates_once_then_cache_results",
    }
    plan["cache_key"] = hashlib.sha256(canonical_json(plan).encode("utf-8")).hexdigest()
    return plan
