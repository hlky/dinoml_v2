from __future__ import annotations

import hashlib
from copy import deepcopy
from typing import Any, Mapping, Sequence

from dinoml.ir import canonical_json
from dinoml.kernels.external import external_kernel_families
from dinoml.kernels.bmm import bmm_op_spec
from dinoml.kernels.providers.cutlass.bmm import cutlass_bmm_candidate_set, cutlass_bmm_candidates
from dinoml.kernels.providers.cutlass.gemm import cutlass_gemm_candidate_set, cutlass_gemm_candidates
from dinoml.kernels.providers.cutlass.alignment import (
    alignment_context_candidate_filter,
    cutlass_bmm_static_alignment_context,
    cutlass_candidate_alignment,
    cutlass_candidate_epilogue_alignment,
    cutlass_gemm_static_alignment_context,
    merge_cutlass_alignment_contexts,
    filter_candidates_by_alignment,
)
from dinoml.kernels.gemm import gemm_op_spec
from dinoml.ops.definitions import get_op_def


KERNEL_MANIFEST_SCHEMA_VERSION = 4
KERNEL_ABI_VERSION = 1
PROFILE_CACHE_SCHEMA_VERSION = 7


def build_kernel_manifest(ir: Mapping[str, Any], target: Mapping[str, Any]) -> dict[str, Any]:
    target_name = target["name"]
    required = []
    seen = set()
    tensor_map = {tensor["name"]: tensor for tensor in ir["tensors"]}
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
        key = (node["op"], kernel_symbol)
        if key in seen:
            continue
        seen.add(key)
        item = {
            "op": node["op"],
            "kernel_symbol": kernel_symbol,
            "kernel_library": resolved.library,
            "profiler_symbol": profiler_symbol,
            "has_profiler": op_def.profiler,
        }
        if candidates:
            item["selected_candidate_id"] = selected_candidate_id
            item["candidates"] = candidates
        if candidate_set:
            item["candidate_set_id"] = candidate_set["candidate_set_id"]
            item["candidate_set_key"] = candidate_set["candidate_set_key"]
            item["candidate_set"] = candidate_set
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
    return _with_kernel_manifest_cache_keys(manifest)


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
        if kernel_library not in {"cutlass_gemm", "cutlass_bmm"}:
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
        if dispatch_group and (key in conflict_keys or key not in selections):
            dispatch_entries = []
            for guarded in dispatch_group:
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
            if manifest_kernel_libraries.get(key, "cutlass_gemm") in {"cutlass_gemm", "cutlass_bmm"}
        )
        if missing_guarded:
            missing_text = ", ".join(f"{op}/{dtype}/{candidate_set_key}" for op, dtype, candidate_set_key in missing_guarded)
            raise ValueError(f"Execution plan guarded selections did not match the kernel manifest: {missing_text}")
    return _with_kernel_manifest_cache_keys(manifest)


def _execution_plan_selection_supported(
    kernel_library: str,
    key: tuple[str, str, str],
    selection: Mapping[str, Any],
    *,
    strict: bool,
) -> bool:
    if kernel_library != "cutlass_bmm":
        return True
    split_k = int(selection.get("split_k", 1) or 1)
    workspace_nbytes = int(selection.get("workspace_nbytes", 0) or 0)
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
    alignment_cap = item.get("cutlass_alignment_cap")
    if check_alignment_cap and alignment_cap is not None and cutlass_candidate_alignment(selected_candidate) > int(alignment_cap):
        if strict:
            raise ValueError(
                "Execution plan selected CUTLASS candidate "
                f"{selected_id!r} for {key[0]} {key[1]} exceeds alignment cap {int(alignment_cap)}"
            )
        return None
    return selected_candidate


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
        "selected_candidate_id": selection.get("selected_candidate_id"),
        "candidate_config_key": selection.get("candidate_config_key") or selected_candidate.get("candidate_config_key"),
        "kernel_symbol": str(selection.get("kernel_symbol") or selected_candidate["kernel_symbol"]),
        "profiler_symbol": str(selection.get("profiler_symbol") or selected_candidate["profiler_symbol"]),
        "shape": dict(selection.get("shape", {})) if isinstance(selection.get("shape"), Mapping) else {},
        "avg_ms": selection.get("avg_ms"),
        "confidence": dict(selection.get("confidence", {})) if isinstance(selection.get("confidence"), Mapping) else {},
        "split_k": int(selection.get("split_k", 1) or 1),
        "workspace_nbytes": int(selection.get("workspace_nbytes", 0) or 0),
    }


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
    support_manifest["required_kernels"] = [
        item for item in result.get("required_kernels", []) if item["kernel_library"] != "model"
    ]
    result["support_cache_key"] = hashlib.sha256(canonical_json(support_manifest).encode("utf-8")).hexdigest()
    return result


def build_support_manifest(
    *,
    target: Mapping[str, Any],
    libraries: Mapping[str, str],
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
