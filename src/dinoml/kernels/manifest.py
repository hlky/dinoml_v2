from __future__ import annotations

import hashlib
from copy import deepcopy
from typing import Any, Mapping

from dinoml.ir import canonical_json
from dinoml.kernels.external import external_kernel_families
from dinoml.kernels.providers.cutlass.gemm import cutlass_gemm_candidate_set, cutlass_gemm_candidates
from dinoml.ops.definitions import get_op_def


KERNEL_MANIFEST_SCHEMA_VERSION = 3
KERNEL_ABI_VERSION = 1
PROFILE_CACHE_SCHEMA_VERSION = 6


def build_kernel_manifest(ir: Mapping[str, Any], target: Mapping[str, Any]) -> dict[str, Any]:
    target_name = target["name"]
    required = []
    seen = set()
    tensor_map = {tensor["name"]: tensor for tensor in ir["tensors"]}
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
        if resolved.library == "cutlass_gemm":
            candidates = [dict(candidate) for candidate in cutlass_gemm_candidates(str(node["op"]), dtype, target=target)]
            candidate_set = cutlass_gemm_candidate_set(str(node["op"]), dtype, target=target)
            kernel_symbol = str(candidates[0]["kernel_symbol"])
            profiler_symbol = str(candidates[0]["profiler_symbol"])
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
            item["selected_candidate_id"] = candidates[0]["candidate_id"]
            item["candidates"] = candidates
        if candidate_set:
            item["candidate_set_id"] = candidate_set["candidate_set_id"]
            item["candidate_set_key"] = candidate_set["candidate_set_key"]
            item["candidate_set"] = candidate_set
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


def apply_execution_plan(
    kernel_manifest: Mapping[str, Any],
    execution_plan: Mapping[str, Any],
    *,
    strict: bool = False,
) -> dict[str, Any]:
    manifest = deepcopy(dict(kernel_manifest))
    selections = _execution_plan_static_selections(execution_plan)
    if not selections:
        return _with_kernel_manifest_cache_keys(manifest)
    applied_keys: set[tuple[str, str, str]] = set()
    for item in manifest.get("required_kernels", []):
        if item.get("kernel_library") != "cutlass_gemm":
            continue
        candidate_set = item.get("candidate_set", {})
        dtype = str(candidate_set.get("dtype", "")) if isinstance(candidate_set, Mapping) else ""
        key = (str(item.get("op", "")), dtype, str(item.get("candidate_set_key", "")))
        selection = selections.get(key)
        if selection is None:
            continue
        applied_keys.add(key)
        selected_id = str(selection.get("selected_candidate_id", ""))
        selected_candidate = _candidate_by_id(item, selected_id)
        if selected_candidate is None:
            if strict:
                raise ValueError(
                    "Execution plan selected unknown CUTLASS candidate "
                    f"{selected_id!r} for {key[0]} {key[1]} candidate set {key[2]}"
                )
            continue
        item["selected_candidate_id"] = selected_id
        item["kernel_symbol"] = str(selection.get("kernel_symbol") or selected_candidate["kernel_symbol"])
        item["profiler_symbol"] = str(selection.get("profiler_symbol") or selected_candidate["profiler_symbol"])
        item["execution_plan_selection"] = {
            "schema_version": int(execution_plan.get("schema_version", 1)),
            "selection_key": selection.get("selection_key"),
            "candidate_set_key": key[2],
            "candidate_config_key": selection.get("candidate_config_key") or selected_candidate.get("candidate_config_key"),
            "shape": dict(selection.get("shape", {})) if isinstance(selection.get("shape"), Mapping) else {},
            "avg_ms": selection.get("avg_ms"),
            "split_k": int(selection.get("split_k", 1) or 1),
            "workspace_nbytes": int(selection.get("workspace_nbytes", 0) or 0),
        }
    if strict:
        missing = sorted(set(selections) - applied_keys)
        if missing:
            missing_text = ", ".join(f"{op}/{dtype}/{candidate_set_key}" for op, dtype, candidate_set_key in missing)
            raise ValueError(f"Execution plan selections did not match the kernel manifest: {missing_text}")
    return _with_kernel_manifest_cache_keys(manifest)


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


def _candidate_by_id(item: Mapping[str, Any], candidate_id: str) -> Mapping[str, Any] | None:
    for candidate in item.get("candidates", []):
        if isinstance(candidate, Mapping) and str(candidate.get("candidate_id")) == candidate_id:
            return candidate
    return None


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
