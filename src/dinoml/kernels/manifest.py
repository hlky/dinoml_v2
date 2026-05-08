from __future__ import annotations

import hashlib
from typing import Any, Mapping

from dinoml.ir import canonical_json
from dinoml.kernels.external import external_kernel_families
from dinoml.ops.definitions import get_op_def


KERNEL_MANIFEST_SCHEMA_VERSION = 3
KERNEL_ABI_VERSION = 1
PROFILE_CACHE_SCHEMA_VERSION = 5


def build_kernel_manifest(ir: Mapping[str, Any], target: Mapping[str, str]) -> dict[str, Any]:
    target_name = target["name"]
    required = []
    seen = set()
    tensor_map = {tensor["name"]: tensor for tensor in ir["tensors"]}
    for node in ir["nodes"]:
        op_def = get_op_def(node["op"])
        binding = op_def.backend_kernels[target_name]
        output_name = str(node["outputs"][0])
        resolved = binding.resolve(str(tensor_map[output_name]["dtype"]))
        key = (node["op"], resolved.symbol)
        if key in seen:
            continue
        seen.add(key)
        item = {
            "op": node["op"],
            "kernel_symbol": resolved.symbol,
            "kernel_library": resolved.library,
            "profiler_symbol": resolved.profiler_symbol,
            "has_profiler": op_def.profiler,
        }
        if resolved.candidates:
            candidates = [dict(candidate) for candidate in resolved.candidates]
            item["selected_candidate_id"] = candidates[0]["candidate_id"]
            item["candidates"] = candidates
        if resolved.candidate_set:
            candidate_set = dict(resolved.candidate_set)
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
    manifest["cache_key"] = hashlib.sha256(canonical_json(manifest).encode("utf-8")).hexdigest()
    support_manifest = dict(manifest)
    support_manifest["required_kernels"] = [
        item for item in required if item["kernel_library"] != "model"
    ]
    manifest["support_cache_key"] = hashlib.sha256(canonical_json(support_manifest).encode("utf-8")).hexdigest()
    return manifest


def build_support_manifest(
    *,
    target: Mapping[str, str],
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


def build_external_kernel_plan(target: Mapping[str, str]) -> dict[str, Any]:
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
