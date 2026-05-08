from __future__ import annotations

import hashlib
from typing import Any, Mapping

from dinoml.ir import canonical_json
from dinoml.ops.definitions import get_op_def


KERNEL_MANIFEST_SCHEMA_VERSION = 1
KERNEL_ABI_VERSION = 1
PROFILE_CACHE_SCHEMA_VERSION = 1


def build_kernel_manifest(ir: Mapping[str, Any], target: Mapping[str, str]) -> dict[str, Any]:
    target_name = target["name"]
    required = []
    seen = set()
    for node in ir["nodes"]:
        op_def = get_op_def(node["op"])
        binding = op_def.backend_kernels[target_name]
        key = (node["op"], binding.symbol)
        if key in seen:
            continue
        seen.add(key)
        required.append(
            {
                "op": node["op"],
                "kernel_symbol": binding.symbol,
                "kernel_library": binding.library,
                "profiler_symbol": binding.profiler_symbol,
                "has_profiler": op_def.profiler,
            }
        )
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
