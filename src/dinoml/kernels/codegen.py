from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from dinoml.kernels.providers.cutlass.bmm import cutlass_bmm_cmake_target, cutlass_bmm_static_library_name, cutlass_bmm_used_candidate_plan
from dinoml.kernels.providers.cutlass.conv import (
    cutlass_conv_cmake_target,
    cutlass_conv_static_library_name,
    cutlass_conv_used_candidate_plan,
    cutlass_conv_wrapper_stages,
)
from dinoml.kernels.providers.cutlass.gemm import cutlass_gemm_static_library_name, cutlass_gemm_used_candidate_plan
from dinoml.kernels.providers.ck.gemm import (
    ck_gemm_cmake_target,
    ck_gemm_profiler_bind_target,
    ck_gemm_profiler_executable_target,
    ck_gemm_profiler_stem,
    ck_gemm_static_library_name,
    ck_gemm_used_candidate_plan,
)
from dinoml.kernels.providers.ck.bmm import (
    ck_bmm_cmake_target,
    ck_bmm_profiler_bind_target,
    ck_bmm_profiler_executable_target,
    ck_bmm_profiler_stem,
    ck_bmm_static_library_name,
    ck_bmm_used_candidate_plan,
)
from dinoml.kernels.providers.ck.conv import (
    ck_conv_cmake_target,
    ck_conv_profiler_bind_target,
    ck_conv_profiler_executable_target,
    ck_conv_profiler_stem,
    ck_conv_static_library_name,
    ck_conv_used_candidate_plan,
)
from dinoml.libgguf_cuda import (
    libgguf_provenance_key,
    libgguf_source_provenance,
)


@dataclass(frozen=True)
class KernelCodegenPlan:
    target: Mapping[str, Any]
    cache_key: str
    support_cache_dir: Path
    kernel_symbols: tuple[str, ...]
    profiler_symbols: tuple[str, ...]
    candidate_profiler_symbols: tuple[str, ...] = ()
    generated_sources: tuple[Mapping[str, Any], ...] = ()
    external_support_libraries: tuple[Mapping[str, Any], ...] = ()
    wrapper_stages: tuple[Mapping[str, Any], ...] = ()

    def to_json(self) -> dict[str, Any]:
        return {
            "target": dict(self.target),
            "cache_key": self.cache_key,
            "support_cache_dir": str(self.support_cache_dir),
            "kernel_symbols": list(self.kernel_symbols),
            "profiler_symbols": list(self.profiler_symbols),
            "candidate_profiler_symbols": list(self.candidate_profiler_symbols),
            "generated_sources": [dict(item) for item in self.generated_sources],
            "external_support_libraries": [dict(item) for item in self.external_support_libraries],
            "wrapper_stages": [dict(item) for item in self.wrapper_stages],
        }


def create_codegen_plan(kernel_manifest: Mapping[str, Any], cache_root: str | Path) -> KernelCodegenPlan:
    target = dict(kernel_manifest["target"])
    target_name = target["name"]
    target_dir = _target_cache_dir(target)
    kernel_symbols = tuple(item["kernel_symbol"] for item in kernel_manifest["required_kernels"])
    profiler_symbols = tuple(
        item["profiler_symbol"]
        for item in kernel_manifest["required_kernels"]
        if item.get("profiler_symbol")
    )
    candidate_profiler_symbols = _candidate_profiler_symbols(kernel_manifest)
    generated_sources = _generated_sources(kernel_manifest)
    support_key = kernel_manifest.get("support_cache_key", kernel_manifest["cache_key"])[:16]
    external_support_libraries = _external_support_libraries(kernel_manifest, Path(cache_root), target_dir, support_key)
    wrapper_stages = _wrapper_stages(kernel_manifest)
    return KernelCodegenPlan(
        target=target,
        cache_key=kernel_manifest["cache_key"],
        support_cache_dir=Path(cache_root) / "support" / target_dir / support_key,
        kernel_symbols=kernel_symbols,
        profiler_symbols=profiler_symbols,
        candidate_profiler_symbols=candidate_profiler_symbols,
        generated_sources=generated_sources,
        external_support_libraries=external_support_libraries,
        wrapper_stages=wrapper_stages,
    )


def _target_cache_dir(target: Mapping[str, Any]) -> str:
    target_name = str(target["name"])
    if target_name not in {"cuda", "rocm"}:
        return target_name
    arch = str(target.get("arch", "native")).replace("sm_", "")
    return f"{target_name}-{_cache_path_segment(arch)}"


def _cache_path_segment(value: str) -> str:
    segment = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value).strip())
    return segment or "native"


def _candidate_profiler_symbols(kernel_manifest: Mapping[str, Any]) -> tuple[str, ...]:
    seen = set()
    symbols = []

    def append_symbol(symbol: Any) -> None:
        if not symbol:
            return
        value = str(symbol)
        if value in seen:
            return
        seen.add(value)
        symbols.append(value)

    for item in kernel_manifest["required_kernels"]:
        execution_plan_selection = item.get("execution_plan_selection")
        if isinstance(execution_plan_selection, Mapping):
            append_symbol(execution_plan_selection.get("profiler_symbol") or item.get("profiler_symbol"))
            continue
        dispatches = [
            selection
            for selection in item.get("execution_plan_dispatch", [])
            if isinstance(selection, Mapping)
        ]
        if dispatches:
            for selection in dispatches:
                append_symbol(selection.get("profiler_symbol") or item.get("profiler_symbol"))
            selected_candidate = item.get("selected_candidate_id")
            for candidate in item.get("candidates", []):
                if (
                    isinstance(candidate, Mapping)
                    and selected_candidate is not None
                    and candidate.get("candidate_id") == selected_candidate
                ):
                    append_symbol(candidate.get("profiler_symbol") or item.get("profiler_symbol"))
                    break
            continue
        for candidate in item.get("candidates", []):
            append_symbol(candidate.get("profiler_symbol"))
    return tuple(symbols)


def _generated_sources(kernel_manifest: Mapping[str, Any]) -> tuple[Mapping[str, Any], ...]:
    sources = []
    for item in kernel_manifest["required_kernels"]:
        generated_source = item.get("generated_source")
        if not isinstance(generated_source, Mapping):
            continue
        entry = {
            "op": str(item["op"]),
            "kernel_symbol": str(item["kernel_symbol"]),
            **{str(key): value for key, value in generated_source.items()},
        }
        sources.append(entry)
    return tuple(sources)


def _external_support_libraries(
    kernel_manifest: Mapping[str, Any],
    cache_root: Path,
    target_dir: str,
    support_key: str,
) -> tuple[Mapping[str, Any], ...]:
    libraries = sorted({item["kernel_library"] for item in kernel_manifest["required_kernels"] if item["kernel_library"] not in {"model"}})
    result = []
    for library in libraries:
        if library == "cutlass_gemm":
            cache_dir = cache_root / "support" / target_dir / "cutlass-gemm" / "cmake-full"
            used_candidate_plan = cutlass_gemm_used_candidate_plan(kernel_manifest)
            modules = _cutlass_gemm_modules(kernel_manifest)
            result.append(
                {
                    "name": library,
                    "cache_dir": str(cache_dir),
                    "modules": modules,
                    "build_mode": "cmake_op_dtype_static_archives",
                    "used_candidate_plan_key": used_candidate_plan["used_candidate_plan_key"],
                    "candidate_set_keys": list(used_candidate_plan["candidate_set_keys"]),
                    "candidate_config_keys": list(used_candidate_plan["candidate_config_keys"]),
                    "kernel_symbols": list(used_candidate_plan["kernel_symbols"]),
                    "profiler_symbols": list(used_candidate_plan["profiler_symbols"]),
                    "entries": [dict(entry) for entry in used_candidate_plan.get("entries", [])],
                }
            )
        elif library == "cutlass_bmm":
            cache_dir = cache_root / "support" / target_dir / "cutlass-bmm" / "cmake-full"
            used_candidate_plan = cutlass_bmm_used_candidate_plan(kernel_manifest)
            modules = _cutlass_bmm_modules(kernel_manifest)
            result.append(
                {
                    "name": library,
                    "cache_dir": str(cache_dir),
                    "modules": modules,
                    "build_mode": "cmake_op_dtype_static_archives",
                    "used_candidate_plan_key": used_candidate_plan["used_candidate_plan_key"],
                    "candidate_set_keys": list(used_candidate_plan["candidate_set_keys"]),
                    "candidate_config_keys": list(used_candidate_plan["candidate_config_keys"]),
                    "kernel_symbols": list(used_candidate_plan["kernel_symbols"]),
                    "profiler_symbols": list(used_candidate_plan["profiler_symbols"]),
                    "entries": [dict(entry) for entry in used_candidate_plan.get("entries", [])],
                }
            )
        elif library == "cutlass_conv":
            cache_dir = cache_root / "support" / target_dir / "cutlass-conv" / "cmake-full"
            used_candidate_plan = cutlass_conv_used_candidate_plan(kernel_manifest)
            modules = _cutlass_conv_modules(kernel_manifest)
            result.append(
                {
                    "name": library,
                    "cache_dir": str(cache_dir),
                    "modules": modules,
                    "build_mode": "cmake_op_dtype_static_archives",
                    "used_candidate_plan_key": used_candidate_plan["used_candidate_plan_key"],
                    "candidate_set_keys": list(used_candidate_plan["candidate_set_keys"]),
                    "candidate_config_keys": list(used_candidate_plan["candidate_config_keys"]),
                    "kernel_symbols": list(used_candidate_plan["kernel_symbols"]),
                    "profiler_symbols": list(used_candidate_plan["profiler_symbols"]),
                    "entries": [dict(entry) for entry in used_candidate_plan.get("entries", [])],
                    "transform_helper_symbols": list(used_candidate_plan.get("transform_helper_symbols", [])),
                }
            )
        elif library == "ck_gemm":
            cache_dir = cache_root / "support" / target_dir / "ck-gemm" / "cmake-full"
            used_candidate_plan = ck_gemm_used_candidate_plan(kernel_manifest)
            modules = _ck_gemm_modules(kernel_manifest)
            result.append(
                {
                    "name": library,
                    "cache_dir": str(cache_dir),
                    "modules": modules,
                    "build_mode": "cmake_op_dtype_static_archives",
                    "used_candidate_plan_key": used_candidate_plan["used_candidate_plan_key"],
                    "candidate_set_keys": list(used_candidate_plan["candidate_set_keys"]),
                    "candidate_config_keys": list(used_candidate_plan["candidate_config_keys"]),
                    "kernel_symbols": list(used_candidate_plan["kernel_symbols"]),
                    "profiler_symbols": list(used_candidate_plan["profiler_symbols"]),
                    "pruned_by_execution_plan": bool(used_candidate_plan.get("pruned_by_execution_plan", False)),
                    "entries": [dict(entry) for entry in used_candidate_plan.get("entries", [])],
                }
            )
        elif library == "ck_bmm":
            cache_dir = cache_root / "support" / target_dir / "ck-bmm" / "cmake-full"
            used_candidate_plan = ck_bmm_used_candidate_plan(kernel_manifest)
            modules = _ck_bmm_modules(kernel_manifest)
            result.append(
                {
                    "name": library,
                    "cache_dir": str(cache_dir),
                    "modules": modules,
                    "build_mode": "cmake_op_dtype_static_archives",
                    "used_candidate_plan_key": used_candidate_plan["used_candidate_plan_key"],
                    "candidate_set_keys": list(used_candidate_plan["candidate_set_keys"]),
                    "candidate_config_keys": list(used_candidate_plan["candidate_config_keys"]),
                    "kernel_symbols": list(used_candidate_plan["kernel_symbols"]),
                    "profiler_symbols": list(used_candidate_plan["profiler_symbols"]),
                    "pruned_by_execution_plan": bool(used_candidate_plan.get("pruned_by_execution_plan", False)),
                    "entries": [dict(entry) for entry in used_candidate_plan.get("entries", [])],
                }
            )
        elif library == "ck_conv":
            cache_dir = cache_root / "support" / target_dir / "ck-conv" / "cmake-full"
            used_candidate_plan = ck_conv_used_candidate_plan(kernel_manifest)
            modules = _ck_conv_modules(kernel_manifest)
            result.append(
                {
                    "name": library,
                    "cache_dir": str(cache_dir),
                    "modules": modules,
                    "build_mode": "cmake_op_dtype_static_archives",
                    "used_candidate_plan_key": used_candidate_plan["used_candidate_plan_key"],
                    "candidate_set_keys": list(used_candidate_plan["candidate_set_keys"]),
                    "candidate_config_keys": list(used_candidate_plan["candidate_config_keys"]),
                    "kernel_symbols": list(used_candidate_plan["kernel_symbols"]),
                    "profiler_symbols": list(used_candidate_plan["profiler_symbols"]),
                    "pruned_by_execution_plan": bool(used_candidate_plan.get("pruned_by_execution_plan", False)),
                    "entries": [dict(entry) for entry in used_candidate_plan.get("entries", [])],
                }
            )
    if _requires_gguf_cuda_native_library(kernel_manifest):
        source_root = Path(__file__).resolve().parents[3] / "third_party" / "libgguf"
        if (source_root / "src" / "libgguf" / "libgguf_cuda" / "csrc" / "libgguf_cuda_native.cu").exists():
            source_provenance = libgguf_source_provenance(source_root)
            source_key = libgguf_provenance_key(source_provenance)
            cache_dir = cache_root / "support" / target_dir / "libgguf-cuda-native" / support_key
            result.append(
                {
                    "name": "gguf_cuda_native",
                    "cache_dir": str(cache_dir / source_key[:16]),
                    "library": "lib/libgguf_cuda_native.a",
                    "manifest": "lib/libgguf_cuda_native_manifest.json",
                    "source_root": str(source_root),
                    "source_kind": "vendored_source",
                    "source_provenance_key": source_key,
                    "source_provenance": source_provenance,
                    "symbols": ["libgguf_cuda_dequantize_rows_on_stream"],
                    "link_mode": "direct",
                    "library_kind": "static",
                }
            )
    return tuple(result)


def _cutlass_bmm_modules(kernel_manifest: Mapping[str, Any]) -> list[dict[str, str]]:
    modules = {}
    for item in kernel_manifest.get("required_kernels", []):
        if item.get("kernel_library") != "cutlass_bmm":
            continue
        op_name = str(item["op"])
        dtype = str(item.get("dtype") or item.get("candidate_set", {}).get("dtype"))
        archive = cutlass_bmm_static_library_name(op_name, dtype)
        modules[archive] = {
            "op": op_name,
            "dtype": dtype,
            "archive": f"lib/{archive}",
            "target": cutlass_bmm_cmake_target(op_name, dtype),
        }
    return [modules[key] for key in sorted(modules)]


def _cutlass_conv_modules(kernel_manifest: Mapping[str, Any]) -> list[dict[str, str]]:
    modules = {}
    for item in kernel_manifest.get("required_kernels", []):
        if item.get("kernel_library") != "cutlass_conv":
            continue
        op_name = str(item["op"])
        dtype = str(item.get("dtype") or item.get("candidate_set", {}).get("dtype"))
        archive = cutlass_conv_static_library_name(op_name, dtype)
        modules[archive] = {
            "op": op_name,
            "dtype": dtype,
            "archive": f"lib/{archive}",
            "target": cutlass_conv_cmake_target(op_name, dtype),
        }
    return [modules[key] for key in sorted(modules)]


def _cutlass_gemm_modules(kernel_manifest: Mapping[str, Any]) -> list[dict[str, str]]:
    modules = {}
    for item in kernel_manifest["required_kernels"]:
        if item.get("kernel_library") != "cutlass_gemm":
            continue
        op_name = str(item["op"])
        dtype = str(item.get("dtype") or item.get("candidate_set", {}).get("dtype"))
        archive = str(item.get("support_archive") or cutlass_gemm_static_library_name(op_name, dtype))
        modules[archive] = {
            "op": op_name,
            "dtype": dtype,
            "archive": f"lib/{archive}",
        }
    return [modules[key] for key in sorted(modules)]


def _ck_gemm_modules(kernel_manifest: Mapping[str, Any]) -> list[dict[str, str]]:
    modules = {}
    for item in kernel_manifest["required_kernels"]:
        if item.get("kernel_library") != "ck_gemm":
            continue
        op_name = str(item["op"])
        dtype = str(item.get("dtype") or item.get("candidate_set", {}).get("dtype"))
        archive = str(item.get("support_archive") or ck_gemm_static_library_name(op_name, dtype))
        modules[archive] = {
            "op": op_name,
            "dtype": dtype,
            "archive": f"lib/{archive}",
            "target": ck_gemm_cmake_target(op_name, dtype),
            "profiler_bind_target": ck_gemm_profiler_bind_target(op_name, dtype),
            "profiler_executable_target": ck_gemm_profiler_executable_target(op_name, dtype),
            "profiler_stem": ck_gemm_profiler_stem(op_name, dtype),
        }
    return [modules[key] for key in sorted(modules)]


def _ck_bmm_modules(kernel_manifest: Mapping[str, Any]) -> list[dict[str, str]]:
    modules = {}
    for item in kernel_manifest["required_kernels"]:
        if item.get("kernel_library") != "ck_bmm":
            continue
        op_name = str(item["op"])
        dtype = str(item.get("dtype") or item.get("candidate_set", {}).get("dtype"))
        archive = str(item.get("support_archive") or ck_bmm_static_library_name(op_name, dtype))
        modules[archive] = {
            "op": op_name,
            "dtype": dtype,
            "archive": f"lib/{archive}",
            "target": ck_bmm_cmake_target(op_name, dtype),
            "profiler_bind_target": ck_bmm_profiler_bind_target(op_name, dtype),
            "profiler_executable_target": ck_bmm_profiler_executable_target(op_name, dtype),
            "profiler_stem": ck_bmm_profiler_stem(op_name, dtype),
        }
    return [modules[key] for key in sorted(modules)]


def _ck_conv_modules(kernel_manifest: Mapping[str, Any]) -> list[dict[str, str]]:
    modules = {}
    for item in kernel_manifest["required_kernels"]:
        if item.get("kernel_library") != "ck_conv":
            continue
        op_name = str(item["op"])
        dtype = str(item.get("dtype") or item.get("candidate_set", {}).get("dtype"))
        archive = str(item.get("support_archive") or ck_conv_static_library_name(op_name, dtype))
        modules[archive] = {
            "op": op_name,
            "dtype": dtype,
            "archive": f"lib/{archive}",
            "target": ck_conv_cmake_target(op_name, dtype),
            "profiler_bind_target": ck_conv_profiler_bind_target(op_name, dtype),
            "profiler_executable_target": ck_conv_profiler_executable_target(op_name, dtype),
            "profiler_stem": ck_conv_profiler_stem(op_name, dtype),
        }
    return [modules[key] for key in sorted(modules)]


def _wrapper_stages(kernel_manifest: Mapping[str, Any]) -> tuple[Mapping[str, Any], ...]:
    return tuple(cutlass_conv_wrapper_stages(kernel_manifest))


def _requires_gguf_cuda_native_library(kernel_manifest: Mapping[str, Any]) -> bool:
    for item in kernel_manifest.get("required_kernels", []):
        if not isinstance(item, Mapping):
            continue
        plan = item.get("gguf_runtime_dequant")
        if not isinstance(plan, Mapping):
            continue
        if str(plan.get("status")) == "lowered_runtime_dequant_scratch":
            return True
    return False
