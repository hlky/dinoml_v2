from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from dinoml.kernels.providers.cutlass.gemm import cutlass_gemm_used_candidate_plan


@dataclass(frozen=True)
class KernelCodegenPlan:
    target: Mapping[str, str]
    cache_key: str
    support_cache_dir: Path
    kernel_symbols: tuple[str, ...]
    profiler_symbols: tuple[str, ...]
    candidate_profiler_symbols: tuple[str, ...] = ()
    external_support_libraries: tuple[Mapping[str, Any], ...] = ()

    def to_json(self) -> dict[str, Any]:
        return {
            "target": dict(self.target),
            "cache_key": self.cache_key,
            "support_cache_dir": str(self.support_cache_dir),
            "kernel_symbols": list(self.kernel_symbols),
            "profiler_symbols": list(self.profiler_symbols),
            "candidate_profiler_symbols": list(self.candidate_profiler_symbols),
            "external_support_libraries": [dict(item) for item in self.external_support_libraries],
        }


def create_codegen_plan(kernel_manifest: Mapping[str, Any], cache_root: str | Path) -> KernelCodegenPlan:
    target = dict(kernel_manifest["target"])
    target_name = target["name"]
    arch = target.get("arch", "native").replace("sm_", "")
    target_dir = f"{target_name}-{arch}" if target_name == "cuda" else target_name
    kernel_symbols = tuple(item["kernel_symbol"] for item in kernel_manifest["required_kernels"])
    profiler_symbols = tuple(
        item["profiler_symbol"]
        for item in kernel_manifest["required_kernels"]
        if item.get("profiler_symbol")
    )
    candidate_profiler_symbols = _candidate_profiler_symbols(kernel_manifest)
    support_key = kernel_manifest.get("support_cache_key", kernel_manifest["cache_key"])[:16]
    external_support_libraries = _external_support_libraries(kernel_manifest, Path(cache_root), target_dir, support_key)
    return KernelCodegenPlan(
        target=target,
        cache_key=kernel_manifest["cache_key"],
        support_cache_dir=Path(cache_root) / "support" / target_dir / support_key,
        kernel_symbols=kernel_symbols,
        profiler_symbols=profiler_symbols,
        candidate_profiler_symbols=candidate_profiler_symbols,
        external_support_libraries=external_support_libraries,
    )


def _candidate_profiler_symbols(kernel_manifest: Mapping[str, Any]) -> tuple[str, ...]:
    seen = set()
    symbols = []
    for item in kernel_manifest["required_kernels"]:
        for candidate in item.get("candidates", []):
            symbol = candidate.get("profiler_symbol")
            if symbol and symbol not in seen:
                seen.add(symbol)
                symbols.append(str(symbol))
    return tuple(symbols)


def _external_support_libraries(
    kernel_manifest: Mapping[str, Any],
    cache_root: Path,
    target_dir: str,
    support_key: str,
) -> tuple[Mapping[str, str], ...]:
    libraries = sorted({item["kernel_library"] for item in kernel_manifest["required_kernels"] if item["kernel_library"] not in {"model"}})
    result = []
    for library in libraries:
        if library == "cutlass_gemm":
            cache_dir = cache_root / "support" / target_dir / "cutlass-gemm" / support_key
            used_candidate_plan = cutlass_gemm_used_candidate_plan(kernel_manifest)
            result.append(
                {
                    "name": library,
                    "cache_dir": str(cache_dir),
                    "library": "lib/libdinoml_cutlass_gemm.so",
                    "used_candidate_plan_key": used_candidate_plan["used_candidate_plan_key"],
                    "candidate_set_keys": list(used_candidate_plan["candidate_set_keys"]),
                    "candidate_config_keys": list(used_candidate_plan["candidate_config_keys"]),
                    "kernel_symbols": list(used_candidate_plan["kernel_symbols"]),
                    "profiler_symbols": list(used_candidate_plan["profiler_symbols"]),
                }
            )
    return tuple(result)
