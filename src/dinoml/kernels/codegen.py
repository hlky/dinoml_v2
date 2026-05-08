from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping


@dataclass(frozen=True)
class KernelCodegenPlan:
    target: Mapping[str, str]
    cache_key: str
    support_cache_dir: Path
    kernel_symbols: tuple[str, ...]
    profiler_symbols: tuple[str, ...]

    def to_json(self) -> dict[str, Any]:
        return {
            "target": dict(self.target),
            "cache_key": self.cache_key,
            "support_cache_dir": str(self.support_cache_dir),
            "kernel_symbols": list(self.kernel_symbols),
            "profiler_symbols": list(self.profiler_symbols),
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
    return KernelCodegenPlan(
        target=target,
        cache_key=kernel_manifest["cache_key"],
        support_cache_dir=Path(cache_root) / "support" / target_dir / kernel_manifest.get("support_cache_key", kernel_manifest["cache_key"])[:16],
        kernel_symbols=kernel_symbols,
        profiler_symbols=profiler_symbols,
    )
