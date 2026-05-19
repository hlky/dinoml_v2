from __future__ import annotations

from typing import Any, Iterable, Mapping


from dinoml.lowering.gpu import render_gpu_module

def render_rocm_module(
    ir: Mapping[str, Any],
    *,
    generated_kernels: Iterable[str] | None = None,
    kernel_manifest: Mapping[str, Any] | None = None,
) -> str:
    return render_gpu_module("rocm", ir, generated_kernels=generated_kernels, kernel_manifest=kernel_manifest)
