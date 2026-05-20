from __future__ import annotations

from dataclasses import dataclass
import os
from importlib import import_module
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping, Protocol


class BackendBuildFunction(Protocol):
    def __call__(
        self,
        ir: Mapping[str, Any],
        *,
        target: Any,
        artifact_dir: Path,
        generated_src_dir: Path,
        kernel_manifest: Mapping[str, Any],
    ) -> Mapping[str, str] | None:
        ...


@dataclass(frozen=True)
class CMakeCapabilities:
    requires_cuda: bool = False
    supports_openmp: bool = False
    supports_cuda_fast_math: bool = False
    support_build_targets: tuple[str, ...] = ()


@dataclass(frozen=True)
class BackendSpec:
    name: str
    default_arch: str
    supported_dtypes: frozenset[str]
    build_function: str
    cmake: CMakeCapabilities
    support_libraries: Mapping[str, str]

    def resolve_build_function(self) -> BackendBuildFunction:
        module_name, function_name = self.build_function.rsplit(".", 1)
        module = import_module(module_name)
        return getattr(module, function_name)


def _shared_library_name(stem: str) -> str:
    if os.name == "nt":
        return f"{stem}.dll"
    if os.uname().sysname == "Darwin":
        return f"lib{stem}.dylib"
    return f"lib{stem}.so"


_BACKENDS: dict[str, BackendSpec] = {
    "cpu": BackendSpec(
        name="cpu",
        default_arch="native",
        supported_dtypes=frozenset({"float16", "float32", "bfloat16", "bool"}),
        build_function="dinoml.backends.cpu.build_cpu_module",
        cmake=CMakeCapabilities(
            supports_openmp=True,
            support_build_targets=("dinoml_runtime", "dinoml_cpu_kernels"),
        ),
        support_libraries=MappingProxyType(
            {
                "runtime_library": f"lib/{_shared_library_name('dinoml_runtime')}",
                "kernel_library": f"lib/{_shared_library_name('dinoml_cpu_kernels')}",
            }
        ),
    ),
    "cuda": BackendSpec(
        name="cuda",
        default_arch="sm_86",
        supported_dtypes=frozenset({"float16", "float32", "bfloat16", "bool"}),
        build_function="dinoml.backends.cuda.build_cuda_module",
        cmake=CMakeCapabilities(
            requires_cuda=True,
            supports_cuda_fast_math=True,
            support_build_targets=("dinoml_runtime", "dinoml_cuda_runtime", "dinoml_cuda_kernels"),
        ),
        support_libraries=MappingProxyType(
            {
                "runtime_library": "lib/libdinoml_runtime.so",
                "cuda_runtime_library": "lib/libdinoml_cuda_runtime.so",
                "kernel_library": "lib/libdinoml_cuda_kernels.so",
            }
        ),
    ),
    "rocm": BackendSpec(
        name="rocm",
        default_arch="gfx1201",
        supported_dtypes=frozenset({"float16", "float32", "bfloat16", "bool"}),
        build_function="dinoml.backends.rocm.build_rocm_module",
        cmake=CMakeCapabilities(
            support_build_targets=("dinoml_runtime", "dinoml_rocm_runtime", "dinoml_rocm_kernels"),
        ),
        support_libraries=MappingProxyType(
            {
                "runtime_library": f"lib/{_shared_library_name('dinoml_runtime')}",
                "rocm_runtime_library": f"lib/{_shared_library_name('dinoml_rocm_runtime')}",
                "kernel_library": f"lib/{_shared_library_name('dinoml_rocm_kernels')}",
            }
        ),
    ),
}


def get_backend_spec(name: str) -> BackendSpec:
    try:
        return _BACKENDS[name]
    except KeyError as exc:
        supported = ", ".join(sorted(_BACKENDS))
        raise ValueError(f"Unsupported DinoML target {name!r}; supported targets: {supported}") from exc


def registered_backend_specs() -> tuple[BackendSpec, ...]:
    return tuple(_BACKENDS[name] for name in sorted(_BACKENDS))


def registered_backend_names() -> tuple[str, ...]:
    return tuple(sorted(_BACKENDS))
