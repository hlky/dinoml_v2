from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping


_CPU_STORAGE_TYPES = MappingProxyType(
    {
        "float32": "float",
        "float16": "dinoml::math::float16",
        "bfloat16": "dinoml::math::bfloat16",
        "int32": "int32_t",
        "int64": "int64_t",
        "bool": "bool",
    }
)

_CUDA_STORAGE_TYPES = MappingProxyType(
    {
        "float32": "float",
        "float16": "half",
        "bfloat16": "dinoml::bfloat16",
        "int32": "int32_t",
        "int64": "int64_t",
        "bool": "bool",
    }
)

_ROCM_STORAGE_TYPES = MappingProxyType(
    {
        "float32": "float",
        "float16": "half",
        "bfloat16": "dinoml::bfloat16",
        "int32": "int32_t",
        "int64": "int64_t",
        "bool": "bool",
    }
)


@dataclass(frozen=True)
class LoweringTargetSpec:
    name: str
    source_extension: str
    op_template_flavor: str
    storage_types: Mapping[str, str]
    generated_module_admitted: bool
    stream_type: str | None = None
    stream_expr: str | None = None
    check_macro: str | None = None
    last_error_call: str | None = None
    memset_async: str | None = None
    warp_full_mask: str | None = None

    @property
    def is_gpu(self) -> bool:
        return self.stream_type is not None

    def storage_type(self, dtype: str) -> str:
        try:
            return self.storage_types[dtype]
        except KeyError as exc:
            raise NotImplementedError(f"{self.name.upper()} lowering does not support dtype {dtype!r}") from exc

    def gpu_template_context(self) -> dict[str, str]:
        if not self.is_gpu:
            return {}
        if (
            self.stream_type is None
            or self.check_macro is None
            or self.last_error_call is None
            or self.memset_async is None
            or self.warp_full_mask is None
        ):
            raise ValueError(f"{self.name.upper()} GPU target spec is incomplete")
        return {
            "gpu_stream_type": self.stream_type,
            "gpu_check_macro": self.check_macro,
            "gpu_last_error_call": self.last_error_call,
            "gpu_memset_async": self.memset_async,
            "gpu_warp_full_mask": self.warp_full_mask,
        }


_TARGET_SPECS = MappingProxyType(
    {
        "cpu": LoweringTargetSpec(
            name="cpu",
            source_extension="cpp",
            op_template_flavor="cpu",
            storage_types=_CPU_STORAGE_TYPES,
            generated_module_admitted=True,
        ),
        "cuda": LoweringTargetSpec(
            name="cuda",
            source_extension="cu",
            op_template_flavor="gpu",
            storage_types=_CUDA_STORAGE_TYPES,
            generated_module_admitted=True,
            stream_type="cudaStream_t",
            stream_expr="session->stream",
            check_macro="DINO_CUDA_CHECK",
            last_error_call="cudaGetLastError()",
            memset_async="cudaMemsetAsync",
            warp_full_mask="0xffffffffu",
        ),
        "rocm": LoweringTargetSpec(
            name="rocm",
            source_extension="hip",
            op_template_flavor="gpu",
            storage_types=_ROCM_STORAGE_TYPES,
            generated_module_admitted=True,
            stream_type="hipStream_t",
            stream_expr="session->stream",
            check_macro="DINO_ROCM_CHECK",
            last_error_call="hipGetLastError()",
            memset_async="hipMemsetAsync",
            warp_full_mask="0xffffffffffffffffull",
        ),
    }
)


def lowering_target_spec(target: str) -> LoweringTargetSpec:
    try:
        return _TARGET_SPECS[target]
    except KeyError as exc:
        raise ValueError(f"Unsupported lowering target: {target}") from exc


def admitted_generated_targets() -> tuple[str, ...]:
    return tuple(name for name, spec in _TARGET_SPECS.items() if spec.generated_module_admitted)


def generated_source_extension(target: str) -> str:
    return lowering_target_spec(target).source_extension


def storage_type(dtype: str, target: str) -> str:
    return lowering_target_spec(target).storage_type(dtype)
