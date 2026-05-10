from __future__ import annotations


def cpu_storage_type(dtype: str) -> str:
    if dtype == "float32":
        return "float"
    if dtype == "float16":
        return "dinoml::math::float16"
    if dtype == "bfloat16":
        return "dinoml::math::bfloat16"
    if dtype == "int32":
        return "int32_t"
    if dtype == "int64":
        return "int64_t"
    if dtype == "bool":
        return "bool"
    raise NotImplementedError(f"CPU lowering does not support dtype {dtype!r}")


def cuda_storage_type(dtype: str) -> str:
    if dtype == "float32":
        return "float"
    if dtype == "float16":
        return "half"
    if dtype == "bfloat16":
        return "__nv_bfloat16"
    if dtype == "int32":
        return "int32_t"
    if dtype == "int64":
        return "int64_t"
    if dtype == "bool":
        return "bool"
    raise NotImplementedError(f"CUDA lowering does not support dtype {dtype!r}")
