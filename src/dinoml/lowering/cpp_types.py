from __future__ import annotations

from dinoml.lowering.target_specs import storage_type


def cpu_storage_type(dtype: str) -> str:
    return storage_type(dtype, "cpu")


def cuda_storage_type(dtype: str) -> str:
    return storage_type(dtype, "cuda")


def rocm_storage_type(dtype: str) -> str:
    return storage_type(dtype, "rocm")
