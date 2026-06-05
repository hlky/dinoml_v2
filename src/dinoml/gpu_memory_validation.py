from __future__ import annotations

import ctypes
import os
from typing import Any, Mapping, Sequence

from dinoml.shapes import validate_runtime_shape

_HIP_RUNTIME_ENV = "DINOML_HIP_RUNTIME_LIBRARY"
_WINDOWS_HIP_RUNTIME_CANDIDATES = (
    r"C:\Windows\System32\amdhip64_7.dll",
    r"C:\Windows\System32\amdhip64_6.dll",
    r"C:\Windows\System32\amdhip64.dll",
)
_POSIX_HIP_RUNTIME_CANDIDATES = (
    "libamdhip64.so",
)


def load_hip_runtime(path: str | os.PathLike[str] | None = None) -> ctypes.CDLL:
    explicit = None if path is None else str(path)
    if explicit:
        return ctypes.CDLL(explicit)
    env_path = os.environ.get(_HIP_RUNTIME_ENV)
    if env_path:
        return ctypes.CDLL(env_path)
    candidates = _WINDOWS_HIP_RUNTIME_CANDIDATES if os.name == "nt" else _POSIX_HIP_RUNTIME_CANDIDATES
    last_error: Exception | None = None
    for candidate in candidates:
        try:
            return ctypes.CDLL(candidate)
        except OSError as exc:
            last_error = exc
    if last_error is not None:
        raise OSError(
            "Unable to load a HIP runtime library; set "
            f"{_HIP_RUNTIME_ENV} to the amdhip64 runtime path"
        ) from last_error
    raise OSError("Unable to load a HIP runtime library")


def hip_mem_get_info(hip_runtime: ctypes.CDLL) -> tuple[int, int]:
    free = ctypes.c_size_t()
    total = ctypes.c_size_t()
    fn = hip_runtime.hipMemGetInfo
    fn.argtypes = [ctypes.POINTER(ctypes.c_size_t), ctypes.POINTER(ctypes.c_size_t)]
    fn.restype = ctypes.c_int
    err = int(fn(ctypes.byref(free), ctypes.byref(total)))
    if err != 0:
        raise RuntimeError(f"hipMemGetInfo failed with error code {err}")
    return int(free.value), int(total.value)


def hip_device_synchronize(hip_runtime: ctypes.CDLL) -> None:
    fn = hip_runtime.hipDeviceSynchronize
    fn.argtypes = []
    fn.restype = ctypes.c_int
    err = int(fn())
    if err != 0:
        raise RuntimeError(f"hipDeviceSynchronize failed with error code {err}")


def input_dim_assignments(
    input_specs: Sequence[Mapping[str, Any]],
    inputs: Mapping[str, Any],
) -> dict[str, int]:
    assignments: dict[str, int] = {}
    for spec in input_specs:
        name = str(spec["name"])
        if name not in inputs:
            raise ValueError(f"Missing validation input {name!r}")
        actual_shape = tuple(int(dim) for dim in getattr(inputs[name], "shape"))
        validated_shape = validate_runtime_shape(name, actual_shape, spec)
        shape_spec = spec.get("shape_spec", spec["shape"])
        for axis, (actual, dim_spec) in enumerate(zip(validated_shape, shape_spec, strict=True)):
            if not isinstance(dim_spec, Mapping) or dim_spec.get("kind") != "dim":
                continue
            dim_name = str(dim_spec["name"])
            existing = assignments.get(dim_name)
            if existing is not None and existing != int(actual):
                raise ValueError(
                    f"Dynamic dimension {dim_name!r} has conflicting values {existing} and {actual} "
                    f"while reading input {name!r} axis {axis}"
                )
            assignments[dim_name] = int(actual)
    return assignments


def matching_bucket_plan(
    metadata: Mapping[str, Any],
    inputs: Mapping[str, Any],
) -> Mapping[str, Any] | None:
    memory_plan = metadata.get("memory_plan", {})
    bucket_plans = memory_plan.get("bucket_plans", [])
    bucket_dimensions = memory_plan.get("bucket_dimensions", [])
    if not isinstance(bucket_plans, list) or not bucket_plans:
        return None
    assignments = input_dim_assignments(metadata.get("inputs", []), inputs)
    for plan in bucket_plans:
        dim_values = plan.get("dim_values", {})
        if not isinstance(dim_values, Mapping):
            continue
        if all(int(assignments.get(str(dim["name"]), -1)) == int(dim_values[str(dim["name"])]) for dim in bucket_dimensions):
            return plan
    raise ValueError(
        "No bucket plan matches runtime input shapes; assignments="
        f"{assignments}, available_plans={[plan.get('bucket_id') for plan in bucket_plans]}"
    )


def align_up(nbytes: int, alignment: int) -> int:
    if nbytes <= 0:
        return 0
    return ((int(nbytes) + int(alignment) - 1) // int(alignment)) * int(alignment)
