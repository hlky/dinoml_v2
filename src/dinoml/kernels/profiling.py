from __future__ import annotations

import ctypes
import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from dinoml.ir import array_to_storage, canonical_json, dtype_nbytes, read_json, write_json
from dinoml.kernels.manifest import PROFILE_CACHE_SCHEMA_VERSION
from dinoml.ops.definitions import get_op_def
from dinoml.shapes import validate_runtime_shape


PROFILE_REPORT_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class GemmProfileWorkload:
    node_id: str
    op: str
    dtype: str
    kernel_symbol: str
    profiler_symbol: str
    a_tensor: str
    b_tensor: str
    output_tensor: str
    a_shape: tuple[int, int]
    b_shape: tuple[int, int]
    output_shape: tuple[int, int]
    m: int
    n: int
    k: int

    def to_json(self) -> dict[str, Any]:
        return {
            "node_id": self.node_id,
            "op": self.op,
            "dtype": self.dtype,
            "kernel_symbol": self.kernel_symbol,
            "profiler_symbol": self.profiler_symbol,
            "inputs": {
                self.a_tensor: list(self.a_shape),
                self.b_tensor: list(self.b_shape),
            },
            "output": {
                self.output_tensor: list(self.output_shape),
            },
            "m": self.m,
            "n": self.n,
            "k": self.k,
        }


def parse_shape_overrides(items: Sequence[str] | None) -> dict[str, tuple[int, ...]]:
    overrides: dict[str, tuple[int, ...]] = {}
    for item in items or ():
        if "=" not in item:
            raise ValueError(f"Expected shape override like name=1,128,768, got {item!r}")
        name, raw_shape = item.split("=", 1)
        dims = tuple(int(part) for part in raw_shape.split(",") if part)
        if not name or not dims or any(dim <= 0 for dim in dims):
            raise ValueError(f"Invalid shape override: {item!r}")
        overrides[name] = dims
    return overrides


def build_profile_workloads(
    graph: Mapping[str, Any],
    kernel_manifest: Mapping[str, Any],
    *,
    input_shapes: Mapping[str, Sequence[int]] | None = None,
) -> list[GemmProfileWorkload]:
    if kernel_manifest.get("target", {}).get("name") != "cuda":
        return []
    tensor_map = {str(tensor["name"]): tensor for tensor in graph["tensors"]}
    required = {
        (str(item["op"]), str(item["kernel_symbol"])): item
        for item in kernel_manifest.get("required_kernels", [])
        if item.get("profiler_symbol")
    }
    overrides = {name: tuple(int(dim) for dim in shape) for name, shape in (input_shapes or {}).items()}
    workloads = []
    for node in graph["nodes"]:
        op_name = str(node["op"])
        if op_name not in {"gemm_rrr", "gemm_rcr"}:
            continue
        output_name = str(node["outputs"][0])
        output_info = tensor_map[output_name]
        dtype = str(output_info["dtype"])
        binding = get_op_def(op_name).backend_kernels["cuda"].resolve(dtype)
        required_item = required.get((op_name, binding.symbol))
        if required_item is None:
            continue
        a_name, b_name = (str(name) for name in node["inputs"])
        a_shape = _runtime_tensor_shape(a_name, tensor_map[a_name], overrides)
        b_shape = _runtime_tensor_shape(b_name, tensor_map[b_name], overrides)
        m, n, k, output_shape = _gemm_problem(op_name, a_shape, b_shape)
        workloads.append(
            GemmProfileWorkload(
                node_id=str(node["id"]),
                op=op_name,
                dtype=dtype,
                kernel_symbol=binding.symbol,
                profiler_symbol=str(required_item["profiler_symbol"]),
                a_tensor=a_name,
                b_tensor=b_name,
                output_tensor=output_name,
                a_shape=(a_shape[0], a_shape[1]),
                b_shape=(b_shape[0], b_shape[1]),
                output_shape=output_shape,
                m=m,
                n=n,
                k=k,
            )
        )
    return workloads


def profile_artifact(
    artifact: str | Path,
    *,
    input_shapes: Mapping[str, Sequence[int]] | None = None,
    iterations: int = 20,
    output: str | Path | None = None,
    seed: int = 2027,
    refresh: bool = False,
) -> dict[str, Any]:
    artifact_dir = Path(artifact)
    manifest = read_json(artifact_dir / "manifest.json")
    if manifest.get("target", {}).get("name") != "cuda":
        raise ValueError("Profiler runner currently supports CUDA artifacts only")
    graph = read_json(artifact_dir / manifest["files"]["graph"])
    kernel_manifest = read_json(artifact_dir / manifest["files"]["kernel_manifest"])
    codegen_plan = read_json(artifact_dir / manifest["files"]["kernel_codegen_plan"])
    workloads = build_profile_workloads(graph, kernel_manifest, input_shapes=input_shapes)
    cache_path = profile_cache_path(codegen_plan)
    cache = _read_profile_cache(cache_path, manifest["target"])
    summary = {"profiled": 0, "cached": 0, "skipped": 0, "failed": 0}
    if not workloads:
        report = _profile_report(artifact_dir, manifest, kernel_manifest, codegen_plan, iterations, [], summary)
        _write_profile_report(report, artifact_dir, output)
        return report

    rng = np.random.default_rng(seed)
    results = []
    profiler = None
    try:
        for workload in workloads:
            key_payload = _profile_key_payload(workload, manifest, kernel_manifest, codegen_plan)
            profile_key = _profile_key(key_payload)
            cached = cache["entries"].get(profile_key)
            if cached is not None and not refresh:
                results.append(_profile_result_from_cache(workload, cached))
                summary["cached"] += 1
            else:
                if profiler is None:
                    profiler = _CudaProfiler(artifact_dir, manifest)
                elapsed_ms = profiler.profile_gemm(workload, iterations=iterations, rng=rng)
                result = _profile_result(workload, elapsed_ms, iterations, profile_key=profile_key, status="ok")
                results.append(result)
                cache["entries"][profile_key] = _cache_entry(workload, result, key_payload)
                summary["profiled"] += 1
    finally:
        if profiler is not None:
            profiler.close()
    if summary["profiled"]:
        _write_profile_cache(cache_path, cache)

    report = _profile_report(artifact_dir, manifest, kernel_manifest, codegen_plan, iterations, results, summary)
    _write_profile_report(report, artifact_dir, output)
    return report


def profile_cache_path(codegen_plan: Mapping[str, Any]) -> Path:
    return Path(str(codegen_plan["support_cache_dir"])) / "profile_cache.v1.json"


def _runtime_tensor_shape(
    name: str,
    tensor: Mapping[str, Any],
    overrides: Mapping[str, Sequence[int]],
) -> tuple[int, ...]:
    if name in overrides:
        return tuple(validate_runtime_shape(name, overrides[name], tensor))
    return tuple(int(dim) for dim in tensor["shape"])


def _gemm_problem(op_name: str, a_shape: Sequence[int], b_shape: Sequence[int]) -> tuple[int, int, int, tuple[int, int]]:
    if len(a_shape) != 2 or len(b_shape) != 2:
        raise ValueError(f"{op_name} profiling requires rank-2 tensors")
    m = int(a_shape[0])
    k = int(a_shape[1])
    if op_name == "gemm_rrr":
        if k != int(b_shape[0]):
            raise ValueError(f"gemm_rrr profiling shape mismatch: {list(a_shape)} vs {list(b_shape)}")
        n = int(b_shape[1])
    elif op_name == "gemm_rcr":
        if k != int(b_shape[1]):
            raise ValueError(f"gemm_rcr profiling shape mismatch: {list(a_shape)} vs {list(b_shape)}")
        n = int(b_shape[0])
    else:
        raise ValueError(f"Unsupported GEMM profiling op: {op_name}")
    return m, n, k, (m, n)


def _profile_result(
    workload: GemmProfileWorkload,
    elapsed_ms: float,
    iterations: int,
    *,
    profile_key: str,
    status: str,
    reason: str = "only_candidate",
) -> dict[str, Any]:
    flops = 2 * workload.m * workload.n * workload.k
    bytes_moved = dtype_nbytes(workload.dtype) * (
        workload.m * workload.k + workload.n * workload.k + workload.m * workload.n
    )
    seconds = max(float(elapsed_ms) / 1000.0, 1e-12)
    tflops = float(flops / seconds / 1.0e12)
    gbps = float(bytes_moved / seconds / 1.0e9)
    payload = workload.to_json()
    payload.update(
        {
            "profile_key": profile_key,
            "status": status,
            "shape": {"m": workload.m, "n": workload.n, "k": workload.k, "source": "runtime_override_or_graph_max_shape"},
            "tensors": {"a": workload.a_tensor, "b": workload.b_tensor, "c": workload.output_tensor},
            "kernel_library": "cutlass_gemm",
            "elapsed_ms": float(elapsed_ms),
            "iterations": int(iterations),
            "flops": int(flops),
            "bytes": int(bytes_moved),
            "gflops": float(tflops * 1000.0),
            "tflops": tflops,
            "gbps": gbps,
            "candidates": [
                {
                    "candidate_id": "manifest_default",
                    "avg_ms": float(elapsed_ms),
                    "gflops": float(tflops * 1000.0),
                    "iterations": int(iterations),
                }
            ],
            "selected": {"candidate_id": "manifest_default", "reason": reason},
        }
    )
    return payload


def _profile_report(
    artifact_dir: Path,
    manifest: Mapping[str, Any],
    kernel_manifest: Mapping[str, Any],
    codegen_plan: Mapping[str, Any],
    iterations: int,
    problems: Sequence[Mapping[str, Any]],
    summary: Mapping[str, int],
) -> dict[str, Any]:
    problem_payloads = [dict(item) for item in problems]
    return {
        "schema_version": PROFILE_REPORT_SCHEMA_VERSION,
        "profile_cache_schema_version": PROFILE_CACHE_SCHEMA_VERSION,
        "artifact": str(artifact_dir.resolve()),
        "target": manifest["target"],
        "kernel_manifest_cache_key": kernel_manifest["cache_key"],
        "codegen_plan_cache_key": codegen_plan["cache_key"],
        "iterations": int(iterations),
        "libraries": _profile_libraries(artifact_dir, manifest, codegen_plan),
        "problems": problem_payloads,
        "workloads": problem_payloads,
        "summary": dict(summary),
    }


def _write_profile_report(report: Mapping[str, Any], artifact_dir: Path, output: str | Path | None) -> None:
    report_path = Path(output) if output is not None else artifact_dir / "debug" / "profile_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    write_json(report_path, dict(report))


def _profile_libraries(
    artifact_dir: Path,
    manifest: Mapping[str, Any],
    codegen_plan: Mapping[str, Any],
) -> list[dict[str, str]]:
    files = manifest["files"]
    libraries = []
    if "cutlass_gemm_library" in files:
        libraries.append(
            {
                "name": "cutlass_gemm",
                "path": str((artifact_dir / files["cutlass_gemm_library"]).resolve()),
            }
        )
    for item in codegen_plan.get("external_support_libraries", []):
        if all(existing["name"] != item["name"] for existing in libraries):
            libraries.append({"name": str(item["name"]), "path": str(item.get("library", ""))})
    return libraries


def _profile_key_payload(
    workload: GemmProfileWorkload,
    manifest: Mapping[str, Any],
    kernel_manifest: Mapping[str, Any],
    codegen_plan: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": PROFILE_CACHE_SCHEMA_VERSION,
        "target": manifest["target"],
        "support_cache_key": kernel_manifest.get("support_cache_key"),
        "codegen_plan_cache_key": codegen_plan["cache_key"],
        "op": workload.op,
        "dtype": workload.dtype,
        "layouts": {
            "a": "row",
            "b": "row" if workload.op == "gemm_rrr" else "column",
            "c": "row",
        },
        "epilogue": "linear_combination",
        "shape": {"m": workload.m, "n": workload.n, "k": workload.k},
        "kernel_symbol": workload.kernel_symbol,
        "profiler_symbol": workload.profiler_symbol,
        "candidate_id": "manifest_default",
    }


def _profile_key(payload: Mapping[str, Any]) -> str:
    return hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()


def _read_profile_cache(path: Path, target: Mapping[str, Any]) -> dict[str, Any]:
    if not path.exists():
        return {"schema_version": PROFILE_CACHE_SCHEMA_VERSION, "target": dict(target), "entries": {}}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {"schema_version": PROFILE_CACHE_SCHEMA_VERSION, "target": dict(target), "entries": {}}
    if payload.get("schema_version") != PROFILE_CACHE_SCHEMA_VERSION:
        return {"schema_version": PROFILE_CACHE_SCHEMA_VERSION, "target": dict(target), "entries": {}}
    if payload.get("target") != dict(target):
        return {"schema_version": PROFILE_CACHE_SCHEMA_VERSION, "target": dict(target), "entries": {}}
    payload.setdefault("entries", {})
    return payload


def _write_profile_cache(path: Path, cache: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    write_json(path, dict(cache))


def _cache_entry(
    workload: GemmProfileWorkload,
    result: Mapping[str, Any],
    key_payload: Mapping[str, Any],
) -> dict[str, Any]:
    candidate = result["candidates"][0]
    return {
        "profile_key": result["profile_key"],
        "key": dict(key_payload),
        "updated_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "op": workload.op,
        "dtype": workload.dtype,
        "shape": {"m": workload.m, "n": workload.n, "k": workload.k},
        "kernel_symbol": workload.kernel_symbol,
        "profiler_symbol": workload.profiler_symbol,
        "best_candidate_id": "manifest_default",
        "avg_ms": float(candidate["avg_ms"]),
        "gflops": float(candidate["gflops"]),
        "iterations": int(candidate["iterations"]),
    }


def _profile_result_from_cache(workload: GemmProfileWorkload, entry: Mapping[str, Any]) -> dict[str, Any]:
    return _profile_result(
        workload,
        float(entry["avg_ms"]),
        int(entry["iterations"]),
        profile_key=str(entry["profile_key"]),
        status="cached",
        reason="cache_hit",
    )


class _CudaProfiler:
    def __init__(self, artifact_dir: Path, manifest: Mapping[str, Any]):
        files = manifest["files"]
        global_mode = getattr(ctypes, "RTLD_GLOBAL", 0) | getattr(ctypes, "RTLD_NOW", 0)
        self._runtime = ctypes.CDLL(str(artifact_dir / files["runtime_library"]), mode=global_mode)
        self._cuda_runtime = ctypes.CDLL(str(artifact_dir / files["cuda_runtime_library"]), mode=global_mode)
        self._cutlass = ctypes.CDLL(str(artifact_dir / files["cutlass_gemm_library"]), mode=global_mode)
        self._buffers: list[ctypes.c_void_p] = []
        self._runtime.dino_get_last_error.restype = ctypes.c_char_p
        self._cuda_runtime.dino_device_malloc.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_size_t]
        self._cuda_runtime.dino_device_malloc.restype = ctypes.c_int
        self._cuda_runtime.dino_device_free.argtypes = [ctypes.c_void_p]
        self._cuda_runtime.dino_device_free.restype = ctypes.c_int
        self._cuda_runtime.dino_copy_host_to_device.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t]
        self._cuda_runtime.dino_copy_host_to_device.restype = ctypes.c_int

    def close(self) -> None:
        while self._buffers:
            ptr = self._buffers.pop()
            self._check(self._cuda_runtime.dino_device_free(ptr))

    def profile_gemm(self, workload: GemmProfileWorkload, *, iterations: int, rng: np.random.Generator) -> float:
        a = self._device_array(_random_storage(workload.a_shape, workload.dtype, rng))
        b = self._device_array(_random_storage(workload.b_shape, workload.dtype, rng))
        c = self._device_array(_zero_storage(workload.output_shape, workload.dtype))
        fn = getattr(self._cutlass, workload.profiler_symbol)
        fn.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_void_p,
        ]
        fn.restype = ctypes.c_float
        return float(
            fn(
                a,
                b,
                c,
                ctypes.c_int(workload.m),
                ctypes.c_int(workload.n),
                ctypes.c_int(workload.k),
                ctypes.c_int(iterations),
                ctypes.c_void_p(0),
            )
        )

    def _device_array(self, array: np.ndarray) -> ctypes.c_void_p:
        contiguous = np.ascontiguousarray(array)
        ptr = ctypes.c_void_p()
        self._check(self._cuda_runtime.dino_device_malloc(ctypes.byref(ptr), ctypes.c_size_t(contiguous.nbytes)))
        self._buffers.append(ptr)
        self._check(
            self._cuda_runtime.dino_copy_host_to_device(
                ptr,
                ctypes.c_void_p(contiguous.ctypes.data),
                ctypes.c_size_t(contiguous.nbytes),
            )
        )
        return ptr

    def _check(self, code: int) -> None:
        if code == 0:
            return
        error = self._runtime.dino_get_last_error()
        message = error.decode("utf-8") if error else f"CUDA profiler helper failed with code {code}"
        raise RuntimeError(message)


def _random_storage(shape: Sequence[int], dtype: str, rng: np.random.Generator) -> np.ndarray:
    values = (rng.standard_normal(tuple(shape)).astype(np.float32) * 0.125)
    return np.ascontiguousarray(array_to_storage(values, dtype))


def _zero_storage(shape: Sequence[int], dtype: str) -> np.ndarray:
    return np.ascontiguousarray(array_to_storage(np.zeros(tuple(shape), dtype=np.float32), dtype))
