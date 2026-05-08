from __future__ import annotations

import ctypes
import hashlib
import json
import os
import re
import shutil
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from dinoml.ir import array_to_storage, canonical_json, dtype_nbytes, read_json, write_json
from dinoml.kernels.manifest import PROFILE_CACHE_SCHEMA_VERSION
from dinoml.ops.definitions import get_op_def
from dinoml.shapes import validate_runtime_shape


PROFILE_REPORT_SCHEMA_VERSION = 5


@dataclass(frozen=True)
class GemmProfileWorkload:
    node_id: str
    op: str
    dtype: str
    kernel_symbol: str
    profiler_symbol: str
    candidate_set_id: str | None
    candidate_set_key: str | None
    candidate_id: str
    candidate_config_key: str | None
    candidate: Mapping[str, Any]
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
            "candidate_set_id": self.candidate_set_id,
            "candidate_set_key": self.candidate_set_key,
            "candidate_id": self.candidate_id,
            "candidate_config_key": self.candidate_config_key,
            "candidate": dict(self.candidate),
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
        candidate = _selected_profile_candidate(required_item)
        a_name, b_name = (str(name) for name in node["inputs"])
        a_shape = _runtime_tensor_shape(a_name, tensor_map[a_name], overrides)
        b_shape = _runtime_tensor_shape(b_name, tensor_map[b_name], overrides)
        m, n, k, output_shape = _gemm_problem(op_name, a_shape, b_shape)
        workloads.append(
            GemmProfileWorkload(
                node_id=str(node["id"]),
                op=op_name,
                dtype=dtype,
                kernel_symbol=str(candidate.get("kernel_symbol") or binding.symbol),
                profiler_symbol=str(candidate.get("profiler_symbol") or required_item["profiler_symbol"]),
                candidate_set_id=(
                    str(required_item["candidate_set_id"]) if required_item.get("candidate_set_id") is not None else None
                ),
                candidate_set_key=(
                    str(required_item["candidate_set_key"]) if required_item.get("candidate_set_key") is not None else None
                ),
                candidate_id=str(candidate["candidate_id"]),
                candidate_config_key=(
                    str(candidate["candidate_config_key"]) if candidate.get("candidate_config_key") is not None else None
                ),
                candidate=candidate,
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


def _selected_profile_candidate(required_item: Mapping[str, Any]) -> dict[str, Any]:
    candidates = [dict(candidate) for candidate in required_item.get("candidates", [])]
    if candidates:
        selected_id = required_item.get("selected_candidate_id")
        for candidate in candidates:
            if candidate.get("candidate_id") == selected_id:
                return candidate
        return candidates[0]
    return {
        "candidate_id": "manifest_default",
        "symbol_id": "manifest_default",
        "provider": "manifest",
        "family": "unknown",
        "op": required_item.get("op"),
        "kernel_symbol": required_item.get("kernel_symbol"),
        "profiler_symbol": required_item.get("profiler_symbol"),
        "candidate_config_key": None,
    }


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
    context = _profile_context(artifact_dir, manifest, codegen_plan)
    summary = {"profiled": 0, "cached": 0, "skipped": 0, "failed": 0}
    if not workloads:
        report = _profile_report(
            artifact_dir,
            manifest,
            kernel_manifest,
            codegen_plan,
            iterations,
            [],
            summary,
            context=context,
        )
        _write_profile_report(report, artifact_dir, output)
        return report

    rng = np.random.default_rng(seed)
    results = []
    profiler = None
    try:
        for workload in workloads:
            key_payload = _profile_key_payload(workload, manifest, kernel_manifest, codegen_plan, context=context)
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

    report = _profile_report(
        artifact_dir,
        manifest,
        kernel_manifest,
        codegen_plan,
        iterations,
        results,
        summary,
        context=context,
    )
    _write_profile_report(report, artifact_dir, output)
    return report


def profile_cache_path(codegen_plan: Mapping[str, Any]) -> Path:
    return Path(str(codegen_plan["support_cache_dir"])) / f"profile_cache.v{PROFILE_CACHE_SCHEMA_VERSION}.json"


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
    candidate_result = dict(workload.candidate)
    candidate_result.update(
        {
            "candidate_id": workload.candidate_id,
            "avg_ms": float(elapsed_ms),
            "gflops": float(tflops * 1000.0),
            "iterations": int(iterations),
        }
    )
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
            "candidates": [candidate_result],
            "selected": {"candidate_id": workload.candidate_id, "reason": reason},
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
    *,
    context: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    problem_payloads = [dict(item) for item in problems]
    profile_context = dict(context or _profile_context(artifact_dir, manifest, codegen_plan))
    return {
        "schema_version": PROFILE_REPORT_SCHEMA_VERSION,
        "profile_cache_schema_version": PROFILE_CACHE_SCHEMA_VERSION,
        "artifact": str(artifact_dir.resolve()),
        "target": manifest["target"],
        "kernel_manifest_cache_key": kernel_manifest["cache_key"],
        "codegen_plan_cache_key": codegen_plan["cache_key"],
        "iterations": int(iterations),
        "fingerprint": profile_context["fingerprint"],
        "hardware": profile_context["fingerprint"]["hardware"],
        "hardware_cache_key": profile_context["fingerprint"]["hardware_key"],
        "libraries": profile_context["fingerprint"]["support_libraries"],
        "support_libraries_cache_key": profile_context["fingerprint"]["support_libraries_key"],
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
) -> list[dict[str, Any]]:
    files = manifest["files"]
    by_name: dict[str, dict[str, Any]] = {}

    def merge(name: str, **fields: Any) -> None:
        entry = by_name.setdefault(name, {"name": name})
        for key, value in fields.items():
            if value not in (None, ""):
                entry[key] = value

    if "cutlass_gemm_library" in files:
        path = artifact_dir / files["cutlass_gemm_library"]
        merge(
            "cutlass_gemm",
            path=str(path.resolve()),
            artifact_path=str(path.resolve()),
            artifact_sha256=_file_sha256(path),
        )
    for item in codegen_plan.get("external_support_libraries", []):
        name = str(item["name"])
        cache_dir = Path(str(item.get("cache_dir", ""))) if item.get("cache_dir") else None
        cache_library = _cache_library_path(item, cache_dir)
        support_manifest = _support_library_manifest_path(name, cache_dir)
        support_payload = _read_optional_json(support_manifest) if support_manifest else {}
        manifest_fields = _support_manifest_fields(support_payload)
        merge(
            name,
            cache_dir=str(cache_dir) if cache_dir else None,
            cache_library=str(cache_library) if cache_library else None,
            cache_library_sha256=_file_sha256(cache_library) if cache_library else None,
            manifest=str(support_manifest) if support_manifest and support_manifest.exists() else None,
            **manifest_fields,
        )
    return [by_name[name] for name in sorted(by_name)]


def _profile_context(
    artifact_dir: Path,
    manifest: Mapping[str, Any],
    codegen_plan: Mapping[str, Any],
) -> dict[str, Any]:
    libraries = _profile_libraries(artifact_dir, manifest, codegen_plan)
    hardware = _cuda_hardware_fingerprint(manifest["target"])
    hardware_cache_payload = _hardware_cache_payload(hardware)
    support_libraries_cache_payload = _support_libraries_cache_payload(libraries)
    hardware_key = _fingerprint_key(hardware_cache_payload)
    support_libraries_key = _fingerprint_key(support_libraries_cache_payload)
    fingerprint_key = _fingerprint_key(
        {
            "hardware_key": hardware_key,
            "support_libraries_key": support_libraries_key,
        }
    )
    return {
        "fingerprint": {
            "schema_version": 1,
            "key": fingerprint_key,
            "hardware_key": hardware_key,
            "support_libraries_key": support_libraries_key,
            "hardware": hardware,
            "support_libraries": libraries,
        },
        "hardware_cache_payload": hardware_cache_payload,
        "support_libraries_cache_payload": support_libraries_cache_payload,
    }


def _cache_library_path(item: Mapping[str, Any], cache_dir: Path | None) -> Path | None:
    if cache_dir is None:
        return None
    library = str(item.get("library", ""))
    if not library:
        return None
    candidate = cache_dir / library
    return candidate if candidate.exists() else None


def _support_library_manifest_path(name: str, cache_dir: Path | None) -> Path | None:
    if cache_dir is None:
        return None
    if name == "cutlass_gemm":
        return cache_dir / "lib" / "cutlass_gemm_manifest.json"
    return cache_dir / "lib" / f"{name}_manifest.json"


def _support_manifest_fields(payload: Mapping[str, Any]) -> dict[str, Any]:
    fields: dict[str, Any] = {}
    for key in (
        "schema_version",
        "provider",
        "source_sha256",
        "library_sha256",
        "cache_key",
        "provenance_key",
        "build_fingerprint",
        "family_cache_key",
        "external_kernel_plan_cache_key",
    ):
        if key in payload:
            fields[f"manifest_{key}" if key in {"schema_version", "cache_key"} else key] = payload[key]
    target = payload.get("target")
    if isinstance(target, Mapping):
        fields["manifest_target"] = dict(target)
    for key in ("compile", "provenance"):
        if isinstance(payload.get(key), Mapping):
            fields[key] = dict(payload[key])
    return fields


def _read_optional_json(path: Path | None) -> dict[str, Any]:
    if path is None or not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _file_sha256(path: Path | None) -> str | None:
    if path is None or not path.exists() or not path.is_file():
        return None
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _cuda_hardware_fingerprint(target: Mapping[str, Any]) -> dict[str, Any]:
    devices = _query_nvidia_smi_devices()
    return {
        "backend": "cuda",
        "target_arch": str(target.get("arch", "")),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        "nvidia_smi": "available" if devices else "unavailable",
        "devices": devices,
        "nvcc": _query_nvcc_version(),
    }


def _query_nvidia_smi_devices() -> list[dict[str, Any]]:
    if shutil.which("nvidia-smi") is None:
        return []
    proc = _run_capture(
        [
            "nvidia-smi",
            "--query-gpu=index,name,compute_cap,driver_version,memory.total",
            "--format=csv,noheader,nounits",
        ],
        timeout=2.0,
    )
    if proc is None or proc.returncode != 0:
        return []
    devices = []
    for line in proc.stdout.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) != 5:
            continue
        devices.append(
            {
                "index": _parse_int(parts[0]),
                "name": parts[1],
                "compute_capability": parts[2],
                "driver_version": parts[3],
                "memory_total_mib": _parse_int(parts[4]),
            }
        )
    return devices


def _query_nvcc_version() -> dict[str, str]:
    if shutil.which("nvcc") is None:
        return {"available": "false"}
    proc = _run_capture(["nvcc", "--version"], timeout=2.0)
    if proc is None or proc.returncode != 0:
        return {"available": "false"}
    release_match = re.search(r"release\s+([0-9.]+)", proc.stdout)
    build_match = re.search(r"V([0-9.]+)", proc.stdout)
    payload = {"available": "true"}
    if release_match:
        payload["release"] = release_match.group(1)
    if build_match:
        payload["build"] = build_match.group(1)
    return payload


def _run_capture(cmd: Sequence[str], *, timeout: float) -> subprocess.CompletedProcess[str] | None:
    try:
        return subprocess.run(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=timeout)
    except (OSError, subprocess.TimeoutExpired):
        return None


def _parse_int(value: str) -> int | None:
    try:
        return int(value)
    except ValueError:
        return None


def _hardware_cache_payload(hardware: Mapping[str, Any]) -> dict[str, Any]:
    devices = []
    for item in hardware.get("devices", []):
        if not isinstance(item, Mapping):
            continue
        devices.append(
            {
                "name": item.get("name"),
                "compute_capability": item.get("compute_capability"),
                "driver_version": item.get("driver_version"),
                "memory_total_mib": item.get("memory_total_mib"),
            }
        )
    return {
        "backend": hardware.get("backend"),
        "target_arch": hardware.get("target_arch"),
        "cuda_visible_devices": hardware.get("cuda_visible_devices", ""),
        "devices": devices,
        "nvcc": dict(hardware.get("nvcc", {})) if isinstance(hardware.get("nvcc"), Mapping) else {},
    }


def _support_libraries_cache_payload(libraries: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    payloads = []
    for library in libraries:
        payloads.append(
            {
                "name": library.get("name"),
                "artifact_sha256": library.get("artifact_sha256"),
                "cache_library_sha256": library.get("cache_library_sha256"),
                "source_sha256": library.get("source_sha256"),
                "library_sha256": library.get("library_sha256"),
                "provenance_key": library.get("provenance_key"),
                "build_fingerprint": library.get("build_fingerprint"),
                "family_cache_key": library.get("family_cache_key"),
                "manifest_cache_key": library.get("manifest_cache_key"),
                "manifest_target": library.get("manifest_target"),
                "external_kernel_plan_cache_key": library.get("external_kernel_plan_cache_key"),
            }
        )
    return sorted(payloads, key=lambda item: str(item.get("name", "")))


def _fingerprint_key(payload: Mapping[str, Any] | Sequence[Mapping[str, Any]]) -> str:
    return hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()


def _profile_key_payload(
    workload: GemmProfileWorkload,
    manifest: Mapping[str, Any],
    kernel_manifest: Mapping[str, Any],
    codegen_plan: Mapping[str, Any],
    *,
    context: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    profile_context = context or _profile_context(Path("."), manifest, codegen_plan)
    return {
        "schema_version": PROFILE_CACHE_SCHEMA_VERSION,
        "target": manifest["target"],
        "hardware_fingerprint_key": profile_context["fingerprint"]["hardware_key"],
        "support_libraries_fingerprint_key": profile_context["fingerprint"]["support_libraries_key"],
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
        "candidate_set_id": workload.candidate_set_id,
        "candidate_set_key": workload.candidate_set_key,
        "candidate_id": workload.candidate_id,
        "candidate_config_key": workload.candidate_config_key,
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
        "best_candidate_id": workload.candidate_id,
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
