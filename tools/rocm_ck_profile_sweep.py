from __future__ import annotations

import argparse
import json
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import dinoml as dml  # noqa: E402
from dinoml.ir import ModelSpec, read_json, write_json  # noqa: E402
from dinoml.kernels.bmm import bmm_op_spec  # noqa: E402
from dinoml.kernels.gemm import gemm_op_spec  # noqa: E402


@dataclass(frozen=True)
class RocmCkProfileSweepCase:
    name: str
    family: str
    op: str
    dtype: str
    build_spec: Callable[[], ModelSpec]


class _GemmSweepModule(dml.Module):
    def __init__(self, op_name: str):
        self._op_name = op_name

    def forward(self, a, b, bias, d0, d1):
        op = getattr(dml.ops, self._op_name)
        values = {"bias": bias, "d0": d0, "d1": d1}
        inputs = [a, b, *(values[name] for name in gemm_op_spec(self._op_name).epilogue.inputs)]
        return {"c": op(*inputs)}


class _BmmSweepModule(dml.Module):
    def __init__(self, op_name: str):
        self._op_name = op_name

    def forward(self, a, b, d0):
        op = getattr(dml.ops, self._op_name)
        inputs = [a, b]
        if bmm_op_spec(self._op_name).epilogue == "add":
            inputs.append(d0)
        return {"c": op(*inputs)}


class _ConvSweepModule(dml.Module):
    def __init__(self, op_name: str):
        self._op_name = op_name

    def forward(self, x, weight, bias, residual):
        op = getattr(dml.ops, self._op_name)
        if self._op_name in {"conv2d_bias_add", "conv2d_bias_add_relu"}:
            return {"y": op(x, weight, bias, residual, padding=1)}
        return {"y": op(x, weight, bias, padding=1)}


def rocm_ck_profile_sweep_cases(dtype: str = "float16") -> list[RocmCkProfileSweepCase]:
    return [
        *_gemm_cases(dtype),
        *_bmm_cases(dtype),
        *_conv_cases(dtype),
    ]


def run_sweep(
    *,
    only: Iterable[str] | None = None,
    artifact_root: str | Path | None = None,
    arch: str | None = None,
    iterations: int = 5,
    repeats: int = 1,
    refresh: bool = False,
    dtype: str = "float16",
    fail_fast: bool = False,
) -> dict[str, Any]:
    selected = _select_cases(rocm_ck_profile_sweep_cases(dtype), only)
    if iterations < 1:
        raise ValueError(f"iterations must be >= 1, got {iterations}")
    if repeats < 1:
        raise ValueError(f"repeats must be >= 1, got {repeats}")
    root = _artifact_root(artifact_root)
    target = dml.Target("rocm", arch=arch)
    results: list[dict[str, Any]] = []
    started = time.perf_counter()
    for case in selected:
        case_started = time.perf_counter()
        artifact_path = root / f"{case.name}.dinoml"
        try:
            artifact = dml.compile(
                case.build_spec(),
                target,
                artifact_path,
                profile=True,
                profile_iterations=iterations,
                profile_repeats=repeats,
                profile_refresh=refresh,
            )
            report = read_json(artifact.path / "debug" / "bootstrap_profile_report.json")
            kernel_manifest = read_json(artifact.path / "kernel_manifest.json")
            results.append(
                {
                    "name": case.name,
                    "family": case.family,
                    "op": case.op,
                    "dtype": case.dtype,
                    "artifact": str(artifact.path),
                    "status": "ok",
                    "profile_summary": dict(report.get("summary", {})),
                    "selected_candidates": _selected_candidates(kernel_manifest),
                    "elapsed_s": time.perf_counter() - case_started,
                }
            )
        except Exception as exc:
            results.append(
                {
                    "name": case.name,
                    "family": case.family,
                    "op": case.op,
                    "dtype": case.dtype,
                    "artifact": str(artifact_path),
                    "status": "error",
                    "error": f"{type(exc).__name__}: {exc}",
                    "elapsed_s": time.perf_counter() - case_started,
                }
            )
            if fail_fast:
                raise
    return {
        "schema_version": 1,
        "kind": "rocm_ck_profile_sweep",
        "target": target.to_json(),
        "iterations": int(iterations),
        "repeats": int(repeats),
        "artifact_root": str(root),
        "summary": {
            "total": len(results),
            "ok": sum(1 for item in results if item["status"] == "ok"),
            "error": sum(1 for item in results if item["status"] != "ok"),
            "elapsed_s": time.perf_counter() - started,
        },
        "cases": results,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Profile representative ROCm CK GEMM/BMM/Conv cases")
    parser.add_argument("--only", action="append", default=[], help="Run a case, op, or family; repeatable")
    parser.add_argument("--artifacts", help="Directory for compiled sweep artifacts")
    parser.add_argument("--arch", default=None, help="ROCm architecture, defaulting to the DinoML ROCm target default")
    parser.add_argument("--dtype", default="float16", choices=("float16", "float32"))
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--refresh", action="store_true", help="Ignore existing profiler cache entries")
    parser.add_argument("--fail-fast", action="store_true")
    parser.add_argument("--out", help="Write the JSON sweep report")
    args = parser.parse_args(argv)

    report = run_sweep(
        only=args.only,
        artifact_root=args.artifacts,
        arch=args.arch,
        iterations=args.iterations,
        repeats=args.repeats,
        refresh=args.refresh,
        dtype=args.dtype,
        fail_fast=args.fail_fast,
    )
    if args.out:
        write_json(Path(args.out), report)
    print(json.dumps(report["summary"], indent=2, sort_keys=True))
    return 0 if report["summary"]["error"] == 0 else 1


def _gemm_cases(dtype: str) -> list[RocmCkProfileSweepCase]:
    return [
        _gemm_case("gemm_rcr", dtype),
        _gemm_case("gemm_rcr_bias", dtype),
        _gemm_case("gemm_rcr_bias_add_relu", dtype),
        _gemm_case("gemm_rcr_bias_add_add_relu", dtype),
    ]


def _gemm_case(op_name: str, dtype: str) -> RocmCkProfileSweepCase:
    m, n, k = 128, 128, 192
    spec = gemm_op_spec(op_name)
    b_shape = [k, n] if spec.base_layout == "rrr" else [n, k]

    def build_spec() -> ModelSpec:
        return dml.trace(
            _GemmSweepModule(op_name),
            {
                "a": dml.TensorSpec([m, k], dtype),
                "b": dml.TensorSpec(b_shape, dtype),
                "bias": dml.TensorSpec([n], dtype),
                "d0": dml.TensorSpec([m, n], dtype),
                "d1": dml.TensorSpec([m, n], dtype),
            },
            name=f"rocm_ck_profile_sweep_{op_name}_{dtype}",
        )

    return RocmCkProfileSweepCase(f"{op_name}_{dtype}", "gemm", op_name, dtype, build_spec)


def _bmm_cases(dtype: str) -> list[RocmCkProfileSweepCase]:
    return [
        _bmm_case("bmm_rcr", dtype),
        _bmm_case("bmm_rcr_add", dtype),
    ]


def _bmm_case(op_name: str, dtype: str) -> RocmCkProfileSweepCase:
    batch, m, n, k = 2, 64, 128, 96
    spec = bmm_op_spec(op_name)
    a_shape = [batch, k, m] if spec.a_layout == "c" else [batch, m, k]
    b_shape = [batch, n, k] if spec.b_layout == "c" else [batch, k, n]
    output_shape = [batch, n, m] if spec.c_layout == "c" else [batch, m, n]

    def build_spec() -> ModelSpec:
        return dml.trace(
            _BmmSweepModule(op_name),
            {
                "a": dml.TensorSpec(a_shape, dtype),
                "b": dml.TensorSpec(b_shape, dtype),
                "d0": dml.TensorSpec(output_shape, dtype),
            },
            name=f"rocm_ck_profile_sweep_{op_name}_{dtype}",
        )

    return RocmCkProfileSweepCase(f"{op_name}_{dtype}", "bmm", op_name, dtype, build_spec)


def _conv_cases(dtype: str) -> list[RocmCkProfileSweepCase]:
    return [
        _conv_case("conv2d_bias", dtype),
        _conv_case("conv2d_bias_relu", dtype),
        _conv_case("conv2d_bias_add", dtype),
        _conv_case("conv2d_bias_add_relu", dtype),
    ]


def _conv_case(op_name: str, dtype: str) -> RocmCkProfileSweepCase:
    def build_spec() -> ModelSpec:
        return dml.trace(
            _ConvSweepModule(op_name),
            {
                "x": dml.TensorSpec([2, 8, 16, 16], dtype),
                "weight": dml.TensorSpec([64, 8, 3, 3], dtype),
                "bias": dml.TensorSpec([64], dtype),
                "residual": dml.TensorSpec([2, 64, 16, 16], dtype),
            },
            name=f"rocm_ck_profile_sweep_{op_name}_{dtype}",
        )

    return RocmCkProfileSweepCase(f"{op_name}_{dtype}", "conv", op_name, dtype, build_spec)


def _select_cases(
    cases: list[RocmCkProfileSweepCase],
    only: Iterable[str] | None,
) -> list[RocmCkProfileSweepCase]:
    filters = {item for item in (only or ()) if item}
    if not filters:
        return cases
    return [
        case
        for case in cases
        if case.name in filters or case.op in filters or case.family in filters
    ]


def _artifact_root(path: str | Path | None) -> Path:
    if path is None:
        return Path(tempfile.mkdtemp(prefix="dinoml_rocm_ck_profile_sweep_"))
    root = Path(path)
    root.mkdir(parents=True, exist_ok=True)
    return root


def _selected_candidates(kernel_manifest: Mapping[str, Any]) -> list[dict[str, Any]]:
    selected = []
    for item in kernel_manifest.get("required_kernels", []):
        selected.append(
            {
                "op": str(item.get("op", "")),
                "kernel_library": str(item.get("kernel_library", "")),
                "dtype": str(item.get("dtype", "")),
                "selected_candidate_id": item.get("selected_candidate_id"),
                "kernel_symbol": item.get("kernel_symbol"),
                "profiler_symbol": item.get("profiler_symbol"),
            }
        )
    return selected


if __name__ == "__main__":
    raise SystemExit(main())
