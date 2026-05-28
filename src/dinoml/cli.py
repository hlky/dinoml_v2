from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np

import dinoml as dml
from dinoml.benchmarks.ops import run_benchmark_suite, write_report
from dinoml.benchmarks.torch_ops import run_torch_benchmark_suite, write_torch_report
from dinoml.backends import registered_backend_names
from dinoml import runtime
from dinoml.ir import read_json
from dinoml.kernels.profiling import parse_shape_overrides, profile_artifact


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="dinoml")
    subparsers = parser.add_subparsers(dest="command", required=True)

    profile_description = (
        "Profiles CUDA CUTLASS or ROCm CK GEMM/BMM/Conv candidates, writes a profile report, "
        "emits an execution plan for the selected candidates, and reports blocked profile items."
    )

    compile_parser = subparsers.add_parser(
        "compile",
        description=(
            "Compile a model artifact. With --profile, DinoML benchmarks supported CUDA CUTLASS "
            "or ROCm CK candidates and rebuilds with the selected execution plan."
        ),
    )
    compile_parser.add_argument("model", help="Python model file defining build_spec()")
    compile_parser.add_argument(
        "--target",
        choices=registered_backend_names(),
        default="cpu",
        help="Compile target backend (default: cpu)",
    )
    compile_parser.add_argument(
        "--arch",
        default=None,
        help="Target architecture; defaults come from the backend registry (CUDA sm_86, ROCm gfx1201)",
    )
    compile_parser.add_argument("--no-tf32", action="store_true", help="Disable optional TF32 CUTLASS GEMM candidates")
    compile_parser.add_argument("--use-fp16-acc", action="store_true", help="Use fp16 accumulation for fp16 CUTLASS GEMM candidates")
    compile_parser.add_argument("--execution-plan", help="Apply a profile-selected execution_plan.json during compile")
    compile_parser.add_argument(
        "--profile",
        action="store_true",
        help=profile_description,
    )
    compile_parser.add_argument("--profile-iterations", type=int, default=20, help="Profiler iterations per candidate")
    compile_parser.add_argument("--profile-repeats", type=int, default=3, help="Profiler timing repeats per candidate")
    compile_parser.add_argument(
        "--profile-shape",
        "--shape",
        dest="profile_shape",
        action="append",
        default=[],
        help="Profile with an input shape override like input=1,128,768; repeat for multiple inputs",
    )
    compile_parser.add_argument("--profile-refresh", action="store_true", help="Ignore existing profiler cache entries")
    compile_parser.add_argument("--constant-load-policy", choices=("eager", "deferred"), default="eager")
    compile_parser.add_argument("--out", required=True)

    inspect_parser = subparsers.add_parser("inspect")
    inspect_parser.add_argument("artifact")

    validate_parser = subparsers.add_parser("validate")
    validate_parser.add_argument("artifact")
    validate_parser.add_argument("--against", required=True)
    validate_parser.add_argument("--atol", type=float, default=1e-4)
    validate_parser.add_argument("--rtol", type=float, default=1e-4)

    benchmark_parser = subparsers.add_parser("benchmark")
    benchmark_parser.add_argument("artifact")
    benchmark_parser.add_argument("--against", help="Python model file defining build_validation_inputs()")
    benchmark_parser.add_argument("--warmup", type=int, default=5)
    benchmark_parser.add_argument("--iterations", "--iters", dest="iterations", type=int, default=20)
    benchmark_parser.add_argument("--out")

    benchmark_ops_parser = subparsers.add_parser("benchmark-ops")
    benchmark_ops_parser.add_argument("target", choices=registered_backend_names())
    benchmark_ops_parser.add_argument(
        "--arch",
        default=None,
        help="Target architecture; defaults come from the backend registry (CUDA sm_86, ROCm gfx1201)",
    )
    benchmark_ops_parser.add_argument("--no-tf32", action="store_true", help="Disable optional TF32 CUTLASS GEMM candidates")
    benchmark_ops_parser.add_argument("--use-fp16-acc", action="store_true", help="Use fp16 accumulation for fp16 CUTLASS GEMM candidates")
    benchmark_ops_parser.add_argument("--warmup", type=int, default=5)
    benchmark_ops_parser.add_argument("--iterations", "--iters", dest="iterations", type=int, default=20)
    benchmark_ops_parser.add_argument(
        "--profile",
        action="store_true",
        help="Compile each benchmark artifact with profiling before benchmarking",
    )
    benchmark_ops_parser.add_argument("--profile-iterations", type=int, default=20, help="Profiler iterations per candidate")
    benchmark_ops_parser.add_argument("--profile-repeats", type=int, default=3, help="Profiler timing repeats per candidate")
    benchmark_ops_parser.add_argument("--profile-refresh", action="store_true", help="Ignore existing profiler cache entries")
    benchmark_ops_parser.add_argument("--only", action="append", default=[], help="Benchmark a case, op, or template name; repeatable")
    benchmark_ops_parser.add_argument("--artifacts", help="Directory for compiled per-op artifacts")
    benchmark_ops_parser.add_argument("--keep-artifacts", action="store_true", help="Keep temporary artifacts when --artifacts is not set")
    benchmark_ops_parser.add_argument(
        "--jobs",
        "-j",
        type=int,
        default=1,
        help="Compile per-op artifacts with up to N workers; device benchmarks still run serially",
    )
    benchmark_ops_parser.add_argument("--fail-fast", action="store_true")
    benchmark_ops_parser.add_argument("--out")

    benchmark_torch_ops_parser = subparsers.add_parser("benchmark-torch-ops")
    benchmark_torch_ops_parser.add_argument("--device", default="cpu", help="PyTorch device to benchmark on")
    benchmark_torch_ops_parser.add_argument("--warmup", type=int, default=5)
    benchmark_torch_ops_parser.add_argument("--iterations", "--iters", dest="iterations", type=int, default=20)
    benchmark_torch_ops_parser.add_argument("--only", action="append", default=[], help="Benchmark a case, op, or template name; repeatable")
    benchmark_torch_ops_parser.add_argument("--fail-fast", action="store_true")
    benchmark_torch_ops_parser.add_argument("--out")

    profile_parser = subparsers.add_parser(
        "profile",
        description=(
            "Profile an existing artifact. Supports CUDA CUTLASS and ROCm CK GEMM/BMM/Conv candidates "
            "and can write an execution_plan.json for a later compile. Blocked profile items explain "
            "why a manifest kernel had no runnable profiler candidate."
        ),
    )
    profile_parser.add_argument("artifact")
    profile_parser.add_argument("--iterations", type=int, default=20, help="Profiler iterations per candidate")
    profile_parser.add_argument("--repeats", type=int, default=3, help="Profiler timing repeats per candidate")
    profile_parser.add_argument(
        "--shape",
        action="append",
        default=[],
        help="Profile with an input shape override like input=1,128,768; repeat for multiple inputs",
    )
    profile_parser.add_argument("--out", help="Write the full profile report JSON, including blocked profile items")
    profile_parser.add_argument("--execution-plan-out", help="Write selected candidates to execution_plan.json")
    profile_parser.add_argument("--refresh", action="store_true", help="Ignore existing profiler cache entries")

    args = parser.parse_args(argv)
    if args.command == "compile":
        return _compile(args)
    if args.command == "inspect":
        return _inspect(args)
    if args.command == "validate":
        return _validate(args)
    if args.command == "benchmark":
        return _benchmark(args)
    if args.command == "benchmark-ops":
        return _benchmark_ops(args)
    if args.command == "benchmark-torch-ops":
        return _benchmark_torch_ops(args)
    if args.command == "profile":
        return _profile(args)
    raise AssertionError(args.command)


def _compile(args: argparse.Namespace) -> int:
    module = _load_python_file(Path(args.model))
    if not hasattr(module, "build_spec"):
        raise RuntimeError(f"{args.model} must define build_spec()")
    spec = module.build_spec()
    compile_kwargs = {
        "target": dml.Target(args.target, arch=args.arch, no_tf32=args.no_tf32, use_fp16_acc=args.use_fp16_acc),
        "output": args.out,
        "execution_plan": args.execution_plan,
        "constant_load_policy": args.constant_load_policy,
    }
    if args.profile:
        compile_kwargs.update(
            {
                "profile": True,
                "profile_iterations": args.profile_iterations,
                "profile_repeats": args.profile_repeats,
                "profile_input_shapes": parse_shape_overrides(args.profile_shape),
                "profile_refresh": args.profile_refresh,
            }
        )
    artifact = dml.compile(spec, **compile_kwargs)
    print(f"Wrote {artifact.path}")
    return 0


def _inspect(args: argparse.Namespace) -> int:
    artifact = Path(args.artifact)
    manifest = read_json(artifact / "manifest.json")
    graph = read_json(artifact / manifest["files"]["graph"])
    print(json.dumps(
        {
            "name": manifest["name"],
            "target": manifest["target"],
            "runtime_abi_version": manifest["runtime_abi_version"],
            "inputs": graph["inputs"],
            "outputs": graph["outputs"],
            "nodes": len(graph["nodes"]),
            "constants": len(graph["constants"]),
        },
        indent=2,
        sort_keys=True,
    ))
    return 0


def _validate(args: argparse.Namespace) -> int:
    module = _load_python_file(Path(args.against))
    for symbol in ("build_validation_inputs", "torch_reference"):
        if not hasattr(module, symbol):
            raise RuntimeError(f"{args.against} must define {symbol}()")
    inputs = module.build_validation_inputs()
    expected = module.torch_reference(inputs)
    if not isinstance(expected, dict):
        expected = {"output_0": expected}

    rt_module = runtime.load(args.artifact, load_constants=True)
    session = None
    try:
        session = rt_module.create_session()
        actual = session.run_numpy(inputs)
        failures = []
        for name, expected_array in expected.items():
            if name not in actual:
                failures.append(f"missing output {name}")
                continue
            expected_np = np.asarray(expected_array, dtype=np.float32)
            actual_np = np.asarray(actual[name], dtype=np.float32)
            max_abs = float(np.max(np.abs(actual_np - expected_np)))
            print(f"{name}: max_abs_diff={max_abs:.6g}")
            if not np.allclose(actual_np, expected_np, atol=args.atol, rtol=args.rtol):
                failures.append(f"{name} mismatch")
    finally:
        if session is not None:
            session.close()
        rt_module.close()
    if failures:
        raise RuntimeError("; ".join(failures))
    print("validation ok")
    return 0


def _benchmark(args: argparse.Namespace) -> int:
    rt_module = runtime.load(args.artifact, load_constants=True)
    session = None
    try:
        input_specs = rt_module.metadata["inputs"]
        if args.against is None:
            if input_specs:
                raise RuntimeError("benchmark requires --against for artifacts with inputs")
            inputs = {}
        else:
            model_module = _load_python_file(Path(args.against))
            if not hasattr(model_module, "build_validation_inputs"):
                raise RuntimeError(f"{args.against} must define build_validation_inputs()")
            inputs = model_module.build_validation_inputs()
        session = rt_module.create_session()
        summary = session.benchmark_numpy(inputs, warmup=args.warmup, iterations=args.iterations)
        report = {
            "artifact": str(Path(args.artifact)),
            "target": rt_module.target_name,
            "inputs": [
                {
                    "name": str(spec["name"]),
                    "shape": list(inputs[str(spec["name"])].shape) if str(spec["name"]) in inputs else list(spec["shape"]),
                    "dtype": str(spec["dtype"]),
                }
                for spec in input_specs
            ],
            "outputs": [
                {
                    "name": str(spec["name"]),
                    "shape": list(spec["shape"]),
                    "dtype": str(spec["dtype"]),
                }
                for spec in rt_module.metadata["outputs"]
            ],
            "session_run": summary,
        }
    finally:
        if session is not None:
            session.close()
        rt_module.close()
    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


def _benchmark_ops(args: argparse.Namespace) -> int:
    report = run_benchmark_suite(
        args.target,
        output_dir=args.artifacts,
        warmup=args.warmup,
        iterations=args.iterations,
        profile=args.profile,
        profile_iterations=args.profile_iterations,
        profile_repeats=args.profile_repeats,
        profile_refresh=args.profile_refresh,
        only=args.only or None,
        arch=args.arch,
        no_tf32=args.no_tf32,
        use_fp16_acc=args.use_fp16_acc,
        keep_artifacts=args.keep_artifacts,
        fail_fast=args.fail_fast,
        jobs=args.jobs,
    )
    if args.out:
        write_report(report, args.out)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["summary"]["error"] == 0 else 1


def _benchmark_torch_ops(args: argparse.Namespace) -> int:
    report = run_torch_benchmark_suite(
        device=args.device,
        warmup=args.warmup,
        iterations=args.iterations,
        only=args.only or None,
        fail_fast=args.fail_fast,
    )
    if args.out:
        write_torch_report(report, args.out)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["summary"]["error"] == 0 else 1


def _profile(args: argparse.Namespace) -> int:
    report = profile_artifact(
        args.artifact,
        input_shapes=parse_shape_overrides(args.shape),
        iterations=args.iterations,
        repeats=args.repeats,
        output=args.out,
        execution_plan_output=args.execution_plan_out,
        refresh=args.refresh,
    )
    print(json.dumps(
        {
            "artifact": report["artifact"],
            "target": report["target"],
            "iterations": report["iterations"],
            "repeats": report.get("repeats", 1),
            "problems": [
                _profile_problem_summary(item)
                for item in report["problems"]
            ],
            "blocked_profile_items": [
                _profile_blocked_item_summary(item)
                for item in report.get("blocked_profile_items", [])
            ],
            "execution_plan": report.get("execution_plan"),
            "summary": report["summary"],
        },
        indent=2,
        sort_keys=True,
    ))
    return 0


def _profile_problem_summary(item: Mapping[str, Any]) -> dict[str, Any]:
    shape = item.get("shape", {})
    selected = item.get("selected", {})
    candidate_id = item.get("candidate_id")
    if candidate_id is None and isinstance(selected, Mapping):
        candidate_id = selected.get("candidate_id")
    payload = {
        "node_id": item["node_id"],
        "op": item["op"],
        "dtype": item["dtype"],
        "kernel_library": item.get("kernel_library"),
        "candidate_id": candidate_id,
        "profiler_symbol": item["profiler_symbol"],
        "shape": dict(shape) if isinstance(shape, Mapping) else {},
        "elapsed_ms": item["elapsed_ms"],
        "workspace_nbytes": item.get("workspace_nbytes", 0),
        "timing": item.get("timing"),
        "tflops": item["tflops"],
    }
    if item.get("iterations") is not None:
        payload["iterations"] = item.get("iterations")
    if item.get("requested_iterations") is not None:
        payload["requested_iterations"] = item.get("requested_iterations")
    adaptive = item.get("adaptive_iterations")
    if isinstance(adaptive, Mapping):
        payload["adaptive_iterations"] = dict(adaptive)
    if item.get("split_k") is not None:
        payload["split_k"] = item.get("split_k")
    conv_config = item.get("conv")
    if isinstance(conv_config, Mapping):
        payload["conv"] = dict(conv_config)
    return payload


def _profile_blocked_item_summary(item: Mapping[str, Any]) -> dict[str, Any]:
    details = item.get("details")
    return {
        "op": item.get("op"),
        "dtype": item.get("dtype"),
        "kernel_library": item.get("kernel_library"),
        "candidate_id": item.get("selected_candidate_id"),
        "candidate_set_id": item.get("candidate_set_id"),
        "kernel_symbol": item.get("kernel_symbol"),
        "profiler_symbol": item.get("profiler_symbol"),
        "reason": item.get("reason"),
        "details": dict(details) if isinstance(details, Mapping) else {},
    }


def _load_python_file(path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(path.stem, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


if __name__ == "__main__":
    raise SystemExit(main())
