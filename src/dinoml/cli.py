from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
from typing import Any

import numpy as np

import dinoml as dml
from dinoml import runtime
from dinoml.ir import read_json
from dinoml.kernels.profiling import parse_shape_overrides, profile_artifact


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="dinoml")
    subparsers = parser.add_subparsers(dest="command", required=True)

    compile_parser = subparsers.add_parser("compile")
    compile_parser.add_argument("model")
    compile_parser.add_argument("--target", default="cuda")
    compile_parser.add_argument("--arch", default="sm_86")
    compile_parser.add_argument("--no-tf32", action="store_true", help="Disable optional TF32 CUTLASS GEMM candidates")
    compile_parser.add_argument("--use-fp16-acc", action="store_true", help="Use fp16 accumulation for fp16 CUTLASS GEMM candidates")
    compile_parser.add_argument("--execution-plan", help="Apply a profile-selected execution_plan.json during compile")
    compile_parser.add_argument("--profile", action="store_true", help="Profile CUTLASS candidates and rebuild with the selected execution plan")
    compile_parser.add_argument("--profile-iterations", type=int, default=20)
    compile_parser.add_argument("--profile-shape", "--shape", dest="profile_shape", action="append", default=[])
    compile_parser.add_argument("--profile-refresh", action="store_true")
    compile_parser.add_argument("--out", required=True)

    inspect_parser = subparsers.add_parser("inspect")
    inspect_parser.add_argument("artifact")

    validate_parser = subparsers.add_parser("validate")
    validate_parser.add_argument("artifact")
    validate_parser.add_argument("--against", required=True)
    validate_parser.add_argument("--atol", type=float, default=1e-4)
    validate_parser.add_argument("--rtol", type=float, default=1e-4)

    profile_parser = subparsers.add_parser("profile")
    profile_parser.add_argument("artifact")
    profile_parser.add_argument("--iterations", type=int, default=20)
    profile_parser.add_argument("--shape", action="append", default=[])
    profile_parser.add_argument("--out")
    profile_parser.add_argument("--execution-plan-out")
    profile_parser.add_argument("--refresh", action="store_true")

    args = parser.parse_args(argv)
    if args.command == "compile":
        return _compile(args)
    if args.command == "inspect":
        return _inspect(args)
    if args.command == "validate":
        return _validate(args)
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
    }
    if args.profile:
        compile_kwargs.update(
            {
                "profile": True,
                "profile_iterations": args.profile_iterations,
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

    rt_module = runtime.load(args.artifact)
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
    session.close()
    rt_module.close()
    if failures:
        raise RuntimeError("; ".join(failures))
    print("validation ok")
    return 0


def _profile(args: argparse.Namespace) -> int:
    report = profile_artifact(
        args.artifact,
        input_shapes=parse_shape_overrides(args.shape),
        iterations=args.iterations,
        output=args.out,
        execution_plan_output=args.execution_plan_out,
        refresh=args.refresh,
    )
    print(json.dumps(
        {
            "artifact": report["artifact"],
            "target": report["target"],
            "iterations": report["iterations"],
            "problems": [
                {
                    "node_id": item["node_id"],
                    "op": item["op"],
                    "dtype": item["dtype"],
                    "profiler_symbol": item["profiler_symbol"],
                    "shape": {"m": item["m"], "n": item["n"], "k": item["k"]},
                    "elapsed_ms": item["elapsed_ms"],
                    "tflops": item["tflops"],
                }
                for item in report["problems"]
            ],
            "execution_plan": report.get("execution_plan"),
            "summary": report["summary"],
        },
        indent=2,
        sort_keys=True,
    ))
    return 0


def _load_python_file(path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(path.stem, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


if __name__ == "__main__":
    raise SystemExit(main())
