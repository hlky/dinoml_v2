from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import math
import shutil
import statistics
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Mapping

import numpy as np

import dinoml as dml
from dinoml import runtime
from dinoml.ir import dtype_runtime_enum
from dinoml.runtime import _DinoTensor, _make_dino_tensor

try:
    import torch
except Exception as exc:  # pragma: no cover
    torch = None
    TORCH_IMPORT_ERROR = repr(exc)
else:
    TORCH_IMPORT_ERROR = None


@dataclass(frozen=True)
class ShapeCase:
    name: str
    shape: tuple[int, ...]
    notes: str


SHAPE_CASES = {
    "tiny": ShapeCase("tiny", (8, 32), "small smoke case"),
    "channels_320": ShapeCase("channels_320", (8192, 320), "UNet-style channel reduction"),
    "tokens_1024": ShapeCase("tokens_1024", (1024, 1024), "attention-row sized reduction"),
    "llm_hidden_4096": ShapeCase("llm_hidden_4096", (2048, 4096), "LLM hidden-dim reduction"),
    "dit_hidden_1152": ShapeCase("dit_hidden_1152", (8192, 1152), "DiT token hidden-dim reduction"),
}

SUITES = {
    "quick": ("tiny", "channels_320", "tokens_1024"),
    "roofline": ("channels_320", "tokens_1024", "llm_hidden_4096", "dit_hidden_1152"),
}

OPS = ("reduce_sum", "reduce_max", "reduce_min", "reduce_mean")


class ReductionModel(dml.Module):
    def __init__(self, op_name: str):
        self.op_name = op_name

    def forward(self, x):
        op = getattr(dml.ops, self.op_name)
        return dml.ops.output(op(x, dim=-1), "y")


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark DinoML generated reduction kernels.")
    parser.add_argument("--out", type=Path, default=Path("tmp/benchmarks_reductions_profile"))
    parser.add_argument("--suite", choices=sorted(SUITES), default="quick")
    parser.add_argument("--shapes", default=None, help="Comma-separated shape case names.")
    parser.add_argument("--ops", default="reduce_sum,reduce_max,reduce_mean", help="Comma-separated reduction ops.")
    parser.add_argument("--targets", default="cpu,cuda", help="Comma-separated targets: cpu,cuda.")
    parser.add_argument("--arch", default="sm_86")
    parser.add_argument("--clean", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    shape_names = _split_arg(args.shapes) or list(SUITES[args.suite])
    op_names = _split_arg(args.ops) or ["reduce_sum", "reduce_max", "reduce_mean"]
    targets = _split_arg(args.targets) or ["cpu", "cuda"]

    unknown_ops = sorted(set(op_names) - set(OPS))
    if unknown_ops:
        raise ValueError(f"Unknown reduction ops: {unknown_ops}")

    if args.clean and args.out.exists():
        shutil.rmtree(args.out)
    args.out.mkdir(parents=True, exist_ok=True)

    results: dict[str, object] = {
        "suite": args.suite,
        "targets": targets,
        "ops": op_names,
        "cuda_available": bool(torch is not None and torch.cuda.is_available()),
        "torch_version": getattr(torch, "__version__", None) if torch is not None else None,
        "torch_import_error": TORCH_IMPORT_ERROR,
        "torch_num_threads": torch.get_num_threads() if torch is not None else None,
        "cases": {},
    }

    for shape_name in shape_names:
        shape_case = SHAPE_CASES[shape_name]
        for op_name in op_names:
            key = f"{op_name}__{shape_case.name}"
            print(f"[bench] {key}", flush=True)
            results["cases"][key] = run_case(
                out_dir=args.out,
                shape_case=shape_case,
                op_name=op_name,
                targets=targets,
                arch=args.arch,
            )

    result_path = args.out / "reductions_profile.json"
    result_path.write_text(json.dumps(results, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(results, indent=2, sort_keys=True))
    print(f"[bench] wrote {result_path}")


def run_case(*, out_dir: Path, shape_case: ShapeCase, op_name: str, targets: list[str], arch: str) -> dict[str, object]:
    seed = int.from_bytes(hashlib.sha256(f"{op_name}:{shape_case.name}".encode("utf-8")).digest()[:8], "little")
    rng = np.random.default_rng(seed)
    x = rng.standard_normal(shape_case.shape).astype(np.float32)
    expected = numpy_reference(op_name, x)
    bytes_info = estimate_reduction_bytes(x.size, expected.size)

    case: dict[str, object] = {
        "shape": list(shape_case.shape),
        "notes": shape_case.notes,
        "op": op_name,
        "numel": int(x.size),
        "cols": int(shape_case.shape[-1]),
        "rows": int(x.size // shape_case.shape[-1]),
        "kernel_bytes": bytes_info,
        "timings_ms": {},
        "throughput_gbs": {},
        "correctness": {},
        "artifacts": {},
    }
    spec = build_spec(shape_case.shape, op_name)

    if "cpu" in targets:
        artifact = out_dir / f"{op_name}__{shape_case.name}__cpu.dinoml"
        dml.compile(spec, dml.Target("cpu"), artifact)
        copy_generated_source(artifact, out_dir / "generated_review" / f"{op_name}__{shape_case.name}__cpu")
        timings, actual = run_cpu_hot(artifact, x, expected.shape)
        add_timing(case, "dinoml_cpu_hot_c_abi", timings, bytes_info)
        case["correctness"]["dinoml_cpu_hot_vs_numpy_max_abs"] = max_abs(actual, expected)
        timings, actual = run_cpu_numpy(artifact, x)
        add_timing(case, "dinoml_cpu_run_numpy_e2e", timings, bytes_info)
        case["correctness"]["dinoml_cpu_runtime_vs_numpy_max_abs"] = max_abs(actual, expected)
        case["artifacts"]["cpu"] = str(artifact)

    if "cuda" in targets and torch is not None and torch.cuda.is_available():
        artifact = out_dir / f"{op_name}__{shape_case.name}__cuda.dinoml"
        dml.compile(spec, dml.Target("cuda", arch=arch), artifact)
        copy_generated_source(artifact, out_dir / "generated_review" / f"{op_name}__{shape_case.name}__cuda")
        timings, actual = run_cuda_hot(artifact, x, expected.shape)
        add_timing(case, "dinoml_cuda_hot_c_abi", timings, bytes_info)
        case["correctness"]["dinoml_cuda_hot_vs_numpy_max_abs"] = max_abs(actual, expected)
        timings, actual = run_cuda_numpy(artifact, x)
        add_timing(case, "dinoml_cuda_run_numpy_e2e_cached", timings, bytes_info)
        case["correctness"]["dinoml_cuda_runtime_vs_numpy_max_abs"] = max_abs(actual, expected)
        case["artifacts"]["cuda"] = str(artifact)

    timings, actual = run_numpy(op_name, x)
    case["timings_ms"]["numpy_e2e"] = timings
    case["correctness"]["numpy_self_check_max_abs"] = max_abs(actual, expected)

    if torch is not None:
        timings, actual = run_torch_cpu(op_name, x)
        case["timings_ms"]["torch_cpu_e2e"] = timings
        case["correctness"]["torch_cpu_vs_numpy_max_abs"] = max_abs(actual, expected)
        if torch.cuda.is_available():
            timings, actual = run_torch_cuda(op_name, x)
            case["timings_ms"]["torch_cuda_hot"] = timings
            add_timing(case, "torch_cuda_hot", timings, bytes_info)
            case["correctness"]["torch_cuda_vs_numpy_max_abs"] = max_abs(actual, expected)

    case["speedups"] = speedups(case["timings_ms"])
    return case


def build_spec(shape: tuple[int, ...], op_name: str):
    return dml.trace(ReductionModel(op_name), inputs={"x": dml.TensorSpec(list(shape), "float32")}, name=f"bench_{op_name}")


def numpy_reference(op_name: str, x: np.ndarray) -> np.ndarray:
    if op_name == "reduce_sum":
        return np.sum(x, axis=-1).astype(np.float32)
    if op_name == "reduce_max":
        return np.max(x, axis=-1).astype(np.float32)
    if op_name == "reduce_min":
        return np.min(x, axis=-1).astype(np.float32)
    if op_name == "reduce_mean":
        return np.mean(x, axis=-1).astype(np.float32)
    raise ValueError(op_name)


def run_cpu_hot(artifact: Path, x: np.ndarray, output_shape: tuple[int, ...]) -> tuple[dict[str, float | int], np.ndarray]:
    module = runtime.load(artifact)
    session = module.create_session()
    output = np.empty(output_shape, dtype=np.float32)
    shape_buffers = []
    input_tensors = (_DinoTensor * 1)()
    tensor, keepalive = _make_dino_tensor(
        ctypes.c_void_p(x.ctypes.data),
        x.shape,
        dtype_runtime_enum("float32"),
        nbytes=x.nbytes,
        device_type=runtime.DINO_DEVICE_CPU,
    )
    shape_buffers.extend(keepalive)
    input_tensors[0] = tensor
    output_tensors = (_DinoTensor * 1)()
    tensor, keepalive = _make_dino_tensor(
        ctypes.c_void_p(output.ctypes.data),
        output.shape,
        dtype_runtime_enum("float32"),
        nbytes=output.nbytes,
        device_type=runtime.DINO_DEVICE_CPU,
    )
    shape_buffers.extend(keepalive)
    output_tensors[0] = tensor

    def run() -> None:
        module._check(
            module._dll.dino_session_run(
                session._handle,
                input_tensors,
                ctypes.c_size_t(1),
                output_tensors,
                ctypes.c_size_t(1),
            )
        )

    timings = bench_ms(run, hot_iters(x.size), warmup=warmup_iters(x.size))
    run()
    session.close()
    module.close()
    return timings, output.copy()


def run_cpu_numpy(artifact: Path, x: np.ndarray) -> tuple[dict[str, float | int], np.ndarray]:
    module = runtime.load(artifact)
    session = module.create_session()
    output = session.run_numpy({"x": x})["y"]
    timings = bench_ms(lambda: session.run_numpy({"x": x}), e2e_iters(x.size), warmup=warmup_iters(x.size))
    output = session.run_numpy({"x": x})["y"]
    session.close()
    module.close()
    return timings, output


def run_cuda_hot(artifact: Path, x: np.ndarray, output_shape: tuple[int, ...]) -> tuple[dict[str, float | int], np.ndarray]:
    module = runtime.load(artifact)
    session = module.create_session()
    output = np.empty(output_shape, dtype=np.float32)
    device_ptrs = []
    try:
        x_ptr = session._device_malloc(x.nbytes)
        y_ptr = session._device_malloc(output.nbytes)
        device_ptrs.extend([x_ptr, y_ptr])
        session._copy_h2d(x_ptr, x)
        shape_buffers = []
        input_tensors = (_DinoTensor * 1)()
        tensor, keepalive = _make_dino_tensor(
            x_ptr,
            x.shape,
            dtype_runtime_enum("float32"),
            nbytes=x.nbytes,
            device_type=runtime.DINO_DEVICE_CUDA,
        )
        shape_buffers.extend(keepalive)
        input_tensors[0] = tensor
        output_tensors = (_DinoTensor * 1)()
        tensor, keepalive = _make_dino_tensor(
            y_ptr,
            output.shape,
            dtype_runtime_enum("float32"),
            nbytes=output.nbytes,
            device_type=runtime.DINO_DEVICE_CUDA,
        )
        shape_buffers.extend(keepalive)
        output_tensors[0] = tensor

        def run() -> None:
            module._check(
                module._dll.dino_session_run(
                    session._handle,
                    input_tensors,
                    ctypes.c_size_t(1),
                    output_tensors,
                    ctypes.c_size_t(1),
                )
            )

        timings = bench_ms(run, cuda_hot_iters(x.size), warmup=warmup_iters(x.size))
        run()
        session._copy_d2h(output, y_ptr)
    finally:
        for ptr in reversed(device_ptrs):
            module._check(module._cuda_runtime_dll.dino_device_free(ptr))
        session.close()
        module.close()
    return timings, output.copy()


def run_cuda_numpy(artifact: Path, x: np.ndarray) -> tuple[dict[str, float | int], np.ndarray]:
    module = runtime.load(artifact)
    session = module.create_session()
    output = session.run_numpy({"x": x})["y"]
    timings = bench_ms(lambda: session.run_numpy({"x": x}), cuda_e2e_iters(x.size), warmup=warmup_iters(x.size))
    output = session.run_numpy({"x": x})["y"]
    session.close()
    module.close()
    return timings, output


def run_numpy(op_name: str, x: np.ndarray) -> tuple[dict[str, float | int], np.ndarray]:
    output = numpy_reference(op_name, x)
    timings = bench_ms(lambda: numpy_reference(op_name, x), e2e_iters(x.size), warmup=warmup_iters(x.size))
    return timings, output


def run_torch_cpu(op_name: str, x: np.ndarray) -> tuple[dict[str, float | int], np.ndarray]:
    tensor = torch.from_numpy(x)
    output = torch_reference(op_name, tensor)
    timings = bench_ms(lambda: torch_reference(op_name, tensor), e2e_iters(x.size), warmup=warmup_iters(x.size))
    return timings, output.numpy().astype(np.float32)


def run_torch_cuda(op_name: str, x: np.ndarray) -> tuple[dict[str, float | int], np.ndarray]:
    tensor = torch.from_numpy(x).cuda()
    output = torch_reference(op_name, tensor)
    timings = bench_torch_cuda_ms(lambda: torch_reference(op_name, tensor), cuda_hot_iters(x.size), warmup=warmup_iters(x.size))
    return timings, output.cpu().numpy().astype(np.float32)


def torch_reference(op_name: str, tensor: "torch.Tensor"):
    if op_name == "reduce_sum":
        return torch.sum(tensor, dim=-1)
    if op_name == "reduce_max":
        return torch.amax(tensor, dim=-1)
    if op_name == "reduce_min":
        return torch.amin(tensor, dim=-1)
    if op_name == "reduce_mean":
        return torch.mean(tensor, dim=-1)
    raise ValueError(op_name)


def estimate_reduction_bytes(input_numel: int, output_numel: int) -> dict[str, int | float]:
    total = (input_numel + output_numel) * 4
    return {
        "logical_bytes": total,
        "unique_floor_bytes": total,
        "logical_gb": total / 1.0e9,
        "unique_floor_gb": total / 1.0e9,
        "notes": "single input read plus output write; arithmetic and cache effects excluded",
    }


def add_timing(case: dict[str, object], name: str, timings: Mapping[str, float | int], bytes_info: Mapping[str, int | float]) -> None:
    case["timings_ms"][name] = dict(timings)
    median_ms = float(timings["median_ms"])
    if median_ms <= 0:
        return
    case["throughput_gbs"][name] = {
        "logical_gbs": float(bytes_info["logical_bytes"]) / (median_ms / 1000.0) / 1.0e9,
        "unique_floor_gbs": float(bytes_info["unique_floor_bytes"]) / (median_ms / 1000.0) / 1.0e9,
    }


def speedups(timings: Mapping[str, object]) -> dict[str, float]:
    medians = {name: float(value["median_ms"]) for name, value in timings.items() if isinstance(value, Mapping)}
    out: dict[str, float] = {}
    if "torch_cuda_hot" in medians and "dinoml_cuda_hot_c_abi" in medians:
        out["dinoml_cuda_vs_torch_cuda"] = medians["torch_cuda_hot"] / medians["dinoml_cuda_hot_c_abi"]
    if "torch_cpu_e2e" in medians and "dinoml_cpu_hot_c_abi" in medians:
        out["dinoml_cpu_hot_vs_torch_cpu"] = medians["torch_cpu_e2e"] / medians["dinoml_cpu_hot_c_abi"]
    if "numpy_e2e" in medians and "dinoml_cpu_hot_c_abi" in medians:
        out["dinoml_cpu_hot_vs_numpy"] = medians["numpy_e2e"] / medians["dinoml_cpu_hot_c_abi"]
    return out


def bench_ms(fn: Callable[[], object], iters: int, warmup: int) -> dict[str, float | int]:
    for _ in range(warmup):
        fn()
    vals = []
    for _ in range(iters):
        t0 = time.perf_counter()
        fn()
        vals.append((time.perf_counter() - t0) * 1000.0)
    return {"median_ms": statistics.median(vals), "min_ms": min(vals), "iters": iters}


def bench_torch_cuda_ms(fn: Callable[[], object], iters: int, warmup: int) -> dict[str, float | int]:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    vals = []
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    for _ in range(iters):
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        vals.append(float(start.elapsed_time(end)))
    return {"median_ms": statistics.median(vals), "min_ms": min(vals), "iters": iters}


def hot_iters(numel: int) -> int:
    if numel < 4096:
        return 2000
    if numel < 3_000_000:
        return 80
    return 20


def cuda_hot_iters(numel: int) -> int:
    if numel < 4096:
        return 2000
    if numel < 3_000_000:
        return 200
    return 60


def e2e_iters(numel: int) -> int:
    if numel < 4096:
        return 1000
    if numel < 3_000_000:
        return 30
    return 8


def cuda_e2e_iters(numel: int) -> int:
    if numel < 4096:
        return 200
    if numel < 3_000_000:
        return 30
    return 8


def warmup_iters(numel: int) -> int:
    return 30 if numel < 4096 else 5


def max_abs(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.max(np.abs(a.astype(np.float32) - b.astype(np.float32))))


def copy_generated_source(artifact: Path, out_dir: Path) -> None:
    generated = artifact / "debug" / "generated_src"
    if out_dir.exists():
        shutil.rmtree(out_dir)
    shutil.copytree(generated, out_dir, ignore=shutil.ignore_patterns("build"))


def _split_arg(value: str | None) -> list[str]:
    if not value:
        return []
    return [item.strip() for item in value.split(",") if item.strip()]


if __name__ == "__main__":
    main()
