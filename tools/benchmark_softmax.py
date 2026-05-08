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
from dinoml.runtime import _DinoTensor, _shape_buffer

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
    "sd_cross_77": ShapeCase("sd_cross_77", (8192, 77), "Stable Diffusion cross-attention rows"),
    "sd_self_1024": ShapeCase("sd_self_1024", (1024, 1024), "UNet self-attention rows"),
    "dit_4096": ShapeCase("dit_4096", (4096, 4096), "DiT square attention rows"),
    "llm_prefill_2048": ShapeCase("llm_prefill_2048", (2048, 2048), "LLM prefill-like rows"),
    "llm_decode_8192": ShapeCase("llm_decode_8192", (1, 8192), "LLM decode single row"),
}

POLICY_K_BUCKETS = (4, 8, 16, 32, 33, 64, 77, 128, 256, 512, 1024, 1408, 1920, 2048, 3840, 4096, 8192, 12500)

SUITES = {
    "quick": ("tiny", "sd_cross_77", "sd_self_1024"),
    "roofline": ("sd_cross_77", "sd_self_1024", "llm_prefill_2048", "dit_4096", "llm_decode_8192"),
    "policy": tuple(f"k{k}" for k in POLICY_K_BUCKETS),
}


class SoftmaxModel(dml.Module):
    def forward(self, x):
        return dml.ops.output(dml.ops.softmax(x, dim=-1), "y")


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark DinoML softmax kernels.")
    parser.add_argument("--out", type=Path, default=Path("tmp/benchmarks_softmax_profile"))
    parser.add_argument("--suite", choices=sorted(SUITES), default="quick")
    parser.add_argument("--shapes", default=None, help="Comma-separated shape case names.")
    parser.add_argument("--targets", default="cpu,cuda", help="Comma-separated targets: cpu,cuda.")
    parser.add_argument("--arch", default="sm_86")
    parser.add_argument("--clean", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    shape_names = _split_arg(args.shapes) or list(SUITES[args.suite])
    targets = _split_arg(args.targets) or ["cpu", "cuda"]

    if args.clean and args.out.exists():
        shutil.rmtree(args.out)
    args.out.mkdir(parents=True, exist_ok=True)

    results: dict[str, object] = {
        "suite": args.suite,
        "targets": targets,
        "cuda_available": bool(torch is not None and torch.cuda.is_available()),
        "torch_version": getattr(torch, "__version__", None) if torch is not None else None,
        "torch_import_error": TORCH_IMPORT_ERROR,
        "torch_num_threads": torch.get_num_threads() if torch is not None else None,
        "cases": {},
    }

    for shape_name in shape_names:
        shape_case = shape_case_for_name(shape_name)
        print(f"[bench] softmax__{shape_case.name}", flush=True)
        results["cases"][shape_case.name] = run_case(
            out_dir=args.out,
            shape_case=shape_case,
            targets=targets,
            arch=args.arch,
        )

    result_path = args.out / "softmax_profile.json"
    result_path.write_text(json.dumps(results, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(results, indent=2, sort_keys=True))
    print(f"[bench] wrote {result_path}")


def run_case(*, out_dir: Path, shape_case: ShapeCase, targets: list[str], arch: str) -> dict[str, object]:
    seed = int.from_bytes(hashlib.sha256(shape_case.name.encode("utf-8")).digest()[:8], "little")
    rng = np.random.default_rng(seed)
    x = (rng.standard_normal(shape_case.shape).astype(np.float32) * np.float32(2.0))
    expected = numpy_reference(x)
    bytes_info = estimate_softmax_bytes(x.size)

    case: dict[str, object] = {
        "shape": list(shape_case.shape),
        "notes": shape_case.notes,
        "numel": int(x.size),
        "cols": int(shape_case.shape[-1]),
        "rows": int(x.size // shape_case.shape[-1]),
        "kernel_bytes": bytes_info,
        "timings_ms": {},
        "throughput_gbs": {},
        "correctness": {},
        "artifacts": {},
    }

    spec = build_spec(shape_case.shape)

    if "cpu" in targets:
        artifact = out_dir / f"softmax__{shape_case.name}__cpu.dinoml"
        dml.compile(spec, dml.Target("cpu"), artifact)
        copy_generated_source(artifact, out_dir / "generated_review" / f"softmax__{shape_case.name}__cpu")
        timings, actual = run_cpu_hot(artifact, x)
        add_timing(case, "dinoml_cpu_hot_c_abi", timings, bytes_info)
        case["correctness"]["dinoml_cpu_hot_vs_numpy_max_abs"] = max_abs(actual, expected)
        timings, actual = run_cpu_numpy(artifact, x)
        add_timing(case, "dinoml_cpu_run_numpy_e2e", timings, bytes_info)
        case["correctness"]["dinoml_cpu_runtime_vs_numpy_max_abs"] = max_abs(actual, expected)
        case["artifacts"]["cpu"] = str(artifact)

    if "cuda" in targets and torch is not None and torch.cuda.is_available():
        artifact = out_dir / f"softmax__{shape_case.name}__cuda.dinoml"
        dml.compile(spec, dml.Target("cuda", arch=arch), artifact)
        copy_generated_source(artifact, out_dir / "generated_review" / f"softmax__{shape_case.name}__cuda")
        timings, actual = run_cuda_hot(artifact, x)
        add_timing(case, "dinoml_cuda_hot_c_abi", timings, bytes_info)
        case["correctness"]["dinoml_cuda_hot_vs_numpy_max_abs"] = max_abs(actual, expected)
        timings, actual = run_cuda_numpy(artifact, x)
        add_timing(case, "dinoml_cuda_run_numpy_e2e_cached", timings, bytes_info)
        case["correctness"]["dinoml_cuda_runtime_vs_numpy_max_abs"] = max_abs(actual, expected)
        case["artifacts"]["cuda"] = str(artifact)

    timings, actual = run_numpy(x)
    case["timings_ms"]["numpy_e2e"] = timings
    case["correctness"]["numpy_self_check_max_abs"] = max_abs(actual, expected)

    if torch is not None:
        timings, actual = run_torch_cpu(x)
        case["timings_ms"]["torch_cpu_e2e"] = timings
        case["correctness"]["torch_cpu_vs_numpy_max_abs"] = max_abs(actual, expected)
        if torch.cuda.is_available():
            timings, actual = run_torch_cuda(x)
            case["timings_ms"]["torch_cuda_hot"] = timings
            add_timing(case, "torch_cuda_hot", timings, bytes_info)
            case["correctness"]["torch_cuda_vs_numpy_max_abs"] = max_abs(actual, expected)

    case["speedups"] = speedups(case["timings_ms"])
    return case


def shape_case_for_name(name: str) -> ShapeCase:
    if name in SHAPE_CASES:
        return SHAPE_CASES[name]
    if name.startswith("k") and name[1:].isdigit():
        k = int(name[1:])
        rows = _policy_rows(k)
        return ShapeCase(name, (rows, k), f"v1-style softmax K-policy bucket with {rows} rows")
    raise KeyError(f"Unknown softmax shape case: {name}")


def _policy_rows(k: int) -> int:
    target_elements = 1_048_576
    rows = max(1, target_elements // max(1, k))
    return min(8192, rows)


def build_spec(shape: tuple[int, ...]):
    return dml.trace(SoftmaxModel(), inputs={"x": dml.TensorSpec(list(shape), "float32")}, name="bench_softmax")


def numpy_reference(x: np.ndarray) -> np.ndarray:
    shifted = x - np.max(x, axis=-1, keepdims=True)
    exp = np.exp(shifted)
    return (exp / np.sum(exp, axis=-1, keepdims=True)).astype(np.float32)


def run_cpu_hot(artifact: Path, x: np.ndarray) -> tuple[dict[str, float | int], np.ndarray]:
    module = runtime.load(artifact)
    session = module.create_session()
    output = np.empty_like(x)
    input_shape = _shape_buffer(x.shape)
    output_shape = _shape_buffer(output.shape)
    input_tensors = (_DinoTensor * 1)()
    input_tensors[0] = _DinoTensor(ctypes.c_void_p(x.ctypes.data), input_shape, x.ndim, dtype_runtime_enum("float32"))
    output_tensors = (_DinoTensor * 1)()
    output_tensors[0] = _DinoTensor(ctypes.c_void_p(output.ctypes.data), output_shape, output.ndim, dtype_runtime_enum("float32"))

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

    timings = bench_ms(run, hot_iters(output.size), warmup=warmup_iters(output.size))
    run()
    session.close()
    module.close()
    return timings, output.copy()


def run_cpu_numpy(artifact: Path, x: np.ndarray) -> tuple[dict[str, float | int], np.ndarray]:
    module = runtime.load(artifact)
    session = module.create_session()
    output = session.run_numpy({"x": x})["y"]
    timings = bench_ms(lambda: session.run_numpy({"x": x}), e2e_iters(output.size), warmup=warmup_iters(output.size))
    output = session.run_numpy({"x": x})["y"]
    session.close()
    module.close()
    return timings, output


def run_cuda_hot(artifact: Path, x: np.ndarray) -> tuple[dict[str, float | int], np.ndarray]:
    module = runtime.load(artifact)
    session = module.create_session()
    output = np.empty_like(x)
    device_ptrs = []
    try:
        x_ptr = session._device_malloc(x.nbytes)
        y_ptr = session._device_malloc(output.nbytes)
        device_ptrs.extend([x_ptr, y_ptr])
        session._copy_h2d(x_ptr, x)
        input_shape = _shape_buffer(x.shape)
        output_shape = _shape_buffer(output.shape)
        input_tensors = (_DinoTensor * 1)()
        input_tensors[0] = _DinoTensor(x_ptr, input_shape, x.ndim, dtype_runtime_enum("float32"))
        output_tensors = (_DinoTensor * 1)()
        output_tensors[0] = _DinoTensor(y_ptr, output_shape, output.ndim, dtype_runtime_enum("float32"))

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

        timings = bench_ms(run, cuda_hot_iters(output.size), warmup=warmup_iters(output.size))
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
    timings = bench_ms(lambda: session.run_numpy({"x": x}), cuda_e2e_iters(output.size), warmup=warmup_iters(output.size))
    output = session.run_numpy({"x": x})["y"]
    session.close()
    module.close()
    return timings, output


def run_numpy(x: np.ndarray) -> tuple[dict[str, float | int], np.ndarray]:
    output = numpy_reference(x)
    timings = bench_ms(lambda: numpy_reference(x), e2e_iters(output.size), warmup=warmup_iters(output.size))
    return timings, output


def run_torch_cpu(x: np.ndarray) -> tuple[dict[str, float | int], np.ndarray]:
    tensor = torch.from_numpy(x)

    def run():
        return torch.softmax(tensor, dim=-1)

    output = run()
    timings = bench_ms(run, e2e_iters(output.numel()), warmup=warmup_iters(output.numel()))
    return timings, output.numpy().astype(np.float32)


def run_torch_cuda(x: np.ndarray) -> tuple[dict[str, float | int], np.ndarray]:
    tensor = torch.from_numpy(x).cuda()

    def run():
        return torch.softmax(tensor, dim=-1)

    output = run()
    timings = bench_torch_cuda_ms(run, cuda_hot_iters(output.numel()), warmup=warmup_iters(output.numel()))
    return timings, output.cpu().numpy().astype(np.float32)


def estimate_softmax_bytes(numel: int) -> dict[str, int | float]:
    logical = numel * 4 * 5
    unique_floor = numel * 4 * 2
    return {
        "logical_bytes": logical,
        "unique_floor_bytes": unique_floor,
        "logical_gb": logical / 1.0e9,
        "unique_floor_gb": unique_floor / 1.0e9,
        "notes": "logical assumes x read for max, x read plus y write for exp, y read/write for normalize",
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
