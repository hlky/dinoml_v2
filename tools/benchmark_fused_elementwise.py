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
from dinoml.ir import dtype_runtime_enum, read_json
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


@dataclass(frozen=True)
class Workload:
    name: str
    description: str


SHAPE_CASES = {
    "tiny": ShapeCase("tiny", (2, 3, 4)),
    "unet_cl_64_64_320": ShapeCase("unet_cl_64_64_320", (2, 64, 64, 320)),
    "unet_cl_32_32_1280": ShapeCase("unet_cl_32_32_1280", (2, 32, 32, 1280)),
    "llm_1_2048_4096": ShapeCase("llm_1_2048_4096", (1, 2048, 4096)),
    "dit_2_4096_1152": ShapeCase("dit_2_4096_1152", (2, 4096, 1152)),
}

WORKLOADS = {
    "scale_bias_relu": Workload("scale_bias_relu", "memory-oriented x * scale + bias followed by relu"),
    "sigmoid_chain": Workload("sigmoid_chain", "x * scale + bias - sigmoid(x), relu, scalar multiply"),
    "gelu_silu": Workload("gelu_silu", "activation-heavy gelu(x) + silu(x)"),
    "binary_full_relu": Workload("binary_full_relu", "two full tensors with add/mul/relu"),
    "scalar_chain": Workload("scalar_chain", "scalar constants with add/mul/relu"),
}

SUITES = {
    "quick": {
        "shapes": ("tiny", "unet_cl_64_64_320", "llm_1_2048_4096"),
        "workloads": ("scale_bias_relu", "sigmoid_chain", "gelu_silu"),
    },
    "roofline": {
        "shapes": ("unet_cl_64_64_320", "unet_cl_32_32_1280", "llm_1_2048_4096", "dit_2_4096_1152"),
        "workloads": ("scale_bias_relu", "sigmoid_chain", "gelu_silu", "binary_full_relu", "scalar_chain"),
    },
}


class BenchmarkModel(dml.Module):
    def __init__(self, workload: str, channels: int):
        self.workload = workload
        if workload in {"scale_bias_relu", "sigmoid_chain"}:
            self.scale = dml.Parameter([channels], dtype="float32")
            self.bias = dml.Parameter([channels], dtype="float32")

    def forward(self, **inputs):
        x = inputs["x"]
        if self.workload == "scale_bias_relu":
            return dml.ops.output(dml.ops.relu(dml.ops.add(dml.ops.mul(x, self.scale), self.bias)), "y")
        if self.workload == "sigmoid_chain":
            y = dml.ops.mul(x, self.scale)
            y = dml.ops.add(y, self.bias)
            y = dml.ops.sub(y, dml.ops.sigmoid(x))
            y = dml.ops.relu(y)
            y = dml.ops.mul(y, 0.5)
            return dml.ops.output(y, "y")
        if self.workload == "gelu_silu":
            return dml.ops.output(dml.ops.add(dml.ops.gelu(x), dml.ops.silu(x)), "y")
        if self.workload == "binary_full_relu":
            z = inputs["z"]
            return dml.ops.output(dml.ops.relu(dml.ops.add(dml.ops.mul(x, z), x)), "y")
        if self.workload == "scalar_chain":
            y = dml.ops.mul(x, 1.125)
            y = dml.ops.add(y, 0.375)
            y = dml.ops.relu(y)
            return dml.ops.output(y, "y")
        raise ValueError(f"Unknown workload: {self.workload}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark DinoML fused_elementwise kernels.")
    parser.add_argument("--out", type=Path, default=Path("tmp/benchmarks_fused_elementwise_profile"))
    parser.add_argument("--suite", choices=sorted(SUITES), default="quick")
    parser.add_argument("--shapes", default=None, help="Comma-separated shape case names.")
    parser.add_argument("--workloads", default=None, help="Comma-separated workload names.")
    parser.add_argument("--targets", default="cpu,cuda", help="Comma-separated targets: cpu,cuda.")
    parser.add_argument("--arch", default="sm_86")
    parser.add_argument("--clean", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    shape_names = _split_arg(args.shapes) or list(SUITES[args.suite]["shapes"])
    workload_names = _split_arg(args.workloads) or list(SUITES[args.suite]["workloads"])
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
        shape_case = SHAPE_CASES[shape_name]
        for workload_name in workload_names:
            workload = WORKLOADS[workload_name]
            key = f"{workload.name}__{shape_case.name}"
            print(f"[bench] {key}", flush=True)
            results["cases"][key] = run_case(
                out_dir=args.out,
                shape_case=shape_case,
                workload=workload,
                targets=targets,
                arch=args.arch,
            )

    result_path = args.out / "fused_elementwise_profile.json"
    result_path.write_text(json.dumps(results, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(results, indent=2, sort_keys=True))
    print(f"[bench] wrote {result_path}")


def run_case(
    *,
    out_dir: Path,
    shape_case: ShapeCase,
    workload: Workload,
    targets: list[str],
    arch: str,
) -> dict[str, object]:
    seed = int.from_bytes(hashlib.sha256(f"{shape_case.name}:{workload.name}".encode("utf-8")).digest()[:8], "little")
    rng = np.random.default_rng(seed)
    constants = make_constants(workload.name, shape_case.shape, rng)
    inputs = make_inputs(workload.name, shape_case.shape, rng)
    spec = build_spec(shape_case.shape, workload.name, constants)
    expected = numpy_reference(workload.name, inputs, constants)

    case: dict[str, object] = {
        "description": workload.description,
        "shape": list(shape_case.shape),
        "numel": int(math.prod(shape_case.shape)),
        "timings_ms": {},
        "throughput_gbs": {},
        "correctness": {},
        "artifacts": {},
    }

    if "cpu" in targets:
        artifact = out_dir / f"{workload.name}__{shape_case.name}__cpu.dinoml"
        dml.compile(spec, dml.Target("cpu"), artifact)
        copy_generated_source(artifact, out_dir / "generated_review" / f"{workload.name}__{shape_case.name}__cpu")
        bytes_info = estimate_kernel_bytes(artifact)
        timings, actual = run_cpu_hot(artifact, inputs)
        add_timing(case, "dinoml_cpu_hot_c_abi", timings, bytes_info)
        case["correctness"]["dinoml_cpu_hot_vs_numpy_max_abs"] = max_abs(actual, expected)
        timings, actual = run_cpu_numpy(artifact, inputs)
        add_timing(case, "dinoml_cpu_run_numpy_e2e", timings, bytes_info)
        case["correctness"]["dinoml_cpu_runtime_vs_numpy_max_abs"] = max_abs(actual, expected)
        case["artifacts"]["cpu"] = str(artifact)

    if "cuda" in targets and torch is not None and torch.cuda.is_available():
        artifact = out_dir / f"{workload.name}__{shape_case.name}__cuda.dinoml"
        dml.compile(spec, dml.Target("cuda", arch=arch), artifact)
        copy_generated_source(artifact, out_dir / "generated_review" / f"{workload.name}__{shape_case.name}__cuda")
        bytes_info = estimate_kernel_bytes(artifact)
        timings, actual = run_cuda_hot(artifact, inputs)
        add_timing(case, "dinoml_cuda_hot_c_abi", timings, bytes_info)
        case["correctness"]["dinoml_cuda_hot_vs_numpy_max_abs"] = max_abs(actual, expected)
        timings, actual = run_cuda_numpy(artifact, inputs)
        add_timing(case, "dinoml_cuda_run_numpy_e2e_cached", timings, bytes_info)
        case["correctness"]["dinoml_cuda_runtime_vs_numpy_max_abs"] = max_abs(actual, expected)
        case["artifacts"]["cuda"] = str(artifact)

    timings, actual = run_numpy(workload.name, inputs, constants)
    case["timings_ms"]["numpy_e2e"] = timings
    case["correctness"]["numpy_self_check_max_abs"] = max_abs(actual, expected)

    if torch is not None:
        timings, actual = run_torch_cpu(workload.name, inputs, constants)
        case["timings_ms"]["torch_cpu_e2e"] = timings
        case["correctness"]["torch_cpu_vs_numpy_max_abs"] = max_abs(actual, expected)
        if torch.cuda.is_available():
            timings, actual = run_torch_cuda(workload.name, inputs, constants)
            case["timings_ms"]["torch_cuda_hot"] = timings
            case["correctness"]["torch_cuda_vs_numpy_max_abs"] = max_abs(actual, expected)

    case["kernel_bytes"] = estimate_kernel_bytes(Path(next(iter(case["artifacts"].values())))) if case["artifacts"] else {}
    return case


def build_spec(shape: tuple[int, ...], workload: str, constants: Mapping[str, np.ndarray]) -> dml.ir.ModelSpec:
    inputs = {"x": dml.TensorSpec(list(shape), "float32")}
    if workload == "binary_full_relu":
        inputs["z"] = dml.TensorSpec(list(shape), "float32")
    return dml.trace(
        BenchmarkModel(workload, shape[-1]),
        inputs=inputs,
        constants=constants,
        name=f"bench_{workload}",
    )


def make_constants(workload: str, shape: tuple[int, ...], rng: np.random.Generator) -> dict[str, np.ndarray]:
    if workload not in {"scale_bias_relu", "sigmoid_chain"}:
        return {}
    channels = shape[-1]
    return {
        "scale": rng.normal(loc=1.0, scale=0.2, size=[channels]).astype(np.float32),
        "bias": rng.normal(loc=0.0, scale=0.1, size=[channels]).astype(np.float32),
    }


def make_inputs(workload: str, shape: tuple[int, ...], rng: np.random.Generator) -> dict[str, np.ndarray]:
    inputs = {"x": rng.standard_normal(shape).astype(np.float32)}
    if workload == "binary_full_relu":
        inputs["z"] = rng.standard_normal(shape).astype(np.float32)
    return inputs


def numpy_reference(workload: str, inputs: Mapping[str, np.ndarray], constants: Mapping[str, np.ndarray]) -> np.ndarray:
    x = inputs["x"].astype(np.float32)
    if workload == "scale_bias_relu":
        return np.maximum(x * constants["scale"] + constants["bias"], 0.0).astype(np.float32)
    if workload == "sigmoid_chain":
        y = x * constants["scale"]
        y = y + constants["bias"]
        y = y - (1.0 / (1.0 + np.exp(-x)))
        y = np.maximum(y, 0.0)
        return (y * np.float32(0.5)).astype(np.float32)
    if workload == "gelu_silu":
        gelu = 0.5 * x * (1.0 + np.tanh(np.float32(0.7978845608028654) * (x + np.float32(0.044715) * x * x * x)))
        silu = x / (1.0 + np.exp(-x))
        return (gelu + silu).astype(np.float32)
    if workload == "binary_full_relu":
        return np.maximum(x * inputs["z"] + x, 0.0).astype(np.float32)
    if workload == "scalar_chain":
        return np.maximum(x * np.float32(1.125) + np.float32(0.375), 0.0).astype(np.float32)
    raise ValueError(f"Unknown workload: {workload}")


def run_cpu_hot(artifact: Path, inputs: Mapping[str, np.ndarray]) -> tuple[dict[str, float | int], np.ndarray]:
    module = runtime.load(artifact)
    session = module.create_session()
    input_specs = module.metadata["inputs"]
    output_specs = module.metadata["outputs"]
    output = np.empty(output_specs[0]["shape"], dtype=np.float32)
    shape_buffers = []
    input_tensors = (_DinoTensor * len(input_specs))()
    for idx, spec in enumerate(input_specs):
        array = inputs[spec["name"]]
        tensor, keepalive = _make_dino_tensor(
            ctypes.c_void_p(array.ctypes.data),
            spec["shape"],
            dtype_runtime_enum("float32"),
            nbytes=array.nbytes,
            device_type=runtime.DINO_DEVICE_CPU,
        )
        shape_buffers.extend(keepalive)
        input_tensors[idx] = tensor
    output_tensors = (_DinoTensor * 1)()
    tensor, keepalive = _make_dino_tensor(
        ctypes.c_void_p(output.ctypes.data),
        output_specs[0]["shape"],
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
                ctypes.c_size_t(len(input_specs)),
                output_tensors,
                ctypes.c_size_t(1),
            )
        )

    timings = bench_ms(run, hot_iters(output.size), warmup=warmup_iters(output.size))
    run()
    session.close()
    module.close()
    return timings, output.copy()


def run_cpu_numpy(artifact: Path, inputs: Mapping[str, np.ndarray]) -> tuple[dict[str, float | int], np.ndarray]:
    module = runtime.load(artifact)
    session = module.create_session()
    output = session.run_numpy(inputs)["y"]
    timings = bench_ms(lambda: session.run_numpy(inputs), e2e_iters(output.size), warmup=warmup_iters(output.size))
    output = session.run_numpy(inputs)["y"]
    session.close()
    module.close()
    return timings, output


def run_cuda_hot(artifact: Path, inputs: Mapping[str, np.ndarray]) -> tuple[dict[str, float | int], np.ndarray]:
    module = runtime.load(artifact)
    session = module.create_session()
    input_specs = module.metadata["inputs"]
    output_specs = module.metadata["outputs"]
    output = np.empty(output_specs[0]["shape"], dtype=np.float32)
    device_ptrs = []
    shape_buffers = []
    try:
        input_tensors = (_DinoTensor * len(input_specs))()
        for idx, spec in enumerate(input_specs):
            array = inputs[spec["name"]]
            ptr = session._device_malloc(array.nbytes)
            device_ptrs.append(ptr)
            session._copy_h2d(ptr, array)
            tensor, keepalive = _make_dino_tensor(
                ptr,
                spec["shape"],
                dtype_runtime_enum("float32"),
                nbytes=array.nbytes,
                device_type=runtime.DINO_DEVICE_CUDA,
            )
            shape_buffers.extend(keepalive)
            input_tensors[idx] = tensor
        output_ptr = session._device_malloc(output.nbytes)
        device_ptrs.append(output_ptr)
        output_tensors = (_DinoTensor * 1)()
        tensor, keepalive = _make_dino_tensor(
            output_ptr,
            output_specs[0]["shape"],
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
                    ctypes.c_size_t(len(input_specs)),
                    output_tensors,
                    ctypes.c_size_t(1),
                )
            )

        timings = bench_ms(run, cuda_hot_iters(output.size), warmup=warmup_iters(output.size))
        run()
        session._copy_d2h(output, output_ptr)
    finally:
        for ptr in reversed(device_ptrs):
            module._check(module._cuda_runtime_dll.dino_device_free(ptr))
        session.close()
        module.close()
    return timings, output.copy()


def run_cuda_numpy(artifact: Path, inputs: Mapping[str, np.ndarray]) -> tuple[dict[str, float | int], np.ndarray]:
    module = runtime.load(artifact)
    session = module.create_session()
    output = session.run_numpy(inputs)["y"]
    timings = bench_ms(lambda: session.run_numpy(inputs), cuda_e2e_iters(output.size), warmup=warmup_iters(output.size))
    output = session.run_numpy(inputs)["y"]
    session.close()
    module.close()
    return timings, output


def run_numpy(workload: str, inputs: Mapping[str, np.ndarray], constants: Mapping[str, np.ndarray]) -> tuple[dict[str, float | int], np.ndarray]:
    output = numpy_reference(workload, inputs, constants)
    timings = bench_ms(lambda: numpy_reference(workload, inputs, constants), e2e_iters(output.size), warmup=warmup_iters(output.size))
    return timings, output


def run_torch_cpu(workload: str, inputs: Mapping[str, np.ndarray], constants: Mapping[str, np.ndarray]) -> tuple[dict[str, float | int], np.ndarray]:
    tensors = {name: torch.from_numpy(value) for name, value in inputs.items()}
    consts = {name: torch.from_numpy(value) for name, value in constants.items()}

    def run():
        return torch_reference(workload, tensors, consts)

    output = run()
    timings = bench_ms(run, e2e_iters(output.numel()), warmup=warmup_iters(output.numel()))
    return timings, output.numpy().astype(np.float32)


def run_torch_cuda(workload: str, inputs: Mapping[str, np.ndarray], constants: Mapping[str, np.ndarray]) -> tuple[dict[str, float | int], np.ndarray]:
    tensors = {name: torch.from_numpy(value).cuda() for name, value in inputs.items()}
    consts = {name: torch.from_numpy(value).cuda() for name, value in constants.items()}

    def run():
        return torch_reference(workload, tensors, consts)

    output = run()
    timings = bench_torch_cuda_ms(run, cuda_hot_iters(output.numel()), warmup=warmup_iters(output.numel()))
    return timings, output.cpu().numpy().astype(np.float32)


def torch_reference(workload: str, tensors: Mapping[str, "torch.Tensor"], constants: Mapping[str, "torch.Tensor"]):
    x = tensors["x"]
    if workload == "scale_bias_relu":
        return torch.relu(x * constants["scale"] + constants["bias"])
    if workload == "sigmoid_chain":
        y = x * constants["scale"]
        y = y + constants["bias"]
        y = y - torch.sigmoid(x)
        y = torch.relu(y)
        return y * 0.5
    if workload == "gelu_silu":
        return torch.nn.functional.gelu(x, approximate="tanh") + torch.nn.functional.silu(x)
    if workload == "binary_full_relu":
        return torch.relu(x * tensors["z"] + x)
    if workload == "scalar_chain":
        return torch.relu(x * 1.125 + 0.375)
    raise ValueError(f"Unknown workload: {workload}")


def estimate_kernel_bytes(artifact: Path) -> dict[str, int | float]:
    graph = read_json(artifact / "graph.dinoir.json")
    tensors = {tensor["name"]: tensor for tensor in graph["tensors"]}
    total_logical = 0
    total_unique_floor = 0
    for node in graph["nodes"]:
        if node["op"] != "fused_elementwise":
            continue
        output_tensor = tensors[node["outputs"][0]]
        output_numel = int(math.prod(output_tensor["shape"]))
        output_bytes = output_numel * 4
        input_logical = len(node["inputs"]) * output_bytes
        input_unique = sum(int(math.prod(tensors[name]["shape"])) * 4 for name in node["inputs"])
        total_logical += input_logical + output_bytes
        total_unique_floor += input_unique + output_bytes
    return {
        "logical_bytes": total_logical,
        "unique_floor_bytes": total_unique_floor,
        "logical_gb": total_logical / 1.0e9,
        "unique_floor_gb": total_unique_floor / 1.0e9,
    }


def add_timing(case: dict[str, object], name: str, timings: dict[str, float | int], bytes_info: Mapping[str, int | float]) -> None:
    case["timings_ms"][name] = timings
    median_ms = float(timings["median_ms"])
    if median_ms <= 0:
        return
    case["throughput_gbs"][name] = {
        "logical_gbs": float(bytes_info["logical_bytes"]) / (median_ms / 1000.0) / 1.0e9,
        "unique_floor_gbs": float(bytes_info["unique_floor_bytes"]) / (median_ms / 1000.0) / 1.0e9,
    }


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
    return 30


def cuda_hot_iters(numel: int) -> int:
    if numel < 4096:
        return 2000
    if numel < 3_000_000:
        return 200
    return 80


def e2e_iters(numel: int) -> int:
    if numel < 4096:
        return 1000
    if numel < 3_000_000:
        return 30
    return 10


def cuda_e2e_iters(numel: int) -> int:
    if numel < 4096:
        return 200
    if numel < 3_000_000:
        return 30
    return 10


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
