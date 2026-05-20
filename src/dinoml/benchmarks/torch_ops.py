from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping

import numpy as np

from dinoml.benchmarks.ops import BenchmarkCase, _select_cases, benchmark_cases
from dinoml.runtime import _summarize_benchmark_samples, _validate_benchmark_counts


@dataclass(frozen=True)
class TorchBenchmarkCase:
    name: str
    inputs: Callable[[], dict[str, np.ndarray]]
    fn: Callable[..., Any]
    op: str
    template: str


def torch_benchmark_cases() -> list[TorchBenchmarkCase]:
    torch_fns = _torch_case_fns()
    cases: list[TorchBenchmarkCase] = []
    for case in benchmark_cases():
        if case.name in _EXCLUDED_TORCH_CASES:
            continue
        if case.name not in torch_fns:
            raise RuntimeError(f"Missing PyTorch benchmark implementation for {case.name}")
        cases.append(
            TorchBenchmarkCase(
                name=case.name,
                inputs=case.inputs,
                fn=torch_fns[case.name],
                op=case.op,
                template=case.template,
            )
        )
    return cases


_EXCLUDED_TORCH_CASES = frozenset()


def run_torch_benchmark_suite(
    *,
    device: str = "cpu",
    warmup: int = 5,
    iterations: int = 20,
    only: Iterable[str] | None = None,
    fail_fast: bool = False,
) -> dict[str, Any]:
    torch = _import_torch()
    warmup_count, iteration_count = _validate_benchmark_counts(warmup, iterations)
    selected = _select_torch_cases(torch_benchmark_cases(), only)
    torch_device = _normalize_device(torch, device)

    results: list[dict[str, Any]] = []
    started = time.perf_counter()
    for idx, case in enumerate(selected):
        print(f"torch::{case.op}::{case.name} [{idx+1}/{len(selected)}]")
        case_started = time.perf_counter()
        try:
            tensors = _torch_inputs(torch, case.inputs(), torch_device)
            summary = _benchmark_torch_call(
                torch,
                torch_device,
                lambda: case.fn(torch=torch, device=torch_device, **tensors),
                warmup=warmup_count,
                iterations=iteration_count,
            )
            outputs = _output_report(case.fn(torch=torch, device=torch_device, **tensors))
            results.append(
                {
                    "name": case.name,
                    "op": case.op,
                    "template": case.template,
                    "framework": "torch",
                    "status": "ok",
                    "inputs": _input_report(tensors),
                    "outputs": outputs,
                    "session_run": summary,
                    "elapsed_s": time.perf_counter() - case_started,
                }
            )
            print(f"torch::{case.op}::{case.name} {summary=}")
        except Exception as exc:
            results.append(
                {
                    "name": case.name,
                    "op": case.op,
                    "template": case.template,
                    "framework": "torch",
                    "status": "error",
                    "error": f"{type(exc).__name__}: {exc}",
                    "elapsed_s": time.perf_counter() - case_started,
                }
            )
            if fail_fast:
                raise

    ok = sum(1 for item in results if item["status"] == "ok")
    return {
        "target": {"framework": "torch", "device": str(torch_device)},
        "warmup": warmup_count,
        "iterations": iteration_count,
        "summary": {
            "total": len(results),
            "ok": ok,
            "error": len(results) - ok,
            "elapsed_s": time.perf_counter() - started,
        },
        "cases": results,
    }


def write_torch_report(report: Mapping[str, Any], output: str | Path) -> None:
    out_path = Path(output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")


def _torch_case_fns() -> dict[str, Callable[..., Any]]:
    return {
        "add": lambda x, y, **_: x + y,
        "sub": lambda x, y, **_: x - y,
        "mul": lambda x, y, **_: x * y,
        "div": lambda x, y, **_: x / (y + 2.0),
        "tanh": lambda x, **_: x.tanh(),
        "cos": lambda x, **_: x.cos(),
        "sin": lambda x, **_: x.sin(),
        "sign": lambda x, **_: x.sign(),
        "abs": lambda x, **_: x.abs(),
        "log": lambda positive, **_: positive.log(),
        "log1p": lambda positive, **_: positive.log1p(),
        "exp": lambda x, **_: (x * 0.1).exp(),
        "sqrt": lambda positive, **_: positive.sqrt(),
        "max": lambda torch, x, y, **_: torch.maximum(x, y),
        "min": lambda torch, x, y, **_: torch.minimum(x, y),
        "sigmoid": lambda x, **_: x.sigmoid(),
        "leaky_relu": lambda torch, x, **_: torch.nn.functional.leaky_relu(x, negative_slope=0.2),
        "hardtanh": lambda torch, x, **_: torch.nn.functional.hardtanh(x, min_val=-0.5, max_val=0.75),
        "relu": lambda torch, x, **_: torch.relu(x),
        "nan_to_num": lambda torch, x, **_: torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0),
        "clamp_nan_to_num": lambda torch, x, **_: torch.nan_to_num(x, nan=0.0).clamp(-1.0, 1.0),
        "silu": lambda torch, x, **_: torch.nn.functional.silu(x),
        "pow": lambda positive, y, **_: positive.pow(y + 2.0),
        "gelu": lambda torch, x, **_: torch.nn.functional.gelu(x, approximate="tanh"),
        "gelu_new": lambda torch, x, **_: _gelu_new(torch, x),
        "fast_gelu": lambda torch, x, **_: 0.5 * x * (1.0 + torch.tanh(0.7978845608 * x * (1.0 + 0.044715 * x * x))),
        "softplus": lambda torch, x, **_: torch.nn.functional.softplus(x),
        "elu": lambda torch, x, **_: torch.nn.functional.elu(x, alpha=1.25),
        "softsign": lambda torch, x, **_: torch.nn.functional.softsign(x),
        "floor_div": lambda torch, x, y, **_: torch.floor((x + 4.0) / (y + 2.0)),
        "celu": lambda torch, x, **_: torch.nn.functional.celu(x, alpha=1.1),
        "floor": lambda x, **_: x.floor(),
        "eq": lambda x, y, **_: x.eq(y),
        "ge": lambda x, y, **_: x.ge(y),
        "gt": lambda x, y, **_: x.gt(y),
        "le": lambda x, y, **_: x.le(y),
        "lt": lambda x, y, **_: x.lt(y),
        "ne": lambda x, y, **_: x.ne(y),
        "where": lambda torch, x, y, condition, **_: torch.where(condition, x, y),
        "cast": lambda condition, **_: condition.float(),
        "full": lambda torch, device, **_: torch.full((1024, 4096), 1.25, dtype=torch.float32, device=device),
        "arange": lambda torch, device, **_: torch.arange(0, 4_194_304, 1, dtype=torch.float32, device=device),
        "randn": lambda torch, device, **_: _randn(torch, (1024, 4096), device=device, seed=17),
        "softmax": lambda torch, x, **_: torch.nn.functional.softmax(x, dim=-1),
        "reduce_sum": lambda x, **_: x.sum(dim=-1),
        "reduce_max": lambda x, **_: x.max(dim=-1).values,
        "reduce_min": lambda x, **_: x.min(dim=-1).values,
        "reduce_mean": lambda x, **_: x.mean(dim=-1),
        "var": lambda x, **_: x.var(dim=-1, unbiased=False),
        "vector_norm": lambda torch, x, **_: torch.linalg.vector_norm(x, dim=-1),
        "avg_pool1d": lambda torch, x, **_: torch.nn.functional.avg_pool1d(x, kernel_size=3, stride=2, padding=1),
        "avg_pool2d": lambda torch, x, **_: torch.nn.functional.avg_pool2d(x, kernel_size=(3, 3), stride=2, padding=1),
        "max_pool2d": lambda torch, x, **_: torch.nn.functional.max_pool2d(x, kernel_size=(3, 3), stride=2, padding=1),
        "argmax": lambda x, **_: x.argmax(dim=-1),
        "topk": lambda x, **_: x.topk(16, dim=-1),
        "layer_norm": lambda torch, x, weight, bias, **_: torch.nn.functional.layer_norm(x, (x.shape[-1],), weight, bias, eps=1e-5),
        "t5_layer_norm": lambda torch, x, weight, **_: _t5_layer_norm(torch, x, weight, eps=1e-6),
        "rms_norm": lambda torch, x, weight, **_: _t5_layer_norm(torch, x, weight, eps=1e-6),
        "get_timestep_embedding": lambda torch, timesteps, **_: _get_timestep_embedding(torch, timesteps, embedding_dim=128),
        "get_1d_rotary_pos_embed": lambda torch, positions, **_: _get_1d_rotary_pos_embed(torch, 128, positions),
        "embedding": lambda torch, table, indices, **_: torch.nn.functional.embedding(indices, table),
        "expand": lambda x, **_: x.expand(16, 128, 768),
        "concatenate": lambda torch, x, y, **_: torch.cat([x, y], dim=1),
        "stack": lambda torch, x, y, **_: torch.stack([x, y], dim=0),
        "flip": lambda x, **_: x.flip((-1,)),
        "repeat_interleave": lambda torch, x, **_: torch.repeat_interleave(x, repeats=2, dim=1),
        "permute": lambda x, **_: x.permute(1, 0, 2),
        "permute021": lambda x, **_: x.permute(0, 2, 1),
        "permute102": lambda x, **_: x.permute(1, 0, 2),
        "permute210": lambda x, **_: x.permute(2, 1, 0),
        "dynamic_slice": lambda x, **_: x[0:16, 32:96, 0:768],
        "index_select": lambda torch, x, **_: torch.index_select(x, 1, torch.arange(0, 128, 2, device=x.device)),
        "gather": lambda torch, x, index, **_: torch.gather(x, 1, index),
        "batch_gather": lambda x, indices, **_: x.gather(1, indices.unsqueeze(-1).expand(-1, -1, x.shape[-1])),
        "slice_scatter": lambda x, update, **_: _slice_scatter(x, update, [0, 48, 0]),
        "pad": lambda torch, x, **_: torch.nn.functional.pad(x, (1, 2), value=-1.0),
    }


def _import_torch() -> Any:
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("benchmark-torch-ops requires PyTorch; install the 'validate' extra or torch") from exc
    return torch


def _normalize_device(torch: Any, device: str) -> Any:
    torch_device = torch.device(device)
    if torch_device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA device requested for PyTorch benchmark, but torch.cuda.is_available() is false")
    return torch_device


def _torch_inputs(torch: Any, inputs: Mapping[str, np.ndarray], device: Any) -> dict[str, Any]:
    tensors = {}
    for name, value in inputs.items():
        tensor = torch.from_numpy(np.asarray(value)).to(device)
        tensors[name] = tensor
    return tensors


def _benchmark_torch_call(
    torch: Any,
    device: Any,
    call: Callable[[], Any],
    *,
    warmup: int,
    iterations: int,
) -> dict[str, float | int]:
    for _ in range(warmup):
        call()
    _synchronize(torch, device)
    samples = []
    for _ in range(iterations):
        started = time.perf_counter()
        out = call()
        if hasattr(out, "contiguous"):
            out.contiguous()
        _synchronize(torch, device)
        samples.append((time.perf_counter() - started) * 1000.0)
    return _summarize_benchmark_samples(samples, warmup=warmup)


def _synchronize(torch: Any, device: Any) -> None:
    if getattr(device, "type", None) == "cuda":
        torch.cuda.synchronize(device)


def _input_report(tensors: Mapping[str, Any]) -> list[dict[str, Any]]:
    return [
        {"name": name, "shape": list(tensor.shape), "dtype": _dtype_name(tensor.dtype)}
        for name, tensor in tensors.items()
    ]


def _output_report(outputs: Any) -> list[dict[str, Any]]:
    if isinstance(outputs, Mapping):
        items = list(outputs.items())
    elif isinstance(outputs, (tuple, list)):
        items = [(f"output_{idx}", value) for idx, value in enumerate(outputs)]
    else:
        items = [("output", outputs)]
    return [
        {"name": name, "shape": list(value.shape), "dtype": _dtype_name(value.dtype)}
        for name, value in items
    ]


def _dtype_name(dtype: Any) -> str:
    name = str(dtype)
    return name.removeprefix("torch.")


def _select_torch_cases(cases: list[TorchBenchmarkCase], only: Iterable[str] | None) -> list[TorchBenchmarkCase]:
    if only is None:
        return cases
    benchmark_like_cases = [
        BenchmarkCase(case.name, lambda: None, case.inputs, case.op, case.template)
        for case in cases
    ]
    selected = _select_cases(benchmark_like_cases, only)
    selected_names = {case.name for case in selected}
    return [case for case in cases if case.name in selected_names]


def _randn(torch: Any, shape: tuple[int, ...], *, device: Any, seed: int) -> Any:
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    return torch.randn(shape, dtype=torch.float32, device=device, generator=generator)


def _gelu_new(torch: Any, x: Any) -> Any:
    return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0))))


def _t5_layer_norm(torch: Any, hidden_states: Any, weight: Any, *, eps: float) -> Any:
    variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
    hidden_states = hidden_states * torch.rsqrt(variance + eps)
    if weight.dtype in (torch.float16, torch.bfloat16):
        hidden_states = hidden_states.to(weight.dtype)
    return weight * hidden_states


def _get_timestep_embedding(
    torch: Any,
    timesteps: Any,
    *,
    embedding_dim: int,
    flip_sin_to_cos: bool = False,
    downscale_freq_shift: float = 1.0,
    scale: float = 1.0,
    max_period: float = 10000.0,
) -> Any:
    half_dim = embedding_dim // 2
    exponent = -math.log(max_period) * torch.arange(
        half_dim,
        dtype=torch.float32,
        device=timesteps.device,
    )
    exponent = exponent / (half_dim - downscale_freq_shift)
    emb = timesteps.float()[:, None] * torch.exp(exponent)[None, :]
    emb = scale * emb
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
    if flip_sin_to_cos:
        emb = torch.cat([emb[:, half_dim:], emb[:, :half_dim]], dim=-1)
    if embedding_dim % 2 == 1:
        emb = torch.nn.functional.pad(emb, (0, 1))
    return emb


def _get_1d_rotary_pos_embed(
    torch: Any,
    dim: int,
    pos: Any,
    *,
    theta: float = 10000.0,
    linear_factor: float = 1.0,
    ntk_factor: float = 1.0,
    repeat_interleave_real: bool = True,
) -> tuple[Any, Any]:
    inv_freq = 1.0 / (
        (theta * ntk_factor)
        ** (torch.arange(0, dim, 2, dtype=torch.float32, device=pos.device)[: dim // 2] / dim)
    )
    freqs = torch.outer(pos.float() / linear_factor, inv_freq)
    cos = freqs.cos()
    sin = freqs.sin()
    if repeat_interleave_real:
        cos = torch.repeat_interleave(cos, repeats=2, dim=-1)
        sin = torch.repeat_interleave(sin, repeats=2, dim=-1)
    else:
        cos = torch.cat([cos, cos], dim=-1)
        sin = torch.cat([sin, sin], dim=-1)
    return cos, sin


def _slice_scatter(x: Any, update: Any, start_indices: list[int]) -> Any:
    out = x.clone()
    slices = tuple(slice(start, start + extent) for start, extent in zip(start_indices, update.shape, strict=True))
    out[slices] = update
    return out
