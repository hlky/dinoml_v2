from __future__ import annotations

import json
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping

import numpy as np

import dinoml as dml
from dinoml import runtime
from dinoml.ir import ModelSpec


@dataclass(frozen=True)
class BenchmarkCase:
    name: str
    build_spec: Callable[[], ModelSpec]
    inputs: Callable[[], dict[str, np.ndarray]]
    op: str
    template: str


@dataclass(frozen=True)
class _CompiledBenchmarkCase:
    index: int
    case: BenchmarkCase
    artifact_path: Path
    status: str
    compile_elapsed_s: float
    error: str | None = None


@dataclass(frozen=True)
class _PreparedBenchmarkCase:
    index: int
    case: BenchmarkCase
    spec: ModelSpec


class _SingleOpModule(dml.Module):
    def __init__(self, fn: Callable[..., Any]):
        self._fn = fn

    def forward(self, **inputs: Any) -> Any:
        result = self._fn(**inputs)
        if isinstance(result, Mapping):
            return {name: dml.ops.output(value, name) for name, value in result.items()}
        if isinstance(result, (tuple, list)):
            return {f"output_{idx}": dml.ops.output(value, f"output_{idx}") for idx, value in enumerate(result)}
        return dml.ops.output(result, "output")


def benchmark_cases() -> list[BenchmarkCase]:
    cases: list[BenchmarkCase] = []

    def add(
        name: str,
        input_specs: Mapping[str, dml.TensorSpec],
        inputs: Mapping[str, np.ndarray] | Callable[[], Mapping[str, np.ndarray]],
        fn: Callable[..., Any],
        *,
        op: str | None = None,
        template: str,
    ) -> None:
        case_op = op or name

        def build_spec() -> ModelSpec:
            return dml.trace(_SingleOpModule(fn), inputs=dict(input_specs), name=f"benchmark_{name}")

        def build_inputs() -> dict[str, np.ndarray]:
            values = inputs() if callable(inputs) else inputs
            return {key: np.asarray(value).copy() for key, value in values.items()}

        cases.append(BenchmarkCase(name, build_spec, build_inputs, case_op, template))

    # Representative transformer activation volume, large enough that GPU timings
    # are dominated by kernel work instead of launch overhead.
    activation_shape = (1024, 4096)

    def elementwise_inputs() -> dict[str, np.ndarray]:
        x = _float_array(activation_shape, -1.25, 0.0001)
        y = _float_array(activation_shape, 0.25, 0.00005)
        return {
            "x": x,
            "y": y,
            "positive": np.abs(x) + 0.25,
            "condition": (np.arange(np.prod(activation_shape)).reshape(activation_shape) % 3 == 0),
        }
    elementwise_specs = {
        "x": dml.TensorSpec(list(activation_shape), "float32"),
        "y": dml.TensorSpec(list(activation_shape), "float32"),
        "positive": dml.TensorSpec(list(activation_shape), "float32"),
        "condition": dml.TensorSpec(list(activation_shape), "bool"),
    }
    elementwise_template = "fused_elementwise_gpu.j2"
    for name, fn in [
        ("add", lambda x, y, **_: x + y),
        ("sub", lambda x, y, **_: x - y),
        ("mul", lambda x, y, **_: x * y),
        ("div", lambda x, y, **_: x / (y + 2.0)),
        ("tanh", lambda x, **_: dml.ops.tanh(x)),
        ("cos", lambda x, **_: dml.ops.cos(x)),
        ("sin", lambda x, **_: dml.ops.sin(x)),
        ("sign", lambda x, **_: dml.ops.sign(x)),
        ("abs", lambda x, **_: dml.ops.abs(x)),
        ("log", lambda positive, **_: dml.ops.log(positive)),
        ("log1p", lambda positive, **_: dml.ops.log1p(positive)),
        ("exp", lambda x, **_: dml.ops.exp(x * 0.1)),
        ("sqrt", lambda positive, **_: dml.ops.sqrt(positive)),
        ("max", lambda x, y, **_: dml.ops.max(x, y)),
        ("min", lambda x, y, **_: dml.ops.min(x, y)),
        ("sigmoid", lambda x, **_: dml.ops.sigmoid(x)),
        ("leaky_relu", lambda x, **_: dml.ops.leaky_relu(x, negative_slope=0.2)),
        ("hardtanh", lambda x, **_: dml.ops.hardtanh(x, min_value=-0.5, max_value=0.75)),
        ("relu", lambda x, **_: dml.ops.relu(x)),
        ("nan_to_num", lambda x, **_: dml.ops.nan_to_num(x, nan_replacement=0.0, posinf_replacement=0.0, neginf_replacement=0.0)),
        ("clamp_nan_to_num", lambda x, **_: dml.ops.clamp_nan_to_num(x, clamp_min=-1.0, clamp_max=1.0, nan_replacement=0.0)),
        ("silu", lambda x, **_: dml.ops.silu(x)),
        ("pow", lambda positive, y, **_: dml.ops.pow(positive, y + 2.0)),
        ("gelu", lambda x, **_: dml.ops.gelu(x)),
        ("gelu_new", lambda x, **_: dml.ops.gelu_new(x)),
        ("fast_gelu", lambda x, **_: dml.ops.fast_gelu(x)),
        ("softplus", lambda x, **_: dml.ops.softplus(x)),
        ("elu", lambda x, **_: dml.ops.elu(x, alpha=1.25)),
        ("softsign", lambda x, **_: dml.ops.softsign(x)),
        ("floor_div", lambda x, y, **_: dml.ops.floor_div(x + 4.0, y + 2.0)),
        ("celu", lambda x, **_: dml.ops.celu(x, alpha=1.1)),
        ("floor", lambda x, **_: dml.ops.floor(x)),
        ("eq", lambda x, y, **_: dml.ops.eq(x, y)),
        ("ge", lambda x, y, **_: dml.ops.ge(x, y)),
        ("gt", lambda x, y, **_: dml.ops.gt(x, y)),
        ("le", lambda x, y, **_: dml.ops.le(x, y)),
        ("lt", lambda x, y, **_: dml.ops.lt(x, y)),
        ("ne", lambda x, y, **_: dml.ops.ne(x, y)),
        ("where", lambda x, y, condition, **_: dml.ops.where(condition, x, y)),
        ("cast", lambda condition, **_: dml.ops.cast(condition, "float32")),
    ]:
        add(name, elementwise_specs, elementwise_inputs, fn, template=elementwise_template)

    add("full", {}, {}, lambda: dml.ops.full([1024, 4096], 1.25, dtype="float32"), template="full_gpu.j2")
    add("arange", {}, {}, lambda: dml.ops.arange(0, 4_194_304, 1, dtype="float32"), template="arange_gpu.j2")
    add("randn", {}, {}, lambda: dml.ops.randn([1024, 4096], dtype="float32", seed=17), template="randn_gpu.j2")

    attention_shape = (32, 128, 1024)
    attention_inputs = lambda: {"x": _float_array(attention_shape, -2.0, 0.0001)}
    add("softmax", {"x": dml.TensorSpec(list(attention_shape), "float32")}, attention_inputs, lambda x: dml.ops.softmax(x, dim=-1), template="softmax_gpu.j2")
    for name, fn in [
        ("reduce_sum", lambda x: dml.ops.reduce_sum(x, dim=-1)),
        ("reduce_max", lambda x: dml.ops.reduce_max(x, dim=-1)),
        ("reduce_min", lambda x: dml.ops.reduce_min(x, dim=-1)),
        ("reduce_mean", lambda x: dml.ops.reduce_mean(x, dim=-1)),
        ("var", lambda x: dml.ops.var(x, dim=-1, unbiased=False)),
        ("vector_norm", lambda x: dml.ops.vector_norm(x, dim=-1)),
    ]:
        add(name, {"x": dml.TensorSpec(list(attention_shape), "float32")}, attention_inputs, fn, template="reduction_gpu.j2")

    add("avg_pool1d", {"x": dml.TensorSpec([16, 64, 1024], "float32")}, lambda: {"x": _float_array((16, 64, 1024), step=0.0001)}, lambda x: dml.ops.avg_pool1d(x, kernel_size=3, stride=2, padding=1), template="avg_pool1d_gpu.j2")
    add("avg_pool2d", {"x": dml.TensorSpec([8, 64, 56, 56], "float32")}, lambda: {"x": _float_array((8, 64, 56, 56), step=0.0001)}, lambda x: dml.ops.avg_pool2d(x, kernel_size=(3, 3), stride=2, padding=1), template="avg_pool2d_gpu.j2")
    add("max_pool2d", {"x": dml.TensorSpec([8, 64, 56, 56], "float32")}, lambda: {"x": _float_array((8, 64, 56, 56), step=0.0001)}, lambda x: dml.ops.max_pool2d(x, kernel_size=(3, 3), stride=2, padding=1), template="max_pool2d_gpu.j2")

    add("argmax", {"x": dml.TensorSpec(list(attention_shape), "float32")}, attention_inputs, lambda x: dml.ops.argmax(x, dim=-1), template="argmax_gpu.j2")
    add("topk", {"x": dml.TensorSpec(list(attention_shape), "float32")}, attention_inputs, lambda x: dml.ops.topk(x, 16, dim=-1), template="topk_gpu.j2")

    def norm_inputs() -> dict[str, np.ndarray]:
        return {
            "x": _float_array((16, 128, 768), -1.0, 0.0001),
            "weight": np.linspace(0.5, 1.5, num=768, dtype=np.float32),
            "bias": np.linspace(-0.25, 0.25, num=768, dtype=np.float32),
        }

    def norm_inputs_without_bias() -> dict[str, np.ndarray]:
        values = norm_inputs()
        del values["bias"]
        return values

    norm_specs = {
        "x": dml.TensorSpec([16, 128, 768], "float32"),
        "weight": dml.TensorSpec([768], "float32"),
        "bias": dml.TensorSpec([768], "float32"),
    }
    add("layer_norm", norm_specs, norm_inputs, lambda x, weight, bias: dml.ops.layer_norm(x, weight, bias, eps=1e-5), template="layer_norm_gpu.j2")
    add("t5_layer_norm", {k: v for k, v in norm_specs.items() if k != "bias"}, norm_inputs_without_bias, lambda x, weight: dml.ops.t5_layer_norm(x, weight, eps=1e-6), template="t5_layer_norm_gpu.j2")
    add("rms_norm", {k: v for k, v in norm_specs.items() if k != "bias"}, norm_inputs_without_bias, lambda x, weight: dml.ops.rms_norm(x, weight, eps=1e-6), op="t5_layer_norm", template="t5_layer_norm_gpu.j2")

    add("get_timestep_embedding", {"timesteps": dml.TensorSpec([4096], "float32")}, {"timesteps": np.arange(4096, dtype=np.float32)}, lambda timesteps: dml.ops.get_timestep_embedding(timesteps, embedding_dim=128), template="get_timestep_embedding_gpu.j2")
    add("get_1d_rotary_pos_embed", {"positions": dml.TensorSpec([4096], "float32")}, {"positions": np.arange(4096, dtype=np.float32)}, lambda positions: dml.ops.get_1d_rotary_pos_embed(128, positions), template="get_1d_rotary_pos_embed_gpu.j2")

    add("embedding", {"table": dml.TensorSpec([32768, 256], "float32"), "indices": dml.TensorSpec([32, 128], "int64")}, lambda: {"table": _float_array((32768, 256), step=0.00001), "indices": np.arange(4096, dtype=np.int64).reshape(32, 128) % 32768}, lambda table, indices: dml.ops.embedding(table, indices), template="embedding_gpu.j2")

    collection_shape = (16, 128, 768)
    collection_specs = {"x": dml.TensorSpec(list(collection_shape), "float32")}
    collection_inputs = lambda: {"x": _float_array(collection_shape, step=0.0001)}
    add("expand", {"x": dml.TensorSpec([1, 128, 768], "float32")}, lambda: {"x": _float_array((1, 128, 768), step=0.0001)}, lambda x: dml.ops.expand(x, [16, 128, 768]), template="expand_gpu.j2")
    add("concatenate", {"x": dml.TensorSpec(list(collection_shape), "float32"), "y": dml.TensorSpec(list(collection_shape), "float32")}, lambda: {"x": _float_array(collection_shape, step=0.0001), "y": _float_array(collection_shape, 10.0, 0.0001)}, lambda x, y: dml.ops.concatenate([x, y], dim=1), template="concatenate_gpu.j2")
    add("stack", {"x": dml.TensorSpec(list(collection_shape), "float32"), "y": dml.TensorSpec(list(collection_shape), "float32")}, lambda: {"x": _float_array(collection_shape, step=0.0001), "y": _float_array(collection_shape, 10.0, 0.0001)}, lambda x, y: dml.ops.stack([x, y], dim=0), template="stack_gpu.j2")
    add("flip", collection_specs, collection_inputs, lambda x: dml.ops.flip(x, dims=(-1,)), template="flip_gpu.j2")
    add("repeat_interleave", collection_specs, collection_inputs, lambda x: dml.ops.repeat_interleave(x, repeats=2, dim=1), template="repeat_interleave_gpu.j2")
    add("permute", collection_specs, collection_inputs, lambda x: dml.ops.permute(x, [1, 0, 2]), template="permute_gpu.j2")
    add("permute021", collection_specs, collection_inputs, lambda x: dml.ops.permute021(x), op="permute021", template="permute_gpu.j2")
    add("permute102", collection_specs, collection_inputs, lambda x: dml.ops.permute102(x), op="permute102", template="permute_gpu.j2")
    add("permute210", collection_specs, collection_inputs, lambda x: dml.ops.permute210(x), op="permute210", template="permute_gpu.j2")
    add("dynamic_slice", collection_specs, collection_inputs, lambda x: dml.ops.dynamic_slice(x, [0, 32, 0], [16, 64, 768]), template="dynamic_slice_gpu.j2")
    add("index_select", collection_specs, collection_inputs, lambda x: dml.ops.index_select(x, dim=1, indices=list(range(0, 128, 2))), template="index_select_gpu.j2")
    add("gather", {"x": dml.TensorSpec(list(collection_shape), "float32"), "index": dml.TensorSpec([16, 64, 768], "int64")}, lambda: {"x": _float_array(collection_shape, step=0.0001), "index": (np.arange(16 * 64 * 768, dtype=np.int64).reshape(16, 64, 768) % 128)}, lambda x, index: dml.ops.gather(x, 1, index), template="gather_gpu.j2")
    add("batch_gather", {"x": dml.TensorSpec([32, 256, 768], "float32"), "indices": dml.TensorSpec([32, 128], "int64")}, lambda: {"x": _float_array((32, 256, 768), step=0.00001), "indices": (np.arange(4096, dtype=np.int64).reshape(32, 128) % 256)}, lambda x, indices: dml.ops.batch_gather(x, indices), template="gather_gpu.j2")
    add("slice_scatter", {"x": dml.TensorSpec(list(collection_shape), "float32"), "update": dml.TensorSpec([16, 32, 768], "float32")}, lambda: {"x": _float_array(collection_shape, step=0.0001), "update": _float_array((16, 32, 768), 100.0, 0.0001)}, lambda x, update: dml.ops.slice_scatter(x, update, [0, 48, 0]), template="slice_scatter_gpu.j2")
    add("pad", collection_specs, collection_inputs, lambda x: dml.ops.pad(x, [1, 2], value=-1.0), template="pad_gpu.j2")
    return cases


def run_benchmark_suite(
    target: str,
    *,
    output_dir: str | Path | None = None,
    warmup: int = 5,
    iterations: int = 20,
    profile: bool = False,
    profile_iterations: int = 20,
    profile_repeats: int = 3,
    profile_refresh: bool = False,
    only: Iterable[str] | None = None,
    arch: str | None = None,
    no_tf32: bool = False,
    use_fp16_acc: bool = False,
    keep_artifacts: bool = False,
    fail_fast: bool = False,
    jobs: int = 1,
) -> dict[str, Any]:
    selected = _select_cases(benchmark_cases(), only)
    if jobs < 1:
        raise ValueError(f"jobs must be >= 1, got {jobs}")
    target_obj = dml.Target(target, arch=arch, no_tf32=no_tf32, use_fp16_acc=use_fp16_acc)
    temp_dir: tempfile.TemporaryDirectory[str] | None = None
    if output_dir is None:
        if keep_artifacts:
            artifact_root = Path(tempfile.mkdtemp(prefix="dinoml_ops_bench_"))
        else:
            temp_dir = tempfile.TemporaryDirectory(prefix="dinoml_ops_bench_", ignore_cleanup_errors=True)
            artifact_root = Path(temp_dir.name)
    else:
        artifact_root = Path(output_dir)
        artifact_root.mkdir(parents=True, exist_ok=True)

    results: list[dict[str, Any]] = []
    started = time.perf_counter()
    try:
        compiled_cases = _compile_benchmark_cases(
            selected,
            target_obj=target_obj,
            artifact_root=artifact_root,
            profile=profile,
            profile_iterations=profile_iterations,
            profile_repeats=profile_repeats,
            profile_refresh=profile_refresh,
            jobs=jobs,
            fail_fast=fail_fast,
        )
        for compiled in compiled_cases:
            case = compiled.case
            if compiled.status != "ok":
                results.append(
                    {
                        "name": case.name,
                        "op": case.op,
                        "template": case.template,
                        "artifact": str(compiled.artifact_path),
                        "status": "error",
                        "error": compiled.error,
                        "compile_elapsed_s": compiled.compile_elapsed_s,
                        "elapsed_s": compiled.compile_elapsed_s,
                    }
                )
                continue
            benchmark_started = time.perf_counter()
            try:
                artifact_path = compiled.artifact_path
                rt_module = runtime.load(artifact_path, load_constants=True)
                session = None
                try:
                    session = rt_module.create_session()
                    summary = session.benchmark_numpy(case.inputs(), warmup=warmup, iterations=iterations)
                    metadata = getattr(rt_module, "metadata", {})
                finally:
                    if session is not None:
                        session.close()
                    rt_module.close()
                benchmark_elapsed_s = time.perf_counter() - benchmark_started
                elapsed_s = compiled.compile_elapsed_s + benchmark_elapsed_s
                results.append(
                    {
                        "name": case.name,
                        "op": case.op,
                        "template": case.template,
                        "artifact": str(artifact_path),
                        "status": "ok",
                        "inputs": _io_report(metadata.get("inputs", [])),
                        "outputs": _io_report(metadata.get("outputs", [])),
                        "session_run": summary,
                        "compile_elapsed_s": compiled.compile_elapsed_s,
                        "benchmark_elapsed_s": benchmark_elapsed_s,
                        "elapsed_s": elapsed_s,
                    }
                )
                print(f"{case.op}::{case.name} {summary=}")
            except Exception as exc:
                benchmark_elapsed_s = time.perf_counter() - benchmark_started
                results.append(
                    {
                        "name": case.name,
                        "op": case.op,
                        "template": case.template,
                        "artifact": str(compiled.artifact_path),
                        "status": "error",
                        "error": f"{type(exc).__name__}: {exc}",
                        "compile_elapsed_s": compiled.compile_elapsed_s,
                        "benchmark_elapsed_s": benchmark_elapsed_s,
                        "elapsed_s": compiled.compile_elapsed_s + benchmark_elapsed_s,
                    }
                )
                if fail_fast:
                    raise
    finally:
        if temp_dir is not None:
            temp_dir.cleanup()

    ok = sum(1 for item in results if item["status"] == "ok")
    return {
        "target": target_obj.to_json(),
        "warmup": warmup,
        "iterations": iterations,
        "profile": {
            "enabled": profile,
            "iterations": profile_iterations,
            "repeats": profile_repeats,
            "refresh": profile_refresh,
        },
        "artifact_root": str(artifact_root),
        "summary": {
            "total": len(results),
            "ok": ok,
            "error": len(results) - ok,
            "elapsed_s": time.perf_counter() - started,
        },
        "cases": results,
    }


def _compile_benchmark_cases(
    cases: list[BenchmarkCase],
    *,
    target_obj: dml.Target,
    artifact_root: Path,
    profile: bool,
    profile_iterations: int,
    profile_repeats: int,
    profile_refresh: bool,
    jobs: int,
    fail_fast: bool,
) -> list[_CompiledBenchmarkCase]:
    prepared, preparation_errors = _prepare_benchmark_cases(cases, fail_fast=fail_fast)
    if not prepared:
        return preparation_errors

    if jobs == 1 or len(cases) <= 1:
        compiled: list[_CompiledBenchmarkCase | None] = [None] * len(cases)
        for item in prepared:
            print(f"{item.case.op}::{item.case.name} compile [{item.index+1}/{len(cases)}]")
            result = _compile_benchmark_case(
                item,
                target_obj=target_obj,
                artifact_root=artifact_root,
                profile=profile,
                profile_iterations=profile_iterations,
                profile_repeats=profile_repeats,
                profile_refresh=profile_refresh,
            )
            if fail_fast and result.status != "ok":
                raise RuntimeError(result.error)
            compiled[item.index] = result
        for error in preparation_errors:
            compiled[error.index] = error
        return [result for result in compiled if result is not None]

    print(f"Compiling {len(prepared)} benchmark cases with jobs={jobs}")
    compiled_results: list[_CompiledBenchmarkCase | None] = [None] * len(cases)
    for error in preparation_errors:
        compiled_results[error.index] = error
    with ThreadPoolExecutor(max_workers=jobs) as executor:
        future_to_case = {
            executor.submit(
                _compile_benchmark_case,
                item,
                target_obj=target_obj,
                artifact_root=artifact_root,
                profile=profile,
                profile_iterations=profile_iterations,
                profile_repeats=profile_repeats,
                profile_refresh=profile_refresh,
            ): item.case
            for item in prepared
        }
        for future in as_completed(future_to_case):
            result = future.result()
            compiled_results[result.index] = result
            print(
                f"{result.case.op}::{result.case.name} compile {result.status} "
                f"elapsed_s={result.compile_elapsed_s:.3f}"
            )
            if fail_fast and result.status != "ok":
                raise RuntimeError(result.error)

    return [result for result in compiled_results if result is not None]


def _prepare_benchmark_cases(
    cases: list[BenchmarkCase],
    *,
    fail_fast: bool,
) -> tuple[list[_PreparedBenchmarkCase], list[_CompiledBenchmarkCase]]:
    prepared: list[_PreparedBenchmarkCase] = []
    errors: list[_CompiledBenchmarkCase] = []
    for idx, case in enumerate(cases):
        started = time.perf_counter()
        try:
            prepared.append(_PreparedBenchmarkCase(index=idx, case=case, spec=case.build_spec()))
        except Exception as exc:
            error = _CompiledBenchmarkCase(
                index=idx,
                case=case,
                artifact_path=Path(f"{case.name}.dinoml"),
                status="error",
                compile_elapsed_s=time.perf_counter() - started,
                error=f"{type(exc).__name__}: {exc}",
            )
            if fail_fast:
                raise RuntimeError(error.error)
            errors.append(error)
    return prepared, errors


def _compile_benchmark_case(
    item: _PreparedBenchmarkCase,
    *,
    target_obj: dml.Target,
    artifact_root: Path,
    profile: bool,
    profile_iterations: int,
    profile_repeats: int,
    profile_refresh: bool,
) -> _CompiledBenchmarkCase:
    index = item.index
    case = item.case
    artifact_path = artifact_root / f"{case.name}.dinoml"
    started = time.perf_counter()
    try:
        compile_kwargs: dict[str, Any] = {}
        if profile:
            compile_kwargs.update(
                {
                    "profile": True,
                    "profile_iterations": profile_iterations,
                    "profile_repeats": profile_repeats,
                    "profile_refresh": profile_refresh,
                }
            )
        artifact = dml.compile(item.spec, target_obj, artifact_path, **compile_kwargs)
        return _CompiledBenchmarkCase(
            index=index,
            case=case,
            artifact_path=Path(artifact.path),
            status="ok",
            compile_elapsed_s=time.perf_counter() - started,
        )
    except Exception as exc:
        return _CompiledBenchmarkCase(
            index=index,
            case=case,
            artifact_path=artifact_path,
            status="error",
            compile_elapsed_s=time.perf_counter() - started,
            error=f"{type(exc).__name__}: {exc}",
        )


def write_report(report: Mapping[str, Any], output: str | Path) -> None:
    out_path = Path(output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")


def _select_cases(cases: list[BenchmarkCase], only: Iterable[str] | None) -> list[BenchmarkCase]:
    if only is None:
        return cases
    requested = {item for item in only}
    selected = [case for case in cases if case.name in requested or case.op in requested or case.template in requested]
    found = {case.name for case in selected} | {case.op for case in selected} | {case.template for case in selected}
    missing = sorted(requested - found)
    if missing:
        raise ValueError(f"Unknown benchmark case filter(s): {', '.join(missing)}")
    return selected


def _io_report(specs: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "name": str(spec["name"]),
            "shape": list(spec["shape"]),
            "dtype": str(spec["dtype"]),
        }
        for spec in specs
    ]


def _float_array(shape: tuple[int, ...], start: float = 0.0, step: float = 0.01) -> np.ndarray:
    values = np.arange(np.prod(shape), dtype=np.float32) * np.float32(step) + np.float32(start)
    return values.reshape(shape)
