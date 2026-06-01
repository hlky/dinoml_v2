from __future__ import annotations

from collections import Counter, defaultdict
from functools import lru_cache

import dinoml as dml  # noqa: F401 - imports and registers public ops.
from dinoml.benchmarks.ops import benchmark_cases
from dinoml.ops.registry import OP_REGISTRY
from tests.cases import standard_cases


FLOAT_DTYPES = ("float16", "float32", "bfloat16")
PUBLIC_DTYPES = (*FLOAT_DTYPES, "bool", "int32", "int64")

EXPECTED_FLOAT_DTYPE_GAPS = {
    "float16": ("_shape_buffer_count_true", "var"),
    "float32": (
        "_shape_buffer_count_true",
        "flash_attention",
        "flash_attention_qkv",
        "flash_attention_static_kv_cache",
    ),
    "bfloat16": (
        "_shape_buffer_count_true",
        "var",
    ),
}

EXPECTED_CUDA_NOT_ROCM = ("flash_attention_static_kv_cache",)
EXPECTED_ROCM_NOT_CUDA: tuple[str, ...] = ()

EXPECTED_BACKEND_BUCKETS = {
    "cpu,cuda,rocm": 52,
    "cuda": 1,
    "cuda,rocm": 51,
    "fused/no direct": 39,
}

EXPECTED_DIRECTLESS_OPS = (
    "abs",
    "add",
    "cast",
    "celu",
    "clamp_nan_to_num",
    "cos",
    "div",
    "elu",
    "eq",
    "exp",
    "fast_gelu",
    "floor",
    "floor_div",
    "ge",
    "gelu",
    "gt",
    "hardtanh",
    "le",
    "leaky_relu",
    "log",
    "log1p",
    "lt",
    "max",
    "min",
    "mul",
    "nan_to_num",
    "ne",
    "pow",
    "relu",
    "sigmoid",
    "sign",
    "silu",
    "sin",
    "softplus",
    "softsign",
    "sqrt",
    "sub",
    "tanh",
    "where",
)

EXPECTED_GPU_ONLY_PROVIDER_PREFIXES = ("bmm_", "flash_attention", "gemm_")
EXPECTED_GPU_ONLY_PROVIDER_COUNT = 52

EXPECTED_CONTRACT_FLOAT_GAP_COUNTS = {
    "float16": 7,
    "float32": 3,
    "bfloat16": 7,
}

EXPECTED_BENCHMARK_FLOAT_GAP_COUNTS = {
    "float16": 7,
    "float32": 3,
    "bfloat16": 7,
}


def test_registered_float_dtype_gaps_are_explicit():
    matrix = _registered_op_matrix()
    missing = {
        dtype: tuple(name for name, entry in matrix.items() if dtype not in entry["dtypes"])
        for dtype in FLOAT_DTYPES
    }

    assert missing == EXPECTED_FLOAT_DTYPE_GAPS, _format_float_gap_report(missing)


def test_cuda_and_rocm_backend_surface_differences_are_explicit():
    matrix = _registered_op_matrix()
    cuda_not_rocm = tuple(
        name
        for name, entry in matrix.items()
        if "cuda" in entry["backends"] and "rocm" not in entry["backends"]
    )
    rocm_not_cuda = tuple(
        name
        for name, entry in matrix.items()
        if "rocm" in entry["backends"] and "cuda" not in entry["backends"]
    )
    backend_buckets = Counter(",".join(entry["backends"]) or "fused/no direct" for entry in matrix.values())

    assert cuda_not_rocm == EXPECTED_CUDA_NOT_ROCM
    assert rocm_not_cuda == EXPECTED_ROCM_NOT_CUDA
    assert dict(backend_buckets) == EXPECTED_BACKEND_BUCKETS


def test_directless_ops_are_known_fused_frontends():
    matrix = _registered_op_matrix()
    directless = tuple(name for name, entry in matrix.items() if not entry["backends"])

    assert directless == EXPECTED_DIRECTLESS_OPS


def test_gpu_only_direct_kernels_are_provider_variants():
    matrix = _registered_op_matrix()
    gpu_only = tuple(
        name
        for name, entry in matrix.items()
        if ("cuda" in entry["backends"] or "rocm" in entry["backends"]) and "cpu" not in entry["backends"]
    )
    unexpected = tuple(name for name in gpu_only if not name.startswith(EXPECTED_GPU_ONLY_PROVIDER_PREFIXES))

    assert unexpected == ()
    assert len(gpu_only) == EXPECTED_GPU_ONLY_PROVIDER_COUNT


def test_contract_and_benchmark_float_dtype_gap_counts_are_visible():
    matrix = _registered_op_matrix(include_internal=False)
    contract_missing = _float_coverage_gaps(matrix, _standard_contract_dtype_coverage())
    benchmark_missing = _float_coverage_gaps(matrix, _benchmark_dtype_coverage())

    contract_counts = {dtype: len(names) for dtype, names in contract_missing.items()}
    benchmark_counts = {dtype: len(names) for dtype, names in benchmark_missing.items()}

    assert contract_counts == EXPECTED_CONTRACT_FLOAT_GAP_COUNTS, _format_coverage_gap_report(
        "contract", contract_missing
    )
    assert benchmark_counts == EXPECTED_BENCHMARK_FLOAT_GAP_COUNTS, _format_coverage_gap_report(
        "benchmark", benchmark_missing
    )


def _registered_op_matrix(*, include_internal: bool = True) -> dict[str, dict[str, tuple[str, ...]]]:
    matrix = {}
    for op_def in sorted(OP_REGISTRY.op_defs(), key=lambda op: op.name):
        if not include_internal and str(op_def.name).startswith("_"):
            continue
        matrix[str(op_def.name)] = {
            "dtypes": tuple(str(dtype) for dtype in op_def.allowed_dtypes),
            "backends": tuple(sorted(str(name) for name in op_def.backend_kernels)),
        }
    return matrix


def _float_coverage_gaps(
    matrix: dict[str, dict[str, tuple[str, ...]]],
    coverage: dict[str, set[str]],
) -> dict[str, tuple[str, ...]]:
    gaps = {}
    for dtype in FLOAT_DTYPES:
        gaps[dtype] = tuple(
            name
            for name, entry in matrix.items()
            if dtype in entry["dtypes"] and dtype not in coverage.get(name, set())
        )
    return gaps


@lru_cache(maxsize=1)
def _standard_contract_dtype_coverage() -> dict[str, set[str]]:
    return _dtype_coverage_from_specs(case.build_spec() for case in standard_cases())


@lru_cache(maxsize=1)
def _benchmark_dtype_coverage() -> dict[str, set[str]]:
    return _dtype_coverage_from_specs(case.build_spec() for case in benchmark_cases())


def _dtype_coverage_from_specs(specs) -> dict[str, set[str]]:
    coverage: dict[str, set[str]] = defaultdict(set)
    for spec in specs:
        tensors = {str(tensor["name"]): tensor for tensor in spec.ir["tensors"]}
        for node in spec.ir["nodes"]:
            op_name = str(node["op"])
            if op_name == "output":
                continue
            for name in [*node.get("inputs", []), *node.get("outputs", [])]:
                tensor = tensors.get(str(name))
                if tensor is None:
                    continue
                dtype = str(tensor["dtype"])
                if dtype in PUBLIC_DTYPES:
                    coverage[op_name].add(dtype)
    return dict(coverage)


def _format_float_gap_report(missing: dict[str, tuple[str, ...]]) -> str:
    lines = ["Registered float dtype support gaps changed:"]
    for dtype in FLOAT_DTYPES:
        lines.append(f"  {dtype}: {', '.join(missing[dtype]) or '<none>'}")
    return "\n".join(lines)


def _format_coverage_gap_report(kind: str, missing: dict[str, tuple[str, ...]]) -> str:
    lines = [f"{kind.title()} float dtype coverage gaps changed:"]
    for dtype in FLOAT_DTYPES:
        names = ", ".join(missing[dtype])
        lines.append(f"  {dtype} ({len(missing[dtype])}): {names or '<none>'}")
    return "\n".join(lines)
