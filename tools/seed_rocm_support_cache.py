from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import dinoml as dml  # noqa: E402
from dinoml.backends import rocm as rocm_backend  # noqa: E402
from dinoml.ir import ModelSpec, canonical_json, write_json  # noqa: E402
from dinoml.kernels.manifest import build_kernel_manifest  # noqa: E402
from dinoml.kernels.bmm import bmm_op_spec  # noqa: E402
from dinoml.kernels.gemm import gemm_op_spec  # noqa: E402


DEFAULT_FAMILIES = ("base", "gemm", "bmm", "conv", "flash_attention")
DEFAULT_DTYPES = ("float16",)


class _SeedGemmModule(dml.nn.Module):
    def forward(self, a, b, bias, d0, d1):
        return {
            "gemm_rcr": dml.ops.output(dml.ops.gemm_rcr(a, b), "gemm_rcr"),
            "gemm_rcr_bias": dml.ops.output(dml.ops.gemm_rcr_bias(a, b, bias), "gemm_rcr_bias"),
            "gemm_rcr_bias_add_relu": dml.ops.output(
                dml.ops.gemm_rcr_bias_add_relu(a, b, bias, d0),
                "gemm_rcr_bias_add_relu",
            ),
            "gemm_rcr_bias_add_add_relu": dml.ops.output(
                dml.ops.gemm_rcr_bias_add_add_relu(a, b, bias, d0, d1),
                "gemm_rcr_bias_add_add_relu",
            ),
        }


class _SeedBmmModule(dml.nn.Module):
    def forward(self, a, b, d0):
        return {
            "bmm_rcr": dml.ops.output(dml.ops.bmm_rcr(a, b), "bmm_rcr"),
            "bmm_rcr_add": dml.ops.output(dml.ops.bmm_rcr_add(a, b, d0), "bmm_rcr_add"),
        }


class _SeedConvModule(dml.nn.Module):
    def forward(self, x, weight, bias, residual):
        return {
            "conv2d_bias": dml.ops.output(dml.ops.conv2d_bias(x, weight, bias, padding=1), "conv2d_bias"),
            "conv2d_bias_relu": dml.ops.output(
                dml.ops.conv2d_bias_relu(x, weight, bias, padding=1),
                "conv2d_bias_relu",
            ),
            "conv2d_bias_add": dml.ops.output(
                dml.ops.conv2d_bias_add(x, weight, bias, residual, padding=1),
                "conv2d_bias_add",
            ),
            "conv2d_bias_add_relu": dml.ops.output(
                dml.ops.conv2d_bias_add_relu(x, weight, bias, residual, padding=1),
                "conv2d_bias_add_relu",
            ),
        }


class _SeedFlashAttentionModule(dml.nn.Module):
    def forward(self, q, k, v, bias):
        return {
            "flash_attention": dml.ops.output(dml.ops.flash_attention(q, k, v, causal=False), "flash_attention"),
            "flash_attention_bias": dml.ops.output(
                dml.ops.flash_attention_bias(q, k, v, bias, causal=False),
                "flash_attention_bias",
            ),
        }


def seed_rocm_support_cache(
    cache_dir: str | Path,
    *,
    arch: str | None = None,
    dtypes: Sequence[str] = DEFAULT_DTYPES,
    families: Sequence[str] = DEFAULT_FAMILIES,
) -> dict[str, Any]:
    family_list = _normalize_choices(families, choices=DEFAULT_FAMILIES, label="family")
    dtype_list = _normalize_dtypes(dtypes)
    target = dml.Target("rocm", arch=arch)
    target_json = target.to_json()
    resolved_arch = str(target_json["arch"])
    cache_root = Path(cache_dir).resolve()
    cache_root.mkdir(parents=True, exist_ok=True)

    results: list[dict[str, Any]] = []
    with _temporary_env("DINOML_CACHE_DIR", str(cache_root)):
        if "base" in family_list:
            support = rocm_backend.ensure_rocm_support_libs(resolved_arch, kernel_manifest=None)
            results.append(
                {
                    "family": "base",
                    "cache_root": str(cache_root / "support" / rocm_backend._rocm_support_cache_dir_name(resolved_arch) / "full"),
                    "artifacts": [
                        str(support.runtime_lib),
                        str(support.rocm_runtime_lib),
                        str(support.kernels_lib),
                    ],
                }
            )
        for family in family_list:
            if family == "base":
                continue
            manifest = _build_family_kernel_manifest(family, target_json=target_json, dtypes=dtype_list)
            artifacts = _seed_family_archives(family, resolved_arch, manifest)
            results.append(
                {
                    "family": family,
                    "dtype_count": len(dtype_list),
                    "required_kernel_count": len(manifest["required_kernels"]),
                    "cache_root": _family_cache_root(cache_root, resolved_arch, family),
                    "artifacts": [str(path) for path in artifacts],
                }
            )
    return {
        "schema_version": 1,
        "kind": "rocm_support_cache_seed",
        "cache_dir": str(cache_root),
        "target": target_json,
        "dtypes": list(dtype_list),
        "families": list(family_list),
        "results": results,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Seed a shared ROCm support cache without populating module/profile caches")
    parser.add_argument("--cache-dir", required=True, help="Cache root to populate, e.g. H:/dinoml_v2_rocm_cache")
    parser.add_argument("--arch", default=None, help="ROCm architecture; default uses DinoML's ROCm target default")
    parser.add_argument(
        "--dtype",
        action="append",
        dest="dtypes",
        default=[],
        help="Dtype to seed; repeatable. Default: float16",
    )
    parser.add_argument(
        "--family",
        action="append",
        dest="families",
        default=[],
        choices=DEFAULT_FAMILIES,
        help="Family to seed; repeatable. Default seeds all supported families.",
    )
    parser.add_argument("--out", help="Write the JSON report")
    args = parser.parse_args(argv)

    report = seed_rocm_support_cache(
        args.cache_dir,
        arch=args.arch,
        dtypes=args.dtypes or DEFAULT_DTYPES,
        families=args.families or DEFAULT_FAMILIES,
    )
    if args.out:
        write_json(Path(args.out), report)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


def _seed_family_archives(family: str, arch: str, manifest: Mapping[str, Any]) -> Sequence[Path]:
    if family == "gemm":
        return rocm_backend._ensure_cmake_ck_gemm_archives(arch, manifest)
    if family == "bmm":
        return rocm_backend._ensure_cmake_ck_bmm_archives(arch, manifest)
    if family == "conv":
        return rocm_backend._ensure_cmake_ck_conv_archives(arch, manifest)
    if family == "flash_attention":
        return rocm_backend._ensure_cmake_flash_attn_ck_archives(arch, manifest)
    raise ValueError(f"Unsupported family: {family}")


def _build_family_kernel_manifest(
    family: str,
    *,
    target_json: Mapping[str, Any],
    dtypes: Sequence[str],
) -> dict[str, Any]:
    manifests = [build_kernel_manifest(spec.ir, target_json) for spec in _family_seed_specs(family, dtypes)]
    if not manifests:
        raise ValueError(f"No manifests built for family {family}")
    required_kernels = []
    for manifest in manifests:
        required_kernels.extend(dict(item) for item in manifest.get("required_kernels", []))
    payload = {
        "target": dict(target_json),
        "required_kernels": required_kernels,
        "cache_key": _seed_manifest_cache_key(family, dtypes, required_kernels),
    }
    return payload


def _family_seed_specs(family: str, dtypes: Sequence[str]) -> list[ModelSpec]:
    specs: list[ModelSpec] = []
    for dtype in dtypes:
        if family == "gemm":
            specs.append(_gemm_seed_spec(dtype))
        elif family == "bmm":
            specs.append(_bmm_seed_spec(dtype))
        elif family == "conv":
            specs.append(_conv_seed_spec(dtype))
        elif family == "flash_attention":
            specs.append(_flash_attention_seed_spec(dtype))
        else:
            raise ValueError(f"Unsupported family: {family}")
    return specs


def _gemm_seed_spec(dtype: str) -> ModelSpec:
    m, n, k = 128, 128, 96
    return dml.trace(
        _SeedGemmModule(),
        inputs={
            "a": dml.TensorSpec([m, k], dtype),
            "b": dml.TensorSpec([n, k] if gemm_op_spec("gemm_rcr").base_layout == "rcr" else [k, n], dtype),
            "bias": dml.TensorSpec([n], dtype),
            "d0": dml.TensorSpec([m, n], dtype),
            "d1": dml.TensorSpec([m, n], dtype),
        },
        name=f"seed_rocm_support_gemm_{dtype}",
    )


def _bmm_seed_spec(dtype: str) -> ModelSpec:
    batch, m, n, k = 2, 64, 128, 96
    spec = bmm_op_spec("bmm_rcr")
    a_shape = [batch, k, m] if spec.a_layout == "c" else [batch, m, k]
    b_shape = [batch, n, k] if spec.b_layout == "c" else [batch, k, n]
    output_shape = [batch, n, m] if spec.c_layout == "c" else [batch, m, n]
    return dml.trace(
        _SeedBmmModule(),
        inputs={
            "a": dml.TensorSpec(a_shape, dtype),
            "b": dml.TensorSpec(b_shape, dtype),
            "d0": dml.TensorSpec(output_shape, dtype),
        },
        name=f"seed_rocm_support_bmm_{dtype}",
    )


def _conv_seed_spec(dtype: str) -> ModelSpec:
    return dml.trace(
        _SeedConvModule(),
        inputs={
            "x": dml.TensorSpec([2, 8, 16, 16], dtype),
            "weight": dml.TensorSpec([64, 8, 3, 3], dtype),
            "bias": dml.TensorSpec([64], dtype),
            "residual": dml.TensorSpec([2, 64, 16, 16], dtype),
        },
        name=f"seed_rocm_support_conv_{dtype}",
    )


def _flash_attention_seed_spec(dtype: str) -> ModelSpec:
    return dml.trace(
        _SeedFlashAttentionModule(),
        inputs={
            "q": dml.TensorSpec([1, 4, 2, 64], dtype),
            "k": dml.TensorSpec([1, 4, 1, 64], dtype),
            "v": dml.TensorSpec([1, 4, 1, 64], dtype),
            "bias": dml.TensorSpec([2, 4, 4], dtype),
        },
        name=f"seed_rocm_support_flash_attention_{dtype}",
    )


def _seed_manifest_cache_key(family: str, dtypes: Sequence[str], required_kernels: Sequence[Mapping[str, Any]]) -> str:
    payload = {
        "kind": "seed_rocm_support_cache",
        "family": family,
        "dtypes": list(dtypes),
        "required": [
            {
                "op": item.get("op"),
                "dtype": item.get("dtype"),
                "kernel_library": item.get("kernel_library"),
                "kernel_symbol": item.get("kernel_symbol"),
            }
            for item in required_kernels
        ],
    }
    return hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()


def _family_cache_root(cache_root: Path, arch: str, family: str) -> str:
    target_dir = rocm_backend._rocm_support_cache_dir_name(arch)
    if family == "gemm":
        return str(cache_root / "support" / target_dir / "ck-gemm" / "cmake-full")
    if family == "bmm":
        return str(cache_root / "support" / target_dir / "ck-bmm" / "cmake-full")
    if family == "conv":
        return str(cache_root / "support" / target_dir / "ck-conv" / "cmake-full")
    if family == "flash_attention":
        return str(cache_root / "support" / target_dir / "flash-attn-ck" / "cmake-full")
    if family == "base":
        return str(cache_root / "support" / target_dir / "full")
    raise ValueError(f"Unsupported family: {family}")


def _normalize_choices(values: Iterable[str], *, choices: Sequence[str], label: str) -> tuple[str, ...]:
    seen: list[str] = []
    for value in values:
        if value not in choices:
            raise ValueError(f"Unsupported {label}: {value}")
        if value not in seen:
            seen.append(value)
    return tuple(seen)


def _normalize_dtypes(dtypes: Sequence[str]) -> tuple[str, ...]:
    seen: list[str] = []
    for dtype in dtypes:
        value = str(dtype)
        if value not in seen:
            seen.append(value)
    if not seen:
        raise ValueError("At least one dtype is required")
    return tuple(seen)


@contextmanager
def _temporary_env(key: str, value: str):
    previous = os.environ.get(key)
    os.environ[key] = value
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = previous


if __name__ == "__main__":
    raise SystemExit(main())
