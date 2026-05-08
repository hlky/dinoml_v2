from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from dinoml.backends.cuda_libraries import require_cuda_library
from dinoml.ir import canonical_json, write_json
from dinoml.kernels.external import external_kernel_families
from dinoml.kernels.manifest import KERNEL_ABI_VERSION, PROFILE_CACHE_SCHEMA_VERSION, build_external_kernel_plan
from dinoml.kernels.providers.cutlass.gemm import cutlass_gemm_used_candidate_plan, render_cutlass_gemm_source


@dataclass(frozen=True)
class CutlassSupportLib:
    library: Path
    include_roots: tuple[Path, ...]
    source: Path
    manifest: Path
    source_manifest: Path


def ensure_cutlass_gemm_support_lib(
    arch: str,
    *,
    cache_key: str | None = None,
    used_candidate_plan: Mapping[str, Any] | None = None,
) -> CutlassSupportLib:
    cutlass = require_cuda_library("cutlass")
    cublaslt = require_cuda_library("cublaslt")
    cache_root = Path(os.environ.get("DINOML_CACHE_DIR", Path.home() / ".cache" / "dinoml_v2"))
    arch_num = _cmake_arch(arch)
    plan = build_external_kernel_plan({"name": "cuda", "arch": f"sm_{arch_num}"})
    families = [family.to_json() for family in external_kernel_families(provider="cutlass", backend="cuda")]
    if used_candidate_plan is None:
        used_candidate_plan = cutlass_gemm_used_candidate_plan(_kernel_manifest_from_families(families, {"name": "cuda", "arch": f"sm_{arch_num}"}))
    family_cache_key = _family_cache_key({"name": "cuda", "arch": f"sm_{arch_num}"}, families)
    used_candidate_plan_key = str(used_candidate_plan["used_candidate_plan_key"])
    manifest_key = cache_key or plan["cache_key"][:16]
    support_root = cache_root / "support" / f"cuda-{arch_num}" / "cutlass-gemm" / manifest_key
    src_dir = support_root / "src"
    lib_dir = support_root / "lib"
    src_dir.mkdir(parents=True, exist_ok=True)
    lib_dir.mkdir(parents=True, exist_ok=True)
    source = src_dir / "dinoml_cutlass_gemm.cu"
    library = lib_dir / "libdinoml_cutlass_gemm.so"
    manifest = lib_dir / "cutlass_gemm_manifest.json"
    source_manifest = src_dir / "source_manifest.json"
    repo_source = _repo_cutlass_gemm_source()
    include_roots = (
        *cutlass.include_roots,
        *(root.parent / "tools" / "util" / "include" for root in cutlass.include_roots if root.name == "include"),
    )
    repo_source_text = repo_source.read_text(encoding="utf-8")
    repo_source_hash = hashlib.sha256(repo_source_text.encode("utf-8")).hexdigest()
    rendered_source = render_cutlass_gemm_source(repo_source_text, used_candidate_plan)
    source_hash = hashlib.sha256(rendered_source.encode("utf-8")).hexdigest()
    compile_flags = _compile_flags(arch_num)
    include_args = [f"-I{root}" for root in include_roots if root.exists()]
    provenance = _build_provenance(
        arch_num=arch_num,
        plan_cache_key=plan["cache_key"],
        family_cache_key=family_cache_key,
        source_hash=source_hash,
        compile_flags=compile_flags,
        include_roots=include_roots,
        cutlass=cutlass,
        cublaslt=cublaslt,
    )
    if (
        library.exists()
        and manifest.exists()
        and source.exists()
        and source_manifest.exists()
        and _file_sha256(source) == source_hash
        and _cached_manifest_matches(
            manifest,
            source_hash,
            provenance["provenance_key"],
            _file_sha256(library),
            family_cache_key,
            used_candidate_plan_key,
        )
    ):
        return CutlassSupportLib(
            library=library,
            include_roots=tuple(root for root in include_roots if root.exists()),
            source=source,
            manifest=manifest,
            source_manifest=source_manifest,
        )
    source.write_text(rendered_source, encoding="utf-8")
    _write_source_manifest(
        source_manifest,
        target={"name": "cuda", "arch": f"sm_{arch_num}"},
        families=families,
        source=source,
        repo_source=repo_source,
        repo_source_hash=repo_source_hash,
        source_hash=source_hash,
        family_cache_key=family_cache_key,
        external_kernel_plan_cache_key=plan["cache_key"],
        used_candidate_plan=used_candidate_plan,
    )

    compile_command = ["nvcc", *compile_flags, *include_args, str(source), "-o", str(library)]
    _run_nvcc(compile_command, cwd=support_root)
    library_hash = _file_sha256(library)
    write_json(
        manifest,
        {
            "schema_version": 2,
            "target": {"name": "cuda", "arch": f"sm_{arch_num}"},
            "provider": "cutlass",
            "families": families,
            "library": library.name,
            "library_sha256": library_hash,
            "source": source.name,
            "source_sha256": source_hash,
            "source_manifest": "../src/source_manifest.json",
            "external_kernel_plan_cache_key": plan["cache_key"],
            "family_cache_key": family_cache_key,
            "used_candidate_plan_key": used_candidate_plan_key,
            "used_candidate_plan": dict(used_candidate_plan),
            "build_fingerprint": provenance["provenance_key"],
            "provenance_key": provenance["provenance_key"],
            "provenance": provenance,
            "compile": {
                "command": compile_command,
                "flags": compile_flags,
                "include_roots": [str(root) for root in include_roots if root.exists()],
            },
            "cache_key": manifest_key,
        },
    )
    return CutlassSupportLib(
        library=library,
        include_roots=tuple(root for root in include_roots if root.exists()),
        source=source,
        manifest=manifest,
        source_manifest=source_manifest,
    )


def _repo_cutlass_gemm_source() -> Path:
    repo_root = Path(__file__).resolve().parents[3]
    source = repo_root / "kernels" / "cuda" / "src" / "cutlass_gemm.cu"
    if not source.exists():
        raise FileNotFoundError(f"Missing CUTLASS GEMM source: {source}")
    return source


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _cached_manifest_matches(
    path: Path,
    source_hash: str,
    provenance_key: str,
    library_hash: str,
    family_cache_key: str,
    used_candidate_plan_key: str,
) -> bool:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return (
        payload.get("schema_version") == 2
        and payload.get("provider") == "cutlass"
        and payload.get("source_sha256") == source_hash
        and payload.get("provenance_key") == provenance_key
        and payload.get("build_fingerprint") == provenance_key
        and payload.get("library_sha256") == library_hash
        and payload.get("family_cache_key") == family_cache_key
        and payload.get("used_candidate_plan_key") == used_candidate_plan_key
    )


def _compile_flags(arch_num: str) -> list[str]:
    return [
        "-std=c++17",
        "-O3",
        "--use_fast_math",
        "--expt-relaxed-constexpr",
        "-shared",
        "-Xcompiler=-fPIC",
        f"-arch=sm_{arch_num}",
    ]


def _build_provenance(
    *,
    arch_num: str,
    plan_cache_key: str,
    family_cache_key: str,
    source_hash: str,
    compile_flags: Sequence[str],
    include_roots: Sequence[Path],
    cutlass: Any,
    cublaslt: Any,
) -> dict[str, Any]:
    dependencies = {
        "cutlass": _library_provenance(cutlass),
        "cublaslt": _library_provenance(cublaslt),
    }
    key_payload = {
        "schema_version": 1,
        "target": {"name": "cuda", "arch": f"sm_{arch_num}"},
        "external_kernel_plan_cache_key": plan_cache_key,
        "family_cache_key": family_cache_key,
        "source_sha256": source_hash,
        "compile_flags": list(compile_flags),
        "nvcc": _nvcc_version_payload(include_stdout=False),
        "dependencies": {
            name: _library_provenance_key_payload(payload)
            for name, payload in dependencies.items()
        },
    }
    return {
        "schema_version": 1,
        "target": {"name": "cuda", "arch": f"sm_{arch_num}"},
        "external_kernel_plan_cache_key": plan_cache_key,
        "family_cache_key": family_cache_key,
        "source_sha256": source_hash,
        "compile_flags": list(compile_flags),
        "include_roots": [str(root) for root in include_roots if root.exists()],
        "nvcc": _nvcc_version_payload(include_stdout=True),
        "dependencies": dependencies,
        "provenance_key": hashlib.sha256(json.dumps(key_payload, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest(),
    }


def _family_cache_key(target: Mapping[str, str], families: Sequence[Mapping[str, Any]]) -> str:
    payload = {
        "schema_version": 1,
        "target": dict(target),
        "provider": "cutlass",
        "families": list(families),
    }
    return hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()


def _kernel_manifest_from_families(families: Sequence[Mapping[str, Any]], target: Mapping[str, str]) -> dict[str, Any]:
    required = []
    for family in families:
        for dtype, candidates in sorted(dict(family.get("candidates_by_dtype", {})).items()):
            candidate_set = dict(family["candidate_sets_by_dtype"][dtype])
            candidates = [dict(candidate) for candidate in candidates]
            required.append(
                {
                    "op": family["op_name"],
                    "kernel_symbol": family["kernel_symbols_by_dtype"][dtype],
                    "kernel_library": "cutlass_gemm",
                    "profiler_symbol": family["profiler_symbols_by_dtype"][dtype],
                    "selected_candidate_id": candidates[0]["candidate_id"],
                    "candidates": candidates,
                    "candidate_set_id": candidate_set["candidate_set_id"],
                    "candidate_set_key": candidate_set["candidate_set_key"],
                    "candidate_set": candidate_set,
                }
            )
    return {
        "target": dict(target),
        "cache_key": hashlib.sha256(canonical_json({"target": dict(target), "required_kernels": required}).encode("utf-8")).hexdigest(),
        "support_cache_key": _family_cache_key(target, families),
        "required_kernels": required,
    }


def _write_source_manifest(
    path: Path,
    *,
    target: Mapping[str, str],
    families: Sequence[Mapping[str, Any]],
    source: Path,
    repo_source: Path,
    repo_source_hash: str,
    source_hash: str,
    family_cache_key: str,
    external_kernel_plan_cache_key: str,
    used_candidate_plan: Mapping[str, Any],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    candidate_sets = [dict(item) for item in used_candidate_plan.get("candidate_sets", [])]
    candidates = [
        {
            **dict(candidate),
            "source_ids": ["cutlass_gemm_static_default"],
            "profiler_source_ids": ["cutlass_gemm_static_default"],
        }
        for candidate in used_candidate_plan.get("candidates", [])
    ]
    manifest = {
        "schema_version": 2,
        "kind": "dinoml.support_source_manifest",
        "target": dict(target),
        "provider": "cutlass",
        "library": "cutlass_gemm",
        "kernel_abi_version": KERNEL_ABI_VERSION,
        "profile_cache_schema_version": PROFILE_CACHE_SCHEMA_VERSION,
        "family_cache_key": family_cache_key,
        "external_kernel_plan_cache_key": external_kernel_plan_cache_key,
        "used_candidate_plan_key": used_candidate_plan["used_candidate_plan_key"],
        "used_candidate_plan": dict(used_candidate_plan),
        "deduplication": {
            "source": "source_key",
            "candidate": "candidate_config_key",
            "candidate_set": "candidate_set_key",
        },
        "sources": [
            {
                "source_id": "cutlass_gemm_static_default",
                "source_key": source_hash,
                "source_role": "support_library",
                "generated": False,
                "language": "cuda",
                "emitted_source_path": source.name,
                "repo_source_path": str(repo_source),
                "repo_source_sha256": repo_source_hash,
                "source_sha256": source_hash,
                "candidate_set_keys": sorted({item["candidate_set_key"] for item in candidate_sets}),
                "candidate_config_keys": sorted({item["candidate_config_key"] for item in candidates}),
                "symbols": _source_symbols(candidates),
            }
        ],
        "candidate_sets": candidate_sets,
        "candidates": candidates,
        "build_units": [
            {
                "build_unit_id": "cutlass_gemm_shared",
                "source_ids": ["cutlass_gemm_static_default"],
                "output_role": "shared_library",
                "expected_outputs": [
                    {
                        "kind": "shared_library",
                        "path": "../lib/libdinoml_cutlass_gemm.so",
                    }
                ],
            }
        ],
    }
    manifest["source_manifest_key"] = hashlib.sha256(
        canonical_json({key: value for key, value in manifest.items() if key != "source_manifest_key"}).encode("utf-8")
    ).hexdigest()
    path.write_text(canonical_json(manifest), encoding="utf-8")


def _source_symbols(candidates: Sequence[Mapping[str, Any]]) -> list[dict[str, str]]:
    symbols = []
    for candidate in candidates:
        config_key = str(candidate["candidate_config_key"])
        symbols.append(
            {
                "kind": "kernel",
                "name": str(candidate["kernel_symbol"]),
                "candidate_config_key": config_key,
            }
        )
        symbols.append(
            {
                "kind": "profiler",
                "name": str(candidate["profiler_symbol"]),
                "candidate_config_key": config_key,
            }
        )
    return sorted(symbols, key=lambda item: (item["kind"], item["name"], item["candidate_config_key"]))


def _flat_candidate_sets(families: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    result = []
    for family in families:
        for dtype, candidate_set in sorted(dict(family.get("candidate_sets_by_dtype", {})).items()):
            result.append(
                {
                    **dict(candidate_set),
                    "op_name": family["op_name"],
                    "dtype": dtype,
                }
            )
    return result


def _flat_candidates(families: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    result = []
    for family in families:
        for dtype, candidates in sorted(dict(family.get("candidates_by_dtype", {})).items()):
            for candidate in candidates:
                result.append(
                    {
                        **dict(candidate),
                        "op_name": family["op_name"],
                        "dtype": dtype,
                        "source_ids": ["cutlass_gemm_static_default"],
                        "profiler_source_ids": ["cutlass_gemm_static_default"],
                    }
                )
    return result


def _library_provenance(library: Any) -> dict[str, Any]:
    include_roots = [Path(root) for root in getattr(library, "include_roots", ())]
    payload = {
        "name": getattr(library, "name", ""),
        "available": bool(getattr(library, "available", False)),
        "include_roots": [str(root) for root in include_roots if root.exists()],
        "headers": _header_provenance(include_roots, getattr(library, "headers", ())),
    }
    if getattr(library, "name", "") == "cutlass":
        root = _cutlass_repo_root(include_roots)
        if root is not None:
            payload["git"] = _git_provenance(root)
    return payload


def _library_provenance_key_payload(payload: Mapping[str, Any]) -> dict[str, Any]:
    result = {
        "name": payload.get("name"),
        "available": payload.get("available"),
        "headers": [
            {"header": item.get("header"), "sha256": item.get("sha256")}
            for item in payload.get("headers", [])
        ],
    }
    if "git" in payload:
        git = payload["git"]
        result["git"] = {
            "commit": git.get("commit"),
            "dirty": git.get("dirty"),
        }
    return result


def _header_provenance(include_roots: Sequence[Path], headers: Sequence[str]) -> list[dict[str, Any]]:
    result = []
    for header in headers:
        found = _find_header(include_roots, header)
        item: dict[str, Any] = {"header": header}
        if found is not None:
            item["path"] = str(found)
            item["sha256"] = _file_sha256(found)
        result.append(item)
    return result


def _find_header(include_roots: Sequence[Path], header: str) -> Path | None:
    for root in include_roots:
        candidate = root / header
        if candidate.exists():
            return candidate
    return None


def _cutlass_repo_root(include_roots: Sequence[Path]) -> Path | None:
    for include_root in include_roots:
        root = include_root.parent if include_root.name == "include" else include_root
        if (root / ".git").exists():
            return root
    return None


def _git_provenance(root: Path) -> dict[str, Any]:
    commit = _run_git(root, ["rev-parse", "HEAD"])
    try:
        dirty_proc = subprocess.run(["git", "-C", str(root), "diff", "--quiet"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=2.0)
        dirty = dirty_proc.returncode != 0
    except (OSError, subprocess.TimeoutExpired):
        dirty = None
    return {
        "root": str(root),
        "commit": commit,
        "dirty": dirty,
    }


def _run_git(root: Path, args: Sequence[str]) -> str | None:
    try:
        proc = subprocess.run(["git", "-C", str(root), *args], text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=2.0)
    except (OSError, subprocess.TimeoutExpired):
        return None
    if proc.returncode != 0:
        return None
    return proc.stdout.strip()


def _nvcc_version_payload(*, include_stdout: bool) -> dict[str, Any]:
    if shutil.which("nvcc") is None:
        return {"available": False}
    try:
        proc = subprocess.run(["nvcc", "--version"], text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=2.0)
    except (OSError, subprocess.TimeoutExpired):
        return {"available": False}
    if proc.returncode != 0:
        return {"available": False}
    release_match = re.search(r"release\s+([0-9.]+)", proc.stdout)
    build_match = re.search(r"V([0-9.]+)", proc.stdout)
    payload: dict[str, Any] = {"available": True}
    if release_match:
        payload["release"] = release_match.group(1)
    if build_match:
        payload["build"] = build_match.group(1)
    if include_stdout:
        payload["stdout"] = proc.stdout
    return payload


def _run_nvcc(cmd: list[str], *, cwd: Path) -> None:
    proc = subprocess.run(cmd, cwd=str(cwd), text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if proc.returncode != 0:
        raise RuntimeError(
            "CUTLASS support build failed\n"
            f"Command: {' '.join(cmd)}\n"
            f"stdout:\n{proc.stdout}\n"
            f"stderr:\n{proc.stderr}"
        )


def _cmake_arch(arch: str) -> str:
    match = re.fullmatch(r"sm_(\d+)", arch)
    if match:
        return match.group(1)
    if re.fullmatch(r"\d+", arch):
        return arch
    raise ValueError(f"Expected CUDA arch like 'sm_86' or '86', got {arch!r}")
