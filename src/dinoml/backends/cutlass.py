from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from dinoml.backends.cuda_libraries import require_cuda_library
from dinoml.ir import canonical_json, write_json
from dinoml.kernels.external import external_kernel_families
from dinoml.kernels.manifest import KERNEL_ABI_VERSION, PROFILE_CACHE_SCHEMA_VERSION, build_external_kernel_plan
from dinoml.kernels.providers.cutlass.bmm import cutlass_bmm_used_candidate_plan, render_cutlass_bmm_source
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
    return _ensure_cutlass_support_lib(
        arch,
        cache_key=cache_key,
        used_candidate_plan=used_candidate_plan,
        library_name="cutlass_gemm",
        family_name="gemm_universal",
        support_dir_name="cutlass-gemm",
        source_name="dinoml_cutlass_gemm.cu",
        library_file_name="libdinoml_cutlass_gemm.so",
        manifest_file_name="cutlass_gemm_manifest.json",
        repo_source=_repo_cutlass_gemm_source(),
        render_source=render_cutlass_gemm_source,
        used_candidate_plan_builder=cutlass_gemm_used_candidate_plan,
        source_id="cutlass_gemm_static_default",
        build_unit_id="cutlass_gemm_shared",
    )


def ensure_cutlass_bmm_support_lib(
    arch: str,
    *,
    cache_key: str | None = None,
    used_candidate_plan: Mapping[str, Any] | None = None,
) -> CutlassSupportLib:
    return _ensure_cutlass_support_lib(
        arch,
        cache_key=cache_key,
        used_candidate_plan=used_candidate_plan,
        library_name="cutlass_bmm",
        family_name="bmm_strided",
        support_dir_name="cutlass-bmm",
        source_name="dinoml_cutlass_bmm.cu",
        library_file_name="libdinoml_cutlass_bmm.so",
        manifest_file_name="cutlass_bmm_manifest.json",
        repo_source=_repo_cutlass_bmm_source(),
        render_source=render_cutlass_bmm_source,
        used_candidate_plan_builder=cutlass_bmm_used_candidate_plan,
        source_id="cutlass_bmm_static_default",
        build_unit_id="cutlass_bmm_shared",
    )


def _ensure_cutlass_support_lib(
    arch: str,
    *,
    cache_key: str | None,
    used_candidate_plan: Mapping[str, Any] | None,
    library_name: str,
    family_name: str,
    support_dir_name: str,
    source_name: str,
    library_file_name: str,
    manifest_file_name: str,
    repo_source: Path,
    render_source: Any,
    used_candidate_plan_builder: Any,
    source_id: str,
    build_unit_id: str,
) -> CutlassSupportLib:
    cutlass = require_cuda_library("cutlass")
    cublaslt = require_cuda_library("cublaslt")
    cache_root = Path(os.environ.get("DINOML_CACHE_DIR", Path.home() / ".cache" / "dinoml_v2"))
    arch_num = _cmake_arch(arch)
    default_target = {"name": "cuda", "arch": f"sm_{arch_num}"}
    families = [
        family.to_json()
        for family in external_kernel_families(provider="cutlass", backend="cuda")
        if family.family == family_name
    ]
    if used_candidate_plan is None:
        used_candidate_plan = used_candidate_plan_builder(
            _kernel_manifest_from_families(families, default_target, library_name=library_name)
        )
    target = dict(used_candidate_plan.get("target", default_target))
    plan = build_external_kernel_plan(target)
    family_cache_key = _family_cache_key(target, families)
    used_candidate_plan_key = str(used_candidate_plan["used_candidate_plan_key"])
    manifest_key = cache_key or plan["cache_key"][:16]
    support_root = cache_root / "support" / f"cuda-{arch_num}" / support_dir_name / manifest_key
    src_dir = support_root / "src"
    lib_dir = support_root / "lib"
    src_dir.mkdir(parents=True, exist_ok=True)
    lib_dir.mkdir(parents=True, exist_ok=True)
    source = src_dir / source_name
    library = lib_dir / library_file_name
    manifest = lib_dir / manifest_file_name
    source_manifest = src_dir / "source_manifest.json"
    include_roots = (
        *cutlass.include_roots,
        *(root.parent / "tools" / "util" / "include" for root in cutlass.include_roots if root.name == "include"),
    )
    repo_source_text = repo_source.read_text(encoding="utf-8")
    repo_source_hash = hashlib.sha256(repo_source_text.encode("utf-8")).hexdigest()
    rendered_source = render_source(repo_source_text, used_candidate_plan)
    source_hash = hashlib.sha256(rendered_source.encode("utf-8")).hexdigest()
    source_metrics = _support_source_metrics(rendered_source, used_candidate_plan)
    compile_flags = _compile_flags(arch_num)
    include_args = [f"-I{root}" for root in include_roots if root.exists()]
    provenance = _build_provenance(
        target=target,
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
        target=target,
        families=families,
        source=source,
        repo_source=repo_source,
        repo_source_hash=repo_source_hash,
        source_hash=source_hash,
        source_metrics=source_metrics,
        family_cache_key=family_cache_key,
        external_kernel_plan_cache_key=plan["cache_key"],
        used_candidate_plan=used_candidate_plan,
        library_name=library_name,
        source_id=source_id,
        build_unit_id=build_unit_id,
        library_file_name=library_file_name,
    )

    compile_command = ["nvcc", *compile_flags, *include_args, str(source), "-o", str(library)]
    compile_started = time.perf_counter()
    _run_nvcc(compile_command, cwd=support_root)
    compile_duration_ms = round((time.perf_counter() - compile_started) * 1000.0, 3)
    library_hash = _file_sha256(library)
    write_json(
        manifest,
        {
            "schema_version": 2,
            "target": target,
            "provider": "cutlass",
            "library_name": library_name,
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
                "duration_ms": compile_duration_ms,
                "source_metrics": source_metrics,
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


def _repo_cutlass_bmm_source() -> Path:
    repo_root = Path(__file__).resolve().parents[3]
    source = repo_root / "kernels" / "cuda" / "src" / "cutlass_bmm.cu"
    if not source.exists():
        raise FileNotFoundError(f"Missing CUTLASS BMM source: {source}")
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
    compile_payload = payload.get("compile")
    return (
        payload.get("schema_version") == 2
        and payload.get("provider") == "cutlass"
        and payload.get("source_sha256") == source_hash
        and payload.get("provenance_key") == provenance_key
        and payload.get("build_fingerprint") == provenance_key
        and payload.get("library_sha256") == library_hash
        and payload.get("family_cache_key") == family_cache_key
        and payload.get("used_candidate_plan_key") == used_candidate_plan_key
        and isinstance(compile_payload, Mapping)
        and isinstance(compile_payload.get("source_metrics"), Mapping)
        and compile_payload.get("duration_ms") is not None
    )


def _compile_flags(arch_num: str) -> list[str]:
    flags = [
        "-std=c++17",
        "-O3",
        "--use_fast_math",
        "--expt-relaxed-constexpr",
        "-shared",
        "-Xcompiler=-fPIC",
        f"-arch=sm_{arch_num}",
    ]
    split_compile = _nvcc_split_compile_flag()
    if split_compile is not None:
        flags.append(split_compile)
    return flags


def _nvcc_split_compile_flag() -> str | None:
    raw_jobs = os.environ.get("DINOML_NVCC_SPLIT_COMPILE", "8")
    try:
        jobs = int(raw_jobs)
    except ValueError:
        return None
    if jobs <= 1 or not _nvcc_supports_option("--split-compile"):
        return None
    return f"--split-compile={jobs}"


def _nvcc_supports_option(option: str) -> bool:
    if shutil.which("nvcc") is None:
        return False
    try:
        result = subprocess.run(
            ["nvcc", "--help"],
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
    except OSError:
        return False
    return result.returncode == 0 and option in result.stdout


def _build_provenance(
    *,
    target: Mapping[str, Any],
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
        "target": dict(target),
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
        "target": dict(target),
        "external_kernel_plan_cache_key": plan_cache_key,
        "family_cache_key": family_cache_key,
        "source_sha256": source_hash,
        "compile_flags": list(compile_flags),
        "include_roots": [str(root) for root in include_roots if root.exists()],
        "nvcc": _nvcc_version_payload(include_stdout=True),
        "dependencies": dependencies,
        "provenance_key": hashlib.sha256(json.dumps(key_payload, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest(),
    }


def _family_cache_key(target: Mapping[str, Any], families: Sequence[Mapping[str, Any]]) -> str:
    payload = {
        "schema_version": 1,
        "target": dict(target),
        "provider": "cutlass",
        "families": list(families),
    }
    return hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()


def _kernel_manifest_from_families(
    families: Sequence[Mapping[str, Any]],
    target: Mapping[str, Any],
    *,
    library_name: str = "cutlass_gemm",
) -> dict[str, Any]:
    required = []
    for family in families:
        for dtype, candidates in sorted(dict(family.get("candidates_by_dtype", {})).items()):
            candidate_set = dict(family["candidate_sets_by_dtype"][dtype])
            candidates = [dict(candidate) for candidate in candidates]
            default_candidates = candidates[:1]
            required.append(
                {
                    "op": family["op_name"],
                    "kernel_symbol": family["kernel_symbols_by_dtype"][dtype],
                    "kernel_library": library_name,
                    "profiler_symbol": family["profiler_symbols_by_dtype"][dtype],
                    "selected_candidate_id": default_candidates[0]["candidate_id"],
                    "candidates": default_candidates,
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


def _support_source_metrics(source_text: str, used_candidate_plan: Mapping[str, Any]) -> dict[str, int]:
    entries = [item for item in used_candidate_plan.get("entries", []) if isinstance(item, Mapping)]
    candidates = [item for item in used_candidate_plan.get("candidates", []) if isinstance(item, Mapping)]
    candidate_sets = [item for item in used_candidate_plan.get("candidate_sets", []) if isinstance(item, Mapping)]
    kernel_symbols = {str(symbol) for symbol in used_candidate_plan.get("kernel_symbols", [])}
    profiler_symbols = {str(symbol) for symbol in used_candidate_plan.get("profiler_symbols", [])}
    return {
        "source_nbytes": len(source_text.encode("utf-8")),
        "source_line_count": len(source_text.splitlines()),
        "entry_count": len(entries),
        "candidate_count": len(candidates),
        "candidate_set_count": len(candidate_sets),
        "kernel_symbol_count": len(kernel_symbols),
        "profiler_symbol_count": len(profiler_symbols),
        "symbol_count": len(kernel_symbols | profiler_symbols),
    }


def _write_source_manifest(
    path: Path,
    *,
    target: Mapping[str, Any],
    families: Sequence[Mapping[str, Any]],
    source: Path,
    repo_source: Path,
    repo_source_hash: str,
    source_hash: str,
    source_metrics: Mapping[str, int],
    family_cache_key: str,
    external_kernel_plan_cache_key: str,
    used_candidate_plan: Mapping[str, Any],
    library_name: str,
    source_id: str,
    build_unit_id: str,
    library_file_name: str,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    candidate_sets = [dict(item) for item in used_candidate_plan.get("candidate_sets", [])]
    candidates = [
        {
            **dict(candidate),
            "source_ids": [source_id],
            "profiler_source_ids": [source_id],
        }
        for candidate in used_candidate_plan.get("candidates", [])
    ]
    manifest = {
        "schema_version": 2,
        "kind": "dinoml.support_source_manifest",
        "target": dict(target),
        "provider": "cutlass",
        "library": library_name,
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
                "source_id": source_id,
                "source_key": source_hash,
                "source_role": "support_library",
                "generated": False,
                "language": "cuda",
                "emitted_source_path": source.name,
                "repo_source_path": str(repo_source),
                "repo_source_sha256": repo_source_hash,
                "source_sha256": source_hash,
                "source_metrics": dict(source_metrics),
                "candidate_set_keys": sorted({item["candidate_set_key"] for item in candidate_sets}),
                "candidate_config_keys": sorted({item["candidate_config_key"] for item in candidates}),
                "symbols": _source_symbols(candidates),
            }
        ],
        "candidate_sets": candidate_sets,
        "candidates": candidates,
        "build_units": [
            {
                "build_unit_id": build_unit_id,
                "source_ids": [source_id],
                "output_role": "shared_library",
                "expected_outputs": [
                    {
                        "kind": "shared_library",
                        "path": f"../lib/{library_file_name}",
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
