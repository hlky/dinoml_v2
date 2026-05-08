from dataclasses import replace
import hashlib
import shutil
from pathlib import Path

import pytest

import dinoml as dml
import dinoml.cli as cli
import dinoml.kernels.profiling as profiling_mod
from dinoml.backends.cuda_libraries import discover_cuda_libraries
from dinoml.ir import read_json, write_json
from dinoml.kernels.codegen import create_codegen_plan
from dinoml.kernels.manifest import PROFILE_CACHE_SCHEMA_VERSION, build_kernel_manifest
from dinoml.kernels.profiling import (
    PROFILE_REPORT_SCHEMA_VERSION,
    _cache_entry,
    _profile_key,
    _profile_key_payload,
    _profile_result,
    build_profile_workloads,
    parse_shape_overrides,
    profile_artifact,
    profile_cache_path,
)
from dinoml.passes import PassManager


class GemmModule(dml.Module):
    def __init__(self, op_name: str):
        self.op_name = op_name

    def forward(self, a, b):
        op = getattr(dml.ops, self.op_name)
        return dml.ops.output(op(a, b), "y")


def test_parse_shape_overrides():
    assert parse_shape_overrides(["x=1,128,768", "tokens=77"]) == {
        "x": (1, 128, 768),
        "tokens": (77,),
    }
    with pytest.raises(ValueError, match="Expected shape override"):
        parse_shape_overrides(["x:1,2"])


def test_build_profile_workloads_uses_runtime_shape_overrides():
    batch = dml.Dim("batch", min=1, max=16)
    tokens = dml.Dim("tokens", min=1, max=24)
    spec = dml.trace(
        GemmModule("gemm_rrr"),
        inputs={"a": dml.TensorSpec([batch, 32], "float16"), "b": dml.TensorSpec([32, tokens], "float16")},
        name="profile_dynamic_gemm",
    )
    lowered, _ = PassManager().run(spec.ir)
    manifest = build_kernel_manifest(lowered, {"name": "cuda", "arch": "sm_86"})

    workloads = build_profile_workloads(lowered, manifest, input_shapes={"a": (7, 32), "b": (32, 11)})

    assert len(workloads) == 1
    workload = workloads[0]
    assert workload.profiler_symbol == "dinoml_profile_cutlass_gemm_rrr_f16"
    assert workload.dtype == "float16"
    assert workload.candidate_set_id == "cutlass_gemm_rrr_f16_linear_combination_v1"
    assert workload.candidate_set_key
    assert workload.candidate_id == "cutlass_default"
    assert workload.candidate["provider"] == "cutlass"
    assert workload.candidate["layouts"] == {"a": "row", "b": "row", "c": "row"}
    assert workload.candidate_config_key
    assert (workload.m, workload.n, workload.k) == (7, 11, 32)
    assert workload.output_shape == (7, 11)


def test_profile_key_changes_with_fingerprint_keys(tmp_path):
    spec = dml.trace(
        GemmModule("gemm_rrr"),
        inputs={"a": dml.TensorSpec([4, 8], "float32"), "b": dml.TensorSpec([8, 6], "float32")},
        name="profile_key_fingerprint",
    )
    lowered, _ = PassManager().run(spec.ir)
    kernel_manifest = build_kernel_manifest(lowered, {"name": "cuda", "arch": "sm_86"})
    codegen_plan = create_codegen_plan(kernel_manifest, tmp_path / "cache").to_json()
    manifest = {"target": {"name": "cuda", "arch": "sm_86"}}
    workload = build_profile_workloads(lowered, kernel_manifest)[0]
    other_candidate = {**workload.candidate, "candidate_id": "cutlass_other", "candidate_config_key": "candidate-b"}
    other_workload = replace(
        workload,
        candidate_id="cutlass_other",
        candidate_config_key="candidate-b",
        candidate_set_key="candidate-set-b",
        candidate=other_candidate,
    )
    context_a = {"fingerprint": {"hardware_key": "hardware-a", "support_libraries_key": "support-a"}}
    context_b = {"fingerprint": {"hardware_key": "hardware-b", "support_libraries_key": "support-a"}}

    payload_a = _profile_key_payload(workload, manifest, kernel_manifest, codegen_plan, context=context_a)
    payload_b = _profile_key_payload(workload, manifest, kernel_manifest, codegen_plan, context=context_b)
    payload_c = _profile_key_payload(other_workload, manifest, kernel_manifest, codegen_plan, context=context_a)

    assert payload_a["hardware_fingerprint_key"] == "hardware-a"
    assert payload_a["support_libraries_fingerprint_key"] == "support-a"
    assert payload_a["candidate_id"] == "cutlass_default"
    assert payload_a["candidate_set_id"] == "cutlass_gemm_rrr_f32_linear_combination_v1"
    assert payload_a["candidate_set_key"] == workload.candidate_set_key
    assert payload_a["candidate_config_key"] == workload.candidate_config_key
    assert _profile_key(payload_a) != _profile_key(payload_b)
    assert _profile_key(payload_a) != _profile_key(payload_c)


def test_profile_artifact_uses_cache_before_running(tmp_path, monkeypatch):
    monkeypatch.setattr(
        profiling_mod,
        "_cuda_hardware_fingerprint",
        lambda target: {
            "backend": "cuda",
            "target_arch": target["arch"],
            "cuda_visible_devices": "",
            "nvidia_smi": "available",
            "devices": [
                {
                    "index": 0,
                    "name": "NVIDIA GeForce RTX 3090",
                    "compute_capability": "8.6",
                    "driver_version": "555.42",
                    "memory_total_mib": 24576,
                }
            ],
            "nvcc": {"available": "true", "release": "12.8", "build": "12.8.42"},
        },
    )
    spec = dml.trace(
        GemmModule("gemm_rrr"),
        inputs={"a": dml.TensorSpec([4, 8], "float32"), "b": dml.TensorSpec([8, 6], "float32")},
        name="cached_profile_gemm",
    )
    lowered, _ = PassManager().run(spec.ir)
    kernel_manifest = build_kernel_manifest(lowered, {"name": "cuda", "arch": "sm_86"})
    codegen_plan = create_codegen_plan(kernel_manifest, tmp_path / "cache").to_json()
    manifest = {
        "artifact_schema_version": 1,
        "runtime_abi_version": 5,
        "name": "cached_profile_gemm",
        "target": {"name": "cuda", "arch": "sm_86"},
        "files": {
            "graph": "graph.dinoir.json",
            "kernel_manifest": "kernel_manifest.json",
            "kernel_codegen_plan": "kernel_codegen_plan.json",
            "cutlass_gemm_library": "lib/libdinoml_cutlass_gemm.so",
        },
    }
    artifact = tmp_path / "cached_profile_gemm.dinoml"
    artifact.mkdir()
    artifact_lib = artifact / "lib" / "libdinoml_cutlass_gemm.so"
    artifact_lib.parent.mkdir()
    artifact_lib.write_bytes(b"artifact cutlass gemm")
    cache_dir = Path(codegen_plan["external_support_libraries"][0]["cache_dir"])
    cache_lib = cache_dir / "lib" / "libdinoml_cutlass_gemm.so"
    cache_lib.parent.mkdir(parents=True)
    cache_lib.write_bytes(b"cached cutlass gemm")
    write_json(
        cache_dir / "lib" / "cutlass_gemm_manifest.json",
        {
            "schema_version": 1,
            "target": {"name": "cuda", "arch": "sm_86"},
            "provider": "cutlass",
            "source_sha256": "source-hash",
            "library_sha256": "library-hash",
            "source_manifest": "../src/source_manifest.json",
            "provenance_key": "provenance-hash",
            "build_fingerprint": "provenance-hash",
            "family_cache_key": "family-hash",
            "external_kernel_plan_cache_key": "external-plan-hash",
            "compile": {"flags": ["-arch=sm_86"]},
            "provenance": {
                "provenance_key": "provenance-hash",
                "compile_flags": ["-arch=sm_86"],
            },
            "cache_key": "cutlass-cache",
        },
    )
    write_json(artifact / "manifest.json", manifest)
    write_json(artifact / "graph.dinoir.json", lowered)
    write_json(artifact / "kernel_manifest.json", kernel_manifest)
    write_json(artifact / "kernel_codegen_plan.json", codegen_plan)

    workload = build_profile_workloads(lowered, kernel_manifest)[0]
    context = profiling_mod._profile_context(artifact, manifest, codegen_plan)
    key_payload = _profile_key_payload(workload, manifest, kernel_manifest, codegen_plan, context=context)
    profile_key = _profile_key(key_payload)
    cached_result = _profile_result(workload, 0.125, 9, profile_key=profile_key, status="ok")
    cache = {
        "schema_version": PROFILE_CACHE_SCHEMA_VERSION,
        "target": {"name": "cuda", "arch": "sm_86"},
        "entries": {profile_key: _cache_entry(workload, cached_result, key_payload)},
    }
    cache_path = profile_cache_path(codegen_plan)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    write_json(cache_path, cache)

    report = profile_artifact(artifact, iterations=3)

    assert report["summary"] == {"cached": 1, "failed": 0, "profiled": 0, "skipped": 0}
    assert report["profile_cache_schema_version"] == PROFILE_CACHE_SCHEMA_VERSION
    assert report["kernel_manifest_cache_key"] == kernel_manifest["cache_key"]
    assert report["codegen_plan_cache_key"] == codegen_plan["cache_key"]
    assert report["fingerprint"]["schema_version"] == 1
    assert report["fingerprint"]["key"]
    assert report["hardware"]["devices"][0]["name"] == "NVIDIA GeForce RTX 3090"
    assert report["fingerprint"]["hardware_key"] == key_payload["hardware_fingerprint_key"]
    assert report["hardware_cache_key"] == key_payload["hardware_fingerprint_key"]
    assert report["libraries"][0]["artifact_sha256"] == hashlib.sha256(b"artifact cutlass gemm").hexdigest()
    assert report["fingerprint"]["support_libraries"][0]["source_sha256"] == "source-hash"
    assert report["fingerprint"]["support_libraries"][0]["library_sha256"] == "library-hash"
    assert report["fingerprint"]["support_libraries"][0]["source_manifest"] == "../src/source_manifest.json"
    assert report["fingerprint"]["support_libraries"][0]["provenance_key"] == "provenance-hash"
    assert report["fingerprint"]["support_libraries"][0]["build_fingerprint"] == "provenance-hash"
    assert report["fingerprint"]["support_libraries"][0]["family_cache_key"] == "family-hash"
    assert report["fingerprint"]["support_libraries"][0]["compile"]["flags"] == ["-arch=sm_86"]
    assert report["fingerprint"]["support_libraries"][0]["provenance"]["provenance_key"] == "provenance-hash"
    assert report["support_libraries_cache_key"] == key_payload["support_libraries_fingerprint_key"]
    assert report["problems"][0]["status"] == "cached"
    assert report["problems"][0]["selected"]["candidate_id"] == "cutlass_default"
    assert report["problems"][0]["selected"]["reason"] == "cache_hit"
    assert report["problems"][0]["candidates"][0]["candidate_config_key"]
    assert report["problems"][0]["profile_key"] == profile_key


@pytest.mark.skipif(shutil.which("nvcc") is None, reason="nvcc is required")
def test_cuda_profile_artifact_writes_cutlass_gemm_report(tmp_path, monkeypatch):
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA device is required")
    if not discover_cuda_libraries()["cutlass"].available:
        pytest.skip("CUTLASS headers are not available")
    monkeypatch.setenv("DINOML_CACHE_DIR", str(tmp_path / "cache"))

    spec = dml.trace(
        GemmModule("gemm_rcr"),
        inputs={"a": dml.TensorSpec([8, 16], "float32"), "b": dml.TensorSpec([12, 16], "float32")},
        name="profile_gemm_rcr",
    )
    artifact = dml.compile(spec, dml.Target("cuda", arch="sm_86"), tmp_path / "profile_gemm_rcr.dinoml")
    report = profile_artifact(artifact.path, iterations=3)

    report_path = artifact.path / "debug" / "profile_report.json"
    assert read_json(report_path) == report
    assert report["schema_version"] == PROFILE_REPORT_SCHEMA_VERSION
    assert report["profile_cache_schema_version"] == PROFILE_CACHE_SCHEMA_VERSION
    assert report["fingerprint"]["schema_version"] == 1
    assert report["hardware"]["backend"] == "cuda"
    assert report["fingerprint"]["hardware_key"] == report["hardware_cache_key"]
    assert report["fingerprint"]["support_libraries_key"] == report["support_libraries_cache_key"]
    assert report["fingerprint"]["support_libraries"][0]["name"] == "cutlass_gemm"
    assert report["libraries"][0]["artifact_sha256"]
    assert report["summary"] == {"cached": 0, "failed": 0, "profiled": 1, "skipped": 0}
    assert len(report["problems"]) == 1
    workload = report["problems"][0]
    assert workload["status"] == "ok"
    assert workload["profiler_symbol"] == "dinoml_profile_cutlass_gemm_rcr_f32"
    assert workload["m"] == 8
    assert workload["n"] == 12
    assert workload["k"] == 16
    assert workload["elapsed_ms"] >= 0.0
    assert workload["flops"] == 2 * 8 * 12 * 16
    assert workload["selected"]["reason"] == "only_candidate"
    assert workload["selected"]["candidate_id"] == "cutlass_default"
    candidate = workload["candidates"][0]
    assert candidate["candidate_id"] == "cutlass_default"
    assert candidate["provider"] == "cutlass"
    assert candidate["family"] == "gemm_universal"
    assert candidate["layouts"] == {"a": "row", "b": "column", "c": "row"}
    assert candidate["kernel_symbol"] == "dinoml_cutlass_gemm_rcr_f32"
    assert candidate["profiler_symbol"] == "dinoml_profile_cutlass_gemm_rcr_f32"
    assert candidate["candidate_config_key"]


def test_cli_profile_smoke(monkeypatch, capsys):
    calls = []

    def fake_profile_artifact(artifact, *, input_shapes, iterations, output, refresh):
        calls.append((artifact, input_shapes, iterations, output, refresh))
        return {
            "artifact": str(artifact),
            "target": {"name": "cuda", "arch": "sm_86"},
            "iterations": iterations,
            "summary": {"cached": 0, "failed": 0, "profiled": 1, "skipped": 0},
            "problems": [
                {
                    "node_id": "n0",
                    "op": "gemm_rrr",
                    "dtype": "float32",
                    "profiler_symbol": "dinoml_profile_cutlass_gemm_rrr_f32",
                    "m": 4,
                    "n": 6,
                    "k": 8,
                    "elapsed_ms": 0.01,
                    "tflops": 0.00384,
                }
            ],
        }

    monkeypatch.setattr(cli, "profile_artifact", fake_profile_artifact)

    assert cli.main(["profile", "artifact.dinoml", "--iterations", "2", "--shape", "a=4,8", "--out", "report.json", "--refresh"]) == 0
    stdout = capsys.readouterr().out

    assert calls == [("artifact.dinoml", {"a": (4, 8)}, 2, "report.json", True)]
    assert "dinoml_profile_cutlass_gemm_rrr_f32" in stdout
