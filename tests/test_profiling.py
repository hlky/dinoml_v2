import shutil

import pytest

import dinoml as dml
import dinoml.cli as cli
from dinoml.backends.cuda_libraries import discover_cuda_libraries
from dinoml.ir import read_json, write_json
from dinoml.kernels.codegen import create_codegen_plan
from dinoml.kernels.manifest import build_kernel_manifest
from dinoml.kernels.profiling import (
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
    assert (workload.m, workload.n, workload.k) == (7, 11, 32)
    assert workload.output_shape == (7, 11)


def test_profile_artifact_uses_cache_before_running(tmp_path):
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
    write_json(artifact / "manifest.json", manifest)
    write_json(artifact / "graph.dinoir.json", lowered)
    write_json(artifact / "kernel_manifest.json", kernel_manifest)
    write_json(artifact / "kernel_codegen_plan.json", codegen_plan)

    workload = build_profile_workloads(lowered, kernel_manifest)[0]
    key_payload = _profile_key_payload(workload, manifest, kernel_manifest, codegen_plan)
    profile_key = _profile_key(key_payload)
    cached_result = _profile_result(workload, 0.125, 9, profile_key=profile_key, status="ok")
    cache = {
        "schema_version": 1,
        "target": {"name": "cuda", "arch": "sm_86"},
        "entries": {profile_key: _cache_entry(workload, cached_result, key_payload)},
    }
    cache_path = profile_cache_path(codegen_plan)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    write_json(cache_path, cache)

    report = profile_artifact(artifact, iterations=3)

    assert report["summary"] == {"cached": 1, "failed": 0, "profiled": 0, "skipped": 0}
    assert report["profile_cache_schema_version"] == 1
    assert report["kernel_manifest_cache_key"] == kernel_manifest["cache_key"]
    assert report["codegen_plan_cache_key"] == codegen_plan["cache_key"]
    assert report["problems"][0]["status"] == "cached"
    assert report["problems"][0]["selected"]["reason"] == "cache_hit"
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
    assert report["schema_version"] == 1
    assert report["profile_cache_schema_version"] == 1
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
    assert workload["candidates"][0]["candidate_id"] == "manifest_default"


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
