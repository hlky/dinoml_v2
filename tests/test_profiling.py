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
from dinoml.kernels.providers.cutlass.gemm import cutlass_gemm_candidates
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


DEFAULT_CUDA_TARGET = {"name": "cuda", "arch": "sm_86"}


def _cutlass_candidates(dtype: str, *, op_name: str = "gemm_rrr", target=None) -> tuple[dict[str, object], ...]:
    return cutlass_gemm_candidates(op_name, dtype, target=target or DEFAULT_CUDA_TARGET)


def _cutlass_candidate_count(dtype: str, *, op_name: str = "gemm_rrr", target=None) -> int:
    return len(_cutlass_candidates(dtype, op_name=op_name, target=target))


def _cutlass_default_candidate_id(dtype: str, *, op_name: str = "gemm_rrr", target=None) -> str:
    return str(_cutlass_candidates(dtype, op_name=op_name, target=target)[0]["candidate_id"])


def _cutlass_default_symbol_id(dtype: str, *, op_name: str = "gemm_rrr", target=None) -> str:
    return str(_cutlass_candidates(dtype, op_name=op_name, target=target)[0]["symbol_id"])


class GemmModule(dml.Module):
    def __init__(self, op_name: str):
        self.op_name = op_name

    def forward(self, a, b):
        op = getattr(dml.ops, self.op_name)
        return dml.ops.output(op(a, b), "y")


class GemmBiasModule(dml.Module):
    def __init__(self, op_name: str):
        self.op_name = op_name

    def forward(self, a, b, bias):
        op = getattr(dml.ops, self.op_name)
        return dml.ops.output(op(a, b, bias), "y")


class GemmResidualModule(dml.Module):
    def __init__(self, op_name: str):
        self.op_name = op_name

    def forward(self, a, b, bias, d0, d1=None):
        op = getattr(dml.ops, self.op_name)
        if d1 is None:
            return dml.ops.output(op(a, b, bias, d0), "y")
        return dml.ops.output(op(a, b, bias, d0, d1), "y")


GEMM_BIAS_RESIDUAL_CASES = tuple(
    (f"gemm_{layout}_bias_{suffix}", layout, epilogue, inputs)
    for layout in ("rcr", "rrr")
    for suffix, epilogue, inputs in (
        ("add", "bias_add", ("bias", "d0")),
        ("add_add", "bias_add_add", ("bias", "d0", "d1")),
        ("mul", "bias_mul", ("bias", "d0")),
        ("mul_add", "bias_mul_add", ("bias", "d0", "d1")),
    )
)
GEMM_BIAS_RESIDUAL_CASES = (
    *GEMM_BIAS_RESIDUAL_CASES,
    ("gemm_rcr_bias_add_relu", "rcr", "bias_add_relu", ("bias", "d0")),
    ("gemm_rcr_bias_add_add_relu", "rcr", "bias_add_add_relu", ("bias", "d0", "d1")),
    ("gemm_rcr_bias_mul_tanh", "rcr", "bias_mul_tanh", ("bias", "d0")),
    ("gemm_rcr_bias_sigmoid_mul", "rcr", "bias_sigmoid_mul", ("bias", "d0")),
    ("gemm_rcr_bias_sigmoid_mul_tanh", "rcr", "bias_sigmoid_mul_tanh", ("bias", "d0")),
)


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

    assert len(workloads) == _cutlass_candidate_count("float16")
    workload = workloads[0]
    assert workload.profiler_symbol == f"dinoml_profile_cutlass_gemm_rrr_float16_{_cutlass_default_symbol_id('float16')}"
    assert workload.dtype == "float16"
    assert workload.candidate_set_id == "cutlass_gemm_rrr_float16_linear_combination_v1"
    assert workload.candidate_set_key
    assert workload.candidate_id == _cutlass_default_candidate_id("float16")
    assert workload.candidate["provider"] == "cutlass"
    assert workload.candidate["layouts"] == {"a": "row", "b": "row", "c": "row"}
    assert workload.candidate["cutlass"]["opclass"] == "tensorop"
    assert workload.candidate["cutlass"]["threadblock"] == [256, 128, 32]
    assert workload.candidate_config_key
    assert (workload.m, workload.n, workload.k) == (7, 11, 32)
    assert workload.output_shape == (7, 11)


def test_build_profile_workloads_uses_manifest_selected_fp16_accumulation_policy():
    target = {**DEFAULT_CUDA_TARGET, "use_fp16_acc": True}
    spec = dml.trace(
        GemmModule("gemm_rrr"),
        inputs={"a": dml.TensorSpec([7, 32], "float16"), "b": dml.TensorSpec([32, 11], "float16")},
        name="profile_fp16_acc_gemm",
    )
    lowered, _ = PassManager().run(spec.ir)
    manifest = build_kernel_manifest(lowered, target)

    workloads = build_profile_workloads(lowered, manifest)

    assert len(workloads) == _cutlass_candidate_count("float16", target=target)
    assert workloads
    assert {workload.candidate["accumulator_dtype"] for workload in workloads} == {"float16"}
    assert all("_f16_" in workload.kernel_symbol for workload in workloads)
    assert all("_f16_" in workload.profiler_symbol for workload in workloads)
    assert all("_f32_" not in workload.kernel_symbol for workload in workloads)
    assert workloads[0].candidate_id == _cutlass_default_candidate_id("float16", target=target)


def test_build_profile_workloads_uses_manifest_selected_no_tf32_policy():
    target = {**DEFAULT_CUDA_TARGET, "no_tf32": True}
    spec = dml.trace(
        GemmModule("gemm_rrr"),
        inputs={"a": dml.TensorSpec([7, 32], "float32"), "b": dml.TensorSpec([32, 11], "float32")},
        name="profile_no_tf32_gemm",
    )
    lowered, _ = PassManager().run(spec.ir)
    manifest = build_kernel_manifest(lowered, target)

    workloads = build_profile_workloads(lowered, manifest)

    assert len(workloads) == 11
    assert {workload.candidate["cutlass"]["opclass"] for workload in workloads} == {"simt"}
    assert {workload.candidate["cutlass"]["math"] for workload in workloads} == {"f32"}
    assert all("simt_sm80_f32" in workload.kernel_symbol for workload in workloads)
    assert all("tf32" not in workload.kernel_symbol for workload in workloads)
    assert workloads[0].candidate_id == _cutlass_default_candidate_id("float32", target=target)


@pytest.mark.parametrize(
    ("op_name", "epilogue"),
    [
        ("gemm_rcr_bias", "bias"),
        ("gemm_rcr_bias_relu", "bias_relu"),
    ],
)
def test_build_profile_workloads_supports_gemm_bias_epilogue(op_name, epilogue):
    spec = dml.trace(
        GemmBiasModule(op_name),
        inputs={
            "a": dml.TensorSpec([7, 32], "float32"),
            "b": dml.TensorSpec([11, 32], "float32"),
            "bias": dml.TensorSpec([11], "float32"),
        },
        name=f"profile_{op_name}",
    )
    lowered, _ = PassManager().run(spec.ir)
    manifest = build_kernel_manifest(lowered, {"name": "cuda", "arch": "sm_86"})

    workloads = build_profile_workloads(lowered, manifest)

    assert len(workloads) == _cutlass_candidate_count("float32")
    workload = workloads[0]
    assert workload.profiler_symbol == f"dinoml_profile_cutlass_{op_name}_float32_{_cutlass_default_symbol_id('float32')}"
    assert workload.candidate_set_id == f"cutlass_{op_name}_float32_{epilogue}_v1"
    assert workload.bias_tensor == "bias"
    assert workload.bias_shape == (11,)
    assert workload.candidate["epilogue"] == epilogue
    assert workload.candidate["epilogue_config"]["inputs"] == ["bias"]
    assert workload.candidate["epilogue_config"]["activation"] == ("relu" if epilogue == "bias_relu" else None)
    assert workload.to_json()["inputs"]["bias"] == [11]


@pytest.mark.parametrize(("op_name", "layout", "epilogue", "epilogue_inputs"), GEMM_BIAS_RESIDUAL_CASES)
def test_build_profile_workloads_supports_gemm_residual_epilogue_inputs(op_name, layout, epilogue, epilogue_inputs):
    input_specs = {
        "a": dml.TensorSpec([7, 32], "float32"),
        "b": dml.TensorSpec([11, 32] if layout == "rcr" else [32, 11], "float32"),
        "bias": dml.TensorSpec([11], "float32"),
        "d0": dml.TensorSpec([7, 11], "float32"),
    }
    if "d1" in epilogue_inputs:
        input_specs["d1"] = dml.TensorSpec([7, 11], "float32")
    spec = dml.trace(GemmResidualModule(op_name), inputs=input_specs, name=f"profile_{op_name}")
    lowered, _ = PassManager().run(spec.ir)
    manifest = build_kernel_manifest(lowered, {"name": "cuda", "arch": "sm_86"})

    workloads = build_profile_workloads(lowered, manifest)

    assert len(workloads) == _cutlass_candidate_count("float32")
    workload = workloads[0]
    assert workload.profiler_symbol == f"dinoml_profile_cutlass_{op_name}_float32_{_cutlass_default_symbol_id('float32')}"
    assert workload.candidate_set_id == f"cutlass_{op_name}_float32_{epilogue}_v1"
    assert workload.bias_tensor == "bias"
    assert workload.bias_shape == (11,)
    assert workload.candidate["epilogue"] == epilogue
    assert workload.candidate["epilogue_config"]["inputs"] == list(epilogue_inputs)
    expected_activation = {
        "bias_add_relu": "relu",
        "bias_add_add_relu": "relu",
        "bias_mul_tanh": "tanh",
        "bias_sigmoid_mul_tanh": "tanh",
    }.get(epilogue)
    expected_pre_activation = "sigmoid" if epilogue in {"bias_sigmoid_mul", "bias_sigmoid_mul_tanh"} else None
    assert workload.candidate["epilogue_config"]["activation"] == expected_activation
    assert workload.candidate["epilogue_config"].get("pre_residual_activation") == expected_pre_activation
    assert workload.candidate["launch_abi"].startswith("dinoml_cutlass_gemm_")
    inputs = workload.to_json()["inputs"]
    assert inputs["bias"] == [11]
    assert inputs["d0"] == [7, 11]
    if "d1" in epilogue_inputs:
        assert inputs["d1"] == [7, 11]
    else:
        assert "d1" not in inputs


@pytest.mark.parametrize(
    ("op_name", "epilogue"),
    [
        ("gemm_rcr_bias_add", "bias_add"),
        ("gemm_rcr_bias_mul", "bias_mul"),
        ("gemm_rcr_bias_mul_tanh", "bias_mul_tanh"),
        ("gemm_rcr_bias_sigmoid_mul", "bias_sigmoid_mul"),
        ("gemm_rcr_bias_sigmoid_mul_tanh", "bias_sigmoid_mul_tanh"),
    ],
)
def test_build_profile_workloads_flattens_gemm_rcr_single_residual_folded_m(op_name, epilogue):
    spec = dml.trace(
        GemmResidualModule(op_name),
        inputs={
            "a": dml.TensorSpec([2, 3, 32], "float32"),
            "b": dml.TensorSpec([11, 32], "float32"),
            "bias": dml.TensorSpec([11], "float32"),
            "d0": dml.TensorSpec([2, 3, 11], "float32"),
        },
        name=f"profile_{op_name}_folded_m",
    )
    lowered, _ = PassManager().run(spec.ir)
    manifest = build_kernel_manifest(lowered, {"name": "cuda", "arch": "sm_86"})

    workloads = build_profile_workloads(lowered, manifest)

    assert len(workloads) == _cutlass_candidate_count("float32")
    workload = workloads[0]
    assert (workload.m, workload.n, workload.k) == (6, 11, 32)
    assert workload.a_shape == (2, 3, 32)
    assert workload.b_shape == (11, 32)
    assert workload.residual_shapes == ((2, 3, 11),)
    assert workload.output_shape == (2, 3, 11)
    assert workload.candidate_set_id == f"cutlass_{op_name}_float32_{epilogue}_v1"
    assert workload.to_json()["inputs"]["a"] == [2, 3, 32]
    assert workload.to_json()["inputs"]["d0"] == [2, 3, 11]
    assert list(workload.to_json()["output"].values()) == [[2, 3, 11]]


@pytest.mark.parametrize(
    ("op_name", "epilogue"),
    [
        ("gemm_rcr_bias_add_add", "bias_add_add"),
        ("gemm_rcr_bias_mul_add", "bias_mul_add"),
        ("gemm_rcr_bias_add_add_relu", "bias_add_add_relu"),
    ],
)
def test_build_profile_workloads_flattens_gemm_rcr_dual_residual_folded_m(op_name, epilogue):
    spec = dml.trace(
        GemmResidualModule(op_name),
        inputs={
            "a": dml.TensorSpec([2, 3, 32], "float32"),
            "b": dml.TensorSpec([11, 32], "float32"),
            "bias": dml.TensorSpec([11], "float32"),
            "d0": dml.TensorSpec([2, 3, 11], "float32"),
            "d1": dml.TensorSpec([2, 3, 11], "float32"),
        },
        name=f"profile_{op_name}_folded_m",
    )
    lowered, _ = PassManager().run(spec.ir)
    manifest = build_kernel_manifest(lowered, {"name": "cuda", "arch": "sm_86"})

    workloads = build_profile_workloads(lowered, manifest)

    assert len(workloads) == _cutlass_candidate_count("float32")
    workload = workloads[0]
    assert (workload.m, workload.n, workload.k) == (6, 11, 32)
    assert workload.residual_tensors == ("d0", "d1")
    assert workload.residual_shapes == ((2, 3, 11), (2, 3, 11))
    assert workload.output_shape == (2, 3, 11)
    assert workload.candidate_set_id == f"cutlass_{op_name}_float32_{epilogue}_v1"
    inputs = workload.to_json()["inputs"]
    assert inputs["d0"] == [2, 3, 11]
    assert inputs["d1"] == [2, 3, 11]
    assert list(workload.to_json()["output"].values()) == [[2, 3, 11]]


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
    assert payload_a["candidate_id"] == _cutlass_default_candidate_id("float32")
    assert payload_a["candidate_set_id"] == "cutlass_gemm_rrr_float32_linear_combination_v1"
    assert payload_a["candidate_set_key"] == workload.candidate_set_key
    assert payload_a["candidate_config_key"] == workload.candidate_config_key
    assert payload_a["layouts"] == workload.candidate["layouts"]
    assert payload_a["epilogue"] == workload.candidate["epilogue"]
    assert payload_a["epilogue_config"] == workload.candidate["epilogue_config"]
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
            "used_candidate_plan_key": "used-plan-hash",
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

    context = profiling_mod._profile_context(artifact, manifest, codegen_plan)
    workloads = build_profile_workloads(lowered, kernel_manifest)
    cached_entries = {}
    key_payload = None
    for workload in workloads:
        key_payload = _profile_key_payload(workload, manifest, kernel_manifest, codegen_plan, context=context)
        profile_key = _profile_key(key_payload)
        cached_result = _profile_result(workload, 0.125, 9, profile_key=profile_key, status="ok")
        cached_entries[profile_key] = _cache_entry(workload, cached_result, key_payload)
    cache = {
        "schema_version": PROFILE_CACHE_SCHEMA_VERSION,
        "target": {"name": "cuda", "arch": "sm_86"},
        "entries": cached_entries,
    }
    cache_path = profile_cache_path(codegen_plan)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    write_json(cache_path, cache)

    report = profile_artifact(artifact, iterations=3)

    assert report["summary"] == {"cached": _cutlass_candidate_count("float32"), "failed": 0, "profiled": 0, "skipped": 0}
    assert report["profile_cache_schema_version"] == PROFILE_CACHE_SCHEMA_VERSION
    assert report["kernel_manifest_cache_key"] == kernel_manifest["cache_key"]
    assert report["codegen_plan_cache_key"] == codegen_plan["cache_key"]
    assert report["fingerprint"]["schema_version"] == 1
    assert report["fingerprint"]["key"]
    assert report["hardware"]["devices"][0]["name"] == "NVIDIA GeForce RTX 3090"
    assert key_payload is not None
    assert report["fingerprint"]["hardware_key"] == key_payload["hardware_fingerprint_key"]
    assert report["hardware_cache_key"] == key_payload["hardware_fingerprint_key"]
    assert report["libraries"][0]["artifact_sha256"] == hashlib.sha256(b"artifact cutlass gemm").hexdigest()
    assert report["fingerprint"]["support_libraries"][0]["source_sha256"] == "source-hash"
    assert report["fingerprint"]["support_libraries"][0]["library_sha256"] == "library-hash"
    assert report["fingerprint"]["support_libraries"][0]["source_manifest"] == "../src/source_manifest.json"
    assert report["fingerprint"]["support_libraries"][0]["provenance_key"] == "provenance-hash"
    assert report["fingerprint"]["support_libraries"][0]["build_fingerprint"] == "provenance-hash"
    assert report["fingerprint"]["support_libraries"][0]["family_cache_key"] == "family-hash"
    assert report["fingerprint"]["support_libraries"][0]["used_candidate_plan_key"] == "used-plan-hash"
    assert report["fingerprint"]["support_libraries"][0]["compile"]["flags"] == ["-arch=sm_86"]
    assert report["fingerprint"]["support_libraries"][0]["provenance"]["provenance_key"] == "provenance-hash"
    assert report["support_libraries_cache_key"] == key_payload["support_libraries_fingerprint_key"]
    assert report["problems"][0]["status"] == "cached"
    assert report["problems"][0]["selected"]["candidate_id"] == _cutlass_default_candidate_id("float32")
    assert report["problems"][0]["selected"]["reason"] == "cache_hit"
    assert report["problems"][0]["candidates"][0]["candidate_config_key"]
    assert report["problems"][0]["profile_key"] in cached_entries


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
    assert report["summary"] == {"cached": 0, "failed": 0, "profiled": _cutlass_candidate_count("float32"), "skipped": 0}
    assert len(report["problems"]) == _cutlass_candidate_count("float32")
    workload = report["problems"][0]
    assert workload["status"] == "ok"
    assert workload["profiler_symbol"] == f"dinoml_profile_cutlass_gemm_rcr_float32_{_cutlass_default_symbol_id('float32')}"
    assert workload["m"] == 8
    assert workload["n"] == 12
    assert workload["k"] == 16
    assert workload["elapsed_ms"] >= 0.0
    assert workload["flops"] == 2 * 8 * 12 * 16
    assert workload["selected"]["reason"] == "only_candidate"
    assert workload["selected"]["candidate_id"] == _cutlass_default_candidate_id("float32")
    candidate = workload["candidates"][0]
    assert candidate["candidate_id"] == _cutlass_default_candidate_id("float32")
    assert candidate["provider"] == "cutlass"
    assert candidate["family"] == "gemm_universal"
    assert candidate["layouts"] == {"a": "row", "b": "column", "c": "row"}
    assert candidate["kernel_symbol"] == f"dinoml_cutlass_gemm_rcr_float32_{_cutlass_default_symbol_id('float32')}"
    assert candidate["profiler_symbol"] == f"dinoml_profile_cutlass_gemm_rcr_float32_{_cutlass_default_symbol_id('float32')}"
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
                    "profiler_symbol": f"dinoml_profile_cutlass_gemm_rrr_float32_{_cutlass_default_symbol_id("float32")}",
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
    assert f"dinoml_profile_cutlass_gemm_rrr_float32_{_cutlass_default_symbol_id("float32")}" in stdout
