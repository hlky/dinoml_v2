from __future__ import annotations

import json
from pathlib import Path

import dinoml.cli as cli
from dinoml.benchmarks.ops import benchmark_cases, run_benchmark_suite


def test_ops_benchmark_cases_are_individual_not_condensed():
    names = {case.name for case in benchmark_cases()}

    assert "add" in names
    assert "sub" in names
    assert "reduce_sum" in names
    assert "reduce_max" in names
    assert "gemm_rcr_bias_add_relu" in names
    assert "bmm_rcr_add" in names
    assert "conv2d_bias_add_relu" in names
    assert "elementwise" not in names
    assert "reductions" not in names
    assert "_shape_buffer_count_true" not in names


def test_provider_benchmark_cases_are_cuda_rocm_only():
    cases = {case.name: case for case in benchmark_cases()}

    assert cases["gemm_rcr_bias_add_relu"].targets == ("cuda", "rocm")
    assert cases["bmm_rcr_add"].targets == ("cuda", "rocm")
    assert cases["conv2d_bias_add_relu"].targets == ("cuda", "rocm")


def test_ops_benchmark_suite_compiles_and_benchmarks_selected_cases(tmp_path, monkeypatch):
    calls = []

    class Artifact:
        def __init__(self, path):
            self.path = Path(path)

    class FakeSession:
        def benchmark_numpy(self, inputs, *, warmup, iterations):
            calls.append(("benchmark", sorted(inputs), warmup, iterations))
            return {
                "count": iterations,
                "warmup": warmup,
                "mean_ms": 1.0,
                "median_ms": 1.0,
                "min_ms": 1.0,
                "max_ms": 1.0,
                "stddev_ms": 0.0,
            }

        def close(self):
            pass

    class FakeRuntimeModule:
        def create_session(self):
            return FakeSession()

        def close(self):
            pass

    def fake_compile(spec, target, output):
        calls.append(("compile", spec.name, target.name, str(output)))
        return Artifact(output)

    monkeypatch.setattr("dinoml.benchmarks.ops.dml.compile", fake_compile)
    monkeypatch.setattr("dinoml.benchmarks.ops.runtime.load", lambda path, load_constants=True: FakeRuntimeModule())

    report = run_benchmark_suite("cpu", output_dir=tmp_path, warmup=1, iterations=2, only=["add", "reduce_sum"])

    assert report["summary"] == {"total": 2, "ok": 2, "error": 0, "elapsed_s": report["summary"]["elapsed_s"]}
    assert [case["name"] for case in report["cases"]] == ["add", "reduce_sum"]
    assert ("compile", "benchmark_add", "cpu", str(tmp_path / "add.dinoml")) in calls
    assert calls.count(("benchmark", ["condition", "positive", "x", "y"], 1, 2)) == 1


def test_ops_benchmark_suite_compiles_provider_cases_on_gpu_targets(tmp_path, monkeypatch):
    calls = []

    class Artifact:
        def __init__(self, path):
            self.path = Path(path)

    class FakeSession:
        def benchmark_numpy(self, inputs, *, warmup, iterations):
            calls.append(("benchmark", sorted(inputs), warmup, iterations))
            return {
                "count": iterations,
                "warmup": warmup,
                "mean_ms": 1.0,
                "median_ms": 1.0,
                "min_ms": 1.0,
                "max_ms": 1.0,
                "stddev_ms": 0.0,
            }

        def close(self):
            pass

    class FakeRuntimeModule:
        def create_session(self):
            return FakeSession()

        def close(self):
            pass

    def fake_compile(spec, target, output):
        ops = {node["op"] for node in spec.ir["nodes"]}
        calls.append(("compile", spec.name, target.name, str(output), ops))
        Path(output).mkdir(parents=True, exist_ok=True)
        op_name = next(iter(ops))
        (Path(output) / "kernel_manifest.json").write_text(
            json.dumps(
                {
                    "required_kernels": [
                        {
                            "op": op_name,
                            "dtype": "float16",
                            "kernel_library": "ck_bmm",
                            "kernel_symbol": f"dinoml_{op_name}_kernel",
                            "profiler_symbol": f"dinoml_{op_name}_profiler",
                            "candidate_set_id": f"{op_name}_candidate_set",
                            "selected_candidate_id": f"{op_name}_candidate",
                            "candidates": [
                                {
                                    "candidate_id": f"{op_name}_candidate",
                                    "candidate_config_key": "xdl_wide_n_v1",
                                }
                            ],
                        }
                    ]
                }
            ),
            encoding="utf-8",
        )
        (Path(output) / "debug").mkdir(parents=True, exist_ok=True)
        execution_plan_path = Path(output) / "debug" / "execution_plan.json"
        execution_plan_path.write_text(
            json.dumps(
                {
                    "summary": {
                        "selection_count": 0,
                        "low_confidence_count": 1,
                        "static_selection_count": 0,
                        "conflict_count": 0,
                    },
                    "selection_policy": "lowest_median_elapsed_ms_per_node_shape",
                    "selection_confidence_policy": {
                        "name": "confidence_interval_margin_v1",
                        "min_repeats": 3,
                    },
                    "selections": [],
                    "low_confidence_selections": [
                        {
                            "node_id": "node0",
                            "op": op_name,
                            "dtype": "float16",
                            "kernel_library": "ck_bmm",
                            "candidate_set_id": f"{op_name}_candidate_set",
                            "selected_candidate_id": f"{op_name}_candidate",
                            "candidate_config_key": "xdl_wide_n_v1",
                            "kernel_symbol": f"dinoml_{op_name}_kernel",
                            "profiler_symbol": f"dinoml_{op_name}_profiler",
                            "avg_ms": 0.012,
                            "gflops": 1250.0,
                            "iterations": 1,
                            "split_k": 1,
                            "workspace_nbytes": 0,
                            "status": "ok",
                            "confidence": {
                                "level": "low",
                                "confident": False,
                                "reasons": ["runner_up_insufficient_repeats"],
                                "selection_metric_ms": 0.012,
                                "runner_up_candidate_id": f"{op_name}_fallback",
                                "runner_up_elapsed_ms": 0.013,
                                "margin_ms": 0.001,
                                "required_margin_ms": 0.002,
                                "relative_speedup_over_runner_up": 0.08,
                                "sample_counts": {"best": 1, "runner_up": 1},
                            },
                        }
                    ],
                    "static_selections": [],
                    "conflicts": [],
                }
            ),
            encoding="utf-8",
        )
        (Path(output) / "debug" / "bootstrap_profile_report.json").write_text(
            json.dumps(
                {
                    "iterations": 1,
                    "repeats": 1,
                    "summary": {"profiled": 1, "failed": 0},
                    "execution_plan": {
                        "schema_version": 1,
                        "execution_plan_key": "test-plan-key",
                        "path": str(execution_plan_path),
                        "selection_count": 0,
                        "low_confidence_count": 1,
                        "static_selection_count": 0,
                        "conflict_count": 0,
                    },
                    "problems": [
                        {
                            "node_id": "node0",
                            "op": op_name,
                            "dtype": "float16",
                            "kernel_library": "ck_bmm",
                            "profiler_symbol": f"dinoml_{op_name}_profiler",
                            "elapsed_ms": 0.012,
                            "tflops": 1.25,
                            "timing": {"avg_ms": 0.012, "repeats": 1},
                            "selected": {"candidate_id": f"{op_name}_candidate"},
                        }
                    ],
                }
            ),
            encoding="utf-8",
        )
        return Artifact(output)

    monkeypatch.setattr("dinoml.benchmarks.ops.dml.compile", fake_compile)
    monkeypatch.setattr("dinoml.benchmarks.ops.runtime.load", lambda path, load_constants=True: FakeRuntimeModule())

    report = run_benchmark_suite(
        "rocm",
        output_dir=tmp_path,
        warmup=1,
        iterations=2,
        only=["gemm_rcr_bias_add_relu", "bmm_rcr_add", "conv2d_bias_add_relu"],
    )

    assert report["summary"]["ok"] == 3
    assert [case["name"] for case in report["cases"]] == [
        "gemm_rcr_bias_add_relu",
        "bmm_rcr_add",
        "conv2d_bias_add_relu",
    ]
    assert ("compile", "benchmark_gemm_rcr_bias_add_relu", "rocm", str(tmp_path / "gemm_rcr_bias_add_relu.dinoml"), {"gemm_rcr_bias_add_relu"}) in calls
    assert ("compile", "benchmark_bmm_rcr_add", "rocm", str(tmp_path / "bmm_rcr_add.dinoml"), {"bmm_rcr_add"}) in calls
    assert ("compile", "benchmark_conv2d_bias_add_relu", "rocm", str(tmp_path / "conv2d_bias_add_relu.dinoml"), {"conv2d_bias_add_relu"}) in calls
    assert report["cases"][1]["provider_kernels"] == [
        {
            "op": "bmm_rcr_add",
            "dtype": "float16",
            "kernel_library": "ck_bmm",
            "kernel_symbol": "dinoml_bmm_rcr_add_kernel",
            "profiler_symbol": "dinoml_bmm_rcr_add_profiler",
            "candidate_set_id": "bmm_rcr_add_candidate_set",
            "selected_candidate_id": "bmm_rcr_add_candidate",
            "split_k": 1,
            "workspace_nbytes": 0,
            "candidate_config_key": "xdl_wide_n_v1",
        }
    ]
    assert report["cases"][1]["provider_profile"] == {
        "iterations": 1,
        "repeats": 1,
        "summary": {"profiled": 1, "failed": 0},
        "problems": [
            {
                "node_id": "node0",
                "op": "bmm_rcr_add",
                "dtype": "float16",
                "kernel_library": "ck_bmm",
                "candidate_id": "bmm_rcr_add_candidate",
                "profiler_symbol": "dinoml_bmm_rcr_add_profiler",
                "elapsed_ms": 0.012,
                "tflops": 1.25,
                "timing": {"avg_ms": 0.012, "repeats": 1},
            }
        ],
        "execution_plan": {
            "summary": {
                "selection_count": 0,
                "low_confidence_count": 1,
                "static_selection_count": 0,
                "conflict_count": 0,
                "schema_version": 1,
                "execution_plan_key": "test-plan-key",
            },
            "selection_policy": "lowest_median_elapsed_ms_per_node_shape",
            "selection_confidence_policy": {
                "name": "confidence_interval_margin_v1",
                "min_repeats": 3,
            },
            "selections": [],
            "low_confidence_selections": [
                {
                    "node_id": "node0",
                    "op": "bmm_rcr_add",
                    "dtype": "float16",
                    "kernel_library": "ck_bmm",
                    "candidate_set_id": "bmm_rcr_add_candidate_set",
                    "selected_candidate_id": "bmm_rcr_add_candidate",
                    "candidate_config_key": "xdl_wide_n_v1",
                    "kernel_symbol": "dinoml_bmm_rcr_add_kernel",
                    "profiler_symbol": "dinoml_bmm_rcr_add_profiler",
                    "avg_ms": 0.012,
                    "gflops": 1250.0,
                    "iterations": 1,
                    "split_k": 1,
                    "workspace_nbytes": 0,
                    "status": "ok",
                    "confidence": {
                        "level": "low",
                        "confident": False,
                        "reasons": ["runner_up_insufficient_repeats"],
                        "selection_metric_ms": 0.012,
                        "runner_up_candidate_id": "bmm_rcr_add_fallback",
                        "runner_up_elapsed_ms": 0.013,
                        "margin_ms": 0.001,
                        "required_margin_ms": 0.002,
                        "relative_speedup_over_runner_up": 0.08,
                        "sample_counts": {"best": 1, "runner_up": 1},
                    },
                }
            ],
            "static_selections": [],
            "conflicts": [],
        },
    }


def test_ops_benchmark_suite_rejects_provider_cases_on_cpu(tmp_path):
    try:
        run_benchmark_suite("cpu", output_dir=tmp_path, only=["gemm_rcr"])
    except ValueError as exc:
        assert "not supported on target cpu: gemm_rcr" in str(exc)
    else:
        raise AssertionError("expected provider benchmark case to be rejected on CPU")


def test_ops_benchmark_suite_parallel_compile_preserves_report_order(tmp_path, monkeypatch):
    calls = []

    class Artifact:
        def __init__(self, path):
            self.path = Path(path)

    class FakeSession:
        def benchmark_numpy(self, inputs, *, warmup, iterations):
            calls.append(("benchmark", sorted(inputs), warmup, iterations))
            return {
                "count": iterations,
                "warmup": warmup,
                "mean_ms": 1.0,
                "median_ms": 1.0,
                "min_ms": 1.0,
                "max_ms": 1.0,
                "stddev_ms": 0.0,
            }

        def close(self):
            pass

    class FakeRuntimeModule:
        def create_session(self):
            return FakeSession()

        def close(self):
            pass

    def fake_compile(spec, target, output):
        calls.append(("compile", spec.name, target.name, str(output)))
        return Artifact(output)

    monkeypatch.setattr("dinoml.benchmarks.ops.dml.compile", fake_compile)
    monkeypatch.setattr("dinoml.benchmarks.ops.runtime.load", lambda path, load_constants=True: FakeRuntimeModule())

    report = run_benchmark_suite("cpu", output_dir=tmp_path, warmup=1, iterations=2, only=["add", "reduce_sum"], jobs=2)

    assert report["summary"] == {"total": 2, "ok": 2, "error": 0, "elapsed_s": report["summary"]["elapsed_s"]}
    assert [case["name"] for case in report["cases"]] == ["add", "reduce_sum"]
    assert all("compile_elapsed_s" in case for case in report["cases"])
    assert all("benchmark_elapsed_s" in case for case in report["cases"])
    assert {call[1] for call in calls if call[0] == "compile"} == {"benchmark_add", "benchmark_reduce_sum"}


def test_ops_benchmark_suite_can_profile_compiles(tmp_path, monkeypatch):
    calls = []

    class Artifact:
        def __init__(self, path):
            self.path = Path(path)

    class FakeSession:
        def benchmark_numpy(self, inputs, *, warmup, iterations):
            calls.append(("benchmark", sorted(inputs), warmup, iterations))
            return {
                "count": iterations,
                "warmup": warmup,
                "mean_ms": 1.0,
                "median_ms": 1.0,
                "min_ms": 1.0,
                "max_ms": 1.0,
                "stddev_ms": 0.0,
            }

        def close(self):
            pass

    class FakeRuntimeModule:
        def create_session(self):
            return FakeSession()

        def close(self):
            pass

    def fake_compile(spec, target, output, **kwargs):
        calls.append(("compile", spec.name, target.name, str(output), kwargs))
        return Artifact(output)

    monkeypatch.setattr("dinoml.benchmarks.ops.dml.compile", fake_compile)
    monkeypatch.setattr("dinoml.benchmarks.ops.runtime.load", lambda path, load_constants=True: FakeRuntimeModule())

    report = run_benchmark_suite(
        "rocm",
        output_dir=tmp_path,
        warmup=1,
        iterations=2,
        profile=True,
        profile_iterations=7,
        profile_repeats=2,
        profile_refresh=True,
        only=["add"],
    )

    assert report["summary"]["ok"] == 1
    assert report["profile"] == {"enabled": True, "iterations": 7, "repeats": 2, "refresh": True}
    assert calls[0] == (
        "compile",
        "benchmark_add",
        "rocm",
        str(tmp_path / "add.dinoml"),
        {
            "profile": True,
            "profile_iterations": 7,
            "profile_repeats": 2,
            "profile_refresh": True,
        },
    )


def test_ops_benchmark_suite_ignores_temp_cleanup_errors(monkeypatch):
    calls = []
    temp_kwargs = []

    class Artifact:
        def __init__(self, path):
            self.path = Path(path)

    class FakeSession:
        def benchmark_numpy(self, inputs, *, warmup, iterations):
            del inputs
            return {
                "count": iterations,
                "warmup": warmup,
                "mean_ms": 1.0,
                "median_ms": 1.0,
                "min_ms": 1.0,
                "max_ms": 1.0,
                "stddev_ms": 0.0,
            }

        def close(self):
            pass

    class FakeRuntimeModule:
        metadata = {"inputs": [], "outputs": []}

        def create_session(self):
            return FakeSession()

        def close(self):
            pass

    class FakeTemporaryDirectory:
        def __init__(self, **kwargs):
            temp_kwargs.append(kwargs)
            self.name = "H:/tmp/dinoml_ops_bench_locked"

        def cleanup(self):
            calls.append(("cleanup", temp_kwargs[-1].get("ignore_cleanup_errors")))
            if not temp_kwargs[-1].get("ignore_cleanup_errors"):
                raise PermissionError("locked ROCm DLL")

    def fake_compile(spec, target, output):
        del spec, target
        return Artifact(output)

    monkeypatch.setattr("dinoml.benchmarks.ops.tempfile.TemporaryDirectory", FakeTemporaryDirectory)
    monkeypatch.setattr("dinoml.benchmarks.ops.dml.compile", fake_compile)
    monkeypatch.setattr("dinoml.benchmarks.ops.runtime.load", lambda path, load_constants=True: FakeRuntimeModule())

    report = run_benchmark_suite("rocm", warmup=1, iterations=1, only=["add"])

    assert report["summary"]["ok"] == 1
    assert temp_kwargs == [{"prefix": "dinoml_ops_bench_", "ignore_cleanup_errors": True}]
    assert calls == [("cleanup", True)]


def test_cli_benchmark_ops_passes_target_to_suite(tmp_path, monkeypatch, capsys):
    captured = {}

    def fake_run_benchmark_suite(target, **kwargs):
        captured["target"] = target
        captured["kwargs"] = kwargs
        return {
            "target": {"name": target, "arch": "sm_86", "no_tf32": False, "use_fp16_acc": False},
            "warmup": kwargs["warmup"],
            "iterations": kwargs["iterations"],
            "artifact_root": str(tmp_path),
            "summary": {"total": 1, "ok": 1, "error": 0, "elapsed_s": 0.0},
            "cases": [],
        }

    monkeypatch.setattr(cli, "run_benchmark_suite", fake_run_benchmark_suite)

    assert cli.main(["benchmark-ops", "cuda", "--warmup", "1", "--iterations", "2", "--only", "add"]) == 0
    capsys.readouterr()

    assert captured["target"] == "cuda"
    assert captured["kwargs"]["only"] == ["add"]
    assert captured["kwargs"]["warmup"] == 1
    assert captured["kwargs"]["iterations"] == 2
    assert captured["kwargs"]["jobs"] == 1


def test_cli_benchmark_ops_passes_jobs_to_suite(tmp_path, monkeypatch, capsys):
    captured = {}

    def fake_run_benchmark_suite(target, **kwargs):
        captured["target"] = target
        captured["kwargs"] = kwargs
        return {
            "target": {"name": target, "arch": "sm_86", "no_tf32": False, "use_fp16_acc": False},
            "warmup": kwargs["warmup"],
            "iterations": kwargs["iterations"],
            "artifact_root": str(tmp_path),
            "summary": {"total": 1, "ok": 1, "error": 0, "elapsed_s": 0.0},
            "cases": [],
        }

    monkeypatch.setattr(cli, "run_benchmark_suite", fake_run_benchmark_suite)

    assert cli.main(["benchmark-ops", "cuda", "--warmup", "1", "--iterations", "2", "--jobs", "4", "--only", "add"]) == 0
    capsys.readouterr()

    assert captured["target"] == "cuda"
    assert captured["kwargs"]["jobs"] == 4


def test_cli_benchmark_ops_passes_profile_options_to_suite(tmp_path, monkeypatch, capsys):
    captured = {}

    def fake_run_benchmark_suite(target, **kwargs):
        captured["target"] = target
        captured["kwargs"] = kwargs
        return {
            "target": {"name": target, "arch": "gfx1201", "no_tf32": False, "use_fp16_acc": False},
            "warmup": kwargs["warmup"],
            "iterations": kwargs["iterations"],
            "profile": {
                "enabled": kwargs["profile"],
                "iterations": kwargs["profile_iterations"],
                "repeats": kwargs["profile_repeats"],
                "refresh": kwargs["profile_refresh"],
            },
            "artifact_root": str(tmp_path),
            "summary": {"total": 1, "ok": 1, "error": 0, "elapsed_s": 0.0},
            "cases": [],
        }

    monkeypatch.setattr(cli, "run_benchmark_suite", fake_run_benchmark_suite)

    assert cli.main(
        [
            "benchmark-ops",
            "rocm",
            "--profile",
            "--profile-iterations",
            "5",
            "--profile-repeats",
            "2",
            "--profile-refresh",
            "--only",
            "add",
        ]
    ) == 0
    capsys.readouterr()

    assert captured["target"] == "rocm"
    assert captured["kwargs"]["profile"] is True
    assert captured["kwargs"]["profile_iterations"] == 5
    assert captured["kwargs"]["profile_repeats"] == 2
    assert captured["kwargs"]["profile_refresh"] is True
