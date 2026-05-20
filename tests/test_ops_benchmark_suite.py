from __future__ import annotations

from pathlib import Path

import dinoml.cli as cli
from dinoml.benchmarks.ops import benchmark_cases, run_benchmark_suite


def test_ops_benchmark_cases_are_individual_not_condensed():
    names = {case.name for case in benchmark_cases()}

    assert "add" in names
    assert "sub" in names
    assert "reduce_sum" in names
    assert "reduce_max" in names
    assert "elementwise" not in names
    assert "reductions" not in names
    assert "_shape_buffer_count_true" not in names


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
