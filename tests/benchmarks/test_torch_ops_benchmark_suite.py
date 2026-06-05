from __future__ import annotations

import numpy as np

import dinoml.cli as cli
from dinoml.benchmarks.ops import benchmark_cases
from dinoml.benchmarks.torch_ops import run_torch_benchmark_suite, torch_benchmark_cases


def test_torch_ops_benchmark_cases_mirror_dinoml_cases():
    assert [case.name for case in torch_benchmark_cases()] == [case.name for case in benchmark_cases()]


def test_torch_ops_benchmark_suite_runs_selected_case(monkeypatch):
    calls = []

    class FakeDevice:
        type = "cpu"

        def __str__(self):
            return "cpu"

    class FakeTensor:
        dtype = "torch.float32"

        def __init__(self, value):
            self.value = np.asarray(value)
            self.shape = self.value.shape

        def __add__(self, other):
            return FakeTensor(self.value + other.value)

        def to(self, device):
            calls.append(("to", str(device), self.shape))
            return self

    class FakeTorch:
        def device(self, device):
            assert device == "cpu"
            return FakeDevice()

        def from_numpy(self, value):
            return FakeTensor(value)

    def fake_benchmark_call(torch, device, call, *, warmup, iterations):
        calls.append(("benchmark", str(device), warmup, iterations))
        call()
        return {
            "count": iterations,
            "warmup": warmup,
            "mean_ms": 1.0,
            "median_ms": 1.0,
            "min_ms": 1.0,
            "max_ms": 1.0,
            "stddev_ms": 0.0,
        }

    monkeypatch.setattr("dinoml.benchmarks.torch_ops._import_torch", lambda: FakeTorch())
    monkeypatch.setattr("dinoml.benchmarks.torch_ops._benchmark_torch_call", fake_benchmark_call)

    report = run_torch_benchmark_suite(device="cpu", warmup=1, iterations=2, only=["add"])

    assert report["target"] == {"framework": "torch", "device": "cpu"}
    assert report["summary"] == {"total": 1, "ok": 1, "error": 0, "elapsed_s": report["summary"]["elapsed_s"]}
    assert report["cases"][0]["name"] == "add"
    assert report["cases"][0]["inputs"][0]["name"] == "x"
    assert ("benchmark", "cpu", 1, 2) in calls


def test_cli_benchmark_torch_ops_passes_options(monkeypatch, tmp_path, capsys):
    captured = {}

    def fake_run_torch_benchmark_suite(**kwargs):
        captured.update(kwargs)
        return {
            "target": {"framework": "torch", "device": kwargs["device"]},
            "warmup": kwargs["warmup"],
            "iterations": kwargs["iterations"],
            "summary": {"total": 1, "ok": 1, "error": 0, "elapsed_s": 0.0},
            "cases": [],
        }

    monkeypatch.setattr(cli, "run_torch_benchmark_suite", fake_run_torch_benchmark_suite)

    assert cli.main(["benchmark-torch-ops", "--device", "cpu", "--warmup", "1", "--iterations", "2", "--only", "add"]) == 0
    capsys.readouterr()

    assert captured["device"] == "cpu"
    assert captured["warmup"] == 1
    assert captured["iterations"] == 2
    assert captured["only"] == ["add"]
    assert captured["fail_fast"] is False
