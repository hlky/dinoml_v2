from __future__ import annotations

import ctypes
import json
import os
import shutil
from pathlib import Path

import numpy as np
import pytest

import dinoml as dml
import dinoml.cli as cli
from dinoml import runtime
from dinoml.runtime import Session, _DinoTensor


class _FakeModule:
    def __init__(self):
        self.calls: list[dict[str, int]] = []
        self._session_benchmark = self._benchmark

    def _check(self, err: int) -> None:
        if err:
            raise RuntimeError(f"runtime error {err}")

    def _benchmark(
        self,
        handle,
        inputs,
        num_inputs,
        outputs,
        num_outputs,
        warmup,
        measured,
        elapsed,
        elapsed_count,
    ) -> int:
        self.calls.append(
            {
                "handle": int(handle.value),
                "num_inputs": int(num_inputs.value),
                "num_outputs": int(num_outputs.value),
                "warmup": int(warmup.value),
                "measured": int(measured.value),
                "elapsed_count": int(elapsed_count.value),
            }
        )
        for idx in range(int(measured.value)):
            elapsed[idx] = float(idx + 1)
        return 0


def _fake_session(module: _FakeModule | None = None) -> Session:
    session = Session.__new__(Session)
    session.module = module or _FakeModule()
    session._handle = ctypes.c_void_p(123)
    return session


def test_session_benchmark_native_summarizes_module_samples():
    module = _FakeModule()
    session = _fake_session(module)
    input_tensors = (_DinoTensor * 0)()
    output_tensors = (_DinoTensor * 0)()

    summary = session._benchmark_native(
        input_tensors,
        0,
        output_tensors,
        0,
        warmup=2,
        iterations=3,
    )

    assert module.calls == [
        {
            "handle": 123,
            "num_inputs": 0,
            "num_outputs": 0,
            "warmup": 2,
            "measured": 3,
            "elapsed_count": 3,
        }
    ]
    assert summary == {
        "count": 3,
        "warmup": 2,
        "mean_ms": 2.0,
        "median_ms": 2.0,
        "min_ms": 1.0,
        "max_ms": 3.0,
        "stddev_ms": pytest.approx(0.816496580927726),
    }


def test_session_benchmark_rejects_non_positive_iteration_count():
    session = _fake_session()
    input_tensors = (_DinoTensor * 0)()
    output_tensors = (_DinoTensor * 0)()

    with pytest.raises(ValueError, match="iterations must be positive"):
        session._benchmark_native(input_tensors, 0, output_tensors, 0, warmup=0, iterations=0)


def test_session_benchmark_requires_recompiled_artifact_symbol():
    module = _FakeModule()
    module._session_benchmark = None
    session = _fake_session(module)
    input_tensors = (_DinoTensor * 0)()
    output_tensors = (_DinoTensor * 0)()

    with pytest.raises(RuntimeError, match="dino_session_benchmark"):
        session._benchmark_native(input_tensors, 0, output_tensors, 0, warmup=0, iterations=1)


def test_cli_benchmark_reports_session_run_summary(tmp_path, monkeypatch, capsys):
    model_path = tmp_path / "benchmark_inputs.py"
    model_path.write_text(
        "\n".join(
            [
                "import numpy as np",
                "",
                "def build_validation_inputs():",
                "    return {'x': np.ones((2, 3), dtype=np.float32)}",
                "",
            ]
        ),
        encoding="utf-8",
    )
    out_path = tmp_path / "benchmark.json"

    class FakeSession:
        def benchmark_numpy(self, inputs, *, warmup, iterations):
            assert inputs["x"].shape == (2, 3)
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
        target_name = "cpu"
        metadata = {
            "inputs": [{"name": "x", "shape": [2, 3], "dtype": "float32"}],
            "outputs": [{"name": "y", "shape": [2, 3], "dtype": "float32"}],
        }

        def create_session(self):
            return FakeSession()

        def close(self):
            pass

    monkeypatch.setattr(cli.runtime, "load", lambda artifact, load_constants=True: FakeRuntimeModule())

    assert cli.main(
        [
            "benchmark",
            "fake.dinoml",
            "--against",
            str(model_path),
            "--warmup",
            "1",
            "--iterations",
            "2",
            "--out",
            str(out_path),
        ]
    ) == 0
    capsys.readouterr()

    report = json.loads(out_path.read_text(encoding="utf-8"))
    assert report["artifact"] == "fake.dinoml"
    assert report["target"] == "cpu"
    assert report["session_run"]["count"] == 2
    assert report["inputs"] == [{"dtype": "float32", "name": "x", "shape": [2, 3]}]


def test_generated_module_templates_export_native_session_benchmark():
    repo_root = Path(__file__).resolve().parents[1]
    cpu_text = (repo_root / "src" / "dinoml" / "templates" / "cpu_module.cpp.j2").read_text(encoding="utf-8")
    gpu_text = (repo_root / "src" / "dinoml" / "templates" / "gpu_module.cu.j2").read_text(encoding="utf-8")
    assert "DINO_EXPORT int dino_session_benchmark" in cpu_text
    assert "DINO_EXPORT int dino_session_benchmark" in gpu_text
    assert "dino_session_run(session, inputs, num_inputs, outputs, num_outputs)" in cpu_text
    assert "dino_session_run_impl(session, inputs, num_inputs, outputs, num_outputs, false, false)" in gpu_text
    assert "{{ event_record }}(events.start, session->stream)" in gpu_text
    assert "{{ event_elapsed_time }}(&elapsed_ms, events.start, events.stop)" in gpu_text
    assert "std::chrono" not in gpu_text


class BenchmarkModule(dml.Module):
    def forward(self, x):
        return dml.ops.output(dml.ops.relu(x + 1.0), "y")


def test_compiled_cpu_artifact_exposes_session_benchmark(tmp_path):
    spec = dml.trace(
        BenchmarkModule(),
        inputs={"x": dml.TensorSpec([2, 3], "float32")},
        name="runtime_benchmark_smoke",
    )
    artifact = dml.compile(spec, dml.Target("cpu"), tmp_path / "runtime_benchmark_cpu.dinoml")
    module = runtime.load(artifact.path)
    session = module.create_session()
    try:
        summary = session.benchmark_numpy(
            {"x": np.array([[-2.0, -0.5, 0.0], [1.0, 2.0, 3.0]], dtype=np.float32)},
            warmup=1,
            iterations=2,
        )
    finally:
        session.close()
        module.close()

    assert summary["count"] == 2
    assert summary["warmup"] == 1
