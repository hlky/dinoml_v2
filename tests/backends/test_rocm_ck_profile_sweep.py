from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


def _load_sweep_module():
    module_path = Path(__file__).resolve().parents[2] / "tools" / "rocm_ck_profile_sweep.py"
    spec = importlib.util.spec_from_file_location("rocm_ck_profile_sweep", module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_rocm_ck_profile_sweep_cases_cover_provider_epilogues():
    sweep = _load_sweep_module()

    cases = sweep.rocm_ck_profile_sweep_cases()
    by_family = {
        family: {case.op for case in cases if case.family == family}
        for family in {"gemm", "bmm", "conv"}
    }

    assert by_family["gemm"] == {
        "gemm_rcr",
        "gemm_rcr_bias",
        "gemm_rcr_bias_add_relu",
        "gemm_rcr_bias_add_add_relu",
    }
    assert by_family["bmm"] == {"bmm_rcr", "bmm_rcr_add"}
    assert by_family["conv"] == {
        "conv2d_bias",
        "conv2d_bias_relu",
        "conv2d_bias_add",
        "conv2d_bias_add_relu",
    }
    assert all(case.dtype == "float16" for case in cases)


def test_rocm_ck_profile_sweep_compiles_selected_cases(tmp_path, monkeypatch):
    sweep = _load_sweep_module()
    calls = []

    class Artifact:
        def __init__(self, path: Path):
            self.path = path

    def fake_compile(spec, target, output, **kwargs):
        output = Path(output)
        output.mkdir(parents=True)
        (output / "debug").mkdir()
        op_name = spec.ir["nodes"][0]["op"]
        calls.append((spec.name, target.to_json(), output.name, kwargs))
        sweep.write_json(
            output / "manifest.json",
            {"target": target.to_json()},
        )
        sweep.write_json(
            output / "kernel_manifest.json",
            {
                "required_kernels": [
                    {
                        "op": op_name,
                        "kernel_library": "ck_bmm",
                        "dtype": "float16",
                        "selected_candidate_id": "ck_bmm_rcr_add_float16_xdl_wide_n_v1",
                        "kernel_symbol": "dinoml_ck_bmm_rcr_add_float16_xdl_wide_n_v1",
                        "profiler_symbol": "dinoml_profile_ck_bmm_rcr_add_float16_xdl_wide_n_v1",
                    }
                ]
            },
        )
        sweep.write_json(
            output / "debug" / "bootstrap_profile_report.json",
            {"summary": {"profiled": 2, "failed": 0, "cached": 0, "blocked": 0, "skipped": 0}},
        )
        return Artifact(output)

    monkeypatch.setattr(sweep.dml, "compile", fake_compile)

    report = sweep.run_sweep(
        only=["bmm_rcr_add"],
        artifact_root=tmp_path,
        iterations=2,
        repeats=1,
        refresh=True,
    )

    assert report["summary"]["total"] == 1
    assert report["summary"]["ok"] == 1
    assert report["summary"]["error"] == 0
    assert report["cases"][0]["op"] == "bmm_rcr_add"
    assert report["cases"][0]["profile_summary"]["profiled"] == 2
    assert report["cases"][0]["selected_candidates"][0]["selected_candidate_id"] == "ck_bmm_rcr_add_float16_xdl_wide_n_v1"
    assert calls == [
        (
            "rocm_ck_profile_sweep_bmm_rcr_add_float16",
            {"name": "rocm", "arch": "gfx1201", "no_tf32": False, "use_fp16_acc": False},
            "bmm_rcr_add_float16.dinoml",
            {
                "profile": True,
                "profile_iterations": 2,
                "profile_repeats": 1,
                "profile_refresh": True,
            },
        )
    ]
