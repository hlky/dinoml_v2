from __future__ import annotations

import importlib.util
import os
from pathlib import Path
from types import SimpleNamespace


def load_module():
    module_path = Path(__file__).resolve().parents[2] / "tools" / "seed_rocm_support_cache.py"
    spec = importlib.util.spec_from_file_location("seed_rocm_support_cache", module_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_seed_rocm_support_cache_scopes_env_and_calls_expected_seeders(monkeypatch, tmp_path):
    module = load_module()
    cache_dir = tmp_path / "rocm_cache"
    calls: list[tuple[str, str, str]] = []
    old_value = os.environ.get("DINOML_CACHE_DIR")

    class FakeTarget:
        def __init__(self, name: str, arch: str | None = None):
            assert name == "rocm"
            self._arch = arch or "gfx1201"

        def to_json(self):
            return {"name": "rocm", "arch": self._arch}

    def fake_base(arch: str, kernel_manifest=None):
        assert kernel_manifest is None
        calls.append(("base", arch, os.environ["DINOML_CACHE_DIR"]))
        return SimpleNamespace(
            runtime_lib=Path("runtime.so"),
            rocm_runtime_lib=Path("rocm_runtime.so"),
            kernels_lib=Path("kernels.so"),
        )

    def fake_manifest(family: str, *, target_json, dtypes):
        assert target_json == {"name": "rocm", "arch": "gfx1201"}
        assert tuple(dtypes) == ("float16",)
        return {"required_kernels": [{"op": family}], "cache_key": f"{family}-seed"}

    monkeypatch.setattr(module.dml, "Target", FakeTarget)
    monkeypatch.setattr(module, "_build_family_kernel_manifest", fake_manifest)
    monkeypatch.setattr(module.rocm_backend, "ensure_rocm_support_libs", fake_base)
    monkeypatch.setattr(
        module.rocm_backend,
        "_ensure_cmake_ck_gemm_archives",
        lambda arch, manifest: calls.append(("gemm", arch, os.environ["DINOML_CACHE_DIR"])) or (Path("gemm.a"),),
    )
    monkeypatch.setattr(
        module.rocm_backend,
        "_ensure_cmake_flash_attn_ck_archives",
        lambda arch, manifest: calls.append(("flash_attention", arch, os.environ["DINOML_CACHE_DIR"])) or (Path("flash.a"),),
    )

    report = module.seed_rocm_support_cache(
        cache_dir,
        arch="gfx1201",
        dtypes=("float16",),
        families=("base", "gemm", "flash_attention"),
    )

    assert calls == [
        ("base", "gfx1201", str(cache_dir.resolve())),
        ("gemm", "gfx1201", str(cache_dir.resolve())),
        ("flash_attention", "gfx1201", str(cache_dir.resolve())),
    ]
    assert [item["family"] for item in report["results"]] == ["base", "gemm", "flash_attention"]
    assert os.environ.get("DINOML_CACHE_DIR") == old_value


def test_build_family_kernel_manifest_merges_required_kernels(monkeypatch):
    module = load_module()
    target_json = {"name": "rocm", "arch": "gfx1201"}
    specs = [SimpleNamespace(ir="ir_a"), SimpleNamespace(ir="ir_b")]

    monkeypatch.setattr(module, "_family_seed_specs", lambda family, dtypes: specs)
    monkeypatch.setattr(
        module,
        "build_kernel_manifest",
        lambda ir, target: {
            "target": dict(target),
            "required_kernels": [{"kernel_symbol": f"symbol_{ir}", "op": str(ir), "dtype": "float16", "kernel_library": "ck"}],
        },
    )

    manifest = module._build_family_kernel_manifest("gemm", target_json=target_json, dtypes=("float16",))

    assert manifest["target"] == target_json
    assert [item["kernel_symbol"] for item in manifest["required_kernels"]] == ["symbol_ir_a", "symbol_ir_b"]
    assert isinstance(manifest["cache_key"], str)
    assert len(manifest["cache_key"]) == 64
