from __future__ import annotations

from pathlib import Path

import pytest


@pytest.fixture(scope="session")
def dinoml_cuda_support_cache_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return tmp_path_factory.mktemp("dinoml_cuda_support_cache")


@pytest.fixture
def use_shared_dinoml_cuda_cache(
    monkeypatch: pytest.MonkeyPatch,
    dinoml_cuda_support_cache_dir: Path,
) -> Path:
    monkeypatch.setenv("DINOML_CACHE_DIR", str(dinoml_cuda_support_cache_dir))
    return dinoml_cuda_support_cache_dir
