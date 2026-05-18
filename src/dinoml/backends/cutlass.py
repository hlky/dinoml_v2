from __future__ import annotations

import functools
import subprocess

from dinoml.backends.build_parallelism import effective_cpu_count


def _compile_flags(arch_num: str) -> list[str]:
    flags = [
        "-shared",
        "-Xcompiler",
        "-fPIC",
        "-O3",
        "-std=c++17",
        f"-arch=sm_{arch_num}",
        "--use_fast_math",
        "--expt-relaxed-constexpr",
        "-diag-suppress=20012",
    ]
    split_compile = _nvcc_split_compile_flag()
    if split_compile is not None:
        flags.append(split_compile)
    return flags


def _nvcc_split_compile_flag() -> str | None:
    cpu_count = max(1, effective_cpu_count())
    if _nvcc_supports_option("--split-compile"):
        return f"--split-compile={cpu_count}"
    if _nvcc_supports_option("--threads"):
        return f"--threads={cpu_count}"
    return None


@functools.lru_cache(maxsize=None)
def _nvcc_supports_option(option: str) -> bool:
    try:
        proc = subprocess.run(
            ["nvcc", "--help"],
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
    except FileNotFoundError:
        return False
    return option in proc.stdout
