# ROCm Backend Plan

This plan records the first ROCm backend integration lane. It is based on the
local Windows environment plus the `hlky/rocm_windows` investigation at commit
`9594294` (`Handle versioned Windows HIP SDK roots`), including
`scripts/Build-Probe.ps1` and `cmake/DinoMLROCmSdk.cmake`.

## Current Scaffold

- Default ROCm target: `dml.Target("rocm")` resolves to `gfx1201`, matching the
  local AMD Radeon RX 9070 XT report from `rocm_sdk targets`.
- The repo has a dedicated ROCm backend spec, CMake SDK resolver, HIP runtime
  helper library, and empty reusable ROCm kernel library.
- The support-library smoke builds `dinoml_runtime`, `dinoml_rocm_runtime`, and
  `dinoml_rocm_kernels` through `.venv/rocm` and writes a support manifest under
  the ROCm support cache.
- Model compilation for `Target("rocm")` still raises before claiming any op
  support. No ROCm generated HIP artifact wrapper, op lowering, runtime Python
  execution path, or CK provider path is admitted yet.

## Windows ROCm Packaging Rules

- Activate `.venv/rocm` before ROCm builds. The active environment must put
  `rocm-sdk`, `hipconfig`, `hipcc`, and `amdclang++` on `PATH`; do not split SDK
  resolution through a separate Python executable override. If the active
  Python environment has the `rocm_sdk` package but no console script, the
  resolver may use `python -m rocm_sdk` from `PATH`.
- Prefer the active `rocm_sdk` package before `hipconfig`. Released PyTorch ROCm
  wheels can expose `hipconfig` paths under `_rocm_sdk_core`, while CMake needs
  the devel payload under `_rocm_sdk_devel` for headers, CMake packages, tools,
  and device libraries.
- Run `rocm-sdk init` before CMake configuration so the devel payload is
  extracted. Regular HIP SDK installs that do not carry the `rocm_sdk` package
  should fall through to `hipconfig`/`HIP_PATH`/`ROCM_PATH`.
- Set `HIP_PLATFORM=amd`, `HIP_PATH`, `ROCM_PATH`, and CMake prefix paths from
  the resolved SDK root.
- Use `CMAKE_HIP_ARCHITECTURES=gfx1201` for the local card unless a later
  validated detection path proves otherwise.
- On Windows, import only the Visual Studio x64 environment keys required by
  clang/MSVC linking: `PATH`, `INCLUDE`, `LIB`, `LIBPATH`, and the VS/Windows
  SDK root/version variables. `rc.exe` is also resolved explicitly before
  configuring Ninja/CMake HIP builds.
- Pass the ROCm device bitcode directory when available; pip/venv SDKs expose it
  under `_rocm_sdk_devel/lib/llvm/amdgcn/bitcode`.

## Validation Ladder

1. Keep `tests/backends/test_rocm_scaffold.py` green without the real toolchain
   smoke, proving target registration and the honest unsupported-op fence.
2. With `.venv/rocm` active, run the opt-in support-library smoke:
   `DINOML_RUN_ROCM_SUPPORT_BUILD_SMOKE=1 python -m pytest -q tests/backends/test_rocm_scaffold.py`.
3. Admit exactly one generated HIP artifact slice, likely fused elementwise,
   only after generated source, CMake module build, library copying, and runtime
   execution are all visible in tests.
4. Add provider work such as CK only after the backend can compile and run a
   minimal model artifact on ROCm.

## Next Bounded Step

Wire a minimal generated HIP module path for one already-supported op family,
preferably fused elementwise `float32`, and prove it with a real `.venv/rocm`
compile/load/run smoke on `gfx1201`. Keep the failure mode explicit for every
other op and dtype combination.
