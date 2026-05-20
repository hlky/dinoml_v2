# ROCm Backend Plan

This plan records the first ROCm backend integration lane. It is based on the
local Windows environment plus the `hlky/rocm_windows` investigation at commit
`9594294` (`Handle versioned Windows HIP SDK roots`), including
`scripts/Build-Probe.ps1` and `cmake/DinoMLROCmSdk.cmake`.

## Current Status

- Default ROCm target: `dml.Target("rocm")` resolves to `gfx1201`, matching the
  local AMD Radeon RX 9070 XT report from `rocm_sdk targets`.
- The repo has a dedicated ROCm backend spec, CMake SDK resolver, HIP runtime
  helper library, and reusable ROCm kernel library.
- The support-library smoke builds `dinoml_runtime`, `dinoml_rocm_runtime`, and
  `dinoml_rocm_kernels` through `.venv/rocm` and writes a support manifest under
  the ROCm support cache.
- Model compilation for `Target("rocm")` now admits the simple
  generated-template op families through shared GPU templates rendered as HIP
  sources. The opt-in ROCm contract compiles, loads, runs, and reference-checks
  every non-provider standard case on the local `.venv/rocm` toolchain.
- CK and provider-backed GEMM/BMM/Conv paths are still not admitted on ROCm.

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
   smoke, proving target registration and backend support contracts.
2. With `.venv/rocm` active, run the opt-in support-library smoke:
   `DINOML_RUN_ROCM_SUPPORT_BUILD_SMOKE=1 python -m pytest -q tests/backends/test_rocm_scaffold.py`.
3. With `.venv/rocm` active, run the opt-in generated artifact contract:
   `DINOML_RUN_ROCM_CONTRACTS=1 python -m pytest -q tests/rocm/test_contracts.py`.
   This is the acceptance gate for simple generated-template ROCm ops because it
   compiles real artifacts, loads them through the runtime, executes on the HIP
   device, and compares against `reference_numpy`.
4. Add provider work such as CK only after it has the same artifact-visible
   compile/load/run proof and does not piggyback on the simple-template
   contract.

## Next Bounded Step

Keep the ROCm lane honest by either hardening the admitted simple generated
surface with focused edge cases or starting one provider-backed lane such as CK
GEMM only when its manifest, support build, generated lowering, runtime load,
and numeric parity proof can all land together. Do not claim GEMM/BMM/Conv ROCm
support from the simple-template contract.
