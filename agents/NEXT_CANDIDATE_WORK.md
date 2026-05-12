# Next Candidate Work

This file should be updated after each major loop.

## Ranked Backlog

1. Add CUDA-backed integration coverage for reported output-shape capacity
   checks when a cheap NVCC/torch fixture is available in CI; the Python
   device-pointer contract now has non-CUDA regression coverage, including
   getter-bound malformed negative reported-shape rejection and rank-growth
   rejection between the two output-shape ABI calls.
2. Rotate to a bounded user-visible workflow or op-parity slice before more
   provider/profile hardening. The CPU CLI quick-start path now has regression
   coverage for `compile`, `inspect`, runtime loading, and `validate`, so the
   next workflow slice should prove a different real path such as deferred
   constants through the CLI or a small non-CUDA example that exercises a
   recently ported primitive.
3. Consider `masked_select` only if the full OP_ADMISSION checklist can be kept
   bounded in one loop: frontend contract, static shape/type limits, CPU
   reference behavior, generated lowering or an explicit bounded helper,
   targeted tests, and checklist updates. If any part is unclear, leave it
   deferred instead of adding frontend-only surface.
4. Continue runtime/container stabilization, but rotate to a fresh concrete
   contract rather than repeatedly polishing the same CUDA helper paths. Useful
   bounded targets include graph-mode lifecycle, runtime pool/session ownership,
   and remaining allocator or constant-state failure behavior that can be
   validated without CUDA CI. Recent hardening has already covered
   closed-session/module guards, output-shape ABI validation, dense constant
   reload preflight, encoded-constant pre-materialization before setter
   application, CUDA staging cleanup retries, CUDA constant setter error
   precedence, profile-cache malformed-entry rejection plus same-target
   stale-writer preservation, scoped CUDA helper/profiler error reporting,
   generated CUDA session-create cleanup for partially allocated session-owned
   buffers, Python session-close retry behavior when staging-buffer cleanup or
   native session destruction fails, Python runtime-module construction cleanup
   when metadata initialization fails after native load succeeds, and the
   non-CUDA CLI quick-start workflow.
5. Continue provider/profile artifact hardening only for concrete,
   project-visible failures or for CUDA-backed profile/report cache coverage
   that becomes cheap enough for CI. Recent coverage rejects or skips stale
   CUTLASS launcher/profiler symbols and malformed guarded dispatch shape
   metadata before attaching execution-plan dispatch to manifests, rejects
   malformed profile cache entries whose embedded `profile_key` is missing or
   inconsistent with the cache map key, and strongly suggests rotating unless a
   new failure is visible.
