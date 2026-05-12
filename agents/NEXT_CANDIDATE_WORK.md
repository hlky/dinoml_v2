# Next Candidate Work

This file should be updated after each major loop.

## Ranked Backlog

1. Add CUDA-backed integration coverage for reported output-shape capacity
   checks when a cheap NVCC/torch fixture is available in CI; the Python
   device-pointer contract now has non-CUDA regression coverage, including
   getter-bound malformed negative reported-shape rejection and rank-growth
   rejection between the two output-shape ABI calls.
2. Continue provider/profile artifact hardening with execution-plan cache key
   preservation or profile report/cache consistency that can be validated
   without CUDA CI. Recent coverage now rejects or skips stale CUTLASS
   launcher/profiler symbols and malformed guarded dispatch shape metadata
   before attaching execution-plan dispatch to manifests.
3. Add profile/report cache regression coverage for sourceable symbolic
   expression shapes once a real CUDA profiling fixture is cheap enough for CI.
4. Continue runtime/container stabilization, but rotate to a fresh concrete
   contract rather than repeatedly polishing the same CUDA helper paths. Useful
   bounded targets include graph-mode lifecycle, runtime pool/session ownership,
   and remaining allocator or constant-state failure behavior that can be
   validated without CUDA CI. Recent hardening has already covered
   closed-session/module guards, output-shape ABI validation, dense constant
   reload preflight, encoded-constant pre-materialization before setter
   application, CUDA staging cleanup retries, CUDA constant setter error
   precedence, profile-cache malformed-entry rejection plus same-target
   stale-writer preservation, scoped CUDA helper/profiler error reporting, and
   generated CUDA session-create cleanup for partially allocated session-owned
   buffers, and Python session-close retry behavior when staging-buffer cleanup
   or native session destruction fails, plus Python runtime-module construction
   cleanup when metadata initialization fails after native load succeeds.
