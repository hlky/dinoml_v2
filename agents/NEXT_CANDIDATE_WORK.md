# Next Candidate Work

This file should be updated after each major loop.

## Ranked Backlog

1. Add CUDA-backed integration coverage for reported output-shape capacity
   checks when a cheap NVCC/torch fixture is available in CI; the Python
   device-pointer contract now has non-CUDA regression coverage, including
   getter-bound malformed negative reported-shape rejection and rank-growth
   rejection between the two output-shape ABI calls.
2. Add profile/report cache regression coverage for sourceable symbolic
   expression shapes once a real CUDA profiling fixture is cheap enough for CI.
3. Continue improving runtime/container contracts for allocator, graph, pool,
   profiling, and constant-state failure behavior before op-specific
   assumptions spread; module-close now invalidates live Python sessions before
   freeing the native module handle, and CUDA staging-buffer grow failures now
   preserve the previously cached session buffer. Dense constant reload now
   rejects truncated constant files before mutating resident constants. Profile
   cache reuse now rejects malformed entry maps, malformed timing/count fields,
   and insufficient timing samples before treating a cached result as
   confidence-policy eligible. CUDA session staging-buffer cleanup now forgets
   successfully freed cached buffers even if a later free fails, so retrying
   close/cleanup does not retain stale freed pointers. Module close now attempts
   every live session close before reporting the first failure, while preserving
   the module handle if any session cleanup failed. CUDA constant updates now
   preserve the primary setter/copy failure when temporary device-buffer cleanup
   also fails. CUDA helper-library allocator/copy/free errors now surface their
   own last-error messages through the Python runtime instead of falling back to
   an unknown runtime error, and CUDA helper call sites now scope last-error
   lookup to that helper library so stale module messages do not mask fresh
   allocator/copy/free failures.
