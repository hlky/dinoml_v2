# Next Candidate Work

This file should be updated after each major loop.

## Ranked Backlog

1. Stop the visible CPU example burst unless a future PM request identifies a
   genuinely distinct workflow. The CLI quick-start path has regression coverage
   for `compile`, `inspect`, runtime loading, and `validate`; deferred constants
   have CLI compile plus validation coverage that explicitly loads constants for
   the correctness run; `examples/image_pooling.py` covers pad/avg-pool/max-pool;
   `examples/candidate_selection.py` covers `topk` plus `batch_gather`;
   `examples/subpixel_upsample.py` covers `pixel_shuffle`; and
   `examples/coordinate_ramp.py` covers creation helpers (`full`/`arange`/
   `meshgrid`) feeding fused elementwise math through the same CPU
   compile/inspect/validate path. Prefer the next non-example project priority
   rather than adding more showcase files by default.
2. Consider `masked_select` only if the full OP_ADMISSION checklist can be kept
   bounded in one loop: frontend contract, static shape/type limits, CPU
   reference behavior, generated lowering or an explicit bounded helper,
   targeted tests, and checklist updates. If any part is unclear, leave it
   deferred instead of adding frontend-only surface.
3. Continue runtime/container stabilization, but rotate to a fresh concrete
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
   when metadata initialization fails after native load succeeds, the non-CUDA
   CLI quick-start workflow, and CUDA-backed direct device-pointer reported
   output-shape capacity rejection through a cheap identity artifact.
4. Continue provider/profile artifact hardening only for concrete,
   project-visible failures or for CUDA-backed profile/report cache coverage
   that becomes cheap enough for CI. Recent coverage rejects or skips stale
   CUTLASS launcher/profiler symbols and malformed guarded dispatch shape
   metadata before attaching execution-plan dispatch to manifests, rejects
   malformed profile cache entries whose embedded `profile_key` is missing or
   inconsistent with the cache map key, and strongly suggests rotating unless a
   new failure is visible.
