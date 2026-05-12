# Next Candidate Work

This file should be updated after each major loop.

## Ranked Backlog

1. Rotate the onboarding path from example quantity to visible compile/run
   usability. The quick-start CPU path should keep using
   `examples/fused_elementwise.py` as the canonical minimal artifact, with
   coverage for both CLI compile/inspect/validate and the direct Python runtime
   API loop (`runtime.load`, `create_session`, `run_numpy`, explicit cleanup).
   `dinoml compile` is now intentionally CPU-first by default, and the
   no-`--target` fused-elementwise CLI path has regression coverage proving the
   artifact target is CPU. Prefer small usability fixes tied to that tested path
   over broad CLI help rewrites or another showcase example.
2. Stop the visible CPU example burst unless a future PM request identifies a
   genuinely distinct workflow. The CLI quick-start path has regression coverage
   for `compile`, `inspect`, runtime loading, and `validate`; deferred constants
   have CLI compile plus validation coverage that explicitly loads constants for
   the correctness run; `examples/image_pooling.py` covers pad/avg-pool/max-pool;
   `examples/candidate_selection.py` covers `topk` plus `batch_gather`;
   `examples/subpixel_upsample.py` covers `pixel_shuffle`; and
   `examples/coordinate_ramp.py` covers creation helpers (`full`/`arange`/
   `meshgrid`) feeding fused elementwise math through the same CPU
   compile/inspect/validate path. A first distinct CUDA-facing model workflow
   now exists in `examples/cuda_linear.py`: it uses explicit `gemm_rrr_bias`,
   dense runtime-settable weight/bias constants, a bucketed dynamic batch
   dimension, and a `--no-tf32` CUTLASS manifest/runtime test to keep provider
   build cost bounded. The same compact model path now has cheap
   profile-assisted compile coverage: a fake profiler selects a non-default
   no-TF32 CUTLASS candidate, `dml.compile(profile=True)` rebuilds from the
   generated execution plan, and the final manifest/codegen plan consume the
   selected candidate without invoking NVCC or a full profile run. Prefer the
   next non-example project priority by default; broader CUDA model workflows
   should add a genuinely new provider/runtime contract rather than another
   showcase file.
3. Leave `masked_select` queued, not admitted. A bounded admission pass found
   that the op's PyTorch/v1 contract has a value-dependent 1D output length in
   `[0, broadcast_numel]`, including all-false masks that produce shape `[0]`.
   V2 `Shape`/`Dim`, caller allocation specs, and normal runtime shape
   validation currently require every shape dimension to be positive, and
   generated modules report output shapes from the caller-provided output
   descriptor rather than an op-updated runtime count. Do not add frontend-only
   `masked_select` or a contract that lies with `min=1`. A first focused
   prerequisite test fixture now proves the Python post-run reported-shape path
   accepts zero-length output reports for `get_output_shape`, NumPy
   materialization, and direct CUDA device-pointer capacity checks while still
   rejecting negative reports. The first internal metadata/codegen slice also
   validates `metadata.output_shape_reports` entries and lets generated CPU/CUDA
   modules report selected output shapes from their generated shape buffers
   rather than the caller-provided output descriptors. CUDA shape-buffer reports
   now preserve the external-stream contract: internally synchronized runs copy
   device shape buffers back to host and make reports available, while
   externally streamed runs avoid the host copy/synchronization and leave those
   reports unavailable. The next admissible slice is an op-local generated
   CPU/CUDA shape-buffer override/counting fixture for a static-rank
   value-dependent output; after that, re-run OP_ADMISSION for a static-rank,
   dense, broadcastable bool-mask `masked_select` helper.
4. Continue runtime/container stabilization, but rotate to a fresh concrete
   contract rather than repeatedly polishing the same CUDA helper paths. Useful
   bounded targets include graph-mode lifecycle, runtime pool/session ownership,
   and remaining allocator or constant-state failure behavior that can be
   validated without CUDA CI. Recent hardening has already covered
   closed-session/module guards, output-shape ABI validation, dense constant
   reload preflight, dense constant setter loaded-state preservation when
   native CPU/CUDA setters fail, encoded-constant pre-materialization before setter
   application, encoded-constant runtime-metadata membership preflight,
   encoded-constant loaded-state rollback when a setter fails, CUDA
   staging cleanup retries, CUDA constant setter error precedence, profile-cache
   malformed-entry rejection plus same-target stale-writer preservation, scoped
   CUDA helper/profiler error reporting, generated CUDA session-create cleanup
   for partially allocated session-owned buffers, Python session-close retry
   behavior when staging-buffer cleanup or native session destruction fails,
   Python runtime-module construction cleanup when metadata initialization fails
   after native load succeeds, generated CPU/CUDA run-start invalidation of
   stale post-run output-shape reports on failed attempted native runs,
   including null input/output arrays and wrong input/output counts, the non-CUDA
   CLI quick-start workflow, and CUDA-backed direct device-pointer reported
   output-shape capacity rejection through a cheap identity artifact. The latest
   input/output map pass also rejects non-mapping caller bindings and unexpected
   tensor names for `run_numpy`, `run_torch`, and direct CUDA pointer execution
   before staging, tensor validation, or pointer packing. `run_torch` now also
   rejects mixed CUDA-device inputs before output allocation or raw pointer
   packing, making the single-device execution assumption project-visible.
   Zero-input CUDA artifacts now fail through an explicit `run_torch` contract
   error before output allocation, because the torch frontend infers output
   placement from caller-provided CUDA inputs.
   Python session
   construction now also destroys a partially created native session handle if
   native creation or session tracking fails before returning a usable
   `Session`, and Python module/session construction rejects successful native
   load/create calls that return null handles before metadata reads or session
   tracking can proceed.
5. Continue provider/profile artifact hardening only for concrete,
   project-visible failures or for CUDA-backed profile/report cache coverage
   that becomes cheap enough for CI. Recent coverage rejects or skips stale
   CUTLASS launcher/profiler symbols and malformed guarded dispatch shape
   metadata before attaching execution-plan dispatch to manifests, rejects
   guarded dispatch selections whose `node_id` is missing or no longer matches
   the profiled manifest node, preserves bucket-derived guarded dispatch shape
   metadata through execution-plan application, rejects
   malformed profile cache entries whose embedded `profile_key` is missing or
   inconsistent with the cache map key, rejects cache hits whose embedded key
   payload no longer matches the current hardware/support/profile key payload,
   proves profile-assisted rebuild consumption on the compact CUDA linear
   model path, verifies keyed execution plans against their payload before
   compile applies provider selections, and records applied execution-plan
   summaries in both `compile_config.json` and top-level `manifest.json`.
   Rotate unless a new
   failure is visible; the next provider/profile slice should be a distinct
   contract such as persistent shared cache behavior, guarded dynamic model
   dispatch with real bucket conflicts, or a concrete runtime/profile error
   path.
