# Classification Rubric

Use this rubric when turning raw candidate lists into selected `.work/op/` tasks.

Assign exactly one label to each candidate.

## `already-covered`

Use when the current v2 surface already supports the behavior directly enough that no new pack is needed.

Typical cases:

- direct frontend op already exists
- existing op family already covers the required semantics
- current frontend spelling differs, but the requested behavior is already exposed in a nearby supported form

Do not mark a candidate `already-covered` only because a related op exists. Verify that the actual behavior is already available.

## `thin-frontend-rewrite`

Use when the missing user-facing spelling can be lowered cleanly to existing DinoML ops without adding new backend claims.

Typical cases:

- aliases of existing creation ops
- view or layout helpers expressible through existing shape or collection ops
- targeted decompositions into existing elementwise, reduction, reshape, split, gather, or concatenate ops

Do not use this label when the rewrite would:

- hide a meaningful semantic contract that should be explicit in IR
- require new runtime behavior
- create misleading backend support claims
- depend on unsupported dynamic outputs or unsupported dtypes

## `new-ir-op`

Use when the semantics are worth representing explicitly in v2 and can plausibly fit the current architecture without broad new representation work.

Typical signs:

- repeated use in important models
- awkward or misleading decomposition through current ops
- clear static-shape contract
- reasonable reference oracle
- plausible lowering path within existing architecture
- the task needs an explicit IR op and should still plan a real backend or kernel implementation rather than a permanent composed fallback

Do not use this label if honest support would immediately require broader type-system, jagged, sparse, or complex-tensor work.
Do not use this label when the pack would stop at a decomposition through existing ops instead of requiring a real backend or kernel endpoint.
Do not use this label when the main unresolved truth is really backend execution strategy or kernel availability.

## `real-backend-lowering-work`

Use when honest support requires lowering, runtime, provider, or backend implementation work.

Typical signs:

- provider stack changes are required
- runtime or kernel manifest support is required
- backend support claims would otherwise be false
- decomposition would be misleading or would dodge the real task
- the op has a meaningful performance identity and honest completion should end in a real provider or kernel implementation rather than a permanent composed fallback

This is the default label for conv, GEMM, BMM, pooling, and similar families when backend truth matters.
It is also the safer default when an op might look decomposable on paper but the expected steady-state implementation should not rely on a broad composed graph.

## `broader-representation-work`

Use when the candidate implies larger structural work outside a routine op task.

Typical signs:

- complex dtype support
- jagged or ragged representation support
- sparse or nested tensor representation
- dynamic output size or compaction semantics that do not fit the current stack cleanly

These should usually be deferred or grouped into a broader representation initiative rather than turned into a small op pack.

## `defer`

Use when the candidate is not the right next pack even if it is real.

Typical reasons:

- low demand relative to cost
- awkward semantics for the current architecture
- broad work with weak usage signal
- likely to distract from higher-value capability gaps

`defer` is a planning decision, not a claim that the op is unimportant forever.

## Required checks before finalizing classification

Before finalizing a candidate:

1. Check whether the current v2 op surface already covers it.
2. Check whether an existing DinoML op can express it cleanly without dishonest support claims.
3. Check whether the op would force broader representation work.
4. Check whether the expected backend truth matches the proposed task size.

## Common examples

- `matmul`: often `already-covered` through existing GEMM or BMM surface
- `zeros`, `ones`, `*_like`: often `thin-frontend-rewrite`
- `split_with_sizes`: often `already-covered` or `thin-frontend-rewrite` if existing `split` already accepts explicit sections
- `einsum`: usually `thin-frontend-rewrite` for selected patterns, not a generic new backend op
- `view_as_complex`, `view_as_real`, `polar`: usually `broader-representation-work`
- `nonzero`: often `broader-representation-work` because output shape depends on data
