# Common Failure Modes

Use this file when drafting pack-local guardrails.

Only include the failure modes relevant to the selected task.

## 1) Frontend-only completion

Failure mode:

- task lands API or trace-level support
- reference path works
- backend truth remains unimplemented

Guardrail:

- say explicitly that frontend or reference parity alone does not complete the task

## 2) Lowering-only support claims

Failure mode:

- codegen or lowering renders successfully
- runtime parity was never run
- backend support is still claimed

Guardrail:

- say explicitly that lowering, render, scaffold, or manifest coverage does not count as backend runtime support

## 3) Architecture switching

Failure mode:

- required provider or lowering stack is hard to extend
- implementation escapes into a different ad hoc path

Guardrail:

- name the required architecture directly
- state that inability to extend it is a blocker, not permission to switch styles

## 4) Silent scope shrink

Failure mode:

- hard part of the task is deferred without rewriting the task contract
- final result claims partial completion as full completion

Guardrail:

- write explicit non-scope
- state which covered ops or backends are required for completion

## 5) Rewrite as dodge

Failure mode:

- real backend or IR work is avoided by overusing decomposition

Guardrail:

- say whether decomposition is acceptable completion
- if only selected decompositions are allowed, say so precisely

## 6) Broad representation trap

Failure mode:

- an apparently small op actually requires complex dtype support, jagged representation, sparse semantics, or dynamic output representation

Guardrail:

- call this out during triage and classify it as broader representation work instead of a routine next-op pack

## 7) CUDA pending treated as complete

Failure mode:

- CPU and ROCm pass
- CUDA remains unverified
- task is still marked done

Guardrail:

- state the required CUDA verification path
- say that pending CUDA verification keeps the task out of done unless CUDA is explicitly out of scope

## 8) Static-shape-only completion

Failure mode:

- task contract requires static-rank dynamic shape or runtime-shape-aware behavior
- implementation only works when relevant extents are compile-time constants
- result is still presented as completing the task

Guardrail:

- state the shape contract explicitly: static-only, static-rank dynamic, runtime-shape-dependent, or deferred
- require verification that exercises runtime-varying extents when dynamic shape support is in scope

## 9) v1 structure imported into v2

Failure mode:

- v1 is used as a structural template rather than a parity reference

Guardrail:

- say that v1 is for behavioral or oracle guidance only unless the task explicitly calls for a provider or runtime pattern that still applies

## 10) Validation mismatch

Failure mode:

- tests run do not actually exercise the changed behavior

Guardrail:

- name the expected narrow checks
- separate reference parity, lowering checks, runtime parity, and throughput work
