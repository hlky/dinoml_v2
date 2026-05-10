# Project Invariants

These rules should not be broken without explicit human approval.

## Architecture

- v2 is not a direct v1 clone.
- v1 is a behavioral reference, not a structural template.
- Public behavior should be represented in explicit schemas where practical.
- Runtime/provider decisions should be artifact-visible.

## Providers

- Provider candidates must be represented in manifests.
- Profile results must produce profile reports.
- Selected candidates must flow through execution plans or explicit manifest overlays.
- Generated code must visibly use selected candidates.
- Provider support libraries must record cache keys and provenance.

## Constants

- Constants must have explicit ownership and residency semantics.
- Encoded constants must preserve source metadata.
- Runtime load/unload/reload must fail clearly when state is invalid.
- Future offload work must build on explicit constant state, not ad hoc pointer nullability.

## Ops

An op is not complete unless it has:
- frontend contract
- shape/type inference
- reference validation or documented reason
- backend lowering or documented bounded helper behavior
- tests or validation
- checklist update

## Docs

- Any behavior change that affects architecture, ops, providers, runtime, constants, or artifacts must update the relevant docs.
