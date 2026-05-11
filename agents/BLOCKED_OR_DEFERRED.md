# Blocked or Deferred Work

These are intentionally deferred unless explicitly requested.

## Deferred

## Needs Design First

- Expression-only symbolic shape source recovery: current generated lowering and
  profiling require every symbolic expression leaf to have a direct runtime
  `Dim` source from an input or constant. Inferring a named dim from only an
  expression result, such as recovering `n` from `n // 2`, is not generally
  invertible and needs an explicit design before implementation.
