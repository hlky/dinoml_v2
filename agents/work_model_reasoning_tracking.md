# Work Model / Reasoning Tracking

Track the model and reasoning effort assigned to committed work so future PM
loops can choose the right capability level for similar tasks.

Format:

```text
{commit_id}, {title}, {model}, {reasoning_effort}
```

## Entries

- bff8139, Document rotary apply exploration and next slice, gpt-5.4, medium
- 3bb83a8, Add helper-only 1D rotary table generation, gpt-5.4, medium
- 191e1d9, Harden CUTLASS conv scaffold transform plans, gpt-5.4, medium
- 7f839d0, Tighten CUTLASS conv scaffold provenance validation, gpt-5.4, medium
- bda4377, Register generated get_timestep_embedding op, gpt-5.4, high
- f50b79c, Finish generated get_1d_rotary_pos_embed slice, gpt-5.4, high
- e93db9e, Fix rotary follow-up provenance and arity contracts, gpt-5.4, high
- 3d05806, Repair named permute specializations, gpt-5.4, high
- bd0ea2a, Align specialized permute schema attrs, gpt-5.4, high
- 34eff08, Extract where frontend module, gpt-5.4, medium
- 0d73929, Compile CUTLASS conv support stubs, gpt-5.5, high
- b6ea294, Fix CUTLASS conv source-only scaffold metadata, gpt-5.5, high
- 199ca28, Exercise CUTLASS conv profiler stub export, gpt-5.5, high
- f6dfe0d, Add CUTLASS conv layout transform helpers, gpt-5.4, high
- 29bf5aa, Expose GGUF dequant scratch session resource, gpt-5.5, high
