# Work Model / Reasoning Tracking

Track the model and reasoning effort assigned to committed work so future PM
loops can choose the right capability level for similar tasks.

Format:

```text
{commit_id}, {title}, {model}, {reasoning_effort}
```

## Entries

- 3a32757, Scope CUTLASS conv provider status names, gpt-5.5, high
- 629d793, Build guarded CUTLASS conv CUDA wrapper, gpt-5.5, high
- 6710c2d, Add CLIP text encoder-layer composition tests, gpt-5.4, high
- 1ba2f2f, Add CLIP contrastive head composition tests, gpt-5.4-mini, medium
- e3e1955, Add CLIP text MLP composition tests, gpt-5.4-mini, medium
- 497c83b, Add CLIP text attention composition tests, gpt-5.4, high
- ceb1f4f, Add CLIP text embedding composition tests, gpt-5.4-mini, medium
- 805133e, Add CLIP text pooling composition tests, gpt-5.4-mini, medium
- 0d460e3, Add bounded integer argmax input support, gpt-5.4, medium
- e69c2cc, Add generated embedding lookup op, gpt-5.4, high
- b059c91, Add generated affine layer_norm op, gpt-5.4, high
- 9a9eac2, Harden dynamic get_1d_rotary_pos_embed CUDA runtime, gpt-5.4, medium
- 625b25c, Harden dynamic get_timestep_embedding CUDA runtime, gpt-5.4, medium
- 6742bd7, Harden named permute runtime regressions, gpt-5.4, medium
- e7b11bf, Harden rms_norm helper runtime regressions, gpt-5.4, high
- edfa639, Emit CUTLASS conv scaffold wrapper debug sources, gpt-5.4, high
- def4984, Record CUTLASS conv wrapper staging metadata, gpt-5.4, high
- 61c5a0f, Add native GGUF runtime-dequant reload regression, gpt-5.4, high
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
- 49a1382, Fix GGUF dequant scratch cache keys, gpt-5.5, high
