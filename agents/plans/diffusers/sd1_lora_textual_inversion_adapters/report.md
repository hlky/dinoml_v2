# SD1 LoRA, Textual Inversion, and Runtime Adapter Mutation Audit

## 1. Source basis

```text
Diffusers commit/version:
  Local checkout X:/H/diffusers at b3a515080752a3ba7ca92161e25530c7f280f629.

Model id(s):
  Base family reference: stable-diffusion-v1-5/stable-diffusion-v1-5.
  Candidate target: sd1_lora_textual_inversion_adapters.

Config sources:
  Reused the base SD1 report for component dimensions and pipeline wiring:
  H:/dinoml_v2/agents/plans/diffusers/stable_diffusion_1_5/report.md.
  No adapter checkpoint configs were required for this mutation-surface audit.

Pipeline files inspected:
  X:/H/diffusers/src/diffusers/pipelines/stable_diffusion/pipeline_stable_diffusion.py

Loader/runtime mutation files inspected:
  X:/H/diffusers/src/diffusers/loaders/lora_pipeline.py
  X:/H/diffusers/src/diffusers/loaders/lora_base.py
  X:/H/diffusers/src/diffusers/loaders/textual_inversion.py
  X:/H/diffusers/src/diffusers/loaders/peft.py
  X:/H/diffusers/src/diffusers/loaders/unet.py
  X:/H/diffusers/src/diffusers/utils/peft_utils.py

Model and attention files inspected:
  X:/H/diffusers/src/diffusers/models/unets/unet_2d_condition.py
  X:/H/diffusers/src/diffusers/models/lora.py
  X:/H/diffusers/src/diffusers/models/attention.py
  X:/H/diffusers/src/diffusers/models/attention_processor.py

Any missing files or assumptions:
  PEFT internals are an external dependency and were inferred only through
  Diffusers call sites. Training, dropout, loss, Control LoRA, IP-Adapter, and
  backend-specific offload variants are out of scope except where they share
  the SD1 adapter mutation APIs.
```

## 2. SD1 integration anchors

`StableDiffusionPipeline` inherits `TextualInversionLoaderMixin` and
`StableDiffusionLoraLoaderMixin` in
`pipeline_stable_diffusion.py:154-160`. The prompt path calls
`maybe_convert_prompt()` before tokenization for both positive and negative
prompts (`pipeline_stable_diffusion.py:391-395`, `474-476`). The `lora_scale`
argument to `encode_prompt()` temporarily scales PEFT LoRA layers on the text
encoder, runs CLIP, then unscales (`pipeline_stable_diffusion.py:373-382`,
`507-510`).

The SD1 LoRA loader surface is anchored by
`StableDiffusionLoraLoaderMixin` in `lora_pipeline.py:133-242`. It declares
`_lora_loadable_modules = ["unet", "text_encoder"]`, fetches and normalizes a
LoRA state dict, then calls:

- `load_lora_into_unet()` -> `unet.load_lora_adapter(...)`
  (`lora_pipeline.py:356-414`).
- `load_lora_into_text_encoder()` -> `_load_lora_into_text_encoder(...)`
  (`lora_pipeline.py:417-472`).

`UNet2DConditionModel` inherits `PeftAdapterMixin` directly
(`unet_2d_condition.py:75-76`), so SD1 UNet adapter behavior is the generic
Diffusers PEFT model mixin path, not a pipeline-local attention-processor-only
implementation.

## 3. Runtime mutation surfaces

| Surface | Source anchors | State mutated | Runtime graph implication |
| --- | --- | --- | --- |
| Load LoRA into UNet | `StableDiffusionLoraLoaderMixin.load_lora_weights`, `load_lora_into_unet`, `PeftAdapterMixin.load_lora_adapter` | Injects PEFT `BaseTunerLayer` wrappers into target UNet modules, adds adapter parameters and `peft_config`, loads adapter tensors. | Changes the executable module graph unless hotswap was prepared before compilation. Dinoml should treat this as artifact/constant-state mutation before compile or explicit runtime adapter state, not hidden Python side effect. |
| Load LoRA into text encoder | `_load_lora_into_text_encoder` in `lora_base.py:321-432` | Calls Transformers `text_encoder.load_adapter(...)`, adds PEFT adapter layers/config, scales to requested `lora_scale`, restores dtype/device. | Affects prompt embedding generation only. If Dinoml first accepts external `prompt_embeds`, text-encoder LoRA is preprocessing/artifact scope. If compiling CLIP, it becomes a CLIP adapter graph variant. |
| `set_adapters` | `lora_base.py:676-777`, `peft.py:452-506`, `peft_utils.py:246-276` | Activates named adapters and changes per-adapter/per-block scales. | Runtime-selected adapter weights change effective Linear/Conv math. Needs explicit adapter activation state in a Dinoml execution plan or force recompile/fuse. |
| `disable_lora` / `enable_lora` | `lora_base.py:779-837`, `peft.py:717-761`, `peft_utils.py:212-221` | Toggles PEFT adapter layers enabled/disabled. | Equivalent to selecting adapter scale 0 or restoring active scales, but implemented as mutable module flags. Dinoml should expose this as adapter-state metadata, not a dynamic Python flag inside compiled graph. |
| `fuse_lora` | `lora_pipeline.py:537-578`, `lora_base.py:537-621`, `peft.py:661-689` | Merges low-rank delta into base weights through PEFT `BaseTunerLayer.merge`; tracks merged adapter names. | Produces new effective base constants. This is the best first Dinoml admission path: precompute merged weights before graph compile/load and run the ordinary SD1 graph. |
| `unfuse_lora` | `lora_pipeline.py:580-594`, `lora_base.py:623-675`, `peft.py:691-700` | Calls `unmerge()` on PEFT tuner layers, subtracting previously merged deltas. | Mutable constant rollback. Dinoml should support only when original base weights and adapter deltas remain artifact-visible; otherwise require reload. |
| `unload_lora_weights` / `unload_lora` | `lora_base.py:514-535`, `peft.py:702-715`, `peft_utils.py:35-103` | Removes PEFT wrapper layers, deletes `peft_config`, resets `_hf_peft_config_loaded`. | Changes graph topology back to base modules. First admission should require a new compile/open after unload unless implemented as explicit optional adapter branches. |
| `delete_adapters` | `lora_base.py:839-875`, `peft.py:763-799` | Deletes selected adapter weights/config from each tuner layer. | Removes named runtime constants. Should invalidate plans that reference deleted adapters. |
| `set_lora_device` | `lora_base.py:932-984` | Moves adapter A/B/magnitude tensors between CPU/GPU. | Residency/offload state only. Dinoml should model adapter constants with explicit residency; using a CPU-resident active adapter should fail clearly or trigger an explicit transfer plan. |
| `enable_lora_hotswap` / hotswap load | `lora_base.py:986-1002`, `peft.py:280-342`, `801-832` | Prepares fixed-rank adapter slots, then replaces adapter tensors in-place. Text encoder hotswap is rejected. | Useful future runtime feature, but first admission should not promise no-recompile hotswap. If admitted later, rank/max-shape slots must be compile-visible. |
| Textual inversion load | `textual_inversion.py:272-460` | Extends tokenizer vocabulary, resizes text encoder input embedding table, writes loaded embeddings into new token rows. | Changes tokenizer state and CLIP embedding constants. For Dinoml SD1 with external prompt embeddings, this is preprocessing. For compiled CLIP, it changes vocab size and embedding constants. |
| Textual inversion prompt conversion | `textual_inversion.py:123-178`, pipeline calls above | Rewrites a prompt containing one multi-vector token into multiple added tokens (`token token_1 ...`). | CPU tokenizer preprocessing that changes token sequence before CLIP. It can affect CLIP length/truncation and therefore prompt embeddings. |
| Textual inversion unload | `textual_inversion.py:467-605` | Removes added tokens, rewrites tokenizer added-token ids, rebuilds a filtered `nn.Embedding`. | Mutates tokenizer id mapping and embedding table shape. Must invalidate cached token ids and prompt embeddings. |

## 4. LoRA weight math and graph effects

Current SD1 LoRA loading requires the PEFT backend
(`lora_pipeline.py:199-200`). The UNet path normalizes checkpoint keys, derives
rank from `lora_B.weight` shapes, creates a `LoraConfig`, then injects adapter
layers and calls `set_peft_model_state_dict()` (`peft.py:216-330`). Diffusers
derives PEFT `target_modules` from adapter state-dict keys in
`peft_utils.py:185-198`, so the exact mutated modules are checkpoint-dependent.
For SD1 checkpoints, common targets are UNet attention Linear projections
(`to_q`, `to_k`, `to_v`, `to_out.0`), feed-forward projections, and sometimes
Conv2d layers. Text encoder loading explicitly scans CLIP modules ending in
`.q_proj`, `.k_proj`, `.v_proj`, `.out_proj`, `.fc1`, and `.fc2`
(`lora_base.py:379-383`).

Unfused LoRA changes a target module from:

```text
y = base(x)
```

to:

```text
y = base(x) + scale(adapter) * B(A(x))
```

where `A` is the rank-down projection and `B` is the rank-up projection. For
Linear, this is two extra GEMMs plus an add per adapted Linear. For Conv2d, the
deprecated Diffusers LoRA-compatible layer shows the same low-rank intent with
`down Conv2d(in -> rank)` followed by `up Conv2d(rank -> out, 1x1)`
(`models/lora.py:271-294`). With PEFT, Conv/Linear details come from PEFT, but
Dinoml should admit the same semantic forms.

Fused LoRA computes an effective base weight:

```text
W_eff = W_base + lora_scale * W_up @ W_down * optional_alpha/rank
```

The deprecated local helpers show the expected Linear and Conv fusion shapes:
Linear fusion uses `torch.bmm(w_up[None, :], w_down[None, :])[0]`
(`models/lora.py:117-142`, `401-431`), while Conv fusion flattens up/down
weights, multiplies them, and reshapes to the original Conv kernel
(`models/lora.py:315-347`). PEFT's `merge()` is authoritative for current
runtime behavior, but these deprecated helpers are useful shape evidence for a
Dinoml offline/fuse implementation.

## 5. Textual inversion token and embedding mutation

`load_textual_inversion()` accepts Hub paths, local files, tensors, or state
dicts. It supports Diffusers one-key state dicts, raw tensor embeddings when a
token is supplied, and Automatic1111 `string_to_param` format
(`textual_inversion.py:204-242`). Multi-vector embeddings are expanded into
multiple tokenizer tokens (`token`, `token_1`, ...), and prompt conversion later
substitutes those extra tokens before tokenization (`textual_inversion.py:245-260`,
`123-178`).

The runtime mutation happens after all validation:

- `text_encoder.resize_token_embeddings(len(tokenizer) + len(tokens))`
  expands the CLIP input embedding table (`textual_inversion.py:445-447`).
- `tokenizer.add_tokens(token)` adds each new token, then
  `input_embeddings.data[token_id] = embedding` writes the loaded vector into
  the embedding table (`textual_inversion.py:450-455`).
- Unload filters tokenizer added tokens and builds a new `nn.Embedding` from
  all retained rows (`textual_inversion.py:558-605`).

For SD1 first slice, textual inversion should be classified as prompt
preprocessing plus text-encoder embedding-table mutation. It does not change
UNet, VAE, scheduler, or denoising-loop ops. It can change the prompt-embedding
cache key because the same prompt string can tokenize differently after
load/unload.

## 6. Preprocessing/artifact loading versus runtime op scope

Preprocessing and artifact-loading scope:

- Fetching LoRA or textual inversion files from Hub/local disk and converting
  A1111/Kohya/Diffusers key formats.
- Computing LoRA rank/alpha metadata and PEFT `LoraConfig`.
- Tokenizer vocabulary edits and textual inversion prompt-string expansion.
- Loading or mutating CLIP embedding rows when Dinoml accepts external
  `prompt_embeds`.
- Fusing LoRA deltas into base weights before compile or before runtime module
  open, if Dinoml records provenance and effective constants.

Runtime op scope:

- Unfused active LoRA in UNet: extra low-rank Linear/Conv branches inside the
  denoising step, including per-adapter scales and optional multi-adapter
  summation.
- Unfused active LoRA in text encoder only if Dinoml compiles CLIP/tokenization
  rather than accepting prompt embeddings.
- Adapter activation, disable/enable, device residency, delete, fuse/unfuse, and
  hotswap are runtime state transitions; they are not ordinary tensor ops and
  should not hide inside a compiled graph.
- Textual inversion prompt conversion and tokenization remain CPU/data-pipeline
  work; embedding lookup is runtime only for a compiled CLIP stage.

## 7. Ops and fusion implications

First Dinoml path, fused/offline:

- No new denoiser ops beyond the base SD1 UNet if LoRA is fused into base
  constants before compile or artifact open.
- Need artifact-visible constant provenance:
  `base_weight`, adapter `A/B`, alpha/rank, adapter scale, adapter name, target
  module path, and resulting `W_eff` hash.
- Need rollback/reload policy for unfuse: either retain base constants and
  adapter deltas or require reopening the base artifact.

Unfused adapter path:

- Linear target: base GEMM plus `x @ A.T @ B.T`, scaled and added. A fused
  provider could lower this as separate GEMMs first, then later as fused
  low-rank epilogue or base-plus-low-rank GEMM.
- Conv2d target: base Conv2d plus rank-down Conv2d and rank-up 1x1 Conv2d,
  scaled and added. Layout passes must respect NCHW/NHWC guards from the base
  SD1 report.
- Multi-adapter activation is a sum over active adapters with per-adapter or
  per-block weights. This affects memory planning because adapter constants can
  be resident, offloaded, or deleted independently of base weights.
- PEFT DoRA and `lora_bias` are detected in Diffusers utility code
  (`peft_utils.py:186-198`, `366-373`). They should be rejected or separately
  admitted; they are not required for a first SD1 LoRA slice.

Textual inversion path:

- Compiled CLIP requires dynamic or bounded embedding-table extension. A simple
  admission can instead materialize prompt embeddings outside Dinoml and treat
  the text encoder as preprocessing.
- Prompt cache keys must include tokenizer added-token state and text-encoder
  embedding-table version, not just the raw prompt string.

## 8. Validation risks

- PEFT is an external dependency. Diffusers call sites define integration
  points, but exact wrapper module classes and merge math are PEFT-version
  dependent. Dinoml should validate against installed PEFT behavior, not only
  deprecated Diffusers helper classes.
- `lora_scale` in `encode_prompt()` temporarily scales and unscales text
  encoder LoRA layers. A failure between scale/unscale would be stateful in
  Python; Dinoml should avoid transient mutable global scales in compiled
  execution.
- Hotswap is intended to avoid recompilation only when prepared before compile
  and rank/alpha compatibility holds; text encoder hotswap is explicitly
  unsupported by Diffusers (`lora_pipeline.py:181-195`,
  `lora_base.py:361-364`).
- Textual inversion unload renumbers added tokens and rebuilds the embedding
  table. Cached token ids, prompt embeddings, and any compiled embedding gather
  plan become stale.
- Fusing multiple adapters is order-sensitive in the sense that selected
  adapter names and scales define the final effective weight. Safe fusing may
  reject NaN deltas. Dinoml must record adapter set and scale in constant cache
  keys.
- `set_lora_device()` can leave an active adapter on CPU while the model is on
  GPU. Diffusers warns this causes device mismatch at inference; Dinoml should
  fail before launch unless an explicit transfer/offload plan exists.

## 9. First Dinoml admission recommendations

1. Admit textual inversion as preprocessing for SD1 first slice. The runtime
   accepts `prompt_embeds` and `negative_prompt_embeds`; tokenizer mutation and
   CLIP embedding-table mutation stay outside the compiled graph.
2. Admit LoRA initially only as an offline/pre-open fused-constant transform for
   UNet weights. Record adapter provenance and generate the same base SD1 graph
   with updated constants.
3. Treat text-encoder LoRA as preprocessing until CLIP compilation is admitted.
   If prompt embeddings are external, text-encoder LoRA has no runtime graph
   effect inside the denoiser artifact.
4. Defer unfused runtime adapters until Dinoml has an explicit adapter-state
   schema: loaded adapters, active adapters, per-component/per-block scales,
   residency, fused/unfused status, and invalidation rules.
5. Defer hotswap/no-recompile semantics. When admitted, require compile-visible
   max rank, fixed target module set, and validation that incoming adapters fit
   the prepared slots.
6. Reject or mark unsupported: DoRA, `lora_bias`, Control LoRA, adapter deletes
   during a live compiled run, and textual inversion unload against compiled
   CLIP without a graph/cache invalidation protocol.

## 10. Implementation checklist

- [ ] Add an adapter artifact schema for SD1 LoRA state: adapter name, target
      component, target module path, rank, alpha, scale, dtype, device/residency,
      and source checkpoint provenance.
- [ ] Implement offline Linear LoRA fusion parity for SD1 UNet attention/MLP
      target modules.
- [ ] Implement offline Conv2d LoRA fusion only after target checkpoint evidence
      proves Conv LoRA is needed; otherwise reject Conv targets clearly.
- [ ] Add fused-weight cache keys containing base weight hash, adapter weight
      hashes, scale, alpha/rank, target path, dtype, and Diffusers/PEFT version.
- [ ] Add validation against Diffusers+PEFT for one UNet attention LoRA target:
      unfused PEFT output versus fused effective weight output.
- [ ] Add a prompt preprocessing cache-key rule for textual inversion: tokenizer
      added-token state plus text-encoder embedding-table version.
- [ ] Keep CLIP/textual inversion outside Dinoml runtime until compiled CLIP is
      admitted.
- [ ] Design explicit runtime adapter state before admitting `set_adapters`,
      `enable_lora`, `disable_lora`, `delete_adapters`, `set_lora_device`, or
      hotswap inside a live Dinoml module.
