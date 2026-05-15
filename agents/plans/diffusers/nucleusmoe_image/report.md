# Diffusers NucleusMoE Image Operator and Integration Report

Candidate slug: `nucleusmoe_image`

## 1. Source basis

```text
Diffusers commit/version:
  Local checkout diffusers at b3a515080752a3ba7ca92161e25530c7f280f629.

Model id(s):
  NucleusAI/NucleusMoE-Image from Diffusers docs/examples.
  Hugging Face Hub resolves the official repo metadata to NucleusAI/Nucleus-Image
  at commit 5e963db4fd0a65c7e4faf53ca2d4eca567c4dcfa.

Config sources:
  H:/configs did not contain a NucleusAI/NucleusMoE-Image or NucleusAI/Nucleus-Image cache.
  Official Hub component JSON was read with huggingface_hub from:
    model_index.json
    transformer/config.json
    scheduler/scheduler_config.json
    vae/config.json
    text_encoder/config.json
    text_encoder/generation_config.json
    processor/config.json
    processor/preprocessor_config.json
    processor/tokenizer_config.json
    processor/video_preprocessor_config.json
    transformer/diffusion_pytorch_model.safetensors.index.json
    text_encoder/model.safetensors.index.json
  The downloaded cache path was under
    C:/Users/user/.cache/huggingface/hub/models--NucleusAI--NucleusMoE-Image/snapshots/5e963db...
  This worker did not copy configs into H:/configs because the task owns only this report path.

Pipeline files inspected:
  diffusers/src/diffusers/pipelines/nucleusmoe_image/pipeline_nucleusmoe_image.py
  diffusers/src/diffusers/pipelines/nucleusmoe_image/pipeline_output.py
  diffusers/src/diffusers/pipelines/nucleusmoe_image/__init__.py

Model files inspected:
  diffusers/src/diffusers/models/transformers/transformer_nucleusmoe_image.py
  diffusers/src/diffusers/models/autoencoders/autoencoder_kl_qwenimage.py

Scheduler/processors/helpers inspected:
  diffusers/src/diffusers/schedulers/scheduling_flow_match_euler_discrete.py
  diffusers/src/diffusers/models/attention_dispatch.py
  diffusers/src/diffusers/models/attention_processor.py
  diffusers/src/diffusers/models/attention.py
  diffusers/src/diffusers/models/normalization.py
  diffusers/src/diffusers/models/embeddings.py
  diffusers/src/diffusers/hooks/text_kv_cache.py
  diffusers/src/diffusers/image_processor.py
  diffusers/tests/pipelines/nucleusmoe_image/test_nucleusmoe_image.py
  diffusers/tests/models/transformers/test_models_transformer_nucleusmoe_image.py

External component configs inspected:
  Qwen3VLForConditionalGeneration / Qwen3VLProcessor configs from the official
  Nucleus repo. Qwen/Qwen3-VL-4B/8B/30B repo metadata was also checked for
  public availability; the Nucleus repo contains its own text_encoder config.

Any missing files or assumptions:
  No official config read was gated or blocked. Only one official full
  NucleusMoE image Diffusers checkpoint/config set was found under NucleusAI.
  Representative variation is therefore source-default versus official config,
  not multiple production checkpoint variants. This report focuses on the
  base text-to-image pipeline. XLA/NPU/MPS/Flax/ONNX, safety, training/loss/
  dropout/gradient checkpointing, callbacks/interrupt, and multi-GPU/context
  parallel are out of scope.
```

## 2. Pipeline and component graph

`NucleusMoEImagePipeline` wires `NucleusMoEImageTransformer2DModel`,
`FlowMatchEulerDiscreteScheduler`, `AutoencoderKLQwenImage`,
`Qwen3VLForConditionalGeneration`, and `Qwen3VLProcessor`. The offload order is
`text_encoder->transformer->vae`.

```text
prompt / negative_prompt
  -> Qwen3VLProcessor chat template and tokenization
  -> Qwen3VLForConditionalGeneration hidden state at return_index, default -8
  -> latent noise [B,1,16,H/8,W/8] source NCTHW with one frame
  -> 2x2 pack to transformer tokens [B,(H/16)*(W/16),64]
  -> denoising loop:
       NucleusMoEImageTransformer2DModel(latent tokens, timestep/1000,
                                         prompt embeds, prompt mask, img_shapes)
       optional true CFG as a second negative transformer call
       CFG renorm over token channel norm
       negate model output
       FlowMatchEulerDiscreteScheduler.step
  -> unpack latent tokens to [B,16,1,H/8,W/8]
  -> AutoencoderKLQwenImage decode(latents / latents_std + latents_mean)
  -> take frame 0, VaeImageProcessor postprocess
```

First-slice required components are prompt embeddings as external tensors,
packed-token transformer forward, expert-choice MoE routing, FlowMatch Euler
step, true CFG arithmetic, latent unpacking, and QwenImage VAE decode boundary.
The Qwen3-VL encoder is independently cacheable and can be admitted later.

Separate candidate reports:

| Surface | Classes/files | Status and runtime delta |
| --- | --- | --- |
| LoRA / PEFT adapters | `PeftAdapterMixin` on `NucleusMoEImageTransformer2DModel`; generic `loaders/peft.py` and `loaders/lora_pipeline.py` | Transformer supports PEFT-style adapter state and LoRA scaling through `attention_kwargs["scale"]`; pipeline has no Nucleus-specific LoRA loader mixin. |
| Textual inversion | No textual inversion mixin or tokenizer mutation path on this pipeline | Not supported by the NucleusMoE image folder. |
| Runtime text K/V cache | `hooks/text_kv_cache.py` with Nucleus block hooks | Supported optimization candidate. Caches per-block text key/value projections by prompt embedding pointer and changes attention kwargs to `cached_txt_key/value`. |
| IP-Adapter | No Nucleus IP mixin or attention processor branch | Not supported in this folder. |
| ControlNet / T2I-Adapter / GLIGEN | No Nucleus model or pipeline classes found | Not supported in this folder. |
| img2img / inpaint / depth2img / upscaling | No non-deprecated Nucleus variants found | Not supported in this folder. |
| Single-file/original conversion | `FromOriginalModelMixin` on transformer and VAE | Separate loader/key-mapping candidate, not base ops. |
| Quantization/offload | Transformer tests include BitsAndBytes and TorchAO mixins; Diffusers group offload tests cover the pipeline | Separate weight/runtime policy candidate. |
| QwenImage VAE codec | `AutoencoderKLQwenImage` | Separate codec report candidate because it is a 3D causal conv VAE used as an image codec with `T=1`. |

## 3. Important config dimensions

Official checkpoint sweep:

| Config | Pipeline | Transformer | Text encoder | VAE | Scheduler | Notes |
| --- | --- | --- | --- | --- | --- | --- |
| `NucleusAI/NucleusMoE-Image` / resolved `NucleusAI/Nucleus-Image` | `NucleusMoEImagePipeline` | 32 single-stream blocks, 2048 hidden, 16 heads, 4 KV heads, 29 MoE blocks | Qwen3-VL 4096 hidden, 36 text layers | QwenImage VAE z=16, 8x spatial compression | FlowMatch Euler, `shift=1`, dynamic shifting disabled | Only official full Diffusers config found. |
| Source defaults | same classes | 24 blocks, 16 heads, KV heads default equal Q heads, 128 experts, first 3 and last dense by default strategy | pipeline expects Qwen3-VL | same VAE source defaults | FlowMatch defaults | Do not use source defaults over official config for production shape. |
| Tiny tests | same classes | 2 dense blocks, 4 heads x 16, hidden 64, in 16, out 4 | tiny Qwen3-VL config, hidden 16 | z=4 | default FlowMatch | Operator smoke only, not production shape. |

Transformer config:

| Field | Official value |
| --- | ---: |
| `patch_size` | 2 |
| VAE latent channels before pack | 16, inferred as `in_channels // 4` |
| packed token `in_channels` / projected output width | 64 / `patch_size * patch_size * out_channels = 64` |
| `out_channels` | 16 latent channels after unpack |
| `num_layers` | 32 |
| `num_attention_heads` x `attention_head_dim` | 16 x 128, inner dim 2048 |
| `num_key_value_heads` | 4, so GQA has 4 query groups per KV head |
| `joint_attention_dim` | 4096 |
| `axes_dims_rope` | `[16,56,56]`, scaled centered image RoPE |
| MoE strategy | `leave_first_three_blocks_dense`; layers 3 through 31 are MoE |
| `num_experts` | 64 |
| `moe_intermediate_dim` | 1344 |
| `capacity_factors` | first 3 zero/dense, layers 3-4 factor 4.0, layers 5-31 factor 2.0 |
| router | softmax scores, `route_scale=2.5`, `use_sigmoid=false` |
| expert compute | `use_grouped_mm=true` in official config |

Text/processor config:

| Component | Dimensions / behavior |
| --- | --- |
| `Qwen3VLForConditionalGeneration` text config | hidden 4096, 36 layers, 32 attention heads, 8 KV heads, head dim 128, intermediate 12288, vocab 151936, max positions 262144, dtype bfloat16. |
| Qwen3-VL vision config | depth 27, hidden 1152, 16 heads, patch size 16, temporal patch size 2, out hidden 4096. Inactive for base text-only prompt encoding. |
| `Qwen3VLProcessor` | tokenizer class `Qwen2Tokenizer`, pad token `<|endoftext|>`, EOS `<|im_end|>`, model max length 262144. Pipeline default `max_sequence_length=1024`. |
| Pipeline prompt hidden state | `outputs.hidden_states[return_index]`, default `return_index=-8`; not the final hidden state by default. |

VAE and scheduler:

| Component | Key fields |
| --- | --- |
| `AutoencoderKLQwenImage` | `base_dim=96`, `z_dim=16`, `dim_mult=[1,2,4,4]`, 2 residual blocks, `temperal_downsample=[false,true,true]`, `latents_mean/std` vectors of length 16, 8x spatial compression, causal Conv3d/RMSNorm/SiLU/mid attention. |
| Scheduler | `FlowMatchEulerDiscreteScheduler`, `num_train_timesteps=1000`, `shift=1.0`, `use_dynamic_shifting=false`, `base_shift=0.5`, `max_shift=1.15`, `base_image_seq_len=256`, `max_image_seq_len=4096`, `time_shift_type=exponential`, no Karras/exponential/beta conversion, no stochastic sampling. |
| Weight metadata | Transformer safetensors index total size 33,845,358,592 bytes; text encoder index total size 17,534,247,392 bytes. Dtype is config/example-derived bfloat16, not a scheduler fact. |

Recommended first Dinoml scheduler slice: FlowMatch Euler with custom sigma
list and static shift. The pipeline always computes and passes `mu`, but the
official scheduler config has `use_dynamic_shifting=false`, so `mu` is ignored
for first parity.

## 3a. Family variation traps

- `in_channels=64` is packed 2x2 latent token width, not VAE latent channels.
- Source latent layout is 5D image-as-one-frame `[B,1,C,H,W]` before packing and `[B,C,1,H,W]` at VAE decode, while the transformer is token-major `[B,S,C]`.
- Official transformer config differs from source defaults: 32 blocks, 64 experts, 4 KV heads, and strategy `leave_first_three_blocks_dense`.
- Official `dense_moe_strategy` makes layers 3 through 31 MoE. Capacity factor zero on the first three layers is inactive because they are dense.
- MoE is expert-choice routing, not token-choice top-k routing. Each expert selects top-C tokens; a token may be selected by multiple experts or none except for the shared expert.
- True CFG is two separate transformer calls, not batch concatenation.
- CFG renorm is always applied when CFG is active: `comb_pred * norm(cond) / norm(comb_pred)` over token channel dim, with no clamp or epsilon in source.
- The model output is negated before the scheduler step.
- Prompt masking can make attention masked even in base text-to-image; if all mask values are true, the pipeline drops the mask.
- Text K/V caching is source-supported through hooks and changes per-block attention kwargs. It should be a separate optimization path.
- Qwen3-VL processor has image/video preprocessing configs, but base `encode_prompt` supplies text only.
- Autoencoder decode is a 3D causal VAE path with `T=1`; do not treat it as standard 2D AutoencoderKL.

## 4. Runtime tensor contract

For 1024x1024, one image per prompt:

| Boundary | Tensor | Shape/layout | Notes |
| --- | --- | --- | --- |
| Formatted prompt | text string | chat-template text | System prompt plus user prompt, `add_generation_prompt=True`. |
| Tokenizer output | `input_ids`, `attention_mask` | `[B,L]`, `L<=1024`, padded to multiple of 8 | CPU/data path. |
| Prompt embeddings | `prompt_embeds` | `[B,L,4096]` | Qwen3-VL hidden state at `return_index=-8` unless overridden. |
| Prompt mask | `prompt_embeds_mask` | `[B,L]` bool/long or `None` | Repeated for images per prompt; dropped if all true. |
| Negative prompt embeddings | same | `[B,Lneg,4096]` | Only when `guidance_scale > 1`; empty string default if no negative prompt. |
| Initial latent map | random or supplied | `[B,1,16,128,128]` source NCTHW | `T=1`; dtype follows prompt embeds. |
| Packed latents | transformer input | `[B,4096,64]` | `view(B,16,64,2,64,2) -> permute(0,2,4,1,3,5) -> reshape`. |
| `img_shapes` | metadata | list of `(1,64,64)` | Frame, packed latent height, packed latent width for RoPE. |
| Timestep | model input | `[B]` | Pipeline passes `t / scheduler.config.num_train_timesteps`; model's `Timesteps` has `scale=1000`. |
| Transformer output | `noise_pred` before negate | `[B,4096,64]` | Same packed token shape. |
| CFG output | `noise_pred` | `[B,4096,64]` | Two calls: cond and negative. |
| Scheduler latents | packed latents | `[B,4096,64]` | Flow Euler update. |
| Unpacked VAE latents | decode input | `[B,16,1,128,128]` | Reverse pack, channel-first with one temporal frame. |
| VAE standardized input | decode input | `[B,16,1,128,128]` | `latents / (1/std) + mean`, equivalently `latents * std + mean`. |
| Decoded image | VAE output then frame slice | `[B,3,1024,1024]` | VAE output `[B,3,1,H,W]`, source takes `[:, :, 0]`. |
| Pipeline output | PIL/NumPy/PT or latent | HWC/CHW by processor, or packed latent if `output_type="latent"` | `output_type="latent"` returns packed scheduler latents before unpack/decode. |

CPU/data-pipeline work: prompt formatting, tokenization, Qwen3-VL execution if
not cached, random seeding, and final image conversion. GPU/runtime work:
transformer denoiser, MoE routing and grouped expert compute, attention,
CFG/renorm/negation, FlowMatch step, latent pack/unpack, and VAE decode.

Cacheable tensors: prompt embeddings and masks, negative prompt embeddings,
text K/V projections per block when using `TextKVCacheConfig`, RoPE frequencies
for fixed `(F,H,W,L)`, scheduler sigma/timestep tables, and VAE latents for
future img2img variants.

Autoencoder encode is not used by the base text-to-image pipeline, but
`AutoencoderKLQwenImage.encode` exists and returns a diagonal Gaussian over
5D latents. Any future img2img/edit candidate should standardize encode parity
with the same `latents_mean/std` convention.

## 5. Operator coverage checklist

Tensor/layout ops:

- 5D latent allocation `[B,1,16,H/8,W/8]`, random normal, dtype/device plumbing.
- 2x2 pack from NCTHW single-frame latent map to `[B,S,64]`.
- 2x2 unpack from `[B,S,64]` to `[B,16,1,H/8,W/8]`.
- Prompt mask repeat/drop, bool conversion, image mask creation, concat into joint key mask `[B,S_img+S_txt]`.
- CFG two-call arithmetic, norm over last token channel, division, output negation.
- MoE gather by global token indices, scatter_add token score sums, scatter_add routed outputs.
- VAE 3D causal conv layout reshapes between `[B,C,T,H,W]` and framewise 2D conv/attention internals.

GEMM/linear ops:

- `img_in`: Linear(64 -> 2048).
- Timestep embedding: sinusoidal 2048 channels, MLP 2048 -> 8192 -> 2048, RMSNorm.
- Per-block modulation: `SiLU + Linear(2048 -> 8192)` producing scale/gate pairs.
- Per-block text projection: Linear(4096 -> 2048).
- Attention image Q/K/V, text added K/V, output projection, all bias-free for the attention projections.
- Dense block SwiGLU FFN with inner dim `int(2048*4*2/3)//128*128 = 5376`.
- MoE router Linear(4096 -> 64), expert grouped GEMMs with weights `(64,2048,2688)` and `(64,1344,2048)`.
- Shared expert SwiGLU on all tokens.
- Final AdaLayerNormContinuous and Linear(2048 -> 64).

Attention primitives:

- Cross-attention-style image queries over concatenated image plus text keys/values in 32 blocks.
- GQA: 16 query heads, 4 KV heads, manual KV repeat to 16 heads.
- Q/K RMSNorm for image Q/K and text K.
- Complex RoPE on image and text keys, using scaled centered image axes and shifted text positions.
- Native/eager `dispatch_attention_fn` parity path; flash-style provider guarded.

Normalization and adaptive conditioning:

- RMSNorm, LayerNorm without affine, AdaLayerNormContinuous, SiLU.
- Gate clamp to `[-2,2]`, `tanh(gate)` residual scale.
- QwenImage VAE RMS norm over channel-first 2D/3D tensors.

Scheduler and guidance arithmetic:

- FlowMatch Euler custom sigmas: pipeline defaults to `linspace(1, 1/steps, steps)`.
- Static shift with `shift=1.0`, terminal zero sigma.
- Per-step `prev = sample + (sigma_next - sigma) * model_output`.
- True CFG and channel-norm renorm.

VAE/postprocessing:

- QwenImageCausalConv3d, framewise Conv2d, nearest-exact upsample, ZeroPad2d+stride Conv2d downsample.
- QwenImageResidualBlock, QwenImageAttentionBlock with single-head SDPA over spatial positions.
- Quant/post-quant 1x1 causal convs, tile/slice policies as separate runtime candidates.

## 6. Denoiser/model breakdown

`NucleusMoEImageTransformer2DModel.forward`:

```text
hidden_states [B,S_img,64] -> img_in -> [B,S_img,2048]
encoder_hidden_states [B,S_txt,4096] -> RMSNorm
prompt mask -> optional joint attention mask [B,S_img+S_txt]
timestep -> sinusoidal projection + MLP + RMSNorm -> temb [B,2048]
img_shapes + text length -> NucleusMoEEmbedRope image/text freqs
32 x NucleusMoEImageTransformerBlock
AdaLayerNormContinuous(hidden, temb) -> Linear 2048->64
```

One transformer block:

```text
temb -> SiLU -> Linear -> scale1, gate1, scale2, gate2
pre_attn LayerNorm(hidden) -> multiply by (1 + scale1)
encoder_proj(text) unless cached text K/V are supplied
image Q/K/V + text K/V -> QK RMSNorm -> RoPE -> GQA repeat -> attention
hidden += tanh(clamp(gate1)) * attention_output
pre_mlp LayerNorm(hidden) -> multiply by (1 + scale2)
if dense: SwiGLU FeedForward
if MoE: expert-choice MoE + shared SwiGLU expert
hidden += tanh(clamp(gate2)) * mlp_output
fp16 clip guard
```

MoE routing in official config:

```text
router_input = cat(temb broadcast to tokens, unmodulated normalized hidden)
scores = softmax(gate(router_input), dim=-1)
affinity = scores.transpose(1, 2)        # [B, E, S]
capacity = ceil(capacity_factor * S / E)
top_indices = topk(affinity, k=capacity, dim=-1)
normalize selected token scores by token_score_sums
routed_input = flattened_hidden[global_token_indices]
experts(routed_input, tokens_per_expert=B*capacity)
out = shared_expert(hidden) + scatter_add(weighted routed expert outputs)
```

At 1024x1024 (`S=4096`, `E=64`), capacity 4.0 gives 256 selected tokens per
expert for layers 3-4, and capacity 2.0 gives 128 selected tokens per expert
for layers 5-31. This is a routing workspace and grouped-GEMM problem, not a
plain MLP replacement.

## 7. Attention requirements

Primary implementation: `NucleusMoEAttnProcessor2_0` in
`transformer_nucleusmoe_image.py`, calling `dispatch_attention_fn`.

Required behavior:

- Query sequence is image tokens only: `[B,S_img,16,128]`.
- Key/value sequence is image tokens followed by text tokens: `[B,S_img+S_txt,4,128]` before GQA repeat.
- KV heads are repeated to match query heads before dispatch; Diffusers does not pass `enable_gqa=True`.
- Image Q/K and text K receive RMSNorm. Text V is unnormalized.
- RoPE is applied to image Q/K and text K. The helper uses complex multiplication (`use_real=False`).
- If prompt masks exist, source passes a bool joint mask over key positions. It is not a causal mask.
- Text K/V cache can skip per-block `encoder_proj`, `add_k_proj`, `add_v_proj`, text K norm, and text RoPE after first computation.

Flash/provider feasibility:

- Unmasked base attention is plausible for a flash-style provider: noncausal, head dim 128, q length 4096, kv length up to about 5120, bf16-friendly.
- Masked prompts require either native SDPA/xFormers/flex or a varlen flash/sage path that exactly reproduces Diffusers' key mask normalization. Normal flash-attn 2/3/hub and Sage non-varlen reject non-None masks.
- GQA is source-expanded before the backend; a Dinoml provider may either preserve repeat-interleave or implement GQA directly with parity guards.
- QK RMSNorm and RoPE must remain explicit provider inputs unless fused by a Nucleus-specific attention kernel.
- Attention projection fusion is possible but text K/V caching changes the block boundary, so cache and fusion need separate admission.

## 8. Scheduler and denoising-loop contract

Pipeline setup:

```text
sigmas = linspace(1.0, 1.0 / num_inference_steps, num_inference_steps)
image_seq_len = packed_latents.shape[1]
mu = calculate_shift(image_seq_len, base_image_seq_len, max_image_seq_len,
                     base_shift, max_shift)
scheduler.set_timesteps(num_inference_steps, sigmas=sigmas, mu=mu)
scheduler.set_begin_index(0)
```

Official config has `use_dynamic_shifting=false`, so `mu` is accepted but not
used. With `shift=1.0`, the supplied sigma list is unchanged before terminal
zero is appended.

Per-step loop:

```text
timestep = t.expand(batch).to(latents.dtype)
cond = transformer(latents, timestep / 1000, prompt_embeds, mask, img_shapes)
if guidance_scale > 1:
  uncond = transformer(latents, timestep / 1000, negative_embeds, negative_mask, img_shapes)
  comb = uncond + guidance_scale * (cond - uncond)
  cond_norm = norm(cond, dim=-1, keepdim=True)
  noise_norm = norm(comb, dim=-1, keepdim=True)
  model_output = comb * (cond_norm / noise_norm)
else:
  model_output = cond
model_output = -model_output
latents = scheduler.step(model_output, t, latents)
```

Scheduler step for the active non-stochastic scalar-timestep path:

```text
prev_sample = sample + (sigma_next - sigma) * model_output
```

Host/runtime split: keep schedule validation/table generation, loop iteration,
and optional two-call CFG orchestration host-visible first. Compile the
transformer step, CFG/renorm/negation, and pointwise FlowMatch update after
tensor parity. Advanced FlowMatch options in the scheduler source
(`use_dynamic_shifting`, Karras/exponential/beta conversion, stochastic
sampling, per-token timesteps) are separate scheduler candidates unless a future
Nucleus config enables them.

## 9. Position, timestep, and custom math

Custom math to reproduce:

- Chat prompt formatting with the fixed Nucleus system prompt and Qwen chat template.
- Prompt embedding selection by `return_index`, default `-8`; admission should not silently use final hidden states.
- Timestep embedding: `Timesteps(num_channels=2048, flip_sin_to_cos=True, downscale_freq_shift=0, scale=1000)` plus `TimestepEmbedding(2048 -> 8192 -> 2048)` and RMSNorm.
- Nucleus RoPE precomputes positive and negative frequency tables length 4096. With `scale_rope=True`, image height/width positions are centered by concatenating negative and positive halves; text positions start at `max_vid_index`.
- `NucleusMoEEmbedRope` warns that batch inference with variable-sized images is not currently supported and uses the first shape for RoPE.
- MoE expert-choice routing with per-token score normalization and `route_scale=2.5`.
- CFG renorm has no epsilon/clamp in source; parity tests should include nonzero-norm guards.
- VAE decode standardization uses `latents * latents_std + latents_mean` through the source expression `latents / (1/std) + mean`.

Precomputable: prompt embeddings, masks, per-block text K/V cache, RoPE
frequencies for fixed image/text shape, scheduler tables. Dynamic: prompt
length/mask, timestep, CFG on/off, image sequence length, MoE top-k token sets,
and VAE tile/slice decisions.

## 10. Preprocessing and input packing

Prompt preprocessing:

- `_format_prompt` builds system/user messages and calls `processor.apply_chat_template(..., tokenize=False, add_generation_prompt=True)`.
- `processor(text=..., padding="longest", pad_to_multiple_of=8, max_length=max_sequence_length, truncation=True, return_attention_mask=True)` creates Qwen inputs.
- The text encoder is called with `use_cache=False`, `return_dict=True`, and `output_hidden_states=True`.
- `prompt_embeds` and optional masks repeat with `repeat_interleave` for `num_images_per_prompt`.
- If all mask elements are true, the mask is replaced by `None`, which changes attention backend feasibility.

Latent packing:

- Pipeline default `height=width=128 * vae_scale_factor = 1024`.
- `vae_scale_factor = 2 ** len(vae.temperal_downsample) = 8`; input dimensions should be divisible by `vae_scale_factor * patch_size = 16`.
- `prepare_latents` samples `[B,1,16,H/8,W/8]` then packs 2x2 spatial latent tiles to `[B,(H/16)*(W/16),64]`.
- `img_shapes` records packed grid size `(1,H/16,W/16)`.

Postprocessing:

- `output_type="latent"` returns packed latents.
- Otherwise unpack to `[B,16,1,H/8,W/8]`, cast to VAE dtype, standardize with VAE mean/std, decode, select frame 0, and postprocess.

## 11. Graph rewrite / lowering opportunities

### Rewrite: Nucleus latent pack/unpack

Source pattern: 5D single-frame latent map and 2x2 NCHW spatial tile packing.

Replacement: explicit `pack2x2_ncthw_t1` / `unpack2x2_to_ncthw_t1` kernels or
canonical reshape/permute/reshape graph.

Preconditions: `T=1`, channel count 16, `patch_size=2`, H/W divisible by 16 in
pixel space, source flatten order `h2,w2,c,dh,dw`, no NHWC layout translation
across the pack boundary.

Failure cases: supplied latents already packed with unknown provenance, future
non-16-channel VAE, or layout pass rewriting axes without changing flatten
order.

Parity test sketch: random `[B,1,16,128,128] -> [B,4096,64] -> [B,16,1,128,128]`
against Diffusers helpers, plus non-square divisible shapes.

### Rewrite: Cross-attention provider island

Source pattern: image QKV plus text K/V, RMSNorm, RoPE, KV repeat for GQA,
optional key mask, noncausal attention, output projection.

Replacement: native attention fallback plus guarded flash/varlen provider.

Preconditions: query tokens image-only, key/value tokens image+text, head dim
128, exact text mask behavior, no IP/control branches, RoPE and QK norm applied
before provider, dtype supported.

Failure cases: non-None mask with a provider that rejects masks, variable
image shapes in one batch, text K/V cache changing projection ownership, or
provider treating GQA differently from repeat-interleave.

Parity test sketch: one dense block with and without prompt mask; compare native
dispatch, varlen/provider path, and text K/V cache path.

### Rewrite: Expert-choice MoE

Source pattern: router over `cat(temb, unmodulated hidden)`, top-C tokens per
expert, score renormalization by selected-token sum, grouped expert SwiGLU,
shared expert, scatter-add.

Replacement: first a reference routing region; later a device-resident routing
plus grouped GEMM provider.

Preconditions: inference mode, fixed `num_experts=64`, capacity factors known
per layer, `use_sigmoid=false`, `route_scale=2.5`, `use_grouped_mm=true` or a
loop fallback with identical ordering.

Failure cases: top-k tie instability, empty/duplicate token selections,
dynamic sequence lengths without workspace sizing, grouped_mm unavailable, or
hidden host sync in fallback loop.

Parity test sketch: force router logits to cover duplicate selections, zero
selection for some tokens, and all experts nonempty; compare routed output and
score normalization.

### Rewrite: FlowMatch Euler Nucleus slice

Source pattern: custom linear sigma list, static shift, negated model output,
pointwise Euler update.

Replacement: host-visible sigma table plus fused pointwise update and CFG/renorm
kernel.

Preconditions: official scheduler config, no stochastic sampling, no per-token
timesteps, scalar timestep path.

Failure cases: future config enabling dynamic shifting or stochastic sampling,
custom timesteps/sigmas not admitted, CFG norm division by zero.

## 12. Kernel fusion candidates

Highest priority:

- Expert-choice MoE routing and grouped SwiGLU expert GEMMs. This is the
  defining Nucleus hot path and needs explicit routing workspace metadata.
- Q/K/V projections + RMSNorm + RoPE + attention provider for image-query/text-KV attention.
- Text K/V cache as a runtime plan: precompute per-block text K/V and reuse
  across denoising steps and CFG branches.
- Ada modulation, clamp/tanh gates, and residual epilogues around attention and MoE/MLP.
- CFG arithmetic + norm renorm + model-output negation.

Medium priority:

- Latent pack/unpack kernels and image/text RoPE frequency generation.
- FlowMatch Euler pointwise scheduler update.
- Dense-block SwiGLU FFN fusion for the first three blocks.
- QwenImage VAE decode Conv3d/RMSNorm/SiLU/resample/mid-attention island.

Lower priority:

- Qwen3-VL text encoder compilation; prompt embeddings are external first.
- VAE tiling/slicing and causal feature cache policy.
- PEFT/LoRA hotswap/fuse/unfuse.
- Single-file/original checkpoint conversion.
- Quantized/offload weight-loading policies.

NHWC/layout candidates:

- Transformer core is token-major `[B,S,C]`; NHWC does not apply inside the denoiser.
- Pack/unpack and VAE boundaries are axis-sensitive and should be protected by a no-layout-translation guard until a layout-aware rewrite exists.
- QwenImage VAE is source channel-first `[B,C,T,H,W]` with framewise Conv2d and causal Conv3d. NDHWC/NHWC optimization may be useful only inside controlled codec islands with Conv2d/Conv3d weight transforms and RMSNorm channel-axis rewrites.
- Scheduler, CFG norm over token channels, MoE routing, and RoPE are not image-layout translation candidates.

## 13. Runtime staging plan

Stage 1: Parse the official Nucleus model index and component configs. Accept
external `prompt_embeds [B,L,4096]` and masks for first transformer parity.

Stage 2: Implement latent pack/unpack and `img_shapes`/RoPE parity for 1024 and
non-square divisible resolutions.

Stage 3: Implement one dense `NucleusMoEImageTransformerBlock` with native
attention fallback, QK RMSNorm, RoPE, Ada modulation, gates, and dense SwiGLU.

Stage 4: Implement one MoE block with expert-choice routing and grouped expert
GEMM fallback/reference parity.

Stage 5: Compile the full 32-block transformer for official config with
external text embeddings and masks.

Stage 6: Add two-call true CFG plus source CFG renorm and output negation.

Stage 7: Add FlowMatch Euler static-shift scheduler one-step and short loop
with scheduler in host control.

Stage 8: Add QwenImage VAE decode boundary and later a separate VAE decode
artifact.

Stage 9: Add text K/V cache, attention provider optimization, then adapter,
single-file, quantized/offload, and full Qwen3-VL text encoder slices.

First Dinoml admission recommendation:
`nucleusmoe_image_denoiser_step_external_text`, with inputs
`latents [B,S,64]`, `timestep [B]`, `prompt_embeds [B,L,4096]`,
optional `prompt_mask [B,L]`, and shape metadata `(F=1,H,W)`. Output is packed
latent derivative `[B,S,64]` before pipeline negation.

## 14. Parity and validation plan

- Config parse tests for official model index, transformer, scheduler, VAE, processor, and text encoder configs.
- Prompt formatting/tokenization boundary tests for default system prompt and `max_sequence_length=1024`.
- Prompt embedding selection parity for `return_index=-8` and an explicit override.
- Latent pack/unpack parity for square and non-square divisible image sizes.
- RoPE parity for `axes_dims_rope=[16,56,56]`, scale-rope centered image positions, and text offset.
- Attention parity with no mask, full mask dropped to `None`, and partial prompt mask.
- Text K/V cache parity against uncached block output.
- Dense block parity for layers 0-2.
- MoE block parity for capacity factors 4.0 and 2.0, grouped_mm and reference loop if available.
- Full transformer forward parity with random external embeddings and fixed latents/timestep.
- CFG two-call arithmetic, CFG renorm, and model-output negation parity.
- FlowMatch sigma table and one-step update parity for official scheduler config.
- QwenImage VAE decode parity for `[1,16,1,128,128]`; encode parity reserved for future variants.
- Short deterministic denoising loop smoke with VAE decode.
- Suggested tolerances: scheduler fp32 `rtol=1e-5, atol=1e-6`; transformer fp32 `rtol=1e-4, atol=1e-5`; bf16/fp16 initially `rtol=2e-2, atol=2e-2`, with MoE routing indices compared exactly.

## 15. Performance probes

- One denoiser step by image sequence length: 1024, 4096, and non-square grids.
- MoE routing breakdown: router/topk/scatter, grouped expert GEMMs, shared expert, and scatter-add.
- Capacity-factor sweep for layers 3-4 versus layers 5-31.
- Attention backend comparison: native SDPA, xFormers/flex, flash varlen, and Dinoml provider under mask/no-mask conditions.
- Text K/V cache speedup and memory usage for one prompt and CFG positive/negative prompts.
- CFG overhead: one transformer call versus two calls plus renorm.
- Flow scheduler/CFG pointwise overhead versus denoiser time.
- Latent pack/unpack memory traffic.
- QwenImage VAE decode throughput and memory for 1024.
- Qwen3-VL text encoder throughput and prompt embedding cache size.
- VRAM/workspace by dtype, prompt length, sequence length, and MoE routing workspace.

## 16. Scope boundary and separate candidates

Separate candidate reports, not ignored:

- `nucleusmoe_image_moe_provider`: expert-choice routing, grouped expert GEMM, score normalization, and workspace planning.
- `nucleusmoe_image_text_kv_cache`: source-supported exact text K/V cache hooks and denoising-loop cache state.
- `nucleusmoe_image_lora_adapters`: PEFT/LoRA adapter state, hotswap/fuse/unfuse, and LoRA scale handling.
- `nucleusmoe_image_single_file_original`: original checkpoint conversion through `FromOriginalModelMixin`.
- `nucleusmoe_image_quantized_offload`: BitsAndBytes/TorchAO/offload weight policy and encoded/runtime loading.
- `nucleusmoe_qwenimage_vae`: QwenImage 3D causal VAE decode/encode, tiling/slicing, mean/std latent contract.
- `nucleusmoe_text_encoder_qwen3vl`: Qwen3-VL text encoder and processor/tokenizer cache admission.
- `scheduler_flowmatch_nucleus`: official static FlowMatch slice plus future dynamic/stochastic/per-token FlowMatch options.

Genuinely ignored/out of scope for this audit:

- Multi-GPU/context parallel paths.
- Callback mutation and interactive interrupt.
- XLA, NPU, MPS, Flax, and ONNX branches.
- Safety checker and NSFW filtering.
- Training, losses, dropout, and gradient checkpointing.
- Textual inversion, IP-Adapter, ControlNet, T2I-Adapter, GLIGEN, img2img, inpaint, depth2img, and upscaling: no active NucleusMoE image implementation was found in the inspected non-deprecated Diffusers folder.

Blockers / validation notes:

- No gated official config blocker. Official configs were available through
  Hugging Face Hub but not present under H:/configs.
- Only one official full Diffusers checkpoint/config set was found, so
  multi-checkpoint variation is limited to source defaults and tests.
- CFG renorm lacks an epsilon in source; validation should include nonzero norm
  assumptions or preserve source behavior exactly.
- `NucleusMoEEmbedRope` warns about variable image sizes within one batch; a
  Dinoml first slice should require a uniform packed image grid per batch.

## 17. Final implementation checklist

- [ ] Parse official Nucleus model index and component configs.
- [ ] Accept external Qwen3-VL prompt embeddings and masks.
- [ ] Implement Nucleus 2x2 latent pack/unpack for single-frame QwenImage latents.
- [ ] Implement Nucleus scaled 3-axis RoPE parity.
- [ ] Implement timestep embedding and Ada modulation path.
- [ ] Implement Nucleus cross-attention native fallback with GQA repeat and mask support.
- [ ] Add text K/V cache parity as a separate optimization slice.
- [ ] Implement dense first-three-block SwiGLU parity.
- [ ] Implement expert-choice MoE routing, score normalization, grouped expert SwiGLU, shared expert, and scatter-add.
- [ ] Add full transformer denoiser-step parity.
- [ ] Implement true CFG two-call arithmetic, CFG renorm, and model-output negation.
- [ ] Implement official FlowMatch Euler static scheduler slice.
- [ ] Add QwenImage VAE mean/std decode boundary.
- [ ] Benchmark MoE, attention, CFG/scheduler, pack/unpack, VAE decode, and text encoder cache.
- [ ] Open separate follow-ups for MoE provider, text K/V cache, LoRA/adapters, single-file conversion, quantized/offload, QwenImage VAE, and Qwen3-VL text encoder.
