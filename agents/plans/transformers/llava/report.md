# LLaVA Transformers Audit

## 1. Source basis

Transformers commit/version: `b75feb2af64c3e29cbbc1bd859958c5432cc7ed4` in `transformers`.

Model id: family `llava`; representative checkpoints inspected from Hugging Face:

- `llava-hf/llava-1.5-7b-hf`
- `llava-hf/llava-1.5-13b-hf`
- `llava-hf/bakLlava-v1-hf`
- `IlyasMoutawwakil/tiny-random-LlavaForConditionalGeneration`
- `llava-hf/llava-interleave-qwen-0.5b-hf`
- `xtuner/llava-phi-3-mini-hf`

Config source: each checkpoint `config.json`, plus available `preprocessor_config.json`, `processor_config.json`, and `special_tokens_map.json`. `processor_config.json` was absent for the tiny-random and Phi-3 Mini mirrors; their processor settings must be inferred from model/preprocessor config or user-supplied processor state.

Source files inspected:

- `src/transformers/models/llava/configuration_llava.py`
- `src/transformers/models/llava/modeling_llava.py`
- `src/transformers/models/llava/processing_llava.py`
- `src/transformers/models/llava/image_processing_llava.py`
- `src/transformers/models/llava/image_processing_pil_llava.py`
- Delegated source for representative towers/decoders: `clip/modeling_clip.py`, `siglip/modeling_siglip.py`, `llama/modeling_llama.py`, `mistral/modeling_mistral.py`, `qwen2/modeling_qwen2.py`, and their configuration files.

Any missing files or assumptions: no remote-code-only files were needed for these representative configs. The `llava` modeling file delegates the text model and vision model through `AutoModel.from_config`, so decoder operator details are text-family-dependent. This report targets inference-time multimodal image+text generation.

## 2. High-level architecture

LLaVA is a vision encoder plus multimodal projector plus causal text decoder. The family wrapper does not implement custom decoder blocks; it builds `vision_tower = AutoModel.from_config(config.vision_config)`, `language_model = AutoModel.from_config(config.text_config)`, and a two-layer `LlavaMultiModalProjector`.

Dataflow:

```text
image/text processor -> pixel_values + expanded input_ids
pixel_values -> vision tower hidden states -> select layer(s) -> optional CLS crop -> projector
input_ids -> token embeddings
image placeholder mask -> masked_scatter(projected image features into token embeddings)
stitched inputs_embeds -> text decoder prefill -> logits / KV cache
decode steps -> text decoder only using cache -> logits / sampling
```

Stage decomposition:

- CPU/data-pipeline: RGB conversion, resize, optional pad, center crop, rescale, normalize, tokenizer, chat template, and expansion of `<image>` into many placeholder tokens.
- Cacheable vision prefix: `pixel_values` through vision tower and projector can be computed independently for a fixed image and selected feature layer policy.
- Prefix construction: projected image features are stitched into text embeddings at placeholder-token positions. This is a small but parity-critical scatter stage.
- Prefill: the text decoder sees one multimodal embedding sequence and produces logits and per-layer KV cache.
- Decode: `prepare_inputs_for_generation` forwards `pixel_values` only on the first generation iteration or when cache is disabled; subsequent cached decode steps are text-only.

The vision/projector output, placeholder stitching, text prefill, and decode cache can be validated independently.

## 3. Important config dimensions

Source defaults from `LlavaConfig`: `image_token_index=32000`, `image_seq_length=576`, `projector_hidden_act="gelu"`, `vision_feature_select_strategy="default"`, `vision_feature_layer=-2`, `multimodal_projector_bias=True`, and `tie_word_embeddings=False`. If omitted, `vision_config` defaults to CLIP vision hidden 1024, 24 layers, 16 heads, image 336, patch 14. If `text_config` is omitted, it defaults to Llama config.

| Checkpoint | Text model | Text dims | KV heads | Context / sliding | Vision tower | Image tokens | Dtype source |
|---|---:|---:|---:|---:|---:|---:|---|
| tiny-random | Llama | hidden 16, 2 layers, 4 heads, MLP 64 | 4 | source default if omitted | CLIP hidden 32, 2 layers, patch 2, image 30 | `(30/2)^2 + 1 - 1 = 225` if processor defaults match config | `config.json` float32 |
| llava-1.5-7b | Llama/Vicuna | config omits hidden/layers/heads; effective Llama defaults are 4096, 32 layers, 32 heads, MLP 11008 | default 32 | max pos 4096 | CLIP hidden 1024, 24 layers, patch 14, image 336 | `24*24 + 1 - 1 = 576` | `config.json` float16 |
| llava-1.5-13b | Llama-2 | hidden 5120, 40 layers, 40 heads, MLP 13824 | 40 | max pos 4096 | CLIP hidden 1024, 24 layers, patch 14, image 336 | 576 | `config.json` float16 |
| bakLlava-v1 | Mistral | config omits hidden/layers/heads; effective Mistral defaults are 4096, 32 layers, 32 heads, MLP 14336 | 8 | max pos 32768, sliding 4096 | CLIP hidden 1024, 24 layers, patch 14, image 336 | 576 | `config.json` bfloat16 |
| llava-interleave-qwen-0.5b | Qwen2 | hidden 1024, 24 layers, 16 heads, MLP 2816 | 16 | max pos 32768, `use_sliding_window=false` so no sliding | SigLIP hidden 1152, 26 layers, patch 14, image 384 | floor `384/14` gives 729 source-computed placeholders; see trap below | `config.json` bfloat16 |
| xtuner/llava-phi-3-mini-hf | Llama class mirror | hidden 3072, MLP 8192, layers/heads omitted and would fall back to Llama defaults if loaded strictly | default hazards | max pos 4096, sliding 2048 field not used by Llama source here | CLIP hidden 1024, 24 layers, patch 14, image 336 | 576 if processor uses patch 14/default strategy | `config.json` float16 |

Head dimensions are `hidden_size // num_attention_heads` unless an explicit `head_dim` exists in the delegated decoder config. CLIP/SigLIP vision head dims are `vision_hidden / vision_heads`, e.g. CLIP 1024/16 = 64 and SigLIP 1152/16 = 72.

## 3a. Family variation traps

- Text model is not fixed. LLaVA 1.5 uses Llama/Vicuna, BakLLaVA uses Mistral with GQA and sliding-window masks, and interleave Qwen uses Qwen2.
- Several configs omit text dimensions and rely on source defaults. Treat omitted `hidden_size`, `num_hidden_layers`, `num_attention_heads`, and `num_key_value_heads` as effective defaults only after instantiating the delegated config class.
- `num_key_value_heads < num_attention_heads` for Mistral/BakLLaVA, so cache shapes are KV-head shapes, not expanded attention-head shapes.
- Qwen2 projections use bias for Q/K/V and no bias for O; Llama/Mistral defaults use no attention bias.
- `vision_feature_select_strategy="default"` crops token 0 from every selected vision hidden state; `"full"` keeps it.
- `vision_feature_layer` may be an int or list. A list concatenates selected hidden states on the feature dimension before the projector, changing `linear_1` input from `vision_hidden` to `vision_hidden * len(layers)`.
- CLIP vision has an explicit class embedding; SigLIP vision has no class embedding. Processor `num_additional_image_tokens` must match the tower convention.
- For SigLIP 384 / patch 14, source placeholder math uses integer floor after preprocessing size, giving `27*27=729`. The source Conv2d with kernel/stride 14 over 384 also produces 27x27 patches because the final pixels are not covered. Do not silently round to 28x28.
- Image preprocessing and vision towers are NCHW in source. NHWC/channel-last should be guarded to local preprocessing/patch-projection regions only, with explicit axis rewrites for crop/normalize/Conv2d/flatten/transpose consumers.
- Placeholder stitching depends on token count equality. Any processor/model mismatch in patch size, extra image tokens, or default/full policy fails at runtime.

## 4. Operator coverage checklist

Tensor/layout ops:

- Image preprocessing: RGB conversion, resize bicubic, optional square pad, center crop, rescale, per-channel normalize, NCHW batch tensor construction.
- Vision embedding: Conv2d patch embedding with `kernel_size=stride=patch_size`, flatten spatial axes, transpose `[B, C, Gh, Gw] -> [B, Gh*Gw, C]`, concat class token for CLIP, add learned positions.
- Feature selection: hidden-state tuple indexing, token slice `[:, 1:]` for default, feature concat on last dim for multi-layer selection, list/cat/split for image batches.
- Text stitch: token embedding lookup, equality mask, `unsqueeze`, `expand_as`, count/numel check, `masked_scatter`.
- Logits: optional last-token slice through `logits_to_keep`, LM head linear.

Neural network primitives:

- CLIP/SigLIP vision: LayerNorm, MHA self-attention, Linear Q/K/V/O, GELU or configured activation MLP, residual adds.
- Projector: `Linear(vision_hidden * num_feature_layers -> text_hidden)`, GELU, `Linear(text_hidden -> text_hidden)`, bias controlled by `multimodal_projector_bias`.
- Text decoders: RMSNorm, Linear Q/K/V/O, gated SiLU MLP for Llama/Mistral/Qwen2, residual adds, final norm, untied/tied LM head.

Attention primitives:

- Vision: noncausal full self-attention over patch sequence plus optional CLS.
- Text: causal self-attention, RoPE, optional GQA/MQA, optional sliding-window masks depending on delegated decoder.
- Optimized backend compatibility is inherited: LLaVA advertises FlashAttention, SDPA, flex attention, and attention backend support, but exact dispatch is delegated.

Generation/cache ops:

- DynamicCache allocation when `use_cache` and no cache is provided.
- Per-layer cache update after RoPE on keys.
- Generation input preparation that drops image inputs after the first cached iteration.
- Position id generation from past seen token count when not supplied.

Preprocessing-coupled ops:

- Placeholder expansion in processor: `sample.replace("<image>", "<image>" * num_image_tokens)`.
- Optional `mm_token_type_ids` from the processor if requested; the LLaVA model forward itself does not consume it.

## 5. Layer/block breakdown

Vision CLIP block, repeated `vision_layers`:

```text
pixel_values [B,3,H,W]
patch = Conv2d(3 -> vision_hidden, kernel=stride=patch)
patch = flatten(2).transpose(1,2)
emb = concat(class_embedding, patch) + learned_pos
x = pre_layernorm(x)
for block:
  y = LayerNorm(x)
  q,k,v = Linear(vision_hidden -> vision_hidden)
  y = full self-attention(q,k,v)
  x = x + Linear(y)
  y = LayerNorm(x)
  y = Linear(vision_hidden -> vision_intermediate) -> activation -> Linear(... -> vision_hidden)
  x = x + y
```

SigLIP is similar but omits the CLIP class embedding/pre-layernorm and uses patch-only position embeddings; representative interleave config disables the SigLIP pooling head.

Projector:

```text
selected_image_feature [num_images, image_tokens, vision_hidden * num_feature_layers]
y = Linear(... -> text_hidden, bias=config.multimodal_projector_bias)
y = GELU(y)
y = Linear(text_hidden -> text_hidden, same bias policy)
```

Text Llama/Mistral/Qwen2 decoder block, repeated `text_layers`:

```text
x = RMSNorm(x)
q = Linear(hidden -> attention_heads * head_dim)
k,v = Linear(hidden -> kv_heads * head_dim)
q,k = RoPE(q,k)
k,v = cache.update(k,v, layer)
attn = causal attention(q,k,v, mask, scaling=head_dim**-0.5)
x = residual + o_proj(attn)
y = RMSNorm(x)
y = down_proj(act(gate_proj(y)) * up_proj(y))
x = residual + y
```

Qwen2 uses bias on Q/K/V. Llama/Mistral source defaults use no Q/K/V/O bias and no MLP bias.

## 6. Attention requirements

Vision attention:

- Full noncausal self-attention.
- CLIP/SigLIP reshape Q/K/V from `[B, T, hidden]` to `[B, heads, T, head_dim]`.
- No KV cache. Dropout is zero in inference.

Text attention:

- Causal self-attention with RoPE on Q and K before cache update.
- Query shape before attention: `[B, num_attention_heads, q_len, head_dim]`.
- Cache K/V shape per layer before repeat: `[B, num_key_value_heads, kv_seq_len, head_dim]`.
- Eager fallback repeats K/V to `[B, num_attention_heads, kv_seq_len, head_dim]` only for attention math. Cache should store the compact KV-head form.
- Attention score order in eager path: `matmul(q, k^T) * scaling`, add mask, softmax with fp32 accumulation, cast back to query dtype, dropout, then matmul with V.
- Llama 7B/13B representative configs are MHA (`kv_heads=heads`).
- BakLLaVA/Mistral is GQA (`heads=32`, `kv_heads=8`, `head_dim=128`) and uses sliding-window causal mask when `sliding_window` is not `None`.
- Qwen2 interleave 0.5B is MHA (`heads=kv_heads=16`, `head_dim=64`) with long RoPE theta 1e6; its representative config sets `use_sliding_window=false`, so `sliding_window` becomes `None`.

Prefill cache after one image+text prompt stores keys/values for the full expanded multimodal token sequence. Decode appends one or more text tokens per step. Pixel values should not re-enter cached decode unless cache is disabled or the generation call starts a new cache continuation.

## 7. Position encoding and custom math

Vision position embeddings are learned. CLIP uses `num_patches + 1` positions including CLS. SigLIP uses patch-only learned positions. Optional interpolation reshapes learned patch positions to `[1, sqrt(N), sqrt(N), C]`, permutes to NCHW, bicubic-interpolates, then flattens back; standard LLaVA preprocessing avoids this by producing the configured image size.

Text RoPE is inherited from the decoder. The shared shape contract is:

```python
def apply_llama_like_rope(q, k, cos, sin):
    # q: [B, heads, T, D], k: [B, kv_heads, T, D]
    cos = cos.unsqueeze(1)
    sin = sin.unsqueeze(1)
    return (q * cos) + (rotate_half(q) * sin), (k * cos) + (rotate_half(k) * sin)
```

Cos/sin depend on `position_ids`, which default to `arange(q_len) + past_seen_tokens`. They can be precomputed by maximum context and RoPE parameter set, but selected rows are dynamic per prefill/decode call.

## 8. Preprocessing and input packing

Processor tensor contract:

- Text output: `input_ids [B, S_expanded]`, optional `attention_mask [B, S_expanded]`, optional `mm_token_type_ids`.
- Image output: `pixel_values [num_images_or_batch, 3, H, W]`, normally float after rescale/normalize.
- CLIP LLaVA 1.5 processors: resize shortest edge 336, center crop 336x336, BICUBIC, RGB, rescale by 1/255, normalize by OpenAI CLIP mean/std, patch 14, `num_additional_image_tokens=1`, default strategy.
- SigLIP interleave processor: resize to 384x384, mean/std 0.5, patch 14, no additional image tokens, full strategy.

Placeholder/stitching contract:

- Processor computes `num_image_tokens = (height // patch_size) * (width // patch_size) + num_additional_image_tokens`, then subtracts one for `vision_feature_select_strategy == "default"`.
- Each textual `<image>` marker is replaced by that many literal `<image>` tokens before tokenization.
- Model computes image features, concatenates all image feature lists on dim 0, casts to text embedding dtype/device, builds `special_image_mask = input_ids == image_token_id`, expands to embedding shape, checks `inputs_embeds[special_image_mask].numel() == image_features.numel()`, and calls `inputs_embeds.masked_scatter(mask, image_features)`.
- If `input_ids` is absent and `inputs_embeds` is supplied, the model finds placeholders by comparing embeddings against the embedding of `image_token_id`; this is fragile for compiled runtimes and should be deferred unless needed.
- Optional `image_sizes` splits projected features by `(image_sizes // vision_tower.patch_size).prod(-1)`, but standard LLaVA processors do not provide this for the inspected configs.

CPU/data-pipeline candidates: all image resize/crop/tokenizer placeholder expansion can remain outside the first GPU graph. GPU/runtime candidates: vision tower, projector, embedding stitch, prefill, decode.

## 9. Graph rewrite / lowering opportunities

### Rewrite: non-overlap vision Conv2d patch embed -> Linear

Source pattern: CLIP/SigLIP `Conv2d(3 -> hidden, kernel_size=patch, stride=patch, padding=0/valid)` followed by flatten spatial and transpose to token-major.

Replacement:

```text
NCHW image -> WindowFlatten([patch,patch,3]) -> GEMM(weight_flat.T) -> optional bias -> [B, Gh*Gw, hidden]
```

Preconditions:

- `kernel_size == stride == patch_size`
- padding is zero or `"valid"`
- dilation 1, groups 1
- source input is NCHW and height/width match or floor-divide exactly as Conv2d semantics require
- flatten order matches PyTorch Conv2d weight layout `[out, in, kh, kw]`

Weight transform:

```python
w = conv.weight.reshape(out_channels, in_channels * kh * kw)
```

Failure cases: dynamic resolutions with interpolation, non-divisible spatial dimensions if a runtime expects all pixels used, non-CLIP/SigLIP custom towers, or NHWC without a matching weight/layout transform.

Parity test sketch: compare Conv2d+flatten+transpose and rewritten GEMM for CLIP 336/14 and SigLIP 384/14, including the SigLIP floor grid 27x27.

### Rewrite: placeholder masked_scatter -> indexed copy

Source pattern: `input_ids == image_token_id`, expand over hidden dimension, `masked_scatter(inputs_embeds, image_features)`.

Replacement:

```text
indices = nonzero(input_ids == image_token_id)
copy image_features.reshape(-1, hidden) into inputs_embeds at indices
```

Preconditions: placeholder count equals image feature token count; all image features are already in token order; no duplicate or missing positions beyond expected repeated placeholders.

Failure cases: `inputs_embeds`-only placeholder detection, multiple images with processor/model order mismatch, batch packing that interleaves variable image counts without explicit descriptors.

### Rewrite: cacheable vision prefix

Source pattern: `pixel_values -> vision_tower -> selected hidden states -> projector`, then scatter into embeddings before prefill.

Replacement: compile/execute vision+projector as a separate subgraph returning `[image_tokens, text_hidden]`, then feed it into a prefill graph with token embeddings and image token positions.

Preconditions: same image bytes/preprocessor output, same feature layer/select strategy, same projector weights, same dtype policy.

Failure cases: requests that pass `image_sizes` with dynamic split behavior, changing `vision_feature_layer` at runtime, or processors that change placeholder expansion.

### Layout opportunity: guarded NCHW -> NHWC in vision-local region

Candidate region: preprocessing normalize/rescale and patch embedding. Axis-sensitive ops include channel normalization axis, crop/pad spatial axes, Conv2d input layout, flatten spatial axes, and transpose to token-major. Downstream vision transformer consumes `[B, T, C]`, so NHWC may be eliminated before token sequence creation. Do not translate the text embedding/projector sequence to NHWC.

## 10. Kernel fusion candidates

Highest priority:

- Text decoder RMSNorm, RoPE, causal/GQA FlashAttention with compact KV cache, and SwiGLU MLP fusion. These dominate prefill/decode.
- Placeholder indexed copy/scatter. It is small, but correctness-sensitive and needed for end-to-end multimodal parity.
- Projector GELU MLP fusion: two GEMMs with GELU over image-token sequence, useful for cacheable vision-prefix throughput.

Medium priority:

- Vision patch Conv2d-as-GEMM and CLIP/SigLIP LayerNorm+attention+MLP kernels.
- Last-token-only logits using `logits_to_keep=1` for decode.
- Processor GPU preprocessing only if CPU preprocessing becomes a bottleneck; otherwise keep it outside runtime graph.

Lower priority:

- Position-embedding interpolation for nonstandard resolutions.
- `inputs_embeds`-only placeholder comparison path.
- Vision hidden-state tuple materialization optimization. Source currently forces `output_hidden_states=True`; first integration can materialize all hidden states for parity, then specialize selected-layer extraction.

## 11. Runtime staging plan

Stage 1: parse LLaVA config, instantiate delegated vision/text config, and load weights for one small checkpoint. Stub tokenizer and image preprocessing with recorded tensors.

Stage 2: run projector-only and placeholder stitch parity with random/projected features and fixed `input_ids`.

Stage 3: run vision tower plus projector parity for CLIP 336/14; add SigLIP 384/14 separately because it has no CLS token and uses full selection.

Stage 4: run text decoder prefill parity using stitched `inputs_embeds`, attention mask, position ids, and no image recomputation inside decode.

Stage 5: implement decode with per-layer compact KV cache. Validate that cached K/V are post-RoPE and shaped `[B, kv_heads, seq, head_dim]`.

Stage 6: enable optimized attention and GEMM/fusion paths for Llama, Mistral, and Qwen2 variants behind config guards.

Stage 7: split vision prefix caching from prefill for production scheduling and continuous batching.

Initially stub: chat template/tokenizer, generation sampling policy, beam search, optional `mm_token_type_ids`, `inputs_embeds`-only placeholder detection, and nonstandard image interpolation.

## 12. Parity and validation plan

- Processor parity: for each representative processor, verify `pixel_values` shape and numeric preprocessing against Transformers for one RGB image and one non-square image.
- Placeholder parity: compare expanded token counts and stitch output for one image, two images, and no-image text.
- Projector parity: random `[N, image_tokens, vision_hidden]` features through Linear-GELU-Linear in fp32/fp16.
- Vision parity: CLIP hidden state `-2` default crop and SigLIP `-1` full selection. Compare projected features.
- Text single-layer parity: delegated Llama/Mistral/Qwen2 one block with `inputs_embeds`.
- Prefill logits parity: full prompt with image placeholders, `logits_to_keep=1` and full logits.
- Decode parity: one-token and multi-token cached decode; assert `pixel_values` absent after first cached iteration.
- Recommended tolerances: fp32 absolute/relative around `1e-4`; fp16/bf16 vision+decoder around `1e-2` for full logits, tighter for isolated GEMMs where accumulation policy matches.

## 13. Performance probes

- CPU preprocessing images/sec by resolution and batch size.
- Vision tower only: CLIP 336/14 and SigLIP 384/14 throughput, separated from projector.
- Projector throughput over image-token counts 225, 576, 729.
- Stitch/scatter latency versus sequence length and number of image placeholders.
- Prefill tokens/sec with expanded multimodal sequence lengths.
- Decode tokens/sec with batch size sweep and `logits_to_keep=1`.
- KV cache memory by text model: `layers * 2 * B * kv_heads * seq * head_dim * dtype_size`.
- Attention backend comparison: eager, SDPA, FlashAttention for Llama MHA, Mistral GQA/sliding, Qwen2 long-context.
- End-to-end requests/hour with and without cached vision prefix.

All probes above are proposed; no DinoML tests or benchmarks were run for this docs-only audit.

## 14. Skip/defer list

- Training, gradients, and gradient checkpointing.
- Beam search and advanced generation controllers beyond greedy/sampling hooks.
- `inputs_embeds`-only image placeholder detection.
- Runtime `image_sizes` split path unless a target checkpoint/processor emits it.
- Dynamic high-resolution position interpolation.
- Multi-GPU tensor parallel, quantization, and speculative decoding.
- Remote-code-only or non-`llava` variants such as `llava_next` unless separately audited.
- Full tokenizer/chat-template parity for first graph runtime; keep as data-pipeline work.

## 15. Final implementation checklist

- [ ] Parse `LlavaConfig` plus delegated `vision_config` and `text_config`.
- [ ] Load CLIP/SigLIP vision tower weights and text decoder weights for one target checkpoint.
- [ ] Implement processor tensor contract fixture for `pixel_values`, expanded `input_ids`, and `attention_mask`.
- [ ] Implement/projector parity: Linear -> GELU -> Linear with configurable bias.
- [ ] Implement image feature selection: layer index/list, default CLS crop, full strategy, concat on feature dim.
- [ ] Implement placeholder count validation and indexed-copy stitch into text embeddings.
- [ ] Implement CLIP vision tower path for 336/14 and SigLIP path for 384/14.
- [ ] Implement Llama/Mistral/Qwen2 prefill from `inputs_embeds`.
- [ ] Implement compact per-layer KV cache shaped `[B, kv_heads, seq, head_dim]`.
- [ ] Add decode path that omits `pixel_values` after first cached iteration.
- [ ] Add Conv2d patch-embed-to-GEMM rewrite behind strict guards.
- [ ] Add cacheable vision-prefix/projector subgraph boundary.
- [ ] Add parity tests for processor, projector, vision features, stitch, prefill logits, and decode logits.
- [ ] Benchmark preprocessing, vision/projector, stitch, prefill, decode, and KV memory separately.
