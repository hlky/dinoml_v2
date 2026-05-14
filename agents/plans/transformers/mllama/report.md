# Mllama Transformers Audit

## 1. Source basis

```text
Transformers commit/version: b75feb2af64c3e29cbbc1bd859958c5432cc7ed4
Model id: meta-llama/Llama-3.2-{11B,90B}-Vision{-Instruct} intended targets; official repos are gated from this environment.
Config source: accessible HF mirrors plus Transformers conversion/config defaults, labeled below.
Source files inspected:
  X:/H/transformers/src/transformers/models/mllama/modeling_mllama.py
  X:/H/transformers/src/transformers/models/mllama/configuration_mllama.py
  X:/H/transformers/src/transformers/models/mllama/processing_mllama.py
  X:/H/transformers/src/transformers/models/mllama/image_processing_mllama.py
  X:/H/transformers/src/transformers/models/mllama/image_processing_pil_mllama.py
  X:/H/transformers/src/transformers/models/mllama/convert_mllama_weights_to_hf.py
  X:/H/transformers/docs/source/en/model_doc/mllama.md
  X:/H/transformers/tests/models/mllama/test_modeling_mllama.py
Source URLs:
  https://github.com/huggingface/transformers/blob/b75feb2af64c3e29cbbc1bd859958c5432cc7ed4/src/transformers/models/mllama/modeling_mllama.py
  https://github.com/huggingface/transformers/blob/b75feb2af64c3e29cbbc1bd859958c5432cc7ed4/src/transformers/models/mllama/configuration_mllama.py
  https://github.com/huggingface/transformers/blob/b75feb2af64c3e29cbbc1bd859958c5432cc7ed4/src/transformers/models/mllama/processing_mllama.py
  https://github.com/huggingface/transformers/blob/b75feb2af64c3e29cbbc1bd859958c5432cc7ed4/src/transformers/models/mllama/image_processing_mllama.py
  https://github.com/huggingface/transformers/blob/b75feb2af64c3e29cbbc1bd859958c5432cc7ed4/src/transformers/models/mllama/image_processing_pil_mllama.py
Config/preprocessor URLs sampled:
  https://huggingface.co/unsloth/Llama-3.2-11B-Vision/raw/main/config.json
  https://huggingface.co/unsloth/Llama-3.2-11B-Vision/raw/main/preprocessor_config.json
  https://huggingface.co/unsloth/Llama-3.2-11B-Vision-Instruct/raw/main/config.json
  https://huggingface.co/unsloth/Llama-3.2-11B-Vision-Instruct/raw/main/preprocessor_config.json
  https://huggingface.co/unsloth/Llama-3.2-90B-Vision-Instruct/raw/main/config.json
  https://huggingface.co/unsloth/Llama-3.2-90B-Vision-Instruct/raw/main/preprocessor_config.json
  https://huggingface.co/intfloat/mmE5-mllama-11b-instruct/raw/main/config.json
  https://huggingface.co/neuralmagic/Llama-3.2-90B-Vision-Instruct-FP8-dynamic/raw/main/config.json
Any missing files or assumptions: no remote code is required for native mllama. Official Meta config/preprocessor files returned 401, so exact production config rows use open mirrors and the native conversion script.
```

Primary runtime target: multimodal image+text to text generation with `MllamaForConditionalGeneration`. The text-only `MllamaForCausalLM` is a useful delegated decoder target and can be staged separately. Training/loss paths are deferred.

## 2. High-level architecture

Mllama is a vision encoder + multimodal projector + causal Llama-style text decoder with inserted gated cross-attention layers. The image token is a text placeholder, not an embedding splice point: image features are not scattered into the token stream. Instead, the processor builds a dense cross-attention mask, the vision tower produces per-tile image tokens, a linear projector maps those to text hidden size, and selected decoder layers attend to those projected image states.

```text
CPU image/text preprocessing
  -> pixel_values/aspect metadata + input_ids/cross_attention_mask
  -> vision Conv2d patch embed + local/global vision encoders
  -> Linear vision_output_dim -> text_hidden_size projector
  -> decoder prefill with causal self-attn + gated image cross-attn
  -> decode with self-attn KV cache and cached cross-attn image K/V
  -> lm_head logits/sampling
```

Stage decomposition:

| Stage | Inputs/outputs | Cacheable independently | First parity target |
|---|---|---:|---|
| CPU/data preprocessing | images/text -> `pixel_values`, `aspect_ratio_ids`, `aspect_ratio_mask`, `cross_attention_mask`, `input_ids` | processor outputs can be reused per prompt | processor tensor shape/golden mask |
| Vision encoder | `[B, I, T, C, H, W]` -> `[B, I, T, patches+1, vision_output_dim]` | yes, per image batch | encoder/projector parity |
| Projector | `vision_output_dim -> text_hidden_size`, then reshape to `[B*I*T, patches+1, Htext]` | yes | projected image state parity |
| Prefill | text tokens + image states/mask -> logits + cache | self-attn and cross-attn cache | prefill logits |
| Decode | next token + caches -> next logits | self-attn grows; cross-attn K/V should be reused | token-by-token parity |

## 3. Important config dimensions

Source defaults from `configuration_mllama.py` describe the 11B native shape unless overridden by checkpoint config:

| Field | 11B/default value | 90B mirror value | Runtime impact |
|---|---:|---:|---|
| text `hidden_size` | 4096 | 8192 | GEMM width, KV cache bytes |
| text layers | 40 total | 100 total | includes cross-attn layers |
| cross-attn layers | `[3,8,13,18,23,28,33,38]` | `[3,8,...,98]` every 5 layers | layer type varies by index |
| self-attn layers | 32 | 80 | causal cached decode |
| cross-attn layer count | 8 | 20 | image K/V cache count |
| attention heads | 32 | 64 | MHA query heads |
| KV heads | 8 | 8 | GQA; repeat factor 4 for 11B, 8 for 90B |
| head dim | 128 | 128 | RoPE and attention head width |
| intermediate size | 14336 | 28672 | SwiGLU GEMMs |
| vocab size / image token | 128256 / 128256 | same | input embedding has `vocab_size + 8`, lm_head has `vocab_size` |
| max positions | 131072 | 131072 | long-context RoPE |
| RoPE | llama3 scaling, theta 500000 | same | non-default RoPE via `rope_parameters`/legacy `rope_scaling` |
| vision hidden / heads | 1280 / 16 | 1280 / 16 | ViT-style encoder |
| vision layers | 32 local + 8 global | same | global layers are gated |
| patch / tile size | 14 / 448 or 560 | 14 / 560 | patch count differs by base/instruct mirror |
| max image tiles | 4 | 4 | dense padding dimension |
| vision output dim | 7680 | 7680 | final + 5 intermediate features concat |
| dtype | bf16 in production mirrors | bf16 | source supports normal Transformers dtypes |
| cache support | `use_cache` true in mirrors; source default true | true | dynamic cache; cross-attn stores image K/V in same layer-indexed cache object |

Representative checkpoint sweep:

| Checkpoint/config inspected | Provenance | Structure highlights |
|---|---|---|
| `unsloth/Llama-3.2-11B-Vision` | open mirror config/preprocessor | 40 text layers, 8 cross layers, image size 448, eos 128001 |
| `unsloth/Llama-3.2-11B-Vision-Instruct` | open mirror config/preprocessor | same operators, image size 560, eos `[128001,128008,128009]` |
| `intfloat/mmE5-mllama-11b-instruct` | open derivative config | native `mllama`, 11B dims, image size 448, `use_cache=false` in config; operator structure same |
| `unsloth/Llama-3.2-90B-Vision-Instruct` | open mirror config/preprocessor | 100 text layers, 20 cross layers, hidden 8192, 64 Q heads, 8 KV heads |
| `neuralmagic/Llama-3.2-90B-Vision-Instruct-FP8-dynamic` | open derivative config | same native graph structure; quantized/FP8 weight format is out of scope for native dense audit |
| Transformers tiny test config | local tests, not HF production | 2 text layers, cross layer `[1]`, vision `image_size=30`, `patch_size=2`, hidden 32/16; useful for CPU parity only |

## 3a. Family variation traps

- Official Meta repos are gated; mirrors may add quantization or derivative metadata. Native source only implements dense PyTorch modules; quantized loader/runtime behavior needs a separate audit.
- 11B and 90B differ in text width/layer count/cross-attn count. Do not hard-code 40 layers or 8 cross layers.
- Cross-attention layers replace normal self-attention layers at configured indices. They do not perform causal self-attention in that layer.
- GQA is required: `num_key_value_heads < num_attention_heads` for production configs.
- Text input embedding is `vocab_size + 8`, while `lm_head` is `vocab_size`; image placeholder labels/logits must be masked or rejected.
- `rope_scaling` appears in mirror configs, while current config class uses `rope_parameters`; Transformers normalizes these through `PreTrainedConfig`. DinoML should canonicalize to effective `rope_parameters`.
- The native processor does not expand image placeholders into hundreds of soft tokens. It creates `cross_attention_mask` ranges per image placeholder.
- Vision tower source consumes NCHW tiles. NHWC is an optimization candidate only inside guarded Conv/attention/MLP regions.
- Vision output shape preserves `[B, max_images, max_tiles, num_patches, vision_output_dim]` before projector. Projector then reshapes to `[-1, num_patches, text_hidden]`, flattening images/tiles into the batch axis for cross-attention.
- Source tests skip some packed generation tests because Mllama applies Q/K norm in cross-attention and has encoder-decoder-like cache behavior. Treat generic Llama packed-cache assumptions as unsafe.

## 4. Operator coverage checklist

Tensor/layout ops:

- reshape/view/flatten/transpose/permute/contiguous, especially `[B,I,T,C,H,W] -> [B*I*T,C,H,W]`, Conv output flatten(2).transpose, head split transpose, image/tile flattening.
- cat/stack/slice/pad/repeat/repeat_interleave/expand/index_select/arange.
- Embedding lookup for tokens, aspect-ratio embeddings, tile embeddings.
- Mask construction: dense boolean/int masks, `masked_fill`, finfo min, row-any reductions, matrix multiply for aspect-ratio attention mask.

Neural primitives:

- Conv2d patch embedding `Linearized Conv2d(C=3, kernel=stride=patch_size, out=1280, bias=False)`.
- Linear without bias for Q/K/V/O in text and vision attention.
- Linear with bias for vision MLP, multimodal projector `7680 -> Htext`, and vision MLP `1280 -> 5120 -> 1280`.
- Text SwiGLU: `down(silu(gate(x)) * up(x))`, e.g. 4096 -> 14336 -> 4096 or 8192 -> 28672 -> 8192.
- LayerNorm for vision, RMSNorm for text and cross-attn Q/K head dims.
- GELU for vision MLP, SiLU for text MLP.
- tanh scalar gates for vision global layers, tile/position embeddings, and text cross-attn residual branches.

Attention primitives:

- Vision noncausal self-attention, MHA, no KV cache.
- Text causal self-attention, GQA, RoPE, dynamic KV cache.
- Text image cross-attention, GQA, no RoPE, Q/K RMSNorm, cross KV cache.
- Eager fallback math: matmul, add mask, fp32 softmax, dropout, matmul V.
- SDPA/Flash/Flex backend dispatch through `ALL_ATTENTION_FUNCTIONS`.

Position/custom math:

- Llama3 RoPE, dynamic rope update support.
- Vision learned absolute position embeddings and learned aspect-ratio/tile embeddings with tanh gates.

Preprocessing-coupled ops:

- image resize to optimal tiled canvas, pad, split to tiles, rescale/normalize.
- `pixel_values`, `aspect_ratio_ids`, `aspect_ratio_mask`, `cross_attention_mask`, `num_tiles`.
- text BOS insertion after leading image tokens.

Generation/cache ops:

- DynamicCache creation and layer-indexed update/get for self and cross layers.
- `prepare_inputs_for_generation` must drop image pixel tensors after first cached iteration.
- `_update_model_kwargs_for_generation` extends `cross_attention_mask` by copying the last row for the new token.

No discrete image codebook, MoE, sliding-window attention, or `cu_seqlens` metadata exists in native mllama source.

## 5. Layer/block breakdown

Vision preprocessing inside model:

```text
pixel_values: [B, I, T, 3, H, W]
x = reshape [B*I*T, 3, H, W]
x = Conv2d(3 -> 1280, kernel=stride=patch, bias=False)
x = flatten patches -> transpose to [B*I*T, P, 1280]
x = reshape [B*I, T, P, 1280]
x += gated aspect-ratio tile embedding
x = cat(cls, x) per tile
x += gated learned patch/tile position embedding
x = LayerNorm
x = pad patch tokens to multiple of 8
x = local vision encoder over flattened T*(P+pad) sequence
x = LayerNorm
x += post tile embedding
x = global gated vision encoder
x = unpad
x = concat(final hidden, local layers [3,7,15,23,30]) -> [B,I,T,P+1,7680]
```

Vision encoder block, repeated 32 local and 8 global:

```text
res = x
x = LayerNorm(x)
q,k,v = Linear(1280 -> 1280, bias=False) split 16 heads of 80
x = noncausal attention(q,k,v, aspect_ratio_attention_mask)
x = Linear(1280 -> 1280, bias=False)
x = res + (tanh(gate_attn) * x if global else x)
res = x
x = LayerNorm(x)
x = Linear(1280 -> 5120, bias=True) -> GELU -> Linear(5120 -> 1280, bias=True)
x = res + (tanh(gate_ffn) * x if global else x)
```

Projector:

```text
image_states = Linear(7680 -> text_hidden_size, bias=True)
image_states = reshape [-1, num_patches, text_hidden_size]
```

Text self-attention decoder layer:

```text
res = x
x = RMSNorm(hidden_size)
q = Linear(H -> H, bias=False)
k,v = Linear(H -> KV_heads*head_dim, bias=False)
q,k = RoPE(q,k)
k,v = cache.update(k,v, layer_idx) if cache
x = GQA causal attention(q,k,v, causal_mask)
x = Linear(H -> H, bias=False)
x = res + x
res = x
x = RMSNorm(H)
x = down(silu(gate(x)) * up(x))
x = res + x
```

Text cross-attention decoder layer:

```text
res = x
x = RMSNorm(H)
q = Linear(H -> H, bias=False); q = RMSNorm(head_dim)
k = Linear(H -> KV_heads*head_dim, bias=False) on image states; k = RMSNorm(head_dim)
v = Linear(H -> KV_heads*head_dim, bias=False) on image states
k,v = cache.update(k,v, layer_idx) when image states are present
x = GQA cross-attention(q,k,v, cross_attention_mask)
x = res + tanh(cross_attn_attn_gate) * Linear(H -> H, bias=False)(x)
res = x
x = RMSNorm(H)
x = down(silu(gate(x)) * up(x))
x = full_text_row_masked_out_mask * x when present
x = res + tanh(cross_attn_mlp_gate) * x
```

## 6. Attention requirements

Vision attention:

- Noncausal self-attention over `max_tiles * padded_patch_tokens`.
- MHA: 16 Q/K/V heads, head dim 80.
- Mask shape after preparation is `[B*I, 1, T*target_length, T*target_length]` with `finfo(dtype).min` on disallowed rows/columns.
- No cache and no RoPE.

Text self-attention:

- Causal GQA with `num_attention_heads / num_key_value_heads` repeat groups.
- Head dim is `hidden_size // num_attention_heads` = 128 in production.
- RoPE is applied to Q/K before cache update, so cached self-attn keys are post-RoPE. Values are unmodified.
- Per self-attn layer cache shape before repeat is `[B, KV_heads, seen_tokens, head_dim]`; after repeat for attention math it is `[B, Q_heads, seen_tokens, head_dim]`.
- Uses `create_causal_mask`, so source mask shape depends on backend/cache.

Text cross-attention:

- Cross GQA from text queries to projected image states.
- Q/K RMSNorm is per head dimension after transpose, before attention.
- Cross K/V cache shape is `[B_or_flat, KV_heads, image_tokens, head_dim]`; source flattens images/tiles into the batch dimension before language model call. Production `image_tokens` is `num_patches = (image_size // patch_size)^2 + 1`, so 1025 for 448 and 1601 for 560 per tile.
- Cross keys are not RoPE encoded. They are K-projected and K-normalized image states.
- If `cross_attention_states` is absent but the layer cache has entries, decode reuses cached cross K/V. If both are absent, cross-attn layer errors; the model skips cross layers only when no states and no cache.

Backend compatibility:

- Source advertises SDPA, FlashAttention, FlexAttention, and eager fallback via `ALL_ATTENTION_FUNCTIONS`.
- Eager order is: repeat KV, `q @ k.T * scaling`, add mask, fp32 softmax cast back, dropout, `attn @ v`.
- Dropout is 0 in inference. Fused attention parity must preserve query scaling and fp32 softmax behavior where applicable.

## 7. Position encoding and custom math

Text RoPE:

```python
def mllama_rope(position_ids, dim, theta, attention_scaling=1.0):
    inv = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
    freqs = (inv[None, :, None] @ position_ids[:, None, :].float()).transpose(1, 2)
    emb = torch.cat((freqs, freqs), dim=-1)
    return emb.cos() * attention_scaling, emb.sin() * attention_scaling

def apply_rope(q, k, cos, sin):
    cos, sin = cos[:, None, :, :], sin[:, None, :, :]
    return q * cos + rotate_half(q) * sin, k * cos + rotate_half(k) * sin
```

Current production configs use llama3 RoPE scaling (`factor=8`, `low_freq_factor=1`, `high_freq_factor=4`, original max 8192, theta 500000). DinoML should canonicalize config values through the same RoPE parameter path and test the llama3 scaling function, not just default RoPE.

Vision positions:

- learned patch/class position embedding: `[num_patches, 1280]`.
- learned tile position embedding: `Embedding(max_aspect_ratio_id+1, max_tiles*num_patches*1280)`.
- aspect-ratio tile embeddings before and after vision encoders: `Embedding(max_aspect_ratio_id+1, max_tiles*1280)`.
- tanh scalar gates blend base patch positions and tile-specific positions.

## 8. Preprocessing and input packing

CPU/data-pipeline preprocessing:

- Images are nested by batch sample: each sample can have variable image count.
- Resize chooses an optimal tiled canvas under `max_image_tiles`; pad to tile grid; split into tiles.
- Output `pixel_values`: `[batch_size, max_num_images, max_image_tiles, 3, tile_height, tile_width]`, float32, NCHW tiles.
- Output `aspect_ratio_ids`: `[batch_size, max_num_images]`, int64, 0 for padded image slots and 1-based aspect ratio ids.
- Output `aspect_ratio_mask`: `[batch_size, max_num_images, max_image_tiles]`, int64, 1 for valid tiles.
- Output `num_tiles`: nested Python list; consumed by processor only to build cross-attention mask.
- Preprocessor config mirrors: base 11B uses tile size 448; instruct 11B/90B use 560. Mean/std are `[0.48145466,0.4578275,0.40821073]` and `[0.26862954,0.26130258,0.27577711]`; resize, pad, rescale, normalize, RGB conversion are enabled.

Text/image coupling:

- Processor counts `<|image|>` in text and requires matching nested image counts.
- It inserts BOS after any leading image tokens if BOS is missing.
- `cross_attention_mask` shape is `[B, text_length, max_num_images, max_image_tiles]`.
- For each image token, the mask allows following text tokens to attend to the corresponding image's valid tiles; consecutive image tokens are treated as a group.
- Model expands tile mask to token mask by `repeat_interleave(num_vision_tokens, dim=3)`, where `num_vision_tokens` is the per-tile patch token count including CLS.
- During generation, `cross_attention_mask` is sliced by current positions and extended by copying the previous last row for new tokens.

No `cu_seqlens`, packed varlen descriptors, modality token type ids, or image codebook logits masks are present in native source.

## 9. Graph rewrite / lowering opportunities

### Rewrite: vision patch Conv2d -> Linear/GEMM

Source pattern: `Conv2d(3 -> 1280, kernel_size=patch_size, stride=patch_size, padding="valid", bias=False)` followed by `flatten(2).transpose(1,2)`.

Replacement:

```text
NCHW Tile -> non-overlap WindowFlatten(C, ph, pw) -> MatMul(weight_flat.T) -> [N, P, 1280]
```

Preconditions:

- `kernel_size == stride == patch_size`.
- padding is valid/zero, dilation 1, groups 1, bias absent.
- `height % patch_size == 0` and `width % patch_size == 0`.
- Source flatten order must match PyTorch NCHW Conv2d output order.

Weight transform:

```python
w = conv.weight.reshape(out_channels, in_channels * ph * pw)
```

Layout constraints/failure cases:

- Initial semantic graph should remain NCHW. NHWC optimization requires a local guarded region with either NHWC WindowFlatten or a weight permutation.
- Reject or fallback if dynamic tile size is not divisible by patch size.

Parity sketch: compare Conv2d+flatten+transpose to WindowFlatten+GEMM for 448 and 560 tiles in fp32/bf16.

### Rewrite: Q/K/V separate Linear -> grouped GEMM

Source pattern: three independent bias-free linear projections followed by head reshape/transpose.

Replacement: grouped GEMM or packed QKV materialization.

Preconditions:

- Same input `x`, no intervening op.
- Bias absent.
- For cross-attention, Q input is text state while K/V input is image state, so only K/V can be grouped together there.
- Weight packing must preserve row order: Q rows, K rows, V rows as separate matrices unless an explicit packed layout is introduced.

Failure cases: quantized/remote-code weight layouts, sharded conversion layouts, or cross-attention Q vs K/V different source tensors.

### Rewrite: RMSNorm + Linear

Source pattern: text RMSNorm into projection or MLP GEMMs.

Replacement: fused RMSNorm kernel feeding GEMM, or persistent normalized tile in shared memory.

Preconditions: last-dim normalization only, eps fixed from config, weight broadcast on hidden/head dim. Cross-attn Q/K norms are over `head_dim` after reshape and must not be fused as hidden-size RMSNorm.

### Rewrite: attention backend lowering

Source pattern: split heads -> optional RoPE/QK norm -> `ALL_ATTENTION_FUNCTIONS` call -> transpose/reshape -> output Linear.

Replacement: native fused prefill/decode attention kernels for:

- causal GQA with post-RoPE cached K.
- cross GQA with static image K/V and per-token cross mask.
- noncausal vision MHA with dense aspect-ratio mask.

Preconditions: dropout 0, inference mode, supported mask dtype/shape, no attentions output.

### Rewrite: layout pass for vision tower

Candidate local region: patch embedding + vision encoder GEMMs/attention.

Required axis rewrites:

- image processor/model input is `[B,I,T,C,H,W]`; NHWC pass would reinterpret only model-internal tile tensor as `[B*I*T,H,W,C]`.
- LayerNorm is over last dim after patch flatten, unaffected once tokenized.
- Attention masks are token-token and should be protected from channel-layout rewrites.

Failure cases: any consumer expecting NCHW pixel tensor, dynamic split-to-tiles in runtime graph, or unguarded `dim=1` channel assumptions. Recommended initial guard: no layout translation around processor outputs and mask construction; optimize only Conv patch embed as a local lowered op.

## 10. Kernel fusion candidates

Highest priority:

- Text RMSNorm and RMSNorm+Linear for decoder throughput.
- GQA causal FlashAttention/paged decode with post-RoPE K cache.
- Cross-attention K/V projection + K RMSNorm cache for image states, because it is reusable across decode.
- SwiGLU MLP fused activation multiply plus GEMM epilogue.
- Last-token-only logits for `logits_to_keep=1` decode.

Medium priority:

- Vision patch Conv2d lowered to GEMM.
- Vision MHA with dense tile/patch mask.
- Projector `Linear(7680 -> Htext)` batched over image tiles.
- Cross-attention mask expansion/slicing fused or precomputed.
- Tanh scalar-gated residual multiply-add.

Lower priority:

- Vision LayerNorm+Linear fusion.
- Aspect-ratio and position embedding add fusion.
- Processor CPU throughput optimization.
- Full logits over long prefill sequences.

## 11. Runtime staging plan

Stage 1: parse config and load weights for `MllamaTextConfig`, including llama3 RoPE canonicalization and image-token embedding/lm_head size mismatch checks.

Stage 2: implement text-only `MllamaForCausalLM` parity for one self-attn block, full prefill, then decode with GQA KV cache.

Stage 3: add cross-attention decoder layers with externally supplied random `cross_attention_states` and `cross_attention_mask`, validating Q/K RMSNorm and cross cache semantics.

Stage 4: add vision encoder/projector parity independently from text. Stub processor with deterministic already-tiled `pixel_values` first.

Stage 5: integrate processor-coupled tensors and full multimodal prefill.

Stage 6: implement cached multimodal decode: no image pixels after first iteration, reuse cross-attn K/V, grow only self-attn cache.

Stage 7: enable optimized attention/GEMM/fusions and guarded vision Conv/layout rewrites.

Initially stub/defer chat templates, sampling policies, quantized checkpoint loaders, and training loss.

## 12. Parity and validation plan

- Unit test `_prepare_cross_attention_mask`: sparse processor mask -> expanded additive mask and row mask for single image, multiple images, consecutive image tokens.
- Unit test `_prepare_aspect_ratio_attention_mask`: valid/padded tile rows and patch padding to multiple of 8.
- Random tensor parity for Conv2d patch rewrite at tile sizes 448 and 560.
- Single vision block parity in fp32, then bf16 tolerance.
- Full vision encoder/projector parity on tiny test config.
- Text self-attn block parity with and without cache; verify cached K is post-RoPE.
- Cross-attn block parity with image states present, then second decode step using only cached K/V.
- Prefill logits parity for tiny config with image inputs.
- Decode token parity for 2-4 steps with `prepare_inputs_for_generation` behavior.
- End-to-end smoke against an accessible mirror if weights are available locally; otherwise document gated-weight blocker.

Suggested tolerances: fp32 `rtol=1e-4, atol=1e-5`; bf16/fp16 block-level `rtol=1e-2, atol=2e-2`, with final logits compared after disabling stochastic sampling.

## 13. Performance probes

- CPU processor images/sec by tile size and image count.
- Vision encoder throughput by tile size 448 vs 560 and valid tile count 1-4.
- Projector throughput for `[B*I*T, P, 7680]`.
- Prefill tokens/sec split into self-attn layers and cross-attn layers.
- Decode tokens/sec with image cache reused; compare with accidental recomputation.
- KV cache memory: self-attn cache vs cross-attn image cache separately.
- Attention backend comparison: eager vs SDPA/Flash/Flex for self, cross, and vision masks.
- Batch sweep: variable image count per sample and variable text length.
- Logits probe: full-sequence logits vs `logits_to_keep=1`.

## 14. Skip/defer list

- Training, labels/loss on image tokens, gradient checkpointing.
- Beam search/speculative/assisted decoding; source tests mark assisted decoding incompatible with current cache crop behavior.
- Quantized/FP8/AWQ/bnb/GGUF weight loaders and kernels.
- Multi-GPU tensor parallel conversion/sharding behavior.
- Remote-code derivatives such as `ocismllama` or `ultravox`.
- General NHWC graph-wide translation. Keep layout optimization local and guarded.
- Packed varlen attention until Q/K norm and mllama cache behavior are explicitly supported.

## 15. Final implementation checklist

- [ ] Parse `MllamaConfig`, `MllamaTextConfig`, and `MllamaVisionConfig`.
- [ ] Canonicalize `rope_scaling`/`rope_parameters` to effective llama3 RoPE parameters.
- [ ] Load text embeddings with `vocab_size + 8` and lm_head with `vocab_size`.
- [ ] Implement text RMSNorm, RoPE, GQA self-attention, DynamicCache-compatible decode.
- [ ] Implement text SwiGLU MLP and last-token logits.
- [ ] Implement cross-attention layers with Q/K head RMSNorm and gated residuals.
- [ ] Implement cross-attn K/V image cache reuse distinct from self-attn cache growth.
- [ ] Implement processor mask expansion and current-position slicing.
- [ ] Implement vision patch embedding, learned tile/position embeddings, vision encoders, and projector.
- [ ] Add guarded Conv2d patch -> GEMM rewrite.
- [ ] Add no-layout-translation guards around processor tensors and mask construction.
- [ ] Add single-block, encoder/projector, prefill, and decode parity tests.
- [ ] Benchmark processor, vision, prefill, decode, logits, and cache memory separately.
