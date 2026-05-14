# fast_vlm Transformers audit

## 1. Source basis

```text
Transformers commit/version: b75feb2af64c3e29cbbc1bd859958c5432cc7ed4 from X:/H/transformers; checkpoint configs say transformers_version=4.56.0.dev0.
Model id: KamilaMila/FastVLM-0.5B, KamilaMila/FastVLM-1.5B, KamilaMila/FastVLM-7B.
Config source: local HF snapshots in agents/plans/transformers/fast_vlm/_sources/.
Source files inspected: fast_vlm configuration/modeling/modular/conversion files; timm_wrapper configuration/modeling; qwen2 configuration/modeling; local installed timm fastvit.py; FastVLM tests/docs.
Any missing files or assumptions: the vision body is delegated to timm, not implemented in Transformers. This report treats `fastvit_mci3` as an allowlisted external body and recommends a separate exact timm-version audit before DinoML owns its kernels.
```

`modeling_fast_vlm.py` and `configuration_fast_vlm.py` are generated from `modular_fast_vlm.py`; future Transformers source edits should target the modular file, but DinoML runtime behavior should follow the generated modeling file.

## 2. High-level architecture

FastVLM is a multimodal projector plus causal LLM:

```text
CPU image/text preprocessing -> timm FastViT vision encoder -> NCHW image map flatten/permute -> 2-layer projector
text token embedding + image placeholder stitch -> Qwen2 prefill/decode with KV cache -> lm_head logits -> generation controller
```

Stage decomposition:

- CPU/data pipeline: chat template inserts `<image>` placeholders; CLIP image processor resizes/crops RGB images to `1024x1024`, rescales by `1/255`, and emits NCHW `pixel_values`.
- Vision encoder: `timm_wrapper` calls timm `fastvit_mci3.forward_features`. Public configs set `inference_mode=True`, so many reparameterizable train-time branches are already materialized as plain convs.
- Projector: FastVLM flattens NCHW feature maps from `[B, C, H, W]` to `[B, H*W, C]`, then applies `Linear(3072 -> text_hidden)`, GELU, `Linear(text_hidden -> text_hidden)`.
- Prefix construction: projected image embeddings are concatenated across images and copied into text embeddings where `input_ids == image_token_index`.
- Decoder: Qwen2 causal decoder handles prefill/decode and returns dynamic KV cache.
- Decode: after first generation iteration, `pixel_values` are not forwarded when cache is enabled because image features are already embedded and cached in the prefix.

Independently stageable validation units: vision feature shape, projector parity, placeholder stitch parity, Qwen2 text-only parity, multimodal prefill logits, then cached decode.

## 3. Important config dimensions

Source defaults from `FastVlmConfig` differ from public checkpoints: source default `image_seq_length=576`, while inspected checkpoints use `256`. Public checkpoints should drive first integration.

| Field | 0.5B | 1.5B | 7B |
|---|---:|---:|---:|
| dtype | bf16 | bf16 | bf16 |
| image token id | 151646 | 151646 | 151646 |
| image seq length | 256 | 256 | 256 |
| vision model | timm/fastvit_mci3.apple_mclip2_dfndr2b | same | same |
| vision hidden / projector input | 3072 | 3072 | 3072 |
| text model | Qwen2-0.5B | Qwen2-1.5B | Qwen2-7B |
| hidden size | 896 | 1536 | 3584 |
| layers | 24 | 28 | 28 |
| attention heads | 14 | 12 | 28 |
| KV heads | 2 | 2 | 4 |
| head dim | 64 | 128 | 128 |
| GQA repeat | 7 | 6 | 7 |
| intermediate size | 4864 | 8960 | 18944 |
| vocab size | 152000 | 152000 | 152128 |
| max positions | 131072 | 131072 | 131072 |
| RoPE theta | 1000000.0 | 1000000.0 | 1000000.0 |
| sliding window | disabled | disabled | disabled |
| tied embeddings | true | true | omitted/false effective unless inherited |

Processor snapshot: `CLIPImageProcessor`, `size.shortest_edge=1024`, `crop_size=1024x1024`, RGB conversion, center crop, bicubic resampling, rescale factor `1/255`, mean `[0,0,0]`, std `[1,1,1]`. `LlavaProcessor` has `patch_size=64`, `num_additional_image_tokens=0`, `image_token="<image>"`.

## 3a. Family variation traps

- `image_seq_length` is not safe to infer from source defaults; public checkpoints use 256 tokens from a `1024/64 = 16` grid.
- FastVLM validates only `vision_feature_select_strategy="full"` and `vision_feature_layer=-1`; reject other values for native FastVLM lowering.
- Vision body is external timm. `timm_wrapper` can wrap arbitrary timm architectures; DinoML should allowlist `fastvit_mci3.apple_mclip2_dfndr2b` first and reject other `vision_config.architecture` values until audited.
- Qwen2 source supports sliding-window attention through `layer_types`, but inspected FastVLM configs set every layer to `full_attention`.
- Qwen2 uses GQA in all public variants; do not assume KV heads equal query heads.
- Q/K/V projections have bias, O projection and MLP projections do not.
- `hidden_size == num_heads * head_dim` in inspected configs, but Qwen2 source allows explicit `head_dim`; parse it if present.
- `tie_word_embeddings` is true for 0.5B/1.5B configs and absent at FastVLM 7B top level; preserve LM-head/embedding alias only when config/runtime weight metadata says it is tied.
- Vision features are NCHW in source; the `flatten(2).permute(0,2,1)` bridge is axis-sensitive and should be protected from blind NHWC translation.
- The docs warn the vision backbone supports only eager attention; setting a global FlashAttention implementation can error. Decoder attention backend may be selected separately through `text_config`.
- Multiple images are technically accepted by concatenating image feature sequences, but docs say the model was not explicitly trained for multiple images; first integration can support one image per prompt and gate multi-image.

## 4. Operator coverage checklist

Tensor/layout ops:

- NCHW image tensor ingestion and dtype/device cast.
- NCHW conv feature map flatten: `[B,C,H,W] -> [B,C,H*W]`.
- Permute/transpose to token sequence: `[B,C,N] -> [B,N,C]`.
- `torch.cat(image_features, dim=0)` after `pooler_output = list(image_features)`.
- Boolean placeholder mask, `unsqueeze(-1)`, `expand_as(inputs_embeds)`.
- `masked_scatter` / indexed copy from `[num_images, image_seq, hidden]` into `[B, text_seq, hidden]`.
- Last-token or selected-token slicing for `logits_to_keep`.

Neural primitives, FastVLM-owned:

- Embedding lookup for Qwen2 token embeddings.
- Projector: `Linear(3072 -> H, bias=True)`, GELU, `Linear(H -> H, bias=True)`, where `H` is 896/1536/3584.
- LM head: `Linear(H -> vocab, bias=False)`, with optional tie to input embedding.

Delegated vision primitives for allowlisted timm `fastvit_mci3`:

- NCHW `Conv2d`, depthwise/grouped convs, 1x1 convs, BatchNorm/LayerNorm2d depending on reparameterization, GELU, SqueezeExcite, adaptive average pooling/head path if used.
- Reparameterized inference blocks: `MobileOneBlock`, `ReparamLargeKernelConv`, `RepMixer`, `RepConditionalPosEnc`.
- Vision attention in last FastViT stages: flatten NCHW to `[B,N,C]`, packed `qkv = Linear(C -> 3C)`, MHA/SDPA or eager attention, output reshape back to NCHW.
- DinoML should not claim full timm_wrapper support from this report alone.

Qwen2 decoder primitives:

- RMSNorm with fp32 variance, residual adds.
- Q/K/V linear projections: `q: H -> n_heads*head_dim`, `k/v: H -> n_kv_heads*head_dim`, all biased.
- RoPE cos/sin generation in fp32 and rotary application to Q/K.
- Causal GQA attention with KV cache update; eager fallback repeats KV heads before matmul.
- Output projection `Linear(n_heads*head_dim -> H, bias=False)`.
- SwiGLU MLP: `down_proj(silu(gate_proj(x)) * up_proj(x))`.
- Final RMSNorm and LM head.

Generation/cache ops:

- `DynamicCache` creation when `use_cache=True`.
- Per-layer cache update after RoPE, before attention backend.
- Position id generation from cached sequence length.
- Causal mask construction; sliding-window mask only if future configs enable it.
- Generation input preparation that forwards image pixels only on first iteration or when cache is disabled.

## 5. Layer/block breakdown

FastVLM multimodal bridge:

```text
pixel_values: [B_img, 3, 1024, 1024]
vision_last_hidden_state = timm_fastvit(pixel_values)        # public configs imply [B_img, 3072, 16, 16]
image_seq = flatten(vision_last_hidden_state, start_dim=2)   # [B_img, 3072, 256]
image_seq = permute(image_seq, [0, 2, 1])                    # [B_img, 256, 3072]
image_emb = linear_2(gelu(linear_1(image_seq)))              # [B_img, 256, H]
inputs_embeds = embed_tokens(input_ids)                      # [B, S, H]
inputs_embeds = masked_scatter(image_placeholder_mask, image_emb)
```

Qwen2 decoder block, repeated N times:

```text
residual = x
x = RMSNorm(x)
q = Linear(H -> Q_heads*D, bias=True).view(B,S,Q_heads,D).transpose(1,2)
k = Linear(H -> KV_heads*D, bias=True).view(B,S,KV_heads,D).transpose(1,2)
v = Linear(H -> KV_heads*D, bias=True).view(B,S,KV_heads,D).transpose(1,2)
q,k = RoPE(q,k, position_ids)
k,v = cache.update(k,v, layer_idx) if cache enabled
attn = causal_attention(q,k,v, mask, scale=D^-0.5, optional sliding_window)
x = residual + Linear(Q_heads*D -> H, bias=False)(attn)
residual = x
x = RMSNorm(x)
x = residual + down_proj(silu(gate_proj(x)) * up_proj(x))
```

FastViT `fastvit_mci3` external body, public allowlist:

```text
stem: 3 MobileOne conv blocks, first two stride 2
stages: depths (2,12,24,4,2), dims (96,192,384,768,1536)
token mixers: repmixer, repmixer, repmixer, attention, attention
downsample between stages with reparameterized large-kernel depthwise conv + 1x1 conv
positional conv in stages 3 and 4
final_conv: MobileOneBlock 1536 -> 3072, depthwise/group_size=1, SE, GELU
```

## 6. Attention requirements

Text decoder attention:

- Causal self-attention, no cross-attention.
- GQA: 0.5B has 14 query heads / 2 KV heads / head dim 64; 1.5B has 12 / 2 / 128; 7B has 28 / 4 / 128.
- Cached K/V are stored after RoPE. Cache shape follows Qwen2 source `[B, KV_heads, cached_seq, head_dim]`; eager attention repeats to query heads before matmul.
- Masking uses Transformers causal mask utilities; for inspected configs every layer is `full_attention`.
- Source supports SDPA/Flash/Flex attention selection for Qwen2 through `ALL_ATTENTION_FUNCTIONS`, but eager fallback uses fp32 softmax then casts to query dtype.
- Rectangular prefill/decode attention is required: query length can be 1 during decode while key length is prefix plus generated tokens.

Vision attention:

- Only inside timm `fastvit_mci3` stages 3 and 4.
- It is image-map self-attention, not generation cache attention.
- Source path flattens NCHW `[B,C,H,W] -> [B,N,C]`, applies packed QKV linear with split order `[q,k,v]`, then reshapes output back to NCHW.
- FastVLM docs state visual backbone supports eager attention only; first DinoML path can route vision tower to an external allowlisted implementation or require eager parity before fusion.

Packed/varlen support:

- No `cu_seqlens` style metadata in FastVLM source. Image tokens are packed into the ordinary text sequence via placeholder IDs.

## 7. Position encoding and custom math

Qwen2 RoPE:

```python
inv_freq = 1.0 / (rope_theta ** (arange(0, head_dim, 2) / head_dim))
freqs = (inv_freq[None, :, None] @ position_ids[:, None, :]).transpose(1, 2)
emb = cat(freqs, freqs, dim=-1)
cos = cos(emb) * attention_scaling
sin = sin(emb) * attention_scaling
q = q * cos[:, None] + rotate_half(q) * sin[:, None]
k = k * cos[:, None] + rotate_half(k) * sin[:, None]
```

Inspected checkpoints use default RoPE with `rope_theta=1000000.0` and no rope scaling. Cos/sin depend on runtime `position_ids` and cached sequence length but can be cached per batch/position range. Future configs with non-default `rope_parameters` must route through the relevant Qwen2/RoPE audit.

Vision positional encoding:

- FastViT `fastvit_mci3` uses `RepConditionalPosEnc` in stages 3 and 4. In inference mode it is a depthwise Conv2d with 7x7 kernel and padding 3. In train-time form it is `pos_enc(x) + x`, reparameterizable into one conv.

## 8. Preprocessing and input packing

CPU/data-pipeline work:

- Chat template should emit `<image>` token(s), plus Qwen-style `<|im_start|>` / `<|im_end|>` tokens.
- Image processor emits NCHW float pixel values after resize/crop/rescale. First DinoML graph can assume preprocessed `pixel_values` and `input_ids` are provided.
- Tokenizer config: `Qwen2Tokenizer`, pad/eos token `<|endoftext|>` id 151643, image token id 151646, model max length 32768 in tokenizer metadata.

GPU/runtime work:

- Check placeholder count equals `num_images * image_seq_length`; source uses a compilable check comparing masked embedding element count with image feature element count.
- Stitch is order-sensitive: image embeddings are flattened in image batch order and copied into placeholder positions in row-major sequence order.
- During generation, `prepare_inputs_for_generation` includes `pixel_values` only on first iteration or when `use_cache=False`. Cached decode must not rerun vision/projector.

Multi-image:

- The code can accept multiple images by concatenating image features along dim 0, but model docs say multi-image prompts were not explicitly trained. Gate it separately from single-image parity.

## 9. Graph rewrite / lowering opportunities

### Rewrite: FastVLM projector to two GEMMs

Source pattern:

```text
Linear(3072 -> H, bias) -> GELU -> Linear(H -> H, bias)
```

Replacement: two row-major GEMMs with bias, with a GELU activation between them.

Preconditions: image feature tensor is contiguous or has a supported `[B,N,C]` stride; projector activation is exactly `gelu`; projector biases enabled as in public configs.

Shape equations: `M = B_img * image_seq_length`, `K0=3072`, `N0=H`, `K1=H`, `N1=H`.

Failure cases: non-`gelu` activation, missing/extra projector layers, different vision hidden size.

Parity sketch: compare projector output for random `[B,256,3072]` bf16/fp32 input against PyTorch.

### Rewrite: image placeholder stitch to indexed copy

Source pattern: boolean mask expansion to `[B,S,H]` then `masked_scatter`.

Replacement: gather image placeholder flat indices, validate count, copy image embeddings into `inputs_embeds` rows.

Preconditions: placeholder mask derives from `input_ids == image_token_id`; image feature count equals placeholder count; no duplicate/output alias issue beyond intended overwrite.

Shape equations: `num_placeholders == B_img * image_seq_length`; copy rows of width `H`.

Failure cases: `inputs_embeds` path with no `input_ids` uses embedding equality to find placeholder token and should be rejected or separately implemented.

Parity sketch: randomized prompts with image tokens at beginning/middle/end, batched left padding, count mismatch error.

### Rewrite: Qwen2 separate Q/K/V projections to packed projection

Source pattern: biased `q_proj`, `k_proj`, `v_proj`, then view/transpose.

Replacement: one packed GEMM with output layout `[Q_all, K_all, V_all]`, then split into Q/K/V.

Preconditions: weights packed in Q, K, V block order; all three projections share input `H`; preserve biases; attention backend accepts GQA KV head count.

Weight transform:

```python
packed_w = cat([q_proj.weight, k_proj.weight, v_proj.weight], dim=0)
packed_b = cat([q_proj.bias, k_proj.bias, v_proj.bias], dim=0)
```

Failure cases: tensor-parallel sharded weights, quantized formats with nonuniform metadata, explicit `head_dim` changing output widths.

### Rewrite: FastViT inference reparameterized convs stay Conv2d

Source pattern: `inference_mode=True` timm `MobileOneBlock`, `ReparamLargeKernelConv`, `RepMixer`, `RepConditionalPosEnc` expose single Conv2d modules.

Replacement: lower as Conv2d/DepthwiseConv2d + activation/SE instead of reconstructing training branches.

Preconditions: checkpoint config has `model_args.inference_mode=true`; state dict keys target `reparam_conv`; exact timm version/body allowlisted.

Failure cases: `inference_mode=false`, custom timm architecture, branch-form weights.

### Rewrite: guarded NCHW to NHWC vision region

Source pattern: all FastViT conv-heavy map ops operate on NCHW, with attention blocks flattening NCHW and returning NCHW.

Replacement: optional local channel-last execution for the entire vision tower plus explicit bridge back to `[B,N,C]`.

Preconditions: whole timm `fastvit_mci3` region is owned by the layout pass; Conv2d, depthwise conv, LayerNorm2d, SE reductions, attention flatten, and final `flatten(2).permute(0,2,1)` are all rewritten together.

Required axis rewrites: NCHW `[0,1,2,3]` becomes NHWC `[0,2,3,1]`; `flatten(2)` over H/W becomes flatten axes 1/2; channel reductions/norms must target C last; attention restore becomes `[B,H,W,C]` not `[B,C,H,W]`.

Failure cases: partial translation across `timm_wrapper`, unknown timm body, `forward_intermediates`, output_hidden_states, or consumers expecting NCHW.

## 10. Kernel fusion candidates

Highest priority:

- Qwen2 RMSNorm: every decoder block plus final norm; fp32 variance and bf16 output matter.
- GQA FlashAttention/SDPA with RoPE and KV cache: dominant prefill/decode cost.
- SwiGLU MLP fusion: `silu(gate) * up -> down` is repeated in every block and maps to two GEMMs plus fused activation multiply.
- Projector GEMM+GELU+GEMM: small but on the image-prefix critical path before TTFT.
- Placeholder indexed copy: avoid materializing a `[B,S,H]` expanded boolean mask.

Medium priority:

- QKV packed projection for Qwen2 with biased Q/K/V.
- Last-token-only logits through `logits_to_keep=1`.
- FastViT reparameterized depthwise/pointwise conv chains, especially stem/downsample/final conv.
- FastViT NCHW flatten-attention-restore fusion in the two attention stages.

Lower priority:

- Vision NHWC/channel-last optimized path after exact timm body allowlist.
- Multi-image packing optimizations.
- Training loss, dropout, gradient checkpointing, and hidden-state/attention outputs.

## 11. Runtime staging plan

1. Parse FastVLM config and gate to public-style `fast_vlm + timm_wrapper fastvit_mci3 + qwen2`.
2. Load text weights and run Qwen2 text-only prefill/decode parity with KV cache.
3. Implement projector and placeholder stitch using synthetic image features.
4. Compose precomputed vision features + text decoder for multimodal prefill parity.
5. Add external/allowlisted FastViT vision path or a separate DinoML FastViT audit/import path.
6. Enable first end-to-end single-image generation: preprocessed pixels, one `<image>` block, cached decode.
7. Add optimized attention, projector fusions, indexed stitch, and last-token logits.
8. Add guarded FastViT graph rewrites/layout work only after exact timm parity is validated.

Stub initially: image preprocessing, timm vision body, multi-image prompts, non-default vision feature selection, training loss.

## 12. Parity and validation plan

- Config parser tests for all three snapshots, including source-default mismatch on `image_seq_length`.
- Projector random tensor parity for `[1,256,3072]` and batched image counts in fp32 and bf16.
- Placeholder stitch parity for single image, batched left padding, and mismatch error.
- Qwen2 single-layer parity for RMSNorm, RoPE, GQA attention, cache update, and SwiGLU.
- Full Qwen2 text-only prefill logits and one-token decode parity.
- Multimodal parity with precomputed image features, then with vision output if owned.
- End-to-end single-image output smoke against `KamilaMila/FastVLM-0.5B` after preprocessing is composed.
- Tolerances: fp32 close to `1e-4` for isolated ops; bf16 use relaxed absolute/relative tolerances around `1e-2` for full graph, tighter for deterministic projector-only tests.

## 13. Performance probes

- Image preprocessing throughput: resize/crop/rescale/tokenizer separately from GPU graph.
- Vision encoder latency for `1024x1024` batch sizes 1, 2, 4.
- Projector latency for `M=B_img*256`, hidden sizes 896/1536/3584.
- Prefill latency split by text length and image-prefix length.
- Decode tokens/sec with and without image prefix cached.
- KV cache memory by variant and sequence length.
- LM-head cost with full logits vs `logits_to_keep=1`.
- Placeholder stitch cost: boolean masked_scatter vs indexed copy.
- Attention backend comparison: eager, SDPA, FlashAttention for decoder only.
- Vision NCHW vs guarded NHWC/channel-last prototype after exact FastViT parity.

## 14. Skip/defer list

- Training, loss parity, dropout behavior, gradient checkpointing.
- Arbitrary `timm_wrapper` architectures.
- `vision_feature_layer` other than `-1` and `vision_feature_select_strategy` other than `full`.
- Multi-image quality parity beyond source shape correctness.
- Vision `output_hidden_states` / `forward_intermediates`.
- Global FlashAttention setting that includes vision tower.
- Sliding-window Qwen2 configs not present in inspected FastVLM checkpoints.
- Quantized or packed weight formats beyond normal dense HF weights.
- Multi-GPU tensor parallel and continuous batching.

## 15. Final implementation checklist

- [ ] Parse `FastVlmConfig` and nested `text_config` / `vision_config`.
- [ ] Gate first target to `vision_config.model_type=timm_wrapper`, `architecture=fastvit_mci3`, `model_args.inference_mode=true`, Qwen2 text config, `image_seq_length=256`.
- [ ] Load/alias token embeddings and LM head according to checkpoint tie metadata.
- [ ] Implement Qwen2 RMSNorm, RoPE, GQA attention, KV cache, SwiGLU, and logits slicing.
- [ ] Implement FastVLM projector `3072 -> H -> H`.
- [ ] Implement image placeholder count validation and indexed embedding stitch.
- [ ] Add precomputed-image-feature multimodal prefill parity.
- [ ] Audit/allowlist timm FastViT `fastvit_mci3` or route vision tower to fallback.
- [ ] Add single-image end-to-end generation parity for 0.5B.
- [ ] Add performance probes for vision, projector, prefill, decode, LM head, and stitch.
- [ ] Add guarded NHWC/channel-last rewrite only after FastViT region is fully owned.
