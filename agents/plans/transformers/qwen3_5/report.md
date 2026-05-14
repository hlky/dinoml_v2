# Qwen3.5 Transformers Family Audit

## 1. Source basis

```text
Transformers commit/version: b75feb2af64c3e29cbbc1bd859958c5432cc7ed4
Model id: Qwen/Qwen3.5-0.8B, Qwen/Qwen3.5-4B, Qwen/Qwen3.5-9B, Qwen/Qwen3.5-27B, Qwen/Qwen3.5-27B-FP8
Config source: Hugging Face raw config.json and preprocessor_config.json fetched May 13, 2026
Source files inspected: qwen3_5/configuration_qwen3_5.py, modeling_qwen3_5.py, modular_qwen3_5.py, tokenization_qwen3_5.py, cache_utils.py, qwen3_vl/processing_qwen3_vl.py, qwen3_vl/video_processing_qwen3_vl.py
Any missing files or assumptions: no qwen3_5-specific processor file; checkpoints point to Qwen3VLProcessor and Qwen2VLImageProcessorFast. processor_config.json returned 404 for checked repos; preprocessor_config.json was available.
```

Primary target for DinoML: multimodal conditional generation, with a useful first subtarget of text-only causal LM over the same hybrid text decoder. The source is generated from `modular_qwen3_5.py`; future upstream edits should inspect modular source first, but runtime parity should follow the generated `modeling_qwen3_5.py`.

Pinned source URLs:

- `https://github.com/huggingface/transformers/blob/b75feb2af64c3e29cbbc1bd859958c5432cc7ed4/src/transformers/models/qwen3_5/modeling_qwen3_5.py`
- `https://github.com/huggingface/transformers/blob/b75feb2af64c3e29cbbc1bd859958c5432cc7ed4/src/transformers/models/qwen3_5/configuration_qwen3_5.py`
- `https://github.com/huggingface/transformers/blob/b75feb2af64c3e29cbbc1bd859958c5432cc7ed4/src/transformers/models/qwen3_5/modular_qwen3_5.py`

## 2. High-level architecture

Qwen3.5 is a multimodal image/video-to-text decoder family:

```text
text/image/video preprocessing -> optional vision encoder/projector -> placeholder embedding stitch -> hybrid text decoder prefill/decode -> logits/sampling
```

Stages:

- CPU/data pipeline: tokenizer, chat template, image/video resize, rescale, normalize, patch packing, placeholder expansion, `mm_token_type_ids`, and `grid_thw`.
- Vision branch: Conv3d patch embedding over prepacked flattened patches, learned/interpolated absolute patch position embedding, noncausal packed vision self-attention, MLP, spatial merge projector to text hidden width.
- Prefix construction: replace image/video placeholder token embeddings with vision features using source `masked_scatter`; this can be guarded down to indexed row copy because the processor controls token counts.
- Text decoder: hybrid repeated layers, where `linear_attention` layers use Gated DeltaNet recurrent state and `full_attention` layers use causal GQA KV cache.
- Decode: first iteration consumes vision tensors; subsequent cached generation drops `pixel_values` and `pixel_values_videos`.

## 3. Important config dimensions

Source defaults from `Qwen3_5TextConfig` and `Qwen3_5VisionConfig`:

| Field | Default |
| --- | --- |
| text hidden size | 4096 |
| text layers | 32 |
| full attention interval | 4, if `layer_types` omitted |
| attention heads / KV heads / head dim | 16 / 4 / 256 |
| full-attention Q content width | `num_attention_heads * head_dim` = 4096 default |
| full-attention Q projection output | `2 * num_attention_heads * head_dim`; second half is sigmoid gate |
| full-attention K/V widths | `num_key_value_heads * head_dim` each |
| MLP intermediate | 12288 |
| activation | `silu` text, `gelu_pytorch_tanh` vision |
| max position embeddings | 32768 source default; 262144 in checked checkpoints |
| RoPE | `rope_parameters`, partial rotary defaulted to 0.25 in `__post_init__` |
| linear-attention heads | key heads 16, value heads 32 default |
| linear-attention dims | key head 128, value head 128, conv kernel 4 |
| vision depth/width | 27 layers, hidden 1152, 16 heads default |
| vision patching | temporal patch 2, spatial patch 16, spatial merge 2 |
| vocab | 248320 |
| cache | `use_cache=True`; mixed dynamic KV and linear-attention states |

Representative checkpoint sweep, from HF `config.json`:

| Repo | Text hidden | Layers | Full layers | Heads/KV/head_dim | Full Q content width | Linear key/value width | Vision depth/hidden/out | Tied embeddings | Notes |
| --- | ---: | ---: | ---: | --- | ---: | --- | --- | --- | --- |
| Qwen3.5-0.8B | 1024 | 24 | 6 | 8/2/256 | 2048 | 2048/2048 | 12/768/1024 | true | `heads * head_dim` is 2x hidden |
| Qwen3.5-4B | 2560 | 32 | 8 | 16/4/256 | 4096 | 2048/4096 | 24/1024/2560 | true | full-attn out projection is 4096 -> 2560 |
| Qwen3.5-9B | 4096 | 32 | 8 | 16/4/256 | 4096 | 2048/4096 | 27/1152/4096 | false | source-default-like text size |
| Qwen3.5-27B | 5120 | 64 | 16 | 24/4/256 | 6144 | 2048/6144 | 27/1152/5120 | false | GQA repeat factor 6 |
| Qwen3.5-27B-FP8 | 5120 | 64 | 16 | 24/4/256 | 6144 | 2048/6144 | 27/1152/5120 | false | checkpoint quantization metadata |

All checked configs use `rope_theta=10000000`, `partial_rotary_factor=0.25`, `mrope_section=[11,11,10]`, `mrope_interleaved=true`, `linear_conv_kernel_dim=4`, `attention_bias=false`, `attention_dropout=0.0`, and text dtype metadata `bfloat16`.

## 3a. Family variation traps

- Do not infer projection width from `hidden_size`: 0.8B, 4B, and 27B have `num_attention_heads * head_dim != hidden_size`.
- Full attention `q_proj` is not standard Q only; it emits query plus an output gate and must be split on the last dim after reshaping to `[..., heads, head_dim * 2]`.
- Text decoder is hybrid by default. A KV-cache-only implementation is wrong for most layers.
- Linear-attention cache state is fixed-size conv plus recurrent matrix state; it does not grow with sequence length.
- `layer_types` is config-significant. Source default derives it from `full_attention_interval=4` only if absent.
- Full attention is GQA: `num_key_value_heads < num_attention_heads` in checked configs.
- RMSNorm weights are zero-initialized offsets used as `(1 + weight)`, not direct gamma.
- Text RoPE is partial and M-RoPE aware; for multimodal input, text/temporal/height/width position IDs have separate semantics.
- `attn_output_gate`, `mlp_only_layers`, `mtp_*`, and `mamba_ssm_dtype` appear in checked configs but are not read by this generated source for the main forward path; treat as ignored metadata for this source basis unless another loader consumes them.
- FP8 checkpoint quantization is a loading/provider contract. The modeling source does not implement FP8 dequant kernels.
- Vision tensors are prepacked flattened patches, not ordinary `[B,C,H,W]` tensors by the time they enter `Qwen3_5VisionModel`.
- Video prompts are timestamp-expanded by the Qwen3-VL processor; Qwen3.5 removes Qwen2-VL `second_per_grid_ts` from generation expansion.

## 4. Operator coverage checklist

Tensor/layout ops:

- Embedding lookup, reshape/view, transpose/permute, chunk/split, concat, repeat, repeat_interleave, expand, cumsum, pad, roll, index/select, boolean equality masks, `masked_scatter` or guarded indexed row copy.
- Layout-sensitive vision patch packing from processor: source packed feature dim is `C * temporal_patch_size * patch_size * patch_size` = `3 * 2 * 16 * 16 = 1536`.

Neural primitives:

- Linear projections with explicit widths:
  - full attention `q_proj: hidden -> 2 * num_attention_heads * head_dim`
  - full attention `k_proj/v_proj: hidden -> num_key_value_heads * head_dim`
  - full attention `o_proj: num_attention_heads * head_dim -> hidden`
  - MLP `gate_proj/up_proj: hidden -> intermediate`, `down_proj: intermediate -> hidden`
  - linear attention `in_proj_qkv: hidden -> 2 * key_dim + value_dim`
  - linear attention `in_proj_z: hidden -> value_dim`, `in_proj_a/b: hidden -> num_value_heads`, `out_proj: value_dim -> hidden`
- RMSNorm over hidden width and over per-head `head_dim`.
- Gated RMSNorm: RMSNorm over `head_v_dim`, then multiply by `silu(z)`.
- SiLU, GELU, sigmoid, softplus, exp, rsqrt, cumsum.
- Depthwise causal Conv1d for linear-attention projected QKV, groups equal channels, kernel 4.
- Vision Conv3d patch embedding with kernel/stride `[2,16,16]`, input channels 3, bias true.
- Vision LayerNorm and MLP with bias.

Attention primitives:

- Full text causal GQA with RoPE-applied Q/K, KV cache update after RoPE.
- Vision noncausal packed self-attention with `cu_seqlens` for FlashAttention path or per-sample splits for other backends.
- Linear-attention Gated DeltaNet chunk and recurrent update kernels; not reducible to softmax attention.

Position/rotary ops:

- Partial RoPE over the first `head_dim * partial_rotary_factor` channels.
- M-RoPE interleaving over temporal/height/width frequency sections.
- Vision rotary position table indexed by per-patch row/column coordinates.
- Vision learned absolute position interpolation from a `sqrt(num_position_embeddings)` grid.

Generation/cache ops:

- Dynamic KV cache for full-attention layers.
- Linear-attention `conv_states` and `recurrent_states`, layer-owned and beam-reorderable.
- `rope_deltas` side state in multimodal wrapper.
- First-step vision feature computation and subsequent decode skipping of vision tensors.

Quantized/packed weight metadata:

- FP8 config has `quant_method=fp8`, dynamic activation scheme, `weight_block_size=[128,128]`, and `modules_to_not_convert`. DinoML should treat this as a separate provider/loading admission path with dense fallback.

## 5. Layer/block breakdown

Text decoder block, repeated `num_hidden_layers`:

```text
residual = x
x = RMSNorm(hidden)(x) with scale (1 + weight)
if layer_type == "linear_attention":
  mixed = Linear(hidden -> 2*key_dim + value_dim)(x)
  mixed = depthwise causal Conv1d(kernel=4, groups=conv_dim)(mixed)
  q,k,v = split(mixed, [key_dim, key_dim, value_dim])
  z = Linear(hidden -> value_dim)(x)
  beta = sigmoid(Linear(hidden -> num_value_heads)(x))
  g = -exp(A_log.float()) * softplus(Linear(hidden -> num_value_heads)(x).float() + dt_bias)
  q,k = reshape to key heads, repeat to value heads if needed
  y,state = gated_delta_rule(q,k,v,g,beta,state)
  y = RMSNormGated(head_v_dim)(y,z)
  y = Linear(value_dim -> hidden)(y)
else layer_type == "full_attention":
  q_and_gate = Linear(hidden -> 2*num_heads*head_dim)(x)
  q, gate = split(q_and_gate)
  q = RMSNorm(head_dim)(q)
  k = RMSNorm(head_dim)(Linear(hidden -> num_kv_heads*head_dim)(x))
  v = Linear(hidden -> num_kv_heads*head_dim)(x)
  q,k = partial M-RoPE(q,k)
  k,v = cache.update(k,v,layer_idx)
  y = causal GQA(q,k,v,mask)
  y = Linear(num_heads*head_dim -> hidden)(y * sigmoid(gate))
x = residual + y
residual = x
x = RMSNorm(hidden)(x)
x = Linear(intermediate -> hidden)(silu(gate_proj(x)) * up_proj(x))
x = residual + x
```

Vision block, repeated `vision_config.depth`:

```text
tokens = Conv3dPatchEmbed(prepacked patches) + interpolated absolute pos
for each vision block:
  tokens = tokens + NoncausalPackedAttention(LayerNorm(tokens), cu_seqlens, 2D RoPE)
  tokens = tokens + MLP(LayerNorm(tokens))
merged = LayerNorm/reshape spatial_merge^2 groups -> Linear -> GELU -> Linear(out_hidden_size)
```

## 6. Attention requirements

Full text attention:

- Causal self-attention only; no cross-attention in decoder.
- GQA with `num_key_value_groups = num_attention_heads // num_key_value_heads`.
- Q width and output-attention width are `num_attention_heads * head_dim`; K/V cache width is `num_key_value_heads * head_dim`.
- Cached keys are stored after Q/K RMSNorm and RoPE; values are stored unrotated.
- Eager fallback repeats K/V before matmul, adds mask, softmaxes in fp32, casts to query dtype, then matmuls V.
- Source dispatch can use FlashAttention or SDPA through `ALL_ATTENTION_FUNCTIONS`.
- Masks are created by `create_causal_mask` using text position IDs.

Linear attention:

- Gated DeltaNet recurrent linear attention with causal depthwise Conv1d over projected Q/K/V.
- Prefill/chunk path uses a chunked gated delta rule with chunk size 64 in the torch fallback.
- Single-token cached decode uses recurrent gated delta rule and in-place conv-state update.
- Q/K L2 normalization is requested inside the gated-delta kernels.
- Padding mask for linear-attention layers is a 2D mask applied by multiplying hidden states before projections; when all tokens attend or cache has previous state, source passes `None`.

Cache manifest:

- Full-attention layer: KV tensors `[B, num_key_value_heads, cache_len, head_dim]`.
- Linear-attention layer conv state: `[B, 2*key_dim + value_dim, linear_conv_kernel_dim]`.
- Linear-attention recurrent state: `[B, linear_num_value_heads, linear_key_head_dim, linear_value_head_dim]`, effectively float32 in torch fallback because the recurrence converts Q/K/V/beta/g to float32.
- Cache reorder indexes batch dimension for both KV and linear states.
- `get_seq_length()` for a hybrid cache resolves to the first full-attention layer; linear layers do not track sequence length.

## 7. Position encoding and custom math

Text RoPE:

```python
rotary_dim = int(head_dim * partial_rotary_factor)
inv_freq = 1.0 / (rope_theta ** (arange(0, rotary_dim, 2) / rotary_dim))
freqs = inv_freq @ position_ids[t_or_h_or_w]
freqs = interleave_mrope_sections(freqs, mrope_section=[11, 11, 10])
cos = cat(freqs, freqs).cos()
sin = cat(freqs, freqs).sin()
q_rot, q_pass = q[..., :rotary_dim], q[..., rotary_dim:]
q = cat(q_rot * cos + rotate_half(q_rot) * sin, q_pass)
```

For plain text, position IDs are expanded to four planes; the first plane feeds causal-mask text positions, while the remaining three planes are used by the rotary module. For multimodal inputs, `get_rope_index` builds temporal/height/width position IDs from `mm_token_type_ids` and `image_grid_thw`/`video_grid_thw`, and caches `rope_deltas` for decode.

Vision RoPE and absolute position:

- Vision rotary uses row/column coordinate IDs derived from `grid_thw` after spatial merge ordering.
- Absolute position uses bilinear interpolation over a learned square table of size `num_position_embeddings` before being permuted into merge-order token layout.

## 8. Preprocessing and input packing

Processor contract:

- Checkpoint preprocessor configs identify `Qwen3VLProcessor` and `Qwen2VLImageProcessorFast`.
- Image/video preprocessing resizes to dimensions divisible by `patch_size * merge_size`, rescales by `1/255`, normalizes with mean/std `[0.5,0.5,0.5]`, and emits flattened patches.
- Model input `pixel_values`/`pixel_values_videos` is rank 2 or rank 3 packed patch data, not raw NHWC/NCHW images. The model then views each row as `[-1, C, temporal_patch_size, patch_size, patch_size]` before Conv3d.
- `image_grid_thw` and `video_grid_thw` are `[num_items, 3]` with grid values after temporal/spatial patching.
- Placeholder expansion count is `grid_thw.prod() // merge_size**2`.
- `mm_token_type_ids`: text 0, image 1, video 2. Qwen3.5 requires these for multimodal M-RoPE if `input_ids` are present.
- Video processor samples frames, pads frame count to a multiple of temporal patch size, and the prompt expansion inserts timestamps per temporal patch group.

Scatter/indexed update:

- Source uses `inputs_embeds.masked_scatter(image_mask, image_embeds)` and same for videos.
- Safe DinoML lowering can be `checked_indexed_row_copy` if guards verify placeholder count, feature count, row-major flatten order, and processor-controlled contiguous expansion. If callers provide arbitrary `inputs_embeds` without `input_ids`, the equality-to-special-embedding path is less robust and should be rejected initially.

## 9. Graph rewrite / lowering opportunities

### Rewrite: multimodal masked_scatter -> checked indexed row copy

Source pattern: boolean mask over `input_ids == image_token_id/video_token_id`, expanded over hidden dim, followed by `masked_scatter`.

Replacement: gather placeholder row indices, verify `len(indices) == features.shape[0]`, copy feature rows into embedding buffer.

Preconditions: `input_ids` present, processor-created placeholders, features are concatenated in processor order, no arbitrary user-supplied `inputs_embeds` matching special embeddings.

Failure cases: mismatched feature/token count, nonstandard processor, hand-edited multimodal prompts.

Parity test: one image and one video prompt with known fake feature rows; compare final `inputs_embeds`.

### Rewrite: processor patch packing + Conv3d patch embed -> Linear

Source pattern: processor already packs each tubelet into a row of length `C*T*P*P`, model views it as `[C,T,P,P]`, then applies Conv3d with kernel/stride equal to `[T,P,P]`.

Replacement: `Linear(1536 -> vision_hidden)` using flattened Conv3d weights plus bias.

Preconditions: `temporal_patch_size=2`, `patch_size=16`, `stride=kernel`, groups 1, no padding/dilation, packed row order matches processor `permute(0,1,4,7,5,8,3,2,6,9)`.

Failure cases: alternate image processor, changed patch sizes, raw image tensors passed directly.

### Rewrite: full-attention q_proj split/gate canonicalization

Source pattern: `Linear(hidden -> 2*Qwidth) -> view(..., heads, 2*head_dim) -> chunk into query and gate`.

Replacement: single GEMM with two output views, or two logical projections sharing one packed weight.

Preconditions: split order is `[query, gate]` along the last per-head dimension; gate is applied after attention output reshape and before `o_proj`.

Failure cases: treating q projection as `hidden -> Qwidth` and dropping gate.

### Rewrite: linear-attention QKV conv bundle

Source pattern: `in_proj_qkv -> transpose -> depthwise causal Conv1d -> split q,k,v`.

Replacement: provider kernel for projected depthwise causal conv plus split, preserving conv-state update.

Preconditions: kernel 4, groups equal channels, bias false in Conv1d, activation SiLU.

Failure cases: trying to lower as softmax attention; ignoring cached conv context in chunked decode.

## 10. Kernel fusion candidates

Highest priority:

- RMSNorm with `(1 + weight)` scale, including per-head Q/K RMSNorm.
- Gated DeltaNet fused kernels for chunk prefill and recurrent decode; this is the central nonstandard runtime requirement.
- Full-attention GQA FlashAttention with partial RoPE, KV cache, and output gate.
- SwiGLU MLP GEMM fusion.
- Last-token-only logits using `logits_to_keep`.

Medium priority:

- Q projection plus gate split, Q/K RMSNorm, partial RoPE preparation.
- Depthwise causal Conv1d update for linear attention.
- Vision packed noncausal attention with `cu_seqlens`.
- Vision Conv3d-patch-as-linear when processor packing is controlled.
- Placeholder indexed row copy.

Lower priority:

- Vision absolute position interpolation on GPU.
- Beam expand/reorder for visual tensors and mixed cache.
- FP8 block weight provider path.
- Sequence classification/token classification heads.

## 11. Runtime staging plan

Stage 1: parse configs and reject unsupported fields explicitly. Admit text-only configs first, with `layer_types` manifest and explicit projection widths.

Stage 2: one text decoder block parity for both `linear_attention` and `full_attention` layer types without cache.

Stage 3: prefill parity for the full hybrid text decoder, using torch-like fallback math for Gated DeltaNet.

Stage 4: decode parity with mixed cache ABI: KV for full-attention layers, conv/recurrent states for linear layers.

Stage 5: add multimodal prefix path with processor-owned packed `pixel_values`, vision encoder, M-RoPE, and checked indexed row copy.

Stage 6: optimized providers: Gated DeltaNet kernels, GQA FlashAttention, vision packed attention, Conv3d-to-linear patch rewrite.

Stage 7: checkpoint quantization/provider work, beginning with dense fallback for FP8 configs and explicit rejection when the provider is unavailable.

## 12. Parity and validation plan

- Unit parity for RMSNorm `(1 + weight)`, Gated RMSNorm, partial RoPE, M-RoPE interleaving, and `get_rope_index`.
- Random tensor parity for Gated DeltaNet no-cache prefill, cached single-token decode, and chunked cached decode.
- Full-attention layer parity with rectangular cache lengths and GQA repeat factors 2/4/6.
- Single-block parity for linear and full layers at fp32, bf16, and fp16 where source supports it.
- Whole text prefill logits parity for 0.8B-style small config with random weights.
- Decode token parity across two or more generated steps, validating cache state mutation and reorder.
- Vision encoder parity with synthetic packed patches and small `grid_thw`.
- Multimodal prefix parity checking placeholder replacement and M-RoPE positions.
- Suggested tolerances: fp32 `1e-5` absolute/relative for primitive tests; bf16/fp16 `2e-2` for whole-block tests, tighter for fp32 reference-only subgraphs.

## 13. Performance probes

- Text prefill tokens/sec split by linear-attention layers versus full-attention layers.
- Decode tokens/sec with state update cost separated into KV attention and Gated DeltaNet recurrence.
- KV cache memory versus linear recurrent-state memory by batch and sequence length.
- Gated DeltaNet chunk size sensitivity, including chunk 64 source fallback behavior.
- Full-attention backend comparison: eager, SDPA, FlashAttention.
- Vision encoder throughput by total packed patch count and `cu_seqlens` distribution.
- Placeholder copy throughput for image/video feature insertion.
- Last-token logits versus full logits.
- FP8 load/dequant/provider benchmark versus dense bf16 fallback, if FP8 admission is added.

## 14. Skip/defer list

- Training, loss, gradient checkpointing, and output attentions.
- Beam search initially, except cache reorder unit tests before generation support claims.
- Sequence/token classification heads.
- Arbitrary `inputs_embeds` multimodal matching by equality to special-token embeddings.
- FP8 checkpoint execution until DinoML has an explicit block-FP8 provider/loading contract.
- Alternative RoPE types beyond default unless configs requiring them are admitted.
- Multi-GPU tensor parallel and pipeline plans.
- Video decode itself; first stage can require already decoded frames or processor-produced tensors.

## 15. Final implementation checklist

- [ ] Parse `Qwen3_5Config`, `Qwen3_5TextConfig`, and `Qwen3_5VisionConfig`.
- [ ] Preserve explicit `head_dim`, Q width, KV width, linear key/value widths, and `layer_types`.
- [ ] Load dense weights with tied embedding/LM-head aliasing when `tie_word_embeddings=true`.
- [ ] Implement RMSNorm with `(1 + weight)` scale.
- [ ] Implement full causal GQA layer with q gate, Q/K RMSNorm, partial M-RoPE, and KV cache.
- [ ] Implement Gated DeltaNet prefill and decode with conv/recurrent state ABI.
- [ ] Add cache manifest for mixed layer types.
- [ ] Implement M-RoPE position ID construction and `rope_deltas`.
- [ ] Implement text-only prefill/decode parity tests.
- [ ] Implement processor-owned vision packed input contract.
- [ ] Implement vision patch embedding, packed noncausal attention, MLP, and merger.
- [ ] Lower multimodal `masked_scatter` to checked indexed row copy.
- [ ] Add multimodal prefill parity with fake/small vision inputs.
- [ ] Add performance probes for Gated DeltaNet, full GQA, vision attention, and logits slicing.
- [ ] Gate FP8 configs behind explicit provider/loading support or dense fallback.
