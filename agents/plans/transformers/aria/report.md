# Aria Transformers Audit

## 1. Source basis

```text
Transformers commit/version: b75feb2af64c3e29cbbc1bd859958c5432cc7ed4
Model id: rhymes-ai/Aria, plus representative public variants listed below
Config source: HF config/preprocessor/tokenizer/generation files saved under _sources/
Source files inspected:
- X:/H/transformers/src/transformers/models/aria/configuration_aria.py
- X:/H/transformers/src/transformers/models/aria/modeling_aria.py
- X:/H/transformers/src/transformers/models/aria/modular_aria.py
- X:/H/transformers/src/transformers/models/aria/processing_aria.py
- X:/H/transformers/src/transformers/models/aria/image_processing_aria.py
- X:/H/transformers/src/transformers/models/aria/image_processing_pil_aria.py
- X:/H/transformers/src/transformers/models/idefics3/configuration_idefics3.py
- X:/H/transformers/src/transformers/models/idefics3/modeling_idefics3.py
Any missing files or assumptions: no gated or 401/403 checkpoints encountered; older Aria repos are remote-code-era configs and must be distinguished from the current in-library implementation.
```

Commit-pinned source URLs:

- [configuration_aria.py](https://github.com/huggingface/transformers/blob/b75feb2af64c3e29cbbc1bd859958c5432cc7ed4/src/transformers/models/aria/configuration_aria.py)
- [modeling_aria.py](https://github.com/huggingface/transformers/blob/b75feb2af64c3e29cbbc1bd859958c5432cc7ed4/src/transformers/models/aria/modeling_aria.py)
- [modular_aria.py](https://github.com/huggingface/transformers/blob/b75feb2af64c3e29cbbc1bd859958c5432cc7ed4/src/transformers/models/aria/modular_aria.py)
- [processing_aria.py](https://github.com/huggingface/transformers/blob/b75feb2af64c3e29cbbc1bd859958c5432cc7ed4/src/transformers/models/aria/processing_aria.py)
- [image_processing_aria.py](https://github.com/huggingface/transformers/blob/b75feb2af64c3e29cbbc1bd859958c5432cc7ed4/src/transformers/models/aria/image_processing_aria.py)
- [configuration_idefics3.py](https://github.com/huggingface/transformers/blob/b75feb2af64c3e29cbbc1bd859958c5432cc7ed4/src/transformers/models/idefics3/configuration_idefics3.py)
- [modeling_idefics3.py](https://github.com/huggingface/transformers/blob/b75feb2af64c3e29cbbc1bd859958c5432cc7ed4/src/transformers/models/idefics3/modeling_idefics3.py)

Authoritative source-edit file: `modeling_aria.py`, `configuration_aria.py`, `processing_aria.py`, and `image_processing_aria.py` are generated from `modular_aria.py`; future Transformers source changes should start from `modular_aria.py`.

Representative HF artifacts inspected:

| Repo | Link | Local snapshot | Notes |
|---|---|---|---|
| rhymes-ai/Aria | [HF](https://huggingface.co/rhymes-ai/Aria) | `_sources/rhymes-ai__Aria/` | Current public in-library-style config, `model_type="aria"`, no `auto_map`; `processor_config.json` present. |
| rhymes-ai/Aria-Chat | [HF](https://huggingface.co/rhymes-ai/Aria-Chat) | `_sources/rhymes-ai__Aria-Chat/` | Remote-code-era config with `auto_map`; no `processor_config.json`. |
| rhymes-ai/Aria-Base-8K | [HF](https://huggingface.co/rhymes-ai/Aria-Base-8K) | `_sources/rhymes-ai__Aria-Base-8K/` | Remote-code-era base; lower RoPE theta than 64K/Chat. |
| rhymes-ai/Aria-Base-64K | [HF](https://huggingface.co/rhymes-ai/Aria-Base-64K) | `_sources/rhymes-ai__Aria-Base-64K/` | Remote-code-era base; 64K max position. |
| rhymes-ai/Aria-sequential_mlp | [HF](https://huggingface.co/rhymes-ai/Aria-sequential_mlp) | `_sources/rhymes-ai__Aria-sequential_mlp/` | Remote-code flag `_attn_implementation="flash_attention_2"`; source still uses current attention backend selection if loaded in-library. |
| rhymes-ai/Aria-torchao-int8wo | [HF](https://huggingface.co/rhymes-ai/Aria-torchao-int8wo) | `_sources/rhymes-ai__Aria-torchao-int8wo/` | Custom-code quantized repo with `quant_int8wo.py`; not an in-library weight-loading contract. |

## 2. High-level architecture

Primary runtime target: multimodal image-to-text conditional generation through `AriaForConditionalGeneration`. Text-only `AriaTextForCausalLM` is independently useful and should be staged first for decoder/MoE parity. Video and audio are not present.

```text
CPU image/text preprocessing
  -> AriaImageProcessor: RGB, resize, pad, normalize, pixel_mask
  -> AriaProcessor: expand <|img|> placeholders to 128 or 256 tokens per crop
  -> Idefics3VisionTransformer: Conv2d patch embedding + bidirectional ViT encoder
  -> AriaProjector: learned queries cross-attend over image patches, then MLP to text hidden size
  -> masked_scatter image embeddings into text token embeddings
  -> AriaText MoE causal decoder prefill/decode
  -> LM head logits/sampling
```

Stage decomposition:

| Stage | Runtime/cache contract |
|---|---|
| CPU/data pipeline | Image resize/pad/rescale/normalize and placeholder expansion. This controls the number of image placeholder tokens and must be parity-tested separately. |
| Vision encoder | Accepts NCHW `pixel_values` and patch mask. Outputs patch sequence hidden states; no KV cache. Can be cached per image/crop before text prefill. |
| Projector | Learned-query cross-attention from `query_num` learned tokens to patch features. Output is exactly `128` tokens for 490-sized crop or `256` for 980-sized crop in inspected configs. Can be cached with the vision output. |
| Prefix construction | `masked_scatter` replaces `<|img|>` token embeddings. Placeholder count must exactly equal projected image feature elements. |
| Text prefill/decode | Autoregressive MoE decoder with standard per-layer self-attention KV cache. `prepare_inputs_for_generation` passes pixels only on first generation iteration or when cache is disabled. |

## 3. Important config dimensions

Main `rhymes-ai/Aria` effective dimensions:

| Field | Value | Provenance |
|---|---:|---|
| text hidden size | 2560 | config.json |
| text layers | 28 | config.json |
| text attention heads | 20 | config.json |
| text KV heads | 20 | config.json |
| text head dim | 128 | source default, `hidden_size // num_attention_heads`; config omits `head_dim` |
| text intermediate size | 1664 | config.json; current in-library MoE expert width |
| shared expert multiplier | 2 | source default; main config omits `moe_num_shared_experts` |
| routed experts | 64 | config.json |
| top-k experts | 6 | config.json |
| vocab size | 100352 | config.json |
| max position embeddings | 65536 | config.json |
| RoPE theta | 5000000 | config field `rope_theta`, normalized into default RoPE parameters by config machinery |
| vision hidden size | 1152 | source default for `idefics3_vision`; main config omits `hidden_size` |
| vision layers | 27 | config.json |
| vision heads | 16 | source default or historical config field |
| vision intermediate size | 4304 | config.json |
| vision image size | 980 | config.json |
| vision patch size | 14 | config.json |
| patch count | 4900 for 980x980, 1225 for 490x490 | derived from image size / patch size |
| projector query map | `{1225: 128, 4900: 256}` | config.json |
| image token id | 9, token `<|img|>` | config/tokenizer_config |
| dtype | bfloat16 | config.json |
| cache support | yes | source `DynamicCache` and `use_cache=True` default |

Checkpoint sweep:

| Repo | In-library scope | Text intermediate | Max positions | RoPE theta | Vision config marker | Processor marker |
|---|---|---:|---:|---:|---|---|
| `rhymes-ai/Aria` | Primary current scope | 1664 | 65536 | 5000000 | `idefics3_vision`; `attention_heads` historical key ignored by current config | `AriaImageProcessor`, processor size conversion `{490:128, 980:256}` |
| `rhymes-ai/Aria-Chat` | Needs remote-code boundary review | 13568 plus `moe_intermediate_size=1664` | 8192 | 5000000 | `aria_vision_model` remote-code key coerced to `idefics3_vision` if loaded in-library | `AriaVisionProcessor`, no processor_config |
| `rhymes-ai/Aria-Base-8K` | Needs remote-code boundary review | 13568 plus `moe_intermediate_size=1664` | 65536 | 100000 | `aria_vision_model` | `AriaVisionProcessor`, no processor_config |
| `rhymes-ai/Aria-Base-64K` | Needs remote-code boundary review | 13568 plus `moe_intermediate_size=1664` | 65536 | 5000000 | `aria_vision_model` | `AriaVisionProcessor`, no processor_config |
| `rhymes-ai/Aria-sequential_mlp` | Remote-code/config flag divergence | 13568 plus `moe_intermediate_size=1664` | 65536 | 5000000 | `_attn_implementation=flash_attention_2` | `AriaVisionProcessor`, no processor_config |
| `rhymes-ai/Aria-torchao-int8wo` | Separate quantized custom-code audit | 13568 plus `moe_intermediate_size=1664` | 65536 | 5000000 | custom code files present | custom `quant_int8wo.py`, not in-library |

## 3a. Family variation traps

- Historical configs use `text_config.model_type="aria_moe_lm"` and `vision_config.model_type="aria_vision_model"`. Current `AriaConfig` routes text into `AriaTextConfig` and forcibly sets dict vision configs to `idefics3_vision`; do not treat remote-code class names as in-library runtime requirements without a separate remote-code audit.
- Current source ignores historical `moe_intermediate_size` and `num_experts_per_tok`; the fields that drive the in-library MoE are `intermediate_size`, `moe_num_experts`, `moe_topk`, and `moe_num_shared_experts`.
- `rhymes-ai/Aria` config sets `intermediate_size=1664`, while older remote-code configs set `intermediate_size=13568` and `moe_intermediate_size=1664`. This is operator-significant for expert GEMM shapes.
- No GQA in inspected configs: `num_key_value_heads == num_attention_heads == 20`. Source supports GQA/MQA through `repeat_kv` when `num_key_value_heads < num_attention_heads`.
- Projection biases differ by branch: text attention and text MLP default to no bias; vision attention/MLP and projector `MultiheadAttention`/linear include biases in several places.
- Projector uses PyTorch `nn.MultiheadAttention` plus explicit q/k/v projections. Weight map contains both explicit `q_proj/k_proj/v_proj` and `multihead_attn.in_proj_*`; DinoML should preserve the actual source path, not infer a single fused projector.
- Placeholder expansion depends on processor image crop size: 490 -> 128 tokens, 980 -> 256 tokens. Mismatch throws at `masked_scatter` preflight.
- Image preprocessing and vision source use NCHW tensors. NHWC/channel-last should be a guarded optimization only around Conv2d/ViT local regions.
- Axis-sensitive ops include pixel mask `unfold(dimension=1/2)`, patch embedding Conv2d on NCHW, `flatten(2).transpose(1,2)`, projector `repeat_interleave` over heads, MoE flatten/sort/index/select/scatter over token axis, and decoder attention transpose/reshape.
- `torchao-int8wo` is custom-code and weight-format-specific; current in-library Aria has no source-coupled quantized weight contract.

## 4. Operator coverage checklist

Tensor/layout ops:

- Embedding lookup for text tokens.
- Reshape/view/flatten/transpose/permute/contiguous.
- NCHW Conv2d patch embedding: `Conv2d(3 -> 1152, kernel=14, stride=14, padding=valid)` for main config.
- Unfold over pixel masks with `size=patch_size`, `step=patch_size`.
- `masked_scatter` image features into token embeddings.
- `index_select`, `argsort`, `index_copy_`, `histc`/bincount-like counts for MoE routing.
- `topk`, `softmax`, `sum`, `chunk`, `cat`, `repeat`, `repeat_interleave`, `expand`.

Neural network primitives:

- RMSNorm over last dim for text.
- LayerNorm for vision and projector.
- Linear/GEMM with and without bias.
- GELU new / tanh GELU for projector MLP; `gelu_pytorch_tanh` in vision defaults; SiLU-gated MLP for shared and routed experts.
- MoE grouped GEMM or sequential fallback for expert weights shaped `[experts, in, out]`.

Attention primitives:

- Vision bidirectional MHA, no cache.
- Projector cross-attention from learned queries to image patches, no cache.
- Text causal MHA/GQA-capable self-attention with RoPE and KV cache.
- Source supports eager, SDPA, FlashAttention, and flex attention through Transformers attention backend registry.

Position/rotary/custom:

- Vision learned positional embedding with fractional/bucketized NaViT-style position ids from patch mask.
- Text default/dynamic RoPE via `ROPE_INIT_FUNCTIONS`, with cos/sin computed in fp32 and cast back.

Generation/cache:

- `DynamicCache` per text decoder layer stores post-RoPE keys and values before KV head repeat.
- `logits_to_keep` last-token or indexed-slice LM head optimization.
- Generation input preparation must only include `pixel_values`/`pixel_mask` on first iteration or when cache is disabled.

Preprocessing-coupled ops:

- RGB conversion, bicubic resize, right/bottom pad, optional split into 490/980-aligned crops, rescale, normalize, boolean `pixel_mask`.
- Placeholder token expansion in text strings before tokenization.

Quantized/packed weight metadata:

- None in current in-library source. `rhymes-ai/Aria-torchao-int8wo` includes custom remote code and PyTorch bin weights; treat as a separate loading/provider contract.

## 5. Layer/block breakdown

Vision tower, repeated 27 times in main config:

```text
pixel_values [Bimg, 3, H, W]
patch_embeds = Conv2d(3 -> 1152, k=14, stride=14)(pixel_values)
x = flatten patches to [Bimg, P, 1152] + learned position_embedding(position_ids)
for each layer:
  y = LayerNorm(x)
  q,k,v = Linear(1152 -> 1152, bias=True)
  y = bidirectional Attention(q,k,v, patch mask)
  x = x + Linear(1152 -> 1152, bias=True)(y)
  y = LayerNorm(x)
  y = Linear(1152 -> 4304, bias=True) -> gelu_pytorch_tanh -> Linear(4304 -> 1152, bias=True)
  x = x + y
x = post LayerNorm(x)
```

Projector:

```text
key_value_states [Bimg, P, 1152], where P in {1225, 4900}
query_num = {1225: 128, 4900: 256}[P]
queries = learned_query[:query_num].repeat(Bimg, 1, 1)
q = Linear(1152 -> 1152, bias=False)(LayerNorm(queries))
k/v = Linear(1152 -> 1152, bias=False)(LayerNorm(key_value_states))
attn = nn.MultiheadAttention(embed=1152, heads=16, batch_first=True)(q,k,v, image mask)
attn = Linear(1152 -> 1152, bias=True)(attn)
out = LayerNorm(attn)
out = Linear(1152 -> 2560, bias=False) -> gelu_new -> Linear(2560 -> 2560, bias=False)
```

Text decoder block, repeated 28 times:

```text
x [B, T, 2560]
y = RMSNorm(x)
q = Linear(2560 -> 2560, bias=False)(y).view(B,T,20,128).transpose(1,2)
k = Linear(2560 -> 2560, bias=False)(y).view(B,T,20,128).transpose(1,2)
v = Linear(2560 -> 2560, bias=False)(y).view(B,T,20,128).transpose(1,2)
q,k = RoPE(q,k)
k,v = cache.update(k,v, layer_idx) when cache is enabled
y = causal Attention(q,k,v, mask, scaling=1/sqrt(128))
x = x + Linear(2560 -> 2560, bias=False)(y)
y = RMSNorm(x)
router_logits = Linear(2560 -> 64, bias=False)(y_flat)
topk = topk(router_logits, k=6); scores = softmax(topk_logits)
expert path:
  sort repeated token assignments by expert
  fc1 = grouped/sequential expert GEMM [64, 2560, 3328]
  projection, gate = chunk(fc1, 2)
  expert = grouped/sequential expert GEMM(silu(projection) * gate) [64, 1664, 2560]
  unpermute, weight by top-k scores, sum across top-k
shared path:
  Linear(2560 -> 3328) and Linear(2560 -> 3328), silu-gated, Linear(3328 -> 2560)
x = x + expert + shared
```

LM head:

```text
logits = Linear(2560 -> 100352, bias=False)(hidden_states[:, slice_indices, :])
```

Tied weights: source declares `lm_head.weight` tied to the token embedding for both text-only and multimodal classes. Preserve this as one logical parameter when weights are tied by loader/config.

## 6. Attention requirements

Vision attention:

- Noncausal self-attention over patch sequence.
- MHA: 16 heads, head dim 72 for hidden 1152.
- Mask comes from patch-level boolean mask derived from pixel mask; source converts it through `create_bidirectional_mask`.
- No KV cache.

Projector attention:

- Cross-attention from learned query tokens to vision patch tokens.
- Query length is 128 or 256; key/value length is 1225 or 4900.
- Uses PyTorch `nn.MultiheadAttention` with `batch_first=True`; source constructs an attention mask by flattening the patch mask, logical-not, then repeating per head and expanding to `[B * heads, query_len, patch_len]`.
- No decode cache, but full projector outputs are independently cacheable per image.

Text attention:

- Causal self-attention.
- Main configs are MHA (`20 q heads`, `20 kv heads`, `head_dim=128`), but source supports GQA/MQA through KV repeat.
- Cache stores `key_states` and `value_states` after RoPE, before `repeat_kv`.
- Per-layer cached tensor shape before repeat: `[B, num_key_value_heads, S_cached, head_dim]`.
- Eager math order: matmul QK^T, multiply by `head_dim**-0.5`, add mask, softmax in fp32, cast to query dtype, dropout, matmul with V.
- Flash/SDPA compatibility: source delegates to `ALL_ATTENTION_FUNCTIONS.get_interface(config._attn_implementation, eager_attention_forward)`.

## 7. Position encoding and custom math

Text RoPE:

```python
def aria_rope(q, k, position_ids, inv_freq, attention_scaling=1.0):
    freqs = (inv_freq[None, :, None].float() @ position_ids[:, None, :].float()).transpose(1, 2)
    emb = torch.cat((freqs, freqs), dim=-1)
    cos = emb.cos() * attention_scaling
    sin = emb.sin() * attention_scaling
    q2 = q * cos[:, None] + rotate_half(q) * sin[:, None]
    k2 = k * cos[:, None] + rotate_half(k) * sin[:, None]
    return q2, k2
```

Default inv-frequencies use `base = rope_theta` and `dim = head_dim`. Advanced RoPE types may be active if config normalization supplies a non-default `rope_parameters["rope_type"]`; this was not observed in saved checkpoint configs.

Vision position ids:

```python
boundaries = arange(1 / patches_per_side, 1.0, 1 / patches_per_side)
nb_h = patch_attention_mask[:, :, 0].sum(1)
nb_w = patch_attention_mask[:, 0, :].sum(1)
bucket_h = bucketize(clamp(arange(max_h) / nb_h[:, None], max=1-1e-6), boundaries, right=True)
bucket_w = bucketize(clamp(arange(max_w) / nb_w[:, None], max=1-1e-6), boundaries, right=True)
pos_ids = bucket_h[:, :, None] * patches_per_side + bucket_w[:, None, :]
```

Precompute opportunities: text RoPE inv-freq and max-length cos/sin tables for common positions; vision learned position embeddings are static, but position id generation depends on dynamic image padding mask.

## 8. Preprocessing and input packing

Image processor contract:

- Input images are converted to RGB.
- Source tensor layout is channel-first/NCHW.
- `max_image_size` must be `490` or `980`; default `980`.
- `min_image_size` default `336`.
- Resize preserves aspect ratio by scaling the longest side to `max_image_size`, with the shorter side at least `min_image_size`.
- Pad right and bottom to `[max_image_size, max_image_size]`.
- `pixel_mask` is boolean `[max_image_size, max_image_size]` with true in valid resized image area.
- Stack crops as `pixel_values [num_crops_total, 3, max_image_size, max_image_size]` and `pixel_mask [num_crops_total, max_image_size, max_image_size]`.
- Rescale/normalize defaults: mean/std `[0.5, 0.5, 0.5]`; `do_rescale=True`, `do_normalize=True`. The saved preprocessor omits `rescale_factor`; source backend defaults should be used.
- Optional `split_image=True` selects a best resolution from 19 grid sizes and divides padded image into crops of `max_image_size`.

Processor/token contract:

- Tokenizer image token is `<|img|>` with id `9`.
- `AriaProcessor` expands each image placeholder string by `num_crops * tokens_per_image`.
- `tokens_per_image = size_conversion[pixel_values.shape[2]]`, default `490 -> 128`, `980 -> 256`.
- Returned model inputs are `input_ids`, `attention_mask`, `pixel_values`, and `pixel_mask`; optional `mm_token_type_ids` can be produced by the processor but the current model forward does not consume it.

Embedding stitch:

```text
inputs_embeds = token_embedding(input_ids)
image_features = projector(vision_tower(pixel_values, patch_attention_mask)).to(inputs_embeds dtype)
special_image_mask = input_ids == image_token_id
assert selected embedding element count == image_features element count
inputs_embeds = inputs_embeds.masked_scatter(special_image_mask[..., None], image_features)
```

## 9. Graph rewrite / lowering opportunities

### Rewrite: vision patch Conv2d -> Linear

Source pattern: `Conv2d(3 -> 1152, kernel=patch_size, stride=patch_size, padding=valid)` followed by flatten/transpose.

Replacement:

```text
NCHW non-overlap WindowFlatten([C, kh, kw]) -> MatMul(weight_flat.T) -> BiasAdd -> [B, P, hidden]
```

Preconditions:

- `kernel_size == stride == patch_size`.
- `padding == valid` / zero padding absent in Conv2d itself.
- `dilation == 1`, `groups == 1`.
- Input height/width divisible by patch size after processor padding.
- Preserve NCHW flatten order unless a guarded layout pass rewrites both activation and weight flatten.

Weight transform:

```python
w = conv.weight.reshape(out_channels, in_channels * kh * kw)
```

Parity test: compare patch embedding output before position add for 490 and 980 square inputs with non-square valid masks.

### Rewrite: text Q/K/V projections -> fused projection

Source pattern: three independent bias-free linears from hidden to q/k/v.

Replacement: one packed GEMM `[hidden, q + k + v]` with split order `[q, k, v]`.

Preconditions:

- Same input tensor and dtype.
- Bias flags identical; inspected configs use `attention_bias=False`.
- Split widths are `num_attention_heads*head_dim`, `num_key_value_heads*head_dim`, `num_key_value_heads*head_dim`.
- Preserve RoPE before cache update.

Failure cases: GQA widths differ; packed split must not assume all three widths equal.

### Rewrite: MoE route/sort/grouped GEMM

Source pattern: `topk -> softmax -> argsort(flatten_indices) -> index_select -> per-expert GEMM -> index_copy -> top-k weighted sum`.

Replacement: provider-backed grouped GEMM with token counts and sorted token buffer.

Preconditions:

- Static `moe_num_experts`, `moe_topk`, expert weight shapes.
- Deterministic tie behavior for `topk`/`argsort` matches PyTorch enough for parity.
- Tokens-per-expert can be represented without CPU `histc` synchronization.

Failure cases: top-k ties and empty experts; source fallback uses CPU `tokens_per_expert.cpu()`, which is not production-appropriate.

### Rewrite: last-token-only logits

Source pattern: `logits_to_keep` slices hidden states before LM head.

Replacement: compile a decode path that runs only selected token rows through `Linear(hidden -> vocab)`.

Preconditions: no loss computation; logits needed only for last token or explicit selected indices.

## 10. Kernel fusion candidates

Highest priority:

- Text RMSNorm + QKV projection.
- RoPE + cached causal attention, including decode cache update.
- MoE router/topk/sort + grouped expert GEMMs.
- SwiGLU expert/shared MLP epilogues.
- Last-token-only LM head.

Medium priority:

- Vision Conv2d patch embedding lowered to GEMM.
- Vision LayerNorm + QKV projection.
- Projector cross-attention with fixed query count and patch-count guard.
- Masked image embedding scatter fused with embedding/prefix construction.

Lower priority:

- CPU image preprocessing acceleration. Important end to end, but first DinoML runtime parity can accept CPU pipeline ownership.
- Optional split-image crop packing. It is a data-pipeline throughput issue unless high-resolution split mode is a first target.
- Flash/flex attention backend parity for vision; eager/SDPA parity is enough for initial validation.

## 11. Runtime staging plan

1. Parse `AriaTextConfig`; reject or explicitly map historical remote-code fields (`aria_moe_lm`, `moe_intermediate_size`) until resolved.
2. Load text-only weights and run one `AriaTextDecoderLayer` with dense/sequential MoE parity.
3. Implement text prefill logits for `AriaTextForCausalLM`, no cache, small sequence.
4. Add `DynamicCache` decode parity for text attention.
5. Add MoE grouped GEMM provider path and router/scatter validation.
6. Compose the Idefics3 vision tower as a separate audited backbone contract; first support NCHW 490/980 square inputs and patch masks.
7. Add Aria projector parity for P=1225 and P=4900.
8. Add placeholder expansion and `masked_scatter` stitch.
9. End-to-end `AriaForConditionalGeneration` prefill logits with one image.
10. Add decode path where image/projector outputs are cached in prefix and pixels are omitted after first iteration.
11. Optimize fusions/layouts after source parity is established.

Stubbable initially: CPU image preprocessing, sampling/beam search, high-resolution split crops, FlashAttention backend, quantized custom-code repos.

## 12. Parity and validation plan

- Config normalization tests: current `rhymes-ai/Aria` plus one historical remote-code config should either map deterministically or be rejected with a clear reason.
- Vision patch embedding test: random NCHW 490 and 980 tensors, compare Conv2d/position ids/patch mask handling.
- Vision one-layer and full-encoder parity with fixed bf16/fp32 tolerance.
- Projector tests for `P=1225 -> 128` and `P=4900 -> 256`, including attention mask shape `[B*heads, Q, P]`.
- Processor/stitch tests: placeholder count exactly equals projected features; mismatch raises.
- Text block parity: attention only, MoE only, then full block.
- MoE routing parity: include empty experts, repeated top-k assignments, and top-k ties if deterministic admission is required.
- Prefill logits parity for text-only and image+text prompts.
- Decode token parity: first step with pixels, later steps without pixels using KV cache.
- Recommended tolerances: fp32 `rtol=1e-4, atol=1e-5`; bf16/fp16 block-level `rtol=5e-2, atol=5e-2`, tighten per-op where possible.

## 13. Performance probes

- CPU image preprocessing images/sec by resolution and split mode.
- Vision encoder throughput for 490 and 980 crops.
- Projector throughput for P=1225/Q=128 and P=4900/Q=256.
- Text prefill tokens/sec by sequence length, image-token prefix length, and batch size.
- Decode tokens/sec with KV cache and last-token-only logits.
- MoE router/topk/sort/grouped GEMM timing; sweep batch tokens and expert imbalance.
- KV cache memory by batch, layers, sequence length, dtype.
- Vision/projector cache memory by number of images/crops.
- Attention backend comparison: eager, SDPA, FlashAttention for text and vision.
- Layout probe: NCHW Conv2d versus lowered patch-GEMM/channel-last guarded path.

## 14. Skip/defer list

- Training, losses, gradient checkpointing.
- Beam search and complex generation controllers beyond basic greedy/sampling.
- Remote-code-only behavior from older `aria_moe_lm`/`aria_vision_model` repos until separately audited.
- `rhymes-ai/Aria-torchao-int8wo` quantized/custom-code loading.
- Optional `split_image=True` high-resolution crop mode for first parity.
- Multi-GPU tensor parallel and pipeline plans.
- Flex attention backend.
- Processor-produced `mm_token_type_ids`, because model forward does not consume them.

## 15. Final implementation checklist

- [ ] Parse current in-library `AriaConfig` and `AriaTextConfig`.
- [ ] Add admission rules for historical remote-code config fields.
- [ ] Load tied text embeddings and LM head without breaking aliasing.
- [ ] Implement text RMSNorm, RoPE, causal MHA, KV cache, and last-token logits.
- [ ] Implement MoE router/topk/sort/token-count/unpermute path.
- [ ] Add grouped expert GEMM provider plan for `[64, 2560, 3328]` and `[64, 1664, 2560]`.
- [ ] Compose or separately audit Idefics3 vision tower support.
- [ ] Implement NCHW patch Conv2d or guarded Conv2d-to-linear rewrite.
- [ ] Implement Aria projector learned-query cross-attention.
- [ ] Implement image placeholder `masked_scatter` stitch.
- [ ] Add processor contract tests for 490/980 token expansion.
- [ ] Add text-only block/prefill/decode parity tests.
- [ ] Add vision/projector parity tests.
- [ ] Add end-to-end one-image prefill and decode parity.
- [ ] Benchmark preprocessing, vision, projector, prefill, decode, and MoE routing separately.
