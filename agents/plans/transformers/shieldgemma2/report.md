# DinoML Transformers Audit: shieldgemma2

## 1. Source basis

```text
Transformers commit/version: b75feb2af64c3e29cbbc1bd859958c5432cc7ed4
Model id: google/shieldgemma-2-4b-it
Config source: official Hub repo is gated/401 without license access; open mirror Nozim6690/hugging-face_shieldgemma-2-4b-it used for config/preprocessor snapshots and labeled as mirror evidence.
Source files inspected:
  X:/H/transformers/src/transformers/models/shieldgemma2/configuration_shieldgemma2.py
  X:/H/transformers/src/transformers/models/shieldgemma2/modeling_shieldgemma2.py
  X:/H/transformers/src/transformers/models/shieldgemma2/processing_shieldgemma2.py
  X:/H/transformers/src/transformers/models/shieldgemma2/convert_shieldgemma2_weights_orbax_to_hf.py
  X:/H/transformers/src/transformers/models/gemma3/modeling_gemma3.py
  X:/H/transformers/src/transformers/models/gemma3/modular_gemma3.py
  X:/H/transformers/src/transformers/models/gemma3/configuration_gemma3.py
  X:/H/transformers/src/transformers/models/gemma3/processing_gemma3.py
  X:/H/transformers/src/transformers/models/gemma3/image_processing_gemma3.py
  X:/H/transformers/src/transformers/models/siglip/modeling_siglip.py
  X:/H/transformers/src/transformers/models/siglip/configuration_siglip.py
Any missing files or assumptions: only one ShieldGemma2 checkpoint was visible; no small/debug native checkpoint found. Gemma3 modeling/configuration files are generated from modular_gemma3.py; modular_gemma3.py is authoritative for future Transformers source edits.
```

Representative config evidence:

| Source | Status | Notes |
| --- | --- | --- |
| [google/shieldgemma-2-4b-it](https://huggingface.co/google/shieldgemma-2-4b-it) | gated, raw files returned 401 | Access would confirm official `config.json`, `preprocessor_config.json`, tokenizer, and weight index. |
| [Nozim6690/hugging-face_shieldgemma-2-4b-it](https://huggingface.co/Nozim6690/hugging-face_shieldgemma-2-4b-it) | open mirror | `config.json`, `preprocessor_config.json`, `processor_config.json`, and safetensors index inspected as mirror evidence. |
| Transformers source defaults | local pinned source | Used when mirror config omits fields that config classes fill in. |

Primary runtime target for this report: image safety classification with `ShieldGemma2ForImageClassification`, not open-ended multimodal generation. The wrapper still executes the Gemma3 image-text decoder and then slices the last-token logits to the Yes/No token pair.

## 2. High-level architecture

ShieldGemma2 is a thin image-classification wrapper over a Gemma3-style vision-language model:

```text
CPU/data pipeline image + policy prompt construction
  -> Gemma3 image processor: RGB/resize/rescale/normalize to NCHW pixel_values
  -> SigLIP vision encoder: patch Conv2d + ViT encoder
  -> Gemma3 multimodal projector: 4096 vision tokens -> 256 soft image tokens
  -> placeholder embedding stitch into text sequence
  -> Gemma3 decoder prefill/decode
  -> LM logits
  -> last-token gather [yes_token_index, no_token_index]
  -> softmax over 2 classes
```

Stage decomposition:

| Stage | Inputs/outputs | Cacheable/independent? | First DinoML target |
| --- | --- | --- | --- |
| Policy prompt construction | image list x policy list -> text prompts, left padded token batch | CPU/data pipeline; no GPU kernel required | Required for end-to-end classifier parity, but can be stubbed with pretokenized inputs first. |
| Image preprocessing | images -> `pixel_values [B_img,3,896,896]` bf16/fp32-compatible tensor | CPU/data pipeline initially | Required shape/normalization contract; GPU resize is optional. |
| Vision tower | `pixel_values` -> `last_hidden_state [B_img,4096,1152]` | Cacheable per image before policy fanout if prompts share the same image | Required for full parity; independently testable. |
| Projector | `[B_img,4096,1152]` -> `[B_img,256,2560]` | Cacheable per image; reused across policy prompts if batching duplicates features | Required. |
| Embedding stitch | text embeddings `[B,S,2560]` with 256 image placeholders per image -> multimodal embeddings | Guarded indexed copy candidate; source uses `masked_scatter` | Required; avoid admitting general boolean scatter if processor guards hold. |
| Decoder/classifier | multimodal sequence -> logits `[B,S,262208]`, then last-token Yes/No logits `[B,2]` | Prefill/decode cache useful; classifier can compute last-token logits only | Required; generation controller optional. |

## 3. Important config dimensions

Mirror checkpoint `Nozim6690/hugging-face_shieldgemma-2-4b-it`:

| Field | Value | Provenance |
| --- | ---: | --- |
| architecture | `ShieldGemma2ForImageClassification` | mirror `config.json` |
| dtype | `bfloat16` | mirror `config.json` `torch_dtype` |
| model_type | `shieldgemma2` | mirror `config.json` |
| vocab_size | 262208 | mirror `text_config` |
| text hidden_size | 2560 | mirror `text_config` |
| text layers | 34 | mirror `text_config` |
| text attention heads | 8 | mirror `text_config` |
| text KV heads | 4 | mirror `text_config` |
| text head_dim | 256 | mirror `text_config` |
| Q width / KV width / O input width | 2048 / 1024 / 2048 | source-derived from heads x head_dim |
| text intermediate_size | 10240 | mirror `text_config` |
| activation | `gelu_pytorch_tanh` | mirror `text_config` |
| max_position_embeddings | 8192 | mirror `text_config` |
| sliding_window | 1024 | mirror `text_config` |
| sliding_window_pattern | 6 | mirror `text_config`; source makes every 6th layer full attention |
| RoPE global theta | 1000000 | mirror `text_config` `rope_theta` |
| RoPE local theta | 10000 | mirror `text_config` `rope_local_base_freq` |
| full-attention RoPE scaling | linear factor 8.0 | mirror `text_config` `rope_scaling`; source applies it to full-attention rope params |
| cache | `use_cache=true`, `cache_implementation=hybrid` | mirror `text_config`; native source creates/updates cache through Transformers cache API |
| mm_tokens_per_image | 256 | mirror `config.json` |
| image_token_index | 262144 | mirror `config.json` |
| boi/eoi token indices | 255999 / 256000 | mirror `config.json` |
| yes/no token indices | 10784 / 3771 | mirror `config.json` |
| vision model_type | `siglip_vision_model` | mirror `vision_config` |
| vision image_size / patch_size | 896 / 14 | mirror `vision_config` |
| vision patch grid / tokens | 64 x 64 / 4096 | source-derived |
| vision hidden_size | 1152 | mirror `vision_config` |
| vision layers | 27 | mirror `vision_config` |
| vision heads / head_dim | 16 / 72 | mirror + source-derived |
| vision intermediate_size | 4304 | mirror `vision_config` |
| vision_use_head | false | mirror `vision_config`; avoids SigLIP pooling head |
| preprocessor size | 896 x 896 | mirror `preprocessor_config.json` |
| image mean/std | `(0.5,0.5,0.5)` / `(0.5,0.5,0.5)` | mirror `preprocessor_config.json` |
| rescale_factor | 1/255 | mirror `preprocessor_config.json` |
| image_seq_length | 256 | mirror preprocessor/processor config |
| safetensors total_size | 8,600,158,944 bytes | mirror safetensors index metadata |

Representative checkpoint sweep:

| Checkpoint/config | Availability | Operator-significant deltas |
| --- | --- | --- |
| `google/shieldgemma-2-4b-it` | gated official | Expected same model; needs licensed access to confirm official config and weights. |
| `Nozim6690/hugging-face_shieldgemma-2-4b-it` | open mirror | Only concrete full config inspected. 4B bf16, SigLIP 896/14, Gemma3 34-layer hybrid attention. |
| `ShieldGemma2Config()` source defaults | local source defaults | Defaults to Gemma3TextConfig 2304 hidden, 26 layers, max position 131072, SigLIP 224/16; useful only for shape-generic code, not checkpoint parity. |

## 3a. Family variation traps

- ShieldGemma2 does not implement its own neural body. It delegates to `AutoModelForImageTextToText.from_config(config=config)`, which maps the wrapper config to Gemma3 behavior. DinoML should route unsupported `text_config.model_type` or `vision_config.model_type` through separate audits or reject them.
- `hidden_size != num_attention_heads * head_dim` for the 4B config: `2560 != 8 * 256 = 2048`. Q/O attention width is 2048, while residual width is 2560.
- GQA is required: KV heads 4, query heads 8, repeat factor 2 after cache update for eager attention.
- Text projections have no bias. SigLIP vision projections and MLP linears do have bias.
- Hybrid attention is required: layers 5, 11, 17, 23, and 29 are full attention; the other 29 layers are sliding-window attention with window 1024.
- Full and sliding attention use separate RoPE parameter sets. Full attention inherits linear scaling factor 8.0 and theta 1e6; sliding attention uses local theta 1e4.
- The source applies Q/K RMSNorm before RoPE, then stores cached keys after RoPE.
- Vision source is NCHW. Any NHWC/channel-last optimization must be a guarded layout rewrite around image preprocessing, patch Conv2d, and projector pooling, not a semantic default.
- The projector assumes a square patch grid and `sqrt(mm_tokens_per_image)` is integral. For 896/14/256 this is `64 -> 16` with AvgPool2d kernel/stride 4.
- Source uses `masked_scatter` for image embedding stitch. Processor should produce exactly 256 contiguous image soft tokens per image placeholder; DinoML can lower to indexed row copy only with placeholder count/order guards.
- `image_token_index` equals 262144 and is outside the text embedding vocab in this config (`vocab_size=262208`, so it is inside the embedding table here; source still has OOV replacement logic for configs where image token is outside vocab).
- `ShieldGemma2Processor` disables pan-and-scan even though Gemma3 processor/image processor can support it. DinoML first integration should reject or ignore `do_pan_and_scan=True` for ShieldGemma2.
- Classification wrapper doc text appears internally inconsistent: code returns `[Yes, No]` logits/probabilities, and comments say Yes means violation, but the docstring says to use `probabilities[:, 1]` for the violative slice. Parity should follow code: Yes is index 0.
- Official repo is gated. Open mirror configs should not be treated as canonical until official files are available.

## 4. Operator coverage checklist

Tensor/layout ops:

- NCHW image tensor input `[B_img,3,896,896]`.
- `Conv2d(3 -> 1152, kernel=14, stride=14, padding=valid)` patch embedding, output `[B_img,1152,64,64]`.
- Flatten spatial patches: `[B,1152,64,64] -> [B,1152,4096] -> transpose -> [B,4096,1152]`.
- Transpose/reshape/contiguous in projector: `[B,4096,1152] -> [B,1152,4096] -> [B,1152,64,64]`.
- AvgPool2d NCHW kernel=stride=4: `[B,1152,64,64] -> [B,1152,16,16]`.
- Flatten/transpose to `[B,256,1152]`.
- Embedding lookup for text tokens; scaled by `sqrt(2560)`.
- Boolean/equality mask for image token IDs, `unsqueeze`, `expand_as`, count check.
- Masked scatter or guarded indexed copy from `[B_img,256,2560]` into `[B,S,2560]`.
- Last-token slice and static vocab index gather: logits `[:, -1, [10784,3771]] -> [B,2]`.

Neural network primitives:

- SigLIP LayerNorm over 1152, eps 1e-6.
- SigLIP MHA, noncausal MHA, 27 layers, `Q/K/V/O Linear(1152 -> 1152)` with bias, 16 heads x 72.
- SigLIP MLP with bias: `Linear(1152 -> 4304)`, `gelu_pytorch_tanh`, `Linear(4304 -> 1152)`.
- Gemma3 RMSNorm over 2560 and 256, eps 1e-6, weight formula `(1 + weight)`.
- Gemma3 GQA projections without bias per layer: `q_proj Linear(2560 -> 2048)`, `k_proj Linear(2560 -> 1024)`, `v_proj Linear(2560 -> 1024)`, `o_proj Linear(2048 -> 2560)`.
- Gemma3 gated MLP without bias: `gate_proj Linear(2560 -> 10240)`, `up_proj Linear(2560 -> 10240)`, `gelu_pytorch_tanh(gate) * up`, `down_proj Linear(10240 -> 2560)`.
- Multimodal projection matmul: `[B,256,1152] @ [1152,2560] -> [B,256,2560]`.
- LM head `Linear(2560 -> 262208, bias=False)`, tied to text embeddings by `_tied_weights_keys`.
- Softmax over final 2 logits for classifier probabilities.

Attention primitives:

- Vision dense noncausal self-attention on 4096 tokens, 27 layers, no cache.
- Text causal full attention on every 6th layer.
- Text causal sliding-window attention on the remaining layers, window 1024.
- GQA repeat of KV heads from `[B,4,T,256]` to `[B,8,T,256]`.
- Attention score scale uses `query_pre_attn_scalar**-0.5 = 1/sqrt(256)`, not `1/sqrt(head_dim)` by coincidence for this config.
- Optional attention logit softcap exists in source but is `null` for the inspected checkpoint.
- Backend dispatch may use eager, SDPA, FlashAttention, or FlexAttention via Transformers `ALL_ATTENTION_FUNCTIONS`; DinoML parity should first implement source math order.

Position/rotary/custom math:

- Per-layer-type RoPE over head_dim 256, with `rotate_half`.
- Cos/sin computed in fp32 then cast to model dtype.
- Full/sliding RoPE params are separate and selected by layer type.

Generation/cache ops:

- `past_key_values.update(k, v, layer_idx)` per text layer.
- Cache key/value logical shapes before repeat: K/V `[B,4,T,256]`.
- Cached keys are after Q/K RMSNorm and RoPE; values are projected values.
- Prefill can use multimodal inputs; decode should omit `pixel_values` once image features are already represented in cache/prefix.

Preprocessing-coupled ops:

- RGB conversion, bilinear resize to 896x896, rescale by 1/255, normalize `(x - 0.5) / 0.5`.
- Prompt expansion: one `<start_of_image>` becomes 256 `<image_soft_token>` placeholders bracketed by BOI/EOI text.
- `token_type_ids` mark image tokens; image block IDs are built with equality, pad, cumsum, and where for mask construction.

Gated/missing links:

- Official checkpoint files are gated; mirror evidence is not canonical.
- DinoML lacks general attention, RMSNorm, LayerNorm, Conv2d, and arbitrary masked scatter in the current v2 checklist. This family should stage around bounded implementations/rewrite paths.

## 5. Layer/block breakdown

Image preprocessing:

```text
input images
  -> RGB convert
  -> bilinear resize to [3,896,896]
  -> rescale by 1/255
  -> normalize mean/std 0.5
  -> pixel_values [B_img,3,896,896] in source NCHW
```

SigLIP vision embedding:

```text
x = Conv2d(3 -> 1152, kernel=14, stride=14, valid)(pixel_values)
x: [B,1152,64,64]
x = flatten(2).transpose(1,2)
x: [B,4096,1152]
x = x + position_embedding[0:4096]
```

SigLIP encoder layer, repeated 27 times:

```text
residual = x
x = LayerNorm(1152, eps=1e-6)(x)
q,k,v = Linear(1152 -> 1152, bias=True)(x)
q,k,v -> [B,16,4096,72]
x = dense noncausal Attention(q,k,v)
x = Linear(1152 -> 1152, bias=True)(x)
x = residual + x
residual = x
x = LayerNorm(1152, eps=1e-6)(x)
x = Linear(1152 -> 4304, bias=True)(x)
x = gelu_pytorch_tanh(x)
x = Linear(4304 -> 1152, bias=True)(x)
x = residual + x
```

Vision post/projector:

```text
x = LayerNorm(1152, eps=1e-6)(vision_last_hidden_state)
x: [B,4096,1152]
x = transpose(1,2).reshape(B,1152,64,64).contiguous()
x = AvgPool2d(kernel=4, stride=4)(x)          # [B,1152,16,16]
x = flatten(2).transpose(1,2)                 # [B,256,1152]
x = RMSNorm(1152, eps=1e-6)(x)
x = matmul(x, mm_input_projection_weight)     # [B,256,2560]
```

Embedding stitch:

```text
input_ids image placeholders are validated against image_features.numel()
inputs_embeds = token_embedding(input_ids or pad-replaced ids) * sqrt(2560)
inputs_embeds = masked_scatter(image_placeholder_mask, image_features)
```

Gemma3 decoder layer, repeated 34 times:

```text
residual = x
x = RMSNorm(2560)(x)
q = Linear(2560 -> 2048, bias=False)(x).view(B,S,8,256).transpose(1,2)
k = Linear(2560 -> 1024, bias=False)(x).view(B,S,4,256).transpose(1,2)
v = Linear(2560 -> 1024, bias=False)(x).view(B,S,4,256).transpose(1,2)
q = RMSNorm(256)(q)
k = RMSNorm(256)(k)
q,k = RoPE(q,k, layer_type-specific cos/sin)
k,v = cache.update(k,v,layer_idx) if cache
x = GQA attention(q, repeat_kv(k), repeat_kv(v), full/sliding mask)
x = Linear(2048 -> 2560, bias=False)(x)
x = RMSNorm(2560)(x)
x = residual + x
residual = x
x = RMSNorm(2560)(x)
x = Linear(10240 -> 2560, bias=False)(
      gelu_pytorch_tanh(Linear(2560 -> 10240, bias=False)(x))
      * Linear(2560 -> 10240, bias=False)(x)
    )
x = RMSNorm(2560)(x)
x = residual + x
```

Classifier head:

```text
x = final RMSNorm(2560)(x)
logits = lm_head(x[:, logits_to_keep, :])      # tied Linear(2560 -> 262208)
selected = logits[:, -1, [10784, 3771]]
probabilities = softmax(selected, dim=-1)
```

## 6. Attention requirements

Vision attention:

- Noncausal dense self-attention.
- Sequence length 4096 for 896/14 images.
- Heads 16, head_dim 72, Q/K/V width 1152.
- No KV cache.
- Mask normally absent for vision tower.
- Source supports SDPA/other backend dispatch through generic Transformers interfaces.

Text attention:

- Autoregressive self-attention for normal classifier/generation path.
- GQA: query heads 8, KV heads 4, head_dim 256, repeat factor 2.
- Q length may be full prefill sequence length or decode length 1; KV length includes cached prefix.
- Full attention layers: every 6th layer, based on `sliding_window_pattern=6`.
- Sliding layers: causal local attention with `sliding_window=1024`.
- If `token_type_ids` are supplied, image token block sequence IDs alter mask construction so image blocks can be handled specially by Transformers mask utilities.
- Packed/varlen support is not explicit in ShieldGemma2 source; FlashAttention may use backend-specific packed paths, but report scope should require dense mask parity first.
- Cached K shape before repeat: `[B,4,T,256]`; cached V shape `[B,4,T,256]`; expanded attention K/V `[B,8,T,256]`.
- Cached keys are stored after RoPE. This matters for cache interoperability and RoPE replay.
- Attention math order for eager parity: Q/K/V projection, Q/K RMSNorm, RoPE, cache update, repeat KV, QK matmul, scale, optional softcap, mask add, fp32 softmax then cast, dropout only in training, AV matmul, transpose/contiguous, output projection.

ShieldGemma2 classifier can initially run prefill-only and compute the final token logits. Decode support is still useful if DinoML wants to share Gemma3 infrastructure, but classification does not require long autoregressive generation beyond one prompt forward.

## 7. Position encoding and custom math

Gemma3 RoPE is standard rotate-half RoPE with per-layer-type parameter tables:

```python
def gemma3_rope(position_ids, inv_freq, attention_scaling, dtype):
    # inv_freq shape [head_dim / 2], position_ids shape [B, S]
    freqs = (inv_freq[None, :, None].float() @ position_ids[:, None, :].float()).transpose(1, 2)
    emb = torch.cat((freqs, freqs), dim=-1)
    return (emb.cos() * attention_scaling).to(dtype), (emb.sin() * attention_scaling).to(dtype)

def apply_rope(q, k, cos, sin):
    cos = cos.unsqueeze(1)
    sin = sin.unsqueeze(1)
    return (q * cos) + (rotate_half(q) * sin), (k * cos) + (rotate_half(k) * sin)
```

For the inspected checkpoint:

- Sliding layers use default RoPE with local theta 10000.
- Full layers use default/linear-scaled RoPE parameters with theta 1000000 and factor 8.0 from mirror config.
- Cos/sin can be precomputed per layer type and position bucket for static max positions, but dynamic decode position IDs and cache offsets must be handled.

Custom RMSNorm:

```python
def gemma3_rms_norm(x, weight, eps):
    y = x.float() * torch.rsqrt((x.float() ** 2).mean(-1, keepdim=True) + eps)
    y = y * (1.0 + weight.float())
    return y.to(x.dtype)
```

Optional softcaps:

```python
scores = tanh(scores / attn_logit_softcapping) * attn_logit_softcapping
logits = tanh(logits / final_logit_softcapping) * final_logit_softcapping
```

Both softcaps are `null` in the inspected mirror checkpoint, but source implements them.

## 8. Preprocessing and input packing

ShieldGemma2Processor behavior:

- Requires images; `text` is not supported as a direct user input for the ShieldGemma2 wrapper.
- Builds a batch for every image-policy pair. Default policies are `dangerous`, `sexual`, and `violence`.
- Requires a chat template. The template emits one image placeholder and a policy text prompt, then starts the model turn.
- Disables pan-and-scan by warning and setting `do_pan_and_scan=False`.
- Forces tokenization padding by default and defaults to `padding_side="left"`.
- Delegates to Gemma3Processor after constructing `images=expanded_images` and `text=rendered_prompts`.

Gemma3Processor packing:

- Replaces each BOI token with:

```text
\n\n<start_of_image><image_soft_token>... repeated 256 times ...<end_of_image>\n\n
```

- Tokenizer returns `input_ids`, `attention_mask`, and, by default, multimodal `token_type_ids`.
- Image processor emits `pixel_values` and `num_crops`; Gemma3Processor removes `num_crops` from model inputs.
- For ShieldGemma2, exactly one image per sample is allowed by the processor. Multiple policies duplicate the image in the batch.

Image processor contract:

- Input decode/fetch and RGB conversion are CPU/data-pipeline work.
- Resize: bilinear, target 896x896 from mirror preprocessor.
- Rescale: multiply by `0.00392156862745098`.
- Normalize: `(x - 0.5) / 0.5` per channel.
- Source tensor layout to model is NCHW `[B,3,H,W]`.

Embedding stitch contract:

- Placeholder token ID: `image_token_index=262144`.
- Expected image feature count: `B_img * 256`.
- Source checks `inputs_embeds[special_image_mask].numel() == image_features.numel()`.
- `masked_scatter` flattens selected embedding elements in row-major tensor order. Under the processor contract, selected positions should be contiguous runs of 256 image soft tokens per prompt. DinoML should implement a stricter indexed copy/prefix-row-copy path with guards:
  - each prompt contains exactly 256 image token positions per image;
  - positions are contiguous in sequence order after tokenization;
  - image feature batch order matches prompt expansion order;
  - reject arbitrary boolean masks.

## 9. Graph rewrite / lowering opportunities

### Rewrite: SigLIP patch Conv2d -> Linear/GEMM

Source pattern:

```text
Conv2d(3 -> 1152, kernel=14, stride=14, padding=valid) on [B,3,896,896]
```

Replacement:

```text
WindowFlatten_NCHW_14x14_stride14 -> MatMul([B*4096, 588] x [588,1152]) -> BiasAdd -> Reshape [B,4096,1152]
```

Preconditions:

- `kernel_size == stride == 14`.
- `padding == valid/0`, `dilation == 1`, `groups == 1`.
- Input H/W divisible by 14 and equal to configured image size unless position interpolation is admitted separately.
- Preserve NCHW flatten order: source Conv2d weight layout `[out_channels, in_channels, kh, kw]`; flatten weight to `[1152, 3*14*14]` then transpose for row-major GEMM.
- If an NHWC preprocessing path is used, rewrite window flatten and weight layout together; otherwise keep NCHW.

Failure cases:

- Dynamic H/W with position interpolation, non-square images, pan-and-scan, or channels-last tensors without a local guarded rewrite.

Parity test sketch:

- Random `[B,3,896,896]`, compare Conv2d output after `flatten(2).transpose(1,2)` against linearized patch embedding in fp32 and bf16.

### Rewrite: Projector AvgPool2d -> token block average

Source pattern:

```text
[B,4096,1152] -> transpose/reshape [B,1152,64,64] -> AvgPool2d(k=4,s=4) -> [B,256,1152]
```

Replacement:

```text
Reshape [B,64,64,1152] -> group 4x4 spatial blocks -> ReduceMean over 16 patch tokens -> [B,16,16,1152] -> flatten [B,256,1152]
```

Preconditions:

- Patch grid square 64x64.
- `mm_tokens_per_image=256`, `tokens_per_side=16`, `kernel_size=4`.
- No padding, ceil mode, or count-exclude-pad behavior.
- Source token order is row-major from NCHW Conv2d flatten.

Layout constraints:

- This is a good local NHWC candidate after vision encoder output because token layout is sequence-major `[B,N,C]`. Do not rewrite SigLIP encoder internals to NHWC unless the whole region is controlled.

### Rewrite: masked_scatter image stitch -> guarded indexed copy

Source pattern:

```text
inputs_embeds.masked_scatter(special_image_mask.expand_as(inputs_embeds), image_features)
```

Replacement:

```text
for each prompt row:
  copy image_features[row_or_image, 0:256, :] into inputs_embeds[row, image_start:image_start+256, :]
```

Preconditions:

- Processor-generated placeholders only.
- Exactly 256 image token positions per image.
- Contiguous placeholder span.
- `image_features.dtype == inputs_embeds.dtype` after explicit cast.
- Batch mapping from images x policies is known.

Failure cases:

- Caller supplies arbitrary `input_ids`/`inputs_embeds` with discontiguous image tokens, multiple images per sample, missing token_type_ids, or custom processor behavior.

### Rewrite: classifier last-token two-column logits

Source pattern:

```text
lm_head(hidden_states[:, -1, :]) -> gather [yes,no] -> softmax
```

Replacement:

```text
GEMM last hidden row against only lm_head rows [10784,3771] -> Softmax2
```

Preconditions:

- Inference target is `ShieldGemma2ForImageClassification`.
- No labels/loss and no full logits requested.
- `logits_to_keep` and return ABI do not require full vocab logits.
- Tied embedding alias remains logical; selecting two rows must not duplicate or mutate tied weights.

### Layout rewrite: guarded NCHW -> NHWC image front-end

Candidate region:

- Image preprocessing output through patch embedding can be made channel-last if DinoML owns resize/normalize and patch extraction.

Required axis rewrites:

- Source Conv2d consumes NCHW. NHWC rewrite changes channel axis from `dim=1` to `dim=-1`.
- Patch flatten order and Conv2d weight transform must preserve source row-major patch token order.
- Projector AvgPool2d source uses NCHW `[B,C,64,64]`; an NHWC projector rewrite would reduce over axes 1/2 spatial blocks and keep channel axis last.

No-layout-translation guard:

- SigLIP encoder tokens `[B,4096,1152]`, Gemma3 decoder `[B,S,2560]`, attention tensors `[B,H,S,D]`, and token ID/mask ops should remain faithful to source axes. Do not reinterpret token sequence axes as image layout.

## 10. Kernel fusion candidates

Highest priority:

- Gemma3 RMSNorm over 2560 and 256, including `(1 + weight)` formula and fp32 accumulation.
- GQA prefill/decode attention with RoPE and KV cache: Q/K RMSNorm -> RoPE -> attention is the critical decoder path.
- Sliding-window causal attention for 29/34 text layers.
- SwiGLU-like gated MLP with `gelu_pytorch_tanh(gate) * up` and down projection.
- Last-token-only two-row classifier head to avoid full vocab logits for classification.

Medium priority:

- SigLIP patch Conv2d lowered to GEMM.
- SigLIP LayerNorm + attention/MLP fusions for 4096-token vision encoder.
- Projector AvgPool2d + RMSNorm + Linear chain.
- Image stitch indexed copy as a bounded special op.
- Token block mask construction from `token_type_ids` if dynamic multimodal masks stay on GPU.

Lower priority:

- Full generation decode helpers and sampling; classifier target only needs Yes/No probabilities.
- Attention logit/final logit softcap fusion; inspected checkpoint disables both.
- Pan-and-scan image crop pipeline; ShieldGemma2 processor disables it.
- Vision positional interpolation; inspected preprocessor fixes 896x896.

## 11. Runtime staging plan

Stage 1: config and weight admission.

- Parse ShieldGemma2 wrapper config, nested Gemma3 text config, nested SigLIP vision config, processor config.
- Reject unsupported nested model types for first target.
- Load tied text embedding/lm_head as one logical parameter.

Stage 2: classifier wrapper without image tower.

- Accept precomputed `inputs_embeds` or pre-stitched multimodal embeddings.
- Run one/few Gemma3 decoder layers and final Yes/No classifier head parity.

Stage 3: Gemma3 text decoder prefill.

- Implement RMSNorm, RoPE, GQA, full/sliding masks, gated MLP.
- Validate full prefill logits on short sequences.

Stage 4: bounded image stitch.

- Implement processor-placeholder guards and indexed copy from `[B,256,2560]` image features into text embeddings.

Stage 5: multimodal projector.

- Implement `[B,4096,1152] -> [B,256,2560]` projector with AvgPool2d/RMSNorm/matmul.

Stage 6: SigLIP vision tower.

- Start with faithful NCHW Conv2d + dense ViT encoder.
- Add Conv2d-to-GEMM and projector pooling rewrites after parity.

Stage 7: cache/decode and optimized attention.

- Add KV cache for Gemma3 to share with other Gemma3/Gemma-family work.
- Enable FlashAttention/sliding-window kernels behind parity checks.

Stage 8: production classifier specialization.

- Last-token-only two-row logits, image-feature caching across policy fanout, and batch scheduling.

Can be stubbed initially: processor text templating, image decode/resize, full generation sampling, labels/loss, output attentions/hidden states.

## 12. Parity and validation plan

- Config parsing test: mirror config produces the exact dimensions in section 3 and layer type pattern with five full-attention layers.
- Custom op random tests:
  - Gemma3 RMSNorm for fp32/bf16 against PyTorch.
  - RoPE cos/sin for full and sliding layer types.
  - GQA repeat/cache update shape tests.
  - Projector pooling rewrite against source projector.
- Single-layer parity:
  - One SigLIP encoder layer with random weights/input `[1,4096,1152]`.
  - One Gemma3 decoder layer with no cache and with synthetic cache.
- Vision parity:
  - Patch embedding Conv2d vs GEMM rewrite.
  - Vision tower final hidden state `[B,4096,1152]`.
- Stitch parity:
  - Processor-generated `input_ids` with one placeholder span; compare masked_scatter to indexed copy.
  - Negative tests for wrong token count and discontiguous image tokens.
- Prefill parity:
  - Full ShieldGemma2 forward on one image and one policy, compare selected logits `[B,2]`.
  - Batch fanout one image x three default policies, verify output shape `[3,2]`.
- Decode/cache parity:
  - If implemented, compare last-token logits with and without cache for text-only and multimodal prefix.
- Tolerances:
  - fp32 custom ops: atol 1e-5, rtol 1e-5.
  - bf16/fp16 end-to-end blocks: atol 2e-2, rtol 2e-2 initially; tighten per op where stable.
  - Classifier probabilities: compare logits first; softmax can amplify tiny two-class differences.

No DinoML tests were run for this report, per task scope.

## 13. Performance probes

- CPU preprocessing throughput: images/sec for decode/RGB/resize/normalize at 896.
- Vision tower throughput: `[B,3,896,896] -> [B,4096,1152]`, batch sweep B=1,2,4,8.
- Projector throughput and memory bandwidth: `[B,4096,1152] -> [B,256,2560]`.
- Stitch overhead: masked scatter vs guarded indexed copy for sequence lengths around ShieldGemma2 prompts.
- Text prefill throughput: sequence length sweep including prompt length + 256 image tokens.
- Sliding vs full layer attention timing split.
- Decode tokens/sec with cache for shared Gemma3 infrastructure, even if classifier does not need long decode.
- Last-token two-row classifier head vs full vocab head.
- Image-feature caching across policy fanout: one image x 1/3/N policies.
- KV cache memory: 34 layers x 2 tensors x `[B,4,T,256]` bf16 plus hybrid/sliding cache policy overhead.
- Attention backend comparison: eager, SDPA, FlashAttention-compatible, DinoML custom full/sliding.
- Layout rewrite probe: NCHW Conv2d vs im2col/GEMM vs guarded NHWC patch extraction.

## 14. Skip/defer list

- Training, labels/loss, gradient checkpointing, dropout.
- Beam search, sampling, speculative decoding, and generation processors for first classifier target.
- Pan-and-scan crops; ShieldGemma2 processor disables them.
- Arbitrary multimodal `masked_scatter`; use guarded image-placeholder copy.
- Dynamic vision resolutions and positional interpolation.
- SigLIP pooling head; `vision_use_head=false` for the inspected checkpoint.
- Non-Gemma3 text configs and non-SigLIP vision configs under ShieldGemma2 until separately audited.
- Quantization and BitsAndBytes 4-bit loading; integration test uses it for memory, but native source operator graph is dense.
- Multi-GPU tensor parallel plan.
- Attention/logit softcaps for the inspected checkpoint because both are null; keep source support as a later option.

## 15. Final implementation checklist

- [ ] Parse ShieldGemma2 wrapper config and nested Gemma3/SigLIP configs.
- [ ] Add gated admission for `text_config.model_type=gemma3_text` and `vision_config.model_type=siglip_vision_model`.
- [ ] Load tied `embed_tokens.weight` / `lm_head.weight` as one logical parameter.
- [ ] Implement Gemma3 RMSNorm with `(1 + weight)` and fp32 accumulation.
- [ ] Implement Gemma3 RoPE with separate full/sliding parameter sets.
- [ ] Implement Gemma3 GQA projections with `hidden_size != q_width`.
- [ ] Implement full and sliding causal attention masks, including image block mask behavior from `token_type_ids`.
- [ ] Implement Gemma3 gated MLP with `gelu_pytorch_tanh`.
- [ ] Implement KV cache shape `[B,4,T,256]` per layer with cached keys after RoPE.
- [ ] Implement SigLIP NCHW patch Conv2d and dense noncausal vision encoder.
- [ ] Implement projector pooling/RMSNorm/matmul from 4096 patch tokens to 256 soft tokens.
- [ ] Implement guarded image-placeholder indexed copy as a replacement for general `masked_scatter`.
- [ ] Implement ShieldGemma2 last-token Yes/No logits and two-class softmax.
- [ ] Add Conv2d-to-GEMM rewrite with NCHW/NHWC guards.
- [ ] Add projector pooling rewrite with token-order guards.
- [ ] Add last-token two-row LM-head specialization.
- [ ] Add parity tests for processor-shaped stitch, one decoder layer, vision projector, prefill selected logits, and batch image-policy fanout.
- [ ] Add performance probes for vision tower, projector, text prefill, sliding attention, and classifier head specialization.
