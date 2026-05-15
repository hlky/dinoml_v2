# CLIP Transformers Family Audit

Primary target: image-text dual encoder contrastive inference for `CLIPModel`, with separately stageable `get_image_features` and `get_text_features`.

## 1. Source basis

```text
Transformers commit/version:
  b75feb2af64c3e29cbbc1bd859958c5432cc7ed4
Local implementation-validation source:
  transformers 5.8.0.dev0 from /workspace/transformers/src/transformers/models/clip/modeling_clip.py
Model id:
  clip family; representative checkpoints listed below
Config source:
  Hugging Face config.json, preprocessor_config.json, tokenizer_config.json, special_tokens_map.json
Source files inspected:
  transformers/src/transformers/models/clip/modeling_clip.py
  transformers/src/transformers/models/clip/configuration_clip.py
  transformers/src/transformers/models/clip/processing_clip.py
  transformers/src/transformers/models/clip/image_processing_clip.py
  transformers/src/transformers/models/clip/tokenization_clip.py
  transformers/src/transformers/image_processing_backends.py
Any missing files or assumptions:
  No remote-code files are required for the inspected OpenAI CLIP checkpoints. The local source is the authoritative implementation for this report.
```

Pinned source URLs:

- `modeling_clip.py`: https://github.com/huggingface/transformers/blob/b75feb2af64c3e29cbbc1bd859958c5432cc7ed4/src/transformers/models/clip/modeling_clip.py
- `configuration_clip.py`: https://github.com/huggingface/transformers/blob/b75feb2af64c3e29cbbc1bd859958c5432cc7ed4/src/transformers/models/clip/configuration_clip.py
- `image_processing_clip.py`: https://github.com/huggingface/transformers/blob/b75feb2af64c3e29cbbc1bd859958c5432cc7ed4/src/transformers/models/clip/image_processing_clip.py
- `tokenization_clip.py`: https://github.com/huggingface/transformers/blob/b75feb2af64c3e29cbbc1bd859958c5432cc7ed4/src/transformers/models/clip/tokenization_clip.py

Bounded v2 text-wrapper validation for this branch also used the local
`transformers 5.8.0.dev0` checkout at
`/workspace/transformers/src/transformers/models/clip/modeling_clip.py`,
specifically the legacy `eos_token_id == 2` pooling branch and
`CLIPModel.get_text_features(...)` text-projection path.

Representative HF configs fetched:

- `hf-internal-testing/tiny-random-clip`: https://huggingface.co/hf-internal-testing/tiny-random-clip/resolve/main/config.json
- `openai/clip-vit-base-patch32`: https://huggingface.co/openai/clip-vit-base-patch32/resolve/main/config.json
- `openai/clip-vit-base-patch16`: https://huggingface.co/openai/clip-vit-base-patch16/resolve/main/config.json
- `openai/clip-vit-large-patch14`: https://huggingface.co/openai/clip-vit-large-patch14/resolve/main/config.json
- `openai/clip-vit-large-patch14-336`: https://huggingface.co/openai/clip-vit-large-patch14-336/resolve/main/config.json

Processor/tokenizer configs were fetched from the same repos using `preprocessor_config.json`, `tokenizer_config.json`, and `special_tokens_map.json`.

## 2. High-level architecture

CLIP is a dual encoder: a ViT-like vision encoder and a causal-masked Transformer text encoder, each followed by a bias-free projection into a shared contrastive embedding dimension. Full `CLIPModel.forward` computes both sides, L2-normalizes the projected vectors, forms a text-by-image dot product matrix, multiplies by `exp(logit_scale)`, and returns both orientations.

Dataflow:

```text
images -> CPU/GPU image processor -> pixel_values[N,3,H,W]
       -> vision patch embedding + CLS + learned position -> vision encoder
       -> CLS pool + post LayerNorm -> visual projection -> image_embeds

text -> byte BPE tokenizer -> input_ids[N,S], attention_mask[N,S]
     -> token + learned position embeddings -> causal text encoder
     -> final LayerNorm -> EOS/EOT pool -> text projection -> text_embeds

image_embeds, text_embeds -> L2 normalize -> matmul -> exp(logit_scale) scale -> logits
```

Stage decomposition:

- CPU/data-pipeline: image resize, center crop, rescale, normalize, RGB conversion; tokenizer normalization, byte-level BPE, BOS/EOS insertion, padding/truncation.
- Independently cacheable encoders: image encoder plus visual projection can run through `get_image_features`; text encoder plus text projection can run through `get_text_features`.
- Contrastive head: L2 norm, batched matrix multiply, scalar exponential scale, transpose. This is small and can be run separately after feature caches are built.
- No decode/prefill stage exists. The text branch uses a causal mask, but it is an encoder-style full-sequence inference pass with no KV cache in this source.

Other heads:

- `CLIPTextModel`, `CLIPVisionModel`, `CLIPTextModelWithProjection`, `CLIPVisionModelWithProjection`: required or directly useful for staged parity.
- `CLIPForImageClassification`: optional/deferred for primary contrastive inference. It reuses the vision encoder but mean-pools patch tokens excluding CLS and adds a classifier head.
- Training contrastive loss: deferred for inference.

## 3. Important config dimensions

Source defaults from `configuration_clip.py`:

| Field | Text default | Vision default | Notes |
| --- | ---: | ---: | --- |
| `vocab_size` | 49408 | n/a | Text embedding rows. |
| `hidden_size` | 512 | 768 | Must divide `num_attention_heads`. |
| `intermediate_size` | 2048 | 3072 | Ungated MLP width. |
| `projection_dim` | 512 | 512 | Top-level `CLIPConfig.projection_dim` drives `CLIPModel` projections. |
| `num_hidden_layers` | 12 | 12 | Separate layer counts. |
| `num_attention_heads` | 8 | 12 | MHA, no GQA/MQA. |
| `head_dim` | 64 | 64 | Derived. |
| `max_position_embeddings` | 77 | n/a | Text sequence cap. |
| `image_size` | n/a | 224 | Square image expected unless interpolation is enabled. |
| `patch_size` | n/a | 32 | Conv2d kernel and stride. |
| `num_channels` | n/a | 3 | Processor converts RGB by default. |
| `hidden_act` | quick_gelu | quick_gelu | `ACT2FN`. |
| `layer_norm_eps` | 1e-5 | 1e-5 | PyTorch LayerNorm. |
| `attention_dropout` | 0.0 | 0.0 | Dropout disabled in inference. |
| `logit_scale_init_value` | n/a | n/a | Top-level default `2.6592`; runtime uses `exp`. |
| Cache support | none | none | No KV cache or recurrent state. |

Representative checkpoint sweep:

| Checkpoint | Text shape | Vision shape | Projection | Image/patch | Vision tokens | Processor crop | Source of facts |
| --- | --- | --- | ---: | --- | ---: | --- | --- |
| `hf-internal-testing/tiny-random-clip` | 5 layers, H=32, heads=4, MLP=37, max text=512, vocab=99 | 5 layers, H=32, heads=4, MLP=37 | 64 | 30 / 2 | 226 | preprocessor says 224 | `config.json` plus preprocessor config; intentionally inconsistent stress fixture |
| `openai/clip-vit-base-patch32` | 12 layers, H=512, heads=8, MLP=2048, max text=77, vocab=49408 | 12 layers, H=768, heads=12, MLP=3072 | 512 | 224 / 32 | 50 | 224 | `config.json`, preprocessor config |
| `openai/clip-vit-base-patch16` | same as Base text | 12 layers, H=768, heads=12, MLP=3072 | 512 | 224 / 16 | 197 | 224 | `config.json`, preprocessor config |
| `openai/clip-vit-large-patch14` | 12 layers, H=768, heads=12, MLP=3072 | 24 layers, H=1024, heads=16, MLP=4096 | 768 | 224 / 14 | 257 | 224 | `config.json`, preprocessor config |
| `openai/clip-vit-large-patch14-336` | same as Large text | 24 layers, H=1024, heads=16, MLP=4096 | 768 | 336 / 14 | 577 | 336 | `config.json`, preprocessor config |

Effective defaults and omitted fields:

- Current source defaults set text `bos_token_id=49406`, `eos_token_id=49407`, `pad_token_id=1`, but many OpenAI checkpoint configs carry older `bos_token_id=0`, `eos_token_id=2`, `pad_token_id=1`. The modeling code has an explicit compatibility branch when `eos_token_id == 2`.
- Current source supports `_attn_implementation` dispatch through the shared Transformers attention interface. Representative configs do not set this field; it is supplied by the base config/runtime.
- Processor configs from older repos use integer `size`/`crop_size`; current source defaults are `size={"shortest_edge": 224}`, `crop_size={"height": 224, "width": 224}`.

## 3a. Family variation traps

- Text and vision widths usually differ. `CLIPModel` has separate `text_projection: Linear(text_hidden -> projection_dim, bias=False)` and `visual_projection: Linear(vision_hidden -> projection_dim, bias=False)`.
- The text model is named an encoder but uses a causal mask. Do not translate it as bidirectional BERT attention.
- `CLIPAttention.is_causal` is initialized `False`, but `CLIPTextModel.forward` passes a causal mask and `is_causal=True` to the encoder. Vision attention receives no mask in the normal path.
- OpenAI checkpoint configs with `eos_token_id == 2` use `input_ids.argmax(dim=-1)` for pooling, not equality against token 2. This relies on EOT being the highest token id in tokenizer output. Newer configs with non-2 EOS pool the first matching EOS position.
- Vision patch embedding is PyTorch NCHW `Conv2d`. The optimized NHWC path is an opportunity, not the semantic source graph.
- Patch sequence length changes materially: Base patch32 has 50 vision tokens, Base patch16 has 197, Large patch14 has 257, and 336px Large has 577.
- Position interpolation is optional and disabled by default. When disabled, the source rejects image sizes that do not match `vision_config.image_size`.
- Vision positional interpolation reshapes the patch position table as square grid, permutes to NCHW for bicubic interpolate, then permutes back. This path is axis-sensitive.
- Processor configs can be inconsistent with tiny test configs. Treat tiny-random as operator stress, not as an end-to-end image preprocessing contract.
- Tokenizer special token strings use `<|startoftext|>` and `<|endoftext|>`; tokenizer model input names are `input_ids` and `attention_mask`.
- No RoPE, ALiBi, relative bias, sliding window, GQA/MQA, MoE, cross-attention, or packed varlen metadata appears in the CLIP source.

## 4. Operator coverage checklist

Tensor/layout ops:

- `view`/flatten of text `input_ids` from arbitrary leading batch dims to `[B, S]`.
- Embedding lookup for token, text position, vision position.
- NCHW Conv2d output flatten: `[B, C, Gh, Gw] -> [B, C, Gh*Gw] -> [B, Gh*Gw, C]`.
- Expand class embedding `[Dv] -> [B, 1, Dv]`.
- Concatenate class token and patch tokens on sequence axis.
- Add position embeddings with broadcasting.
- Attention reshape/transposes: `[B, T, D] -> [B, H, T, Dh]`, then output transpose/contiguous/reshape back.
- Index/gather for text pooling by per-row EOT/EOS position.
- Slice/index vision CLS token `last_hidden_state[:, 0, :]`.
- Transpose logits matrix for `logits_per_image`.
- Optional bicubic interpolation path for positional embeddings.

Neural network primitives:

- NCHW `Conv2d(Cin=3, Cout=vision_hidden, kernel=patch, stride=patch, padding=0, bias=False)`.
- Dense GEMMs for Q, K, V, output projection. All attention linear layers have bias in the source.
- Dense GEMMs for MLP `hidden -> intermediate -> hidden`, both with bias.
- Bias-free projection heads: text `hidden -> projection_dim`, vision `hidden -> projection_dim`.
- LayerNorm over last dimension with learned weight/bias and epsilon 1e-5.
- `quick_gelu`.
- Residual adds.
- L2 normalization as square, sum over last dim with keepdim, square root, divide.
- Scalar `exp(logit_scale)`, multiply logits.
- Optional contrastive loss and image classification loss are deferred for inference.

Attention primitives:

- Standard MHA self-attention, Q/K/V shape `[B, heads, T, head_dim]`.
- Eager math: `softmax((Q @ K^T) * head_dim^-0.5 + mask, dim=-1, dtype=float32).to(query.dtype) @ V`.
- Text branch: causal additive mask combined with optional padding attention mask.
- Vision branch: no causal mask and no padding mask in the normal source path.
- Transformers backend dispatch can target eager, SDPA, FlashAttention, or FlexAttention through `ALL_ATTENTION_FUNCTIONS`; Dinoml should first reproduce eager-visible semantics.

Preprocessing-coupled ops:

- Image: RGB conversion, resize shortest edge or square legacy size, center crop, rescale, normalize with OpenAI mean/std, output `pixel_values`.
- Text: byte-level BPE, NFC normalization, whitespace collapse, lowercase, BOS/EOS post-processing, padding/truncation to model max length.

No required generation/cache, scatter multimodal stitch, packed sequence metadata, or distributed/tensor-parallel ops for the primary target.

## 5. Layer/block breakdown

Vision embeddings:

```text
pixel_values: [B, 3, H, W] NCHW
patch = Conv2d(3 -> Dv, kernel=P, stride=P, bias=False)(pixel_values)
patch = flatten spatial then transpose: [B, Dv, Gh, Gw] -> [B, Gh*Gw, Dv]
cls = class_embedding.expand(B, 1, Dv)
x = concat([cls, patch], dim=1)                         # [B, 1+Gh*Gw, Dv]
x = x + learned_position[0 : 1+Gh*Gw]
```

Vision encoder:

```text
x = pre_layernorm(x)
repeat Nv layers:
  r = x
  x = LayerNorm(x)
  q,k,v = Linear(Dv -> Dv, bias=True)(x)
  x = MHA(q,k,v, mask=None)
  x = r + Linear(Dv -> Dv, bias=True)(x)
  r = x
  x = LayerNorm(x)
  x = r + Linear(Iv -> Dv, bias=True)(quick_gelu(Linear(Dv -> Iv, bias=True)(x)))
pooled = post_layernorm(x[:, 0, :])
image_features = Linear(Dv -> projection_dim, bias=False)(pooled)
```

Text embeddings:

```text
input_ids: [B, S]
position_ids default: [1, S] = arange(S)
x = token_embedding[input_ids] + position_embedding[position_ids]  # [B, S, Dt]
```

Text encoder:

```text
mask = create_causal_mask(config, inputs_embeds=x, attention_mask=attention_mask, past_key_values=None)
repeat Nt layers:
  r = x
  x = LayerNorm(x)
  q,k,v = Linear(Dt -> Dt, bias=True)(x)
  x = causal MHA(q,k,v, mask)
  x = r + Linear(Dt -> Dt, bias=True)(x)
  r = x
  x = LayerNorm(x)
  x = r + Linear(It -> Dt, bias=True)(quick_gelu(Linear(Dt -> It, bias=True)(x)))
x = final_layer_norm(x)
pool index = argmax(input_ids) if eos_token_id == 2 else first equality match for eos_token_id
pooled = x[batch_arange, pool_index]
text_features = Linear(Dt -> projection_dim, bias=False)(pooled)
```

Contrastive head:

```text
image_embeds = image_features / sqrt(sum(image_features ** 2, dim=-1, keepdim=True))
text_embeds = text_features / sqrt(sum(text_features ** 2, dim=-1, keepdim=True))
logits_per_text = (text_embeds @ image_embeds.T) * exp(logit_scale)
logits_per_image = logits_per_text.T
```

## 6. Attention requirements

Vision attention:

- Noncausal self-attention.
- MHA only: `num_key_value_heads` is not present; K/V heads equal Q heads.
- Shapes: Base vision `[B, 12, 50, 64]` for patch32, `[B, 12, 197, 64]` for patch16; Large vision `[B, 16, 257 or 577, 64]`.
- No padding mask in normal image path.
- No position bias or RoPE; learned absolute positions are already added to hidden states.

Text attention:

- Causal self-attention despite no generation cache.
- MHA only. Base text `[B, 8, S<=77, 64]`; Large text `[B, 12, S<=77, 64]`; tiny `[B, 4, S<=512, 8]`.
- Masking: `create_causal_mask` receives `attention_mask` and produces an additive mask consumed by the attention backend. Eager attention adds the mask before softmax.
- Softmax is explicitly computed in `float32` in eager mode and cast back to query dtype.
- Dropout is zero in inference and also zero for representative OpenAI configs.
- No KV cache. There are no before/after cache shapes to reproduce.
- FlashAttention/SDPA compatibility: source advertises support through Transformers attention backend dispatch. Dinoml fused attention should preserve causal flag, additive padding mask behavior, query scaling, and fp32 softmax parity.

## 7. Position encoding and custom math

Text positions are learned absolute embeddings indexed by default `position_ids[:, :S]`. Vision positions are learned absolute embeddings over CLS plus patch grid.

Optional vision position interpolation:

```python
def interpolate_clip_vision_pos(position_table, embeddings, height, width, patch_size):
    class_pos = position_table[:, :1]
    patch_pos = position_table[:, 1:]
    old_grid = int((patch_pos.shape[1]) ** 0.5)
    new_h = height // patch_size
    new_w = width // patch_size
    patch_pos = patch_pos.reshape(1, old_grid, old_grid, embeddings.shape[-1])
    patch_pos = patch_pos.permute(0, 3, 1, 2)
    patch_pos = bicubic_interpolate(patch_pos, size=(new_h, new_w), align_corners=False)
    patch_pos = patch_pos.permute(0, 2, 3, 1).view(1, -1, embeddings.shape[-1])
    return concat([class_pos, patch_pos], dim=1)
```

This can be precomputed for fixed alternate image sizes. It depends on runtime `height`, `width` only when `interpolate_pos_encoding=True` or tracing dynamic shapes. For first integration, keep image size fixed and skip interpolation.

Custom math:

- `quick_gelu` from Transformers activation registry.
- L2 norm uses `pow(x, 2)`, `sum(dim=-1, keepdim=True)`, `pow(sum, 0.5)` rather than `torch.linalg.norm`; match this for export parity.
- Logit scale is a learned scalar parameter stored in log space and applied as `exp(logit_scale)`.

## 8. Preprocessing and input packing

Image processor contract:

- Source class defaults: bicubic resize, OpenAI CLIP mean `[0.48145466, 0.4578275, 0.40821073]`, std `[0.26862954, 0.26130258, 0.27577711]`, resize enabled, center crop enabled, rescale enabled, normalize enabled, RGB conversion enabled.
- OpenAI configs use crop/size 224 except `clip-vit-large-patch14-336`, which uses 336.
- The backend returns `BatchFeature({"pixel_values": processed_images})`; resize/normalize/crop helpers use `ChannelDimension.FIRST`, so the tensor contract for PyTorch model entry is `pixel_values: float [B, 3, H, W]`.
- The model casts `pixel_values` to the patch embedding weight dtype before Conv2d.
- GPU runtime graph can start at `pixel_values`; image decoding and standard processor transforms can remain CPU/data-pipeline for Stage 1.

Text tokenizer contract:

- Tokenizer is byte-level BPE with NFC normalization, whitespace collapse, lowercase, regex split, and ByteLevel pre-tokenization.
- `RobertaProcessing` adds BOS and EOS. Tokenizer config uses `model_max_length=77` for OpenAI checkpoints.
- Model input names are `input_ids` and `attention_mask`. `position_ids` is optional and usually omitted.
- Special token strings are BOS `<|startoftext|>` and EOS/UNK/PAD `<|endoftext|>`.
- Important mismatch: current config class defaults use CLIP vocab ids `49406/49407`, but representative OpenAI checkpoint `text_config` has old `bos_token_id=0`, `eos_token_id=2`, `pad_token_id=1`. The source handles old `eos_token_id=2` by pooling at `argmax(input_ids)`, which points at the high-id EOT token produced by the tokenizer.
- Padding affects attention through `attention_mask` and can also affect new-EOS pooling if `pad_token_id == eos_token_id`; the source equality branch intentionally pools the first EOS match.

There are no modality placeholder tokens, token type IDs, image/text embedding stitch, grid metadata, packed patch rows, or cu-seqlens-style descriptors.

## 9. Graph rewrite / lowering opportunities

### Rewrite: non-overlap Conv2d patch embedding -> GEMM

Source pattern:

```text
Conv2d(Cin=3, Cout=Dv, kernel=P, stride=P, padding=0, dilation=1, groups=1, bias=False)
flatten(2).transpose(1, 2)
```

Replacement:

```text
WindowFlatten NCHW patches [B, Gh, Gw, 3*P*P]
-> MatMul(weight_flat.T) [3*P*P -> Dv]
-> Reshape [B, Gh*Gw, Dv]
```

Preconditions:

- `kernel_size == stride == patch_size`.
- `padding == 0`, `dilation == 1`, `groups == 1`, `bias is None`.
- Input `H` and `W` equal configured `image_size` for default path, or are divisible by `patch_size` when interpolation is enabled.
- Flatten order must match PyTorch Conv2d NCHW receptive-field order: channels, kernel height, kernel width.

Weight transform:

```python
w = conv.weight.reshape(out_channels, in_channels * patch_h * patch_w)
```

Layout constraints:

- Faithful source layout is NCHW. NHWC optimization may use an NHWC patch extractor, but must transform window flatten order or the weight layout accordingly.
- Downstream consumer requires sequence-major hidden states `[B, Gh*Gw, Dv]`.

Failure cases:

- Non-square/list patch sizes need generalized `Gh`, `Gw`; source type allows lists, but implementation uses scalar arithmetic in `num_patches`.
- Dynamic image sizes without position interpolation must fail, not silently run.

Parity test sketch:

- Compare Conv2d+flatten+transpose against WindowFlatten+GEMM for Base patch32, Base patch16, and Large patch14.
- Include fp32 and fp16 weights, batch > 1, nonzero input, exact configured image size.

### Rewrite: packed QKV projection

Source pattern:

```text
q = Linear(D -> D)(x)
k = Linear(D -> D)(x)
v = Linear(D -> D)(x)
reshape each to [B,H,T,Dh]
```

Replacement:

```text
Linear(D -> 3D) with concatenated weights/biases -> split q,k,v
```

Preconditions:

- Same input tensor and dtype for q/k/v.
- All three projections have bias, and bias order is preserved as q, k, v.
- No hooks/observability requirements for separate projection outputs.

Weight transform:

```python
Wqkv = concat([Wq, Wk, Wv], dim=0)
bqkv = concat([bq, bk, bv], dim=0)
```

Failure cases:

- Debug modes requiring per-projection outputs or attention recording should disable the rewrite.

Parity test sketch:

- One CLIP attention layer, eager backend, random hidden states, compare q/k/v and final attention output.

### Rewrite: LayerNorm + attention/MLP residual regions

Source pattern:

```text
x = residual + Attention(LayerNorm(x))
x = x + MLP(LayerNorm(x))
```

Replacement:

- Preserve pre-norm topology. Fuse LayerNorm with adjacent GEMM input preparation where profitable; fuse residual add with output projection epilogue when candidate kernels support it.

Preconditions:

- LayerNorm axis is last dimension only.
- Residual input and projected output shapes exactly match.

Failure cases:

- Output hidden-state capture or attentions may require materializing intermediate tensors.

### Rewrite: L2 normalize + contrastive matmul

Source pattern:

```text
x / sqrt(sum(x*x, -1, keepdim=True))
y / sqrt(sum(y*y, -1, keepdim=True))
(text @ image.T) * exp(logit_scale)
```

Replacement:

- Fused row-wise norm kernel for image/text embeddings, followed by GEMM and scalar epilogue multiply.

Preconditions:

- Last dimension equals projection dim.
- Norm epsilon is absent in source; do not add one unless a guarded approximate mode is explicitly selected.

Parity test sketch:

- Compare logits for random projected features against source math, including small norms but avoid zero vectors for first pass.

### Layout rewrite: NCHW image ingress -> local NHWC patch region

Opportunity:

- Processor/model source contract is NCHW. A layout pass can convert the local patch embedding region to NHWC/channel-last for better memory coalescing.

Required guards:

- Layout region must include only processor output or explicit transpose, patch extraction/Conv2d, flatten, and immediate projection to sequence tokens.
- Axis rewrite needed for Conv2d channel axis: source `dim=1` channel becomes last channel in NHWC.
- Flatten order and weight transform must preserve PyTorch NCHW Conv2d semantics.
- Position embedding and Transformer encoder operate on `[B, tokens, hidden]` and are layout-neutral after patch projection.

No-layout-translation guards:

- Positional interpolation path uses NCHW around `nn.functional.interpolate`; either keep it source-layout or implement a dedicated equivalent.
- Any external caller providing `pixel_values` directly must satisfy the advertised NCHW API unless a separate NHWC API is introduced.

## 10. Kernel fusion candidates

Highest priority:

- LayerNorm over `[B, T, D]`: appears before every attention and MLP plus final text/post vision norms.
- Dense GEMM epilogues with bias and residual add for attention output and MLP second projection.
- QKV packed projection for both encoders.
- Fused attention for small/medium sequence lengths with causal flag support for text and noncausal support for vision.
- Conv patch embedding lowered to GEMM or a specialized non-overlap patch kernel, especially for patch16/patch14.

Medium priority:

- `quick_gelu` fused into MLP first/second GEMM path.
- Row-wise L2 normalization plus contrastive GEMM scalar epilogue.
- Vision CLS gather/post LayerNorm/projection as a small fused tail.
- Text EOT/EOS pooling gather plus projection, including the old `argmax(input_ids)` compatibility branch.

Lower priority:

- Bicubic positional interpolation; useful for dynamic image sizes but not required for fixed OpenAI checkpoint inference.
- Image classification head mean-pool excluding CLS.
- Contrastive training loss.

## 11. Runtime staging plan

Stage 1: config, tokenizer/processor metadata, and weights.

- Parse nested `CLIPConfig`, `CLIPTextConfig`, and `CLIPVisionConfig`.
- Load separate text, vision, projection, and logit scale weights.
- Keep preprocessing outside DinoML runtime and accept `input_ids`, `attention_mask`, and `pixel_values`.

Stage 2: independent text encoder parity.

- Implement token/position embeddings, causal mask construction, MHA, LayerNorm, MLP, final norm, EOT/EOS pooling, and text projection.
- Validate `get_text_features` before contrastive logits.

2026-05-15 bounded update:

- A narrow DinoML wrapper now covers the legacy text-only
  `get_text_features` path with explicit `position_ids`, causal + padding-mask
  behavior, final LayerNorm, legacy argmax pooling, and text projection through
  existing DinoML ops only.
- Wrapper-level parity is now pinned against the local Transformers
  `CLIPModel.get_text_features(...)` implementation for a tiny one-layer config
  with weights flowing through the wrapper API.
- The newer non-2 EOS first-match pooling branch remains a follow-up because
  this slice stayed aligned with the already-landed legacy pooling coverage.

Stage 3: independent vision encoder parity.

- Implement NCHW patch embedding, CLS/position add, pre LayerNorm, noncausal encoder, CLS pool, post LayerNorm, and visual projection.
- Validate `get_image_features` for fixed image size first.

Stage 4: contrastive head.

- Add L2 normalization, image/text feature matrix multiply, `exp(logit_scale)` scaling, transpose output.
- Validate logits for multiple image/text batch sizes.

Stage 5: optimized kernels.

- Add packed QKV, fused attention, patch Conv2d-to-GEMM, LayerNorm/MLP fusions.
- Keep shape and layout guards artifact-visible.

Stage 6: optional extensions.

- Position interpolation for non-default image sizes.
- `CLIPTextModelWithProjection`, `CLIPVisionModelWithProjection`, and image classification head.

Initially stubbable:

- `return_loss`, hidden-state/attention recording, gradient checkpointing, output tuple variants, image classification head, and positional interpolation.

## 12. Parity and validation plan

Unit parity:

- `quick_gelu` against Transformers activation registry.
- L2 norm helper against `_get_vector_norm` math.
- Text pooling branch for `eos_token_id == 2` using `argmax(input_ids)`.
- Text pooling branch for non-2 EOS using first equality match.
- Causal additive mask shape and values for padded and unpadded text.
- NCHW patch Conv2d+flatten+transpose against rewritten patch GEMM.

Layer parity:

- One text `CLIPEncoderLayer` with causal mask, fp32 tolerance around `1e-5` to `1e-4`.
- One vision `CLIPEncoderLayer` with no mask.
- Full text encoder for Base patch32 prompt batch with max length 77.
- Full vision encoder for Base patch32 and patch16 images.

End-to-end parity:

- `get_text_features` for 2-4 prompts with padding.
- `get_image_features` for 1 and batched images after HF processor.
- Full `CLIPModel.forward` logits for unequal image/text batch sizes, verifying both `logits_per_text` and transpose `logits_per_image`.
- Checkpoint sweep: tiny-random for small shapes, Base patch32, Base patch16, Large patch14, Large patch14-336.

Suggested tolerances:

- fp32: `atol=1e-4`, `rtol=1e-4` for layer/logit parity.
- fp16/bf16 optimized kernels: start with `atol=2e-2`, `rtol=2e-2` for full logits, then tighten per kernel.
- Attention backend parity should separately compare eager math before enabling FlashAttention-style kernels.

No DinoML tests were run for this docs-only audit.

## 13. Performance probes

- Processor throughput: images/sec for resize/crop/normalize on CPU versus optional GPU preprocessing.
- Vision patch embedding throughput: Conv2d source path versus patch-GEMM rewrite for patch32, patch16, patch14.
- Vision encoder throughput: batch sweep for 50, 197, 257, and 577 vision tokens.
- Text encoder throughput: prompt batch sweep for `S <= 77`, plus tiny fixture `S <= 512` if supporting generic configs.
- Attention backend comparison: eager, SDPA-like, FlashAttention-like for causal text and noncausal vision.
- Projection and contrastive head: text batch by image batch matrix sizes, especially many-text/few-image retrieval.
- End-to-end image-text retrieval: separate image feature caching, text feature caching, and logits-only recompute.
- Memory probes: activation memory by vision token count and batch size; no KV cache memory probe is needed.

## 14. Skip/defer list

- Training loss and gradient checkpointing.
- Hidden state and attention output capture.
- Beam search, sampling, decode loops, and KV cache.
- Positional interpolation for non-configured image sizes.
- Image classification head.
- Remote-code variants outside official CLIP implementation.
- Quantization and multi-GPU tensor parallel.
- Full processor implementation inside DinoML runtime; first integration can consume already-preprocessed tensors.

## 15. Final implementation checklist

- [ ] Parse nested CLIP text/vision/top-level config.
- [ ] Load text encoder, vision encoder, projection heads, and `logit_scale` weights.
- [ ] Accept processor outputs: `pixel_values[B,3,H,W]`, `input_ids[B,S]`, `attention_mask[B,S]`.
- [ ] Implement CLIP tokenizer metadata handling enough to validate BOS/EOS/PAD and pooling branch.
- [ ] Implement NCHW vision patch embedding and shape checks.
- [ ] Implement CLS token append and learned absolute vision positions.
- [ ] Implement text token/position embeddings and max-position guard.
- [ ] Implement causal text mask with optional padding mask.
- [ ] Implement MHA with fp32 softmax parity and both causal/noncausal modes.
- [ ] Implement LayerNorm, quick_gelu MLP, residual adds.
- [ ] Implement vision CLS pool and post LayerNorm.
- [ ] Implement text EOT/EOS pooling, including old `eos_token_id == 2` argmax behavior.
- [ ] Implement bias-free text and visual projection heads.
- [ ] Implement L2 feature normalization, contrastive matmul, `exp(logit_scale)` scale, and transpose.
- [ ] Add one-layer text and vision parity tests.
- [ ] Add `get_text_features`, `get_image_features`, and full logits parity tests.
- [ ] Add checkpoint sweep coverage for tiny-random, Base patch32, Base patch16, Large patch14, and Large patch14-336.
- [ ] Add guarded Conv2d patch embedding to GEMM rewrite.
- [ ] Add packed QKV rewrite with source-observability guard.
- [ ] Add benchmarks separating preprocessing, image encoder, text encoder, and logits head.
