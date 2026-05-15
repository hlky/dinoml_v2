# ViP-LLaVA Transformers Audit

Primary target: inference-time multimodal image+text generation with `VipLlavaForConditionalGeneration`. This report covers the native in-library `vipllava` wrapper at the pinned Transformers checkout, plus the delegated CLIP vision tower and Llama-family decoder contracts needed by the public `llava-hf` checkpoints.

## 1. Source basis

```text
Transformers commit/version:
  b75feb2af64c3e29cbbc1bd859958c5432cc7ed4
Model id:
  vipllava family; primary native checkpoints are llava-hf/vip-llava-7b-hf and llava-hf/vip-llava-13b-hf
Config source:
  Hugging Face config.json, preprocessor_config.json, processor_config.json, tokenizer_config.json
Source files inspected:
  transformers/src/transformers/models/vipllava/configuration_vipllava.py
  transformers/src/transformers/models/vipllava/modeling_vipllava.py
  transformers/src/transformers/models/vipllava/modular_vipllava.py
  transformers/src/transformers/models/llava/processing_llava.py
  transformers/src/transformers/models/clip/image_processing_clip.py
  transformers/src/transformers/models/clip/modeling_clip.py
  transformers/src/transformers/models/clip/configuration_clip.py
  transformers/src/transformers/models/llama/modeling_llama.py
  transformers/src/transformers/models/llama/configuration_llama.py
Any missing files or assumptions:
  No remote-code files were needed for the native `vipllava` audit. `modeling_vipllava.py` is generated from `modular_vipllava.py`; runtime behavior was read from the generated file, while future source edits should target the modular file. HF configs were fetched from public model repos on 2026-05-13.
```

Pinned source URLs:

- `modeling_vipllava.py`: https://github.com/huggingface/transformers/blob/b75feb2af64c3e29cbbc1bd859958c5432cc7ed4/src/transformers/models/vipllava/modeling_vipllava.py
- `modular_vipllava.py`: https://github.com/huggingface/transformers/blob/b75feb2af64c3e29cbbc1bd859958c5432cc7ed4/src/transformers/models/vipllava/modular_vipllava.py
- `configuration_vipllava.py`: https://github.com/huggingface/transformers/blob/b75feb2af64c3e29cbbc1bd859958c5432cc7ed4/src/transformers/models/vipllava/configuration_vipllava.py
- `processing_llava.py`: https://github.com/huggingface/transformers/blob/b75feb2af64c3e29cbbc1bd859958c5432cc7ed4/src/transformers/models/llava/processing_llava.py
- `image_processing_clip.py`: https://github.com/huggingface/transformers/blob/b75feb2af64c3e29cbbc1bd859958c5432cc7ed4/src/transformers/models/clip/image_processing_clip.py
- `modeling_clip.py`: https://github.com/huggingface/transformers/blob/b75feb2af64c3e29cbbc1bd859958c5432cc7ed4/src/transformers/models/clip/modeling_clip.py
- `modeling_llama.py`: https://github.com/huggingface/transformers/blob/b75feb2af64c3e29cbbc1bd859958c5432cc7ed4/src/transformers/models/llama/modeling_llama.py

Representative HF configs:

- `llava-hf/vip-llava-7b-hf`: https://huggingface.co/llava-hf/vip-llava-7b-hf/resolve/main/config.json
- `llava-hf/vip-llava-13b-hf`: https://huggingface.co/llava-hf/vip-llava-13b-hf/resolve/main/config.json
- Processor/preprocessor/tokenizer configs from the same repos.
- Local small/debug config from `tests/models/vipllava/test_modeling_vipllava.py`.
- Historical original-family repos such as `mucai/vip-llava-7b-base`, `mucai/vip-llava-7b-pretrain`, and `mucai/vip-llava-llama-3-8b` were checked as variation traps only. They do not use native `model_type="vipllava"` and should not be treated as covered by this report.

## 2. High-level architecture

ViP-LLaVA is a CLIP-style vision encoder plus a multimodal projector plus a causal Llama-family language decoder. Compared with regular LLaVA, the distinguishing wrapper behavior is the vision feature pyramid: the model selects one or more intermediate vision hidden states, drops the CLS token from each selected layer, concatenates selected layer features on the channel dimension, then applies a LayerNorm + two-layer MLP projector.

Dataflow:

```text
image/text processor -> pixel_values + expanded input_ids
pixel_values[N,3,H,W] -> CLIP vision tower hidden states
selected hidden layers -> drop CLS -> concat last dim -> projector LN/MLP
input_ids -> token embeddings
image placeholder mask -> masked_scatter(projected image features into token embeddings)
stitched inputs_embeds -> Llama decoder prefill -> logits / KV cache
decode steps -> Llama decoder only, using cache -> logits / sampling
```

Stage decomposition:

- CPU/data pipeline: RGB conversion, bicubic resize, center crop, rescale by `1/255`, CLIP mean/std normalization, tokenizer, chat template, and expansion of each `<image>` into image-placeholder token runs.
- Cacheable vision/projector stage: `pixel_values -> vision_tower hidden_states -> selected patch features -> projector`. For a fixed image, selected layer list, and dtype, this can be computed independently of text decode.
- Prefix construction: projected image features replace token embeddings at `image_token_id` positions via `masked_scatter`. This is small but parity-critical.
- Prefill: the Llama-family decoder consumes one multimodal embedding sequence and creates self-attention KV cache.
- Decode: `prepare_inputs_for_generation` forwards `pixel_values` only on the first generation iteration, or when `use_cache=False`; cached subsequent decode is text-only.

Independently stageable validation points are image preprocessing, CLIP hidden-state selection, projector output, placeholder stitch, Llama prefill logits, and cached decode logits.

## 3. Important config dimensions

Source defaults from `VipLlavaConfig`: `image_token_index=32000`, `projector_hidden_act="gelu"`, `projector_layernorm_eps=1e-5`, `vision_feature_layers=(-2, -5, -8, -11, 6)`, `image_seq_length=576`, and `tie_word_embeddings=False`. If omitted, the vision tower defaults to CLIP vision hidden 1024, 24 layers, 16 heads, image 336, patch 14; text defaults to `LlamaConfig`.

| Field | Native 7B effective value | Native 13B value | Source/default notes |
| --- | ---: | ---: | --- |
| Top model type | `vipllava` | `vipllava` | From `config.json`. |
| Text model type | `llama` / Vicuna | `llama` / Llama-2-13B | Delegated via `AutoModel.from_config(config.text_config)`. |
| Text hidden size | 4096 inferred from `LlamaConfig` default | 5120 | 7B config omits hidden size; current source default applies unless loader backfills from checkpoint-specific config state. |
| Text layers | 32 inferred default | 40 | 7B omits; 13B explicit. |
| Attention heads | 32 inferred default | 40 | Head dim 128 for both. |
| KV heads | 32 inferred default | 40 | No GQA for native public 7B/13B configs. |
| Text MLP intermediate | 11008 inferred default | 13824 | Llama SwiGLU FFN. |
| Context | 4096 | 4096 | From `text_config.max_position_embeddings`. |
| Text vocab | 32064 | 32064 | `<image>` id 32000, `<pad>` id 32001. |
| Vision tower | CLIP vision | CLIP vision | Native public configs use `clip_vision_model`. |
| Vision hidden/layers/heads | 1024 / 24 / 16 | 1024 / 24 / 16 | Head dim 64. |
| Vision MLP intermediate | 4096 | 4096 | CLIP MLP uses `quick_gelu` unless config overrides. |
| Image size / patch | 336 / 14 | 336 / 14 | Patch grid 24x24. |
| Patch tokens after CLS drop | 576 | 576 | `336 / 14 = 24`, then `24*24`. |
| Selected vision layers | legacy config trap; source default is `[-2,-5,-8,-11,6]` | `[-2,-5,-8,-11,6]` | Current source reads `vision_feature_layers`, not LLaVA's `vision_feature_layer`. |
| Projector input width | 5120 for 5 selected layers | 5120 | `5 * vision_hidden`. |
| Projector hidden/output width | 4096 | 5120 | Matches text hidden size. |
| Dtype | `float16` in config | `float16` in config | Dtype source is HF `config.json`. |
| Cache support | `use_cache=True` from Llama default | `use_cache=True` if not overridden | Wrapper passes cache through to text model. |

Representative sweep:

| Config source | Native scope | Text dims | Vision feature policy | Processor image tokens | Operator-significant notes |
| --- | --- | ---: | --- | ---: | --- |
| Local tiny test config | In scope as debug | hidden 32, 2 layers, 4 heads, MLP 37 | `[0,0,1,1,0]` | `(8/2)^2 = 16` | Repeats layer indices intentionally; projector input is still `5 * vision_hidden`. |
| `llava-hf/vip-llava-7b-hf` | In scope, with legacy-key guard | effective Llama default 4096/32/32/11008 from current source when omitted | config contains old `vision_feature_layer=-2` and `vision_feature_select_strategy="default"`; current source ignores these and defaults to `vision_feature_layers` | 576 | Treat legacy keys as compatibility risk; safetensors index contains projector LayerNorm and MLP weights. |
| `llava-hf/vip-llava-13b-hf` | In scope | 5120/40/40/13824 | `[-2,-5,-8,-11,6]` | 576 | Clean native config for current source. |
| `mucai/vip-llava-7b-base` | Out of scope for native `vipllava` | Llama/Vicuna-like 4096/32/32/11008 | remote/original fields such as `mm_hidden_size=5120`, `mm_vision_tower=clip_4layers_336` | not native | `model_type="llava"`, architecture names are not native `VipLlavaForConditionalGeneration`. |
| `mucai/vip-llava-llama-3-8b` | Separate audit target | Llama-3-like 4096/32 heads, 8 KV heads, MLP 14336, theta 500000 | remote/original fields | not native | `model_type="llava_llama"` and GQA/large vocab change decoder requirements. |

## 3a. Family variation traps

- `modeling_vipllava.py` is generated; edit `modular_vipllava.py` upstream, but audit generated runtime behavior.
- Native ViP-LLaVA always drops the first vision token from every selected layer in `get_image_features`; it does not implement LLaVA's `vision_feature_select_strategy="full"` branch.
- The config key is `vision_feature_layers`. The public 7B config still contains LLaVA-style `vision_feature_layer` and `vision_feature_select_strategy`; current native source does not read those fields. DinoML should normalize or reject such legacy-key-only configs rather than silently building a one-layer projector.
- The projector always has `projector_layernorm`, `linear_1`, activation, and `linear_2`, with biases on both Linear layers. A historical `projector_layernorm=false` field in the 7B config is ignored by current native source.
- Multiple selected vision layers concatenate on the last dimension, so projector `linear_1.in_features = len(vision_feature_layers) * vision_hidden_size`. Repeated indices are allowed by the local test config and duplicate feature channels.
- CLIP image preprocessing/modeling is NCHW. NHWC/channel-last optimization is only safe inside guarded local regions; semantic axes remain NCHW through preprocessing, Conv2d patch embedding, `flatten(2)`, and `transpose(1,2)`.
- Processor token count must match source selection. For the public processor config, `num_additional_image_tokens=1` and `vision_feature_select_strategy="default"` make token expansion `24*24 + 1 - 1 = 576`, matching source's CLS drop.
- Placeholder replacement is global over the batch: `n_image_tokens = special_image_mask.sum()` must equal `image_features.shape[0] * image_features.shape[1]`. Multi-image or multi-sample packing is accepted only if total placeholder count matches flattened projected features.
- If `inputs_embeds` is supplied without `input_ids`, the placeholder mask is computed by exact embedding equality against the image-token embedding. That path is brittle for compiled runtimes; first integration should require `input_ids` for multimodal runs.
- Decoder details are delegated. Native public 7B/13B are MHA Llama-style, but historical/non-native ViP-LLaVA variants can use GQA, different RoPE theta, larger vocab, or remote model classes.
- `logits_to_keep` slices decoder hidden states before `lm_head`, enabling last-token-only logits. DinoML should preserve this as an optimization surface.

## 4. Operator coverage checklist

Tensor/layout ops:

- NCHW image tensor ingest: `pixel_values[N,3,336,336]`.
- Conv2d patch embedding: CLIP `Conv2d(3 -> 1024, kernel=14, stride=14, bias=False)`.
- Flatten spatial patch grid and transpose: `[N,1024,24,24] -> [N,576,1024]`.
- CLS concat: learned `[1024]` class embedding expanded to `[N,1,1024]`, concatenate on sequence dim.
- Position embedding add for CLIP sequence length 577.
- Hidden-state tuple capture for every CLIP layer, not just last hidden state.
- Slice/drop CLS: `hidden_states[layer][:, 1:]`.
- Concatenate selected layer features on last dim: five `[N,576,1024] -> [N,576,5120]`.
- Placeholder mask: compare `input_ids == image_token_id`, unsqueeze, expand to embedding shape.
- Masked scatter/indexed copy: replace `[N,S,H_text]` token embeddings with flattened `[N,576,H_text]` image features.
- Last-token or indexed logits slicing for `logits_to_keep`.

Neural network primitives:

- Embedding lookup for Llama token embeddings.
- CLIP LayerNorm before encoder and per block; CLIP final pooled LayerNorm is not used by ViP-LLaVA projector path except vision model still returns it.
- CLIP MHA attention and MLP: Linear Q/K/V/O with bias, GELU/quick-GELU MLP.
- Projector: `LayerNorm(5120) -> Linear(5120 -> text_hidden, bias=True) -> GELU -> Linear(text_hidden -> text_hidden, bias=True)`.
- Llama decoder RMSNorm, RoPE Q/K projection, causal attention, SwiGLU MLP, residual adds, final RMSNorm.
- LM head `Linear(text_hidden -> vocab_size, bias=False)`; weight is tied logically to `model.language_model.embed_tokens.weight` by `_tied_weights_keys`, while top config has `tie_word_embeddings=False`.

Attention primitives:

- CLIP vision noncausal self-attention, MHA, no KV cache, sequence length 577 for 336/14.
- Llama causal self-attention, native public configs MHA with `num_key_value_heads == num_attention_heads`.
- SDPA/Flash/Flex attention backend dispatch is declared supported by wrappers, but eager fallback must remain correct.
- Additive attention masks and causal masks are owned by the delegated Llama/CLIP implementations.

Position/rotary ops:

- CLIP learned absolute position embedding over CLS+patch tokens; interpolation exists in CLIP source but public ViP-LLaVA path uses fixed 336 unless `interpolate_pos_encoding` is passed through kwargs.
- Llama RoPE computes cos/sin in fp32 and applies to Q/K before cache update.

Generation/cache ops:

- `DynamicCache`/`Cache` through delegated Llama model.
- Per-layer cache stores K/V after RoPE and before KV-head repeat expansion.
- `prepare_inputs_for_generation` suppresses `pixel_values` after the first cached iteration.

Preprocessing-coupled ops:

- CLIP image processor: convert RGB, resize shortest edge 336, center crop 336x336, rescale, normalize with OpenAI CLIP mean/std.
- LLaVA processor: replace each textual `<image>` marker with repeated image-token strings before tokenization.
- Llama tokenizer coupling: `<image>` id 32000, `<pad>` id 32001, left padding in public tokenizer configs.

Scatter/indexed update ops:

- Required for multimodal embedding stitch: boolean mask expand + masked scatter or equivalent stable indexed copy. The replacement order must match PyTorch's flattened mask order.

Packed/varlen metadata:

- No `cu_seqlens` or packed image-grid metadata in native ViP-LLaVA. Variable prompt lengths use normal tokenizer padding and attention masks.

Distributed/tensor-parallel:

- No ViP-LLaVA-specific tensor parallel code. Delegated Llama config carries TP hints for q/k/v/o and MLP projections; first DinoML target can defer multi-GPU TP.

## 5. Layer/block breakdown

CLIP vision tower:

```text
pixel_values[N,3,336,336]
patch = Conv2d(3 -> 1024, kernel=14, stride=14, no bias) -> [N,1024,24,24]
patch = flatten(2).transpose(1,2) -> [N,576,1024]
tokens = concat(CLS[N,1,1024], patch) + pos_embed[1,577,1024]
x = pre_layernorm(tokens)
repeat 24 times:
  residual = x
  x = LayerNorm(x)
  q,k,v = Linear(1024 -> 1024, bias=True)
  x = noncausal MHA(q,k,v, seq=577, heads=16, head_dim=64)
  x = residual + Linear(1024 -> 1024, bias=True)
  residual = x
  x = LayerNorm(x)
  x = residual + Linear(4096 -> 1024)(quick_gelu(Linear(1024 -> 4096)(x)))
return all hidden states
```

ViP-LLaVA feature pyramid and projector:

```text
features = []
for layer in vision_feature_layers:            # default [-2,-5,-8,-11,6]
  features.append(vision_hidden_states[layer][:, 1:])  # drop CLS
image_features = cat(features, dim=-1)         # [N,576,5120] for CLIP-1024 and 5 layers
image_features = LayerNorm(5120, eps=1e-5)
image_features = Linear(5120 -> text_hidden, bias=True)
image_features = GELU(image_features)
image_features = Linear(text_hidden -> text_hidden, bias=True)
```

Multimodal stitch:

```text
inputs_embeds = embed_tokens(input_ids)         # [N,S,text_hidden]
mask = input_ids == image_token_id              # [N,S]
assert mask.sum() == N_images * 576
inputs_embeds = masked_scatter(mask[...,None].expand_as(inputs_embeds), image_features)
```

Llama decoder block for native 7B/13B:

```text
repeat N_text_layers:
  residual = x
  x_norm = RMSNorm(x)
  q = Linear(hidden -> heads*head_dim, no bias)(x_norm)
  k = Linear(hidden -> kv_heads*head_dim, no bias)(x_norm)
  v = Linear(hidden -> kv_heads*head_dim, no bias)(x_norm)
  q,k = RoPE(q,k, position_ids)
  if cache: k,v = cache.update(k,v, layer_idx)
  attn = causal_attention(q,k,v, mask, scale=head_dim**-0.5)
  x = residual + Linear(heads*head_dim -> hidden, no bias)(attn)
  residual = x
  x_norm = RMSNorm(x)
  mlp = down_proj(silu(gate_proj(x_norm)) * up_proj(x_norm))
  x = residual + mlp
x = final RMSNorm(x)
logits = lm_head(x[:, slice_indices, :])
```

Native public 7B inferred shapes: hidden 4096, layers 32, heads 32, KV heads 32, head dim 128, intermediate 11008. Native 13B shapes: hidden 5120, layers 40, heads 40, KV heads 40, head dim 128, intermediate 13824.

## 6. Attention requirements

CLIP vision attention:

- Noncausal self-attention only.
- MHA: 16 heads, head dim 64 for public configs.
- Sequence length is fixed at 577 for 336x336/14 images, but hidden-state output from multiple layers is required.
- Eager math: `matmul(q, k.T) * scale`, add mask if present, softmax in fp32, cast back, dropout only in training, matmul with V, output projection.
- No KV cache. CLIP vision/projector outputs can be cached at the multimodal-prefix level.

Llama decoder attention:

- Causal self-attention.
- Native public 7B/13B use MHA (`num_key_value_heads == num_attention_heads`), but DinoML should keep GQA support because delegated Llama source supports `num_key_value_heads < num_attention_heads`.
- Q shape before attention: `[batch, num_attention_heads, q_len, head_dim]`.
- K/V cache shape before repeat expansion: `[batch, num_key_value_heads, kv_len, head_dim]`.
- If GQA is active, eager attention repeats cached K/V with `repeat_kv` to `[batch, num_attention_heads, kv_len, head_dim]` only for attention math; cache storage remains KV-head shape.
- Cached keys are stored after RoPE because `past_key_values.update` happens after `apply_rotary_pos_emb`.
- Eager fallback upcasts softmax to fp32 and downcasts to query dtype.
- Optimized backends are routed through Transformers `ALL_ATTENTION_FUNCTIONS` according to `_attn_implementation`; ViP-LLaVA declares flash attention, SDPA, flex attention, and generic attention backend support.
- Decode with cache should not rerun vision/projector after the first cached iteration.

## 7. Position encoding and custom math

CLIP vision uses learned absolute position embeddings. For fixed 336x336 public configs, position ids cover 577 tokens. CLIP source has bicubic interpolation support if `interpolate_pos_encoding=True`; first integration can reject variable image sizes unless that flag is explicitly in scope.

Llama RoPE:

```python
def apply_llama_rope(q, k, cos, sin):
    cos = cos.unsqueeze(1)
    sin = sin.unsqueeze(1)
    def rotate_half(x):
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return cat([-x2, x1], dim=-1)
    return (q * cos) + (rotate_half(q) * sin), (k * cos) + (rotate_half(k) * sin)
```

RoPE cos/sin computation uses fp32 matmul between inverse frequencies and `position_ids`, then casts to the model dtype. Public native configs use default Llama RoPE parameters unless omitted fields are filled by `LlamaConfig`: `rope_theta` effectively 10000 for Llama defaults. Historical Llama-3 ViP-LLaVA variants advertise `rope_theta=500000` but are not native `vipllava`.

Projector custom math is standard LayerNorm + GELU MLP. The custom part is feature selection:

```python
def vipllava_select_features(hidden_states, layers):
    selected = [hidden_states[i][:, 1:] for i in layers]
    return cat(selected, dim=-1)
```

This depends on runtime-selected `vision_feature_layers` and on the vision tower returning hidden states.

## 8. Preprocessing and input packing

Public processor configs:

- `processor_class="LlavaProcessor"`.
- `image_processor_type="CLIPImageProcessor"`.
- `patch_size=14`.
- `num_additional_image_tokens=1`.
- `vision_feature_select_strategy="default"`.
- `image_token="<image>"`.

Image pipeline:

```text
input image -> RGB -> bicubic resize shortest_edge=336 -> center_crop 336x336
-> rescale by 0.00392156862745098
-> normalize by mean [0.48145466, 0.4578275, 0.40821073]
   and std [0.26862954, 0.26130258, 0.27577711]
-> pixel_values[N,3,336,336]
```

Text/tokenizer pipeline:

- Tokenizer class in public configs is `LlamaTokenizer`.
- `<image>` token id is 32000 and is recorded as `extra_special_tokens.image_token`.
- `<pad>` token id is 32001, public tokenizer configs use left padding.
- `add_bos_token=true`, `add_eos_token=false`.
- `model_max_length` is 4096 for 7B tokenizer config; 13B tokenizer config uses a very large sentinel despite model context 4096, so model config/context should win for runtime admission.

Placeholder packing:

```text
num_image_tokens = (height // patch_size) * (width // patch_size) + num_additional_image_tokens
if vision_feature_select_strategy == "default":
  num_image_tokens -= 1
```

For 336x336 and patch 14 this gives `24*24 + 1 - 1 = 576`. The processor repeats the literal image token string before tokenization. The model later checks total placeholder count against projected feature count and raises on mismatch.

CPU/data-pipeline candidates: image resize/crop/rescale/normalize and string token expansion. GPU/runtime candidates: CLIP vision tower, feature selection, projector, embedding stitch, Llama prefill/decode.

End-to-end generation controller behavior is ordinary Transformers generation plus ViP-LLaVA's `prepare_inputs_for_generation` pixel gating. No family-specific forced decoder ids, timestamp processors, or suppress-token lists were found.

## 9. Graph rewrite / lowering opportunities

### Rewrite: CLIP patch Conv2d -> Linear

Source pattern:

```text
Conv2d(C=3 -> D=1024, kernel=14, stride=14, padding=0, dilation=1, groups=1, bias=False)
flatten(2).transpose(1,2)
```

Replacement:

```text
WindowFlatten NHWC/NCHW local patches [N,576,3*14*14]
-> MatMul(weight_flat.T) -> [N,576,1024]
```

Preconditions:

- `kernel_size == stride == patch_size`.
- `padding == 0`, `dilation == 1`, `groups == 1`.
- Input H/W match or are divisible by patch size with the same floor behavior as Conv2d.
- Weight transform preserves PyTorch Conv2d cross-correlation order.
- Bias absent for CLIP patch embedding.

Weight transform:

```python
w = conv.weight.reshape(out_channels, in_channels * kh * kw)
out = patch_matrix @ w.T
```

Layout constraints: source is NCHW. NHWC is safe only if the patch extractor and weight flatten use matching channel/spatial order and downstream sequence layout remains `[N,num_patches,D]`.

Failure cases: non-divisible image sizes if interpolation/variable image support is enabled, any nonzero padding/dilation/groups, or alternate vision towers.

Parity test sketch: random 336x336 fp32/fp16 image, compare Conv2d+flatten+transpose to patch-matrix GEMM before position embedding.

### Rewrite: ViP-LLaVA feature pyramid gather

Source pattern:

```text
[hidden_states[layer][:, 1:] for layer in layers] -> cat(dim=-1)
```

Replacement:

```text
Static layer-output taps -> CLS crop -> channel concat
```

Preconditions:

- Vision tower returns all requested hidden states.
- All selected states have the same batch, patch sequence, and hidden width.
- Layer indices are normalized against the hidden-state tuple. Negative and repeated indices must be preserved.

Shape equations: `[N, 1+P, Hv] * L -> [N, P, L*Hv]`.

Failure cases: layer index out of range, non-CLIP tower without CLS token, source config requiring `full` selection.

Parity test sketch: instrument CLIP hidden states from PyTorch and compare DinoML selected concat byte/order against PyTorch.

### Rewrite: projector MLP into fused epilogue regions

Source pattern:

```text
LayerNorm(5120) -> Linear(5120,Ht) -> GELU -> Linear(Ht,Ht)
```

Replacement: keep LayerNorm explicit; lower both Linear ops to GEMM, fuse GELU into first GEMM epilogue when supported.

Preconditions:

- Static projector widths.
- Biases present for both Linear layers.
- Activation exactly GELU from `ACT2FN`.

Failure cases: alternate `projector_hidden_act`, different number of vision layers changing first GEMM width.

Parity test sketch: random `[N,576,5120]` features, compare projector output in fp32 and fp16.

### Rewrite: masked_scatter image stitch -> indexed copy

Source pattern:

```text
mask = input_ids == image_token_id
inputs_embeds.masked_scatter(mask[...,None].expand_as(inputs_embeds), image_features)
```

Replacement:

```text
positions = nonzero(input_ids == image_token_id) in row-major order
copy flattened image_features into inputs_embeds[positions, :]
```

Preconditions:

- `input_ids` is available.
- `positions.numel() == image_features.shape[0] * image_features.shape[1]`.
- Flatten order matches PyTorch boolean masked scatter.

Failure cases: `inputs_embeds`-only path, token/feature count mismatch, mixed multi-image packing where ordering is ambiguous without preserving row-major nonzero order.

Parity test sketch: batch with two samples and multiple image-token runs; compare stitched embeddings exactly in fp32.

### Rewrite: decode pixel guard

Source pattern: `prepare_inputs_for_generation` forwards `pixel_values` only on first iteration or when cache disabled.

Replacement: explicit generation-stage state:

```text
prefill_or_uncached -> run image/projector/stitch
cached_decode -> text decoder only
```

Preconditions: cache is valid and contains the multimodal prefix.

Failure cases: cache continuation where first iteration has precomputed system prompt but still needs current image tokens; source comment notes first iteration is not necessarily full prefill.

Parity test sketch: generate one token with image prefill, then next token with cache and verify vision tower is not called.

## 10. Kernel fusion candidates

Highest priority:

- CLIP patch embedding Conv2d-to-GEMM or direct patch GEMM. It is the first vision bottleneck and has strict non-overlap guards.
- CLIP encoder LayerNorm + QKV projection + attention backend. Vision prefill is fixed-size and batch-throughput friendly.
- ViP feature selection + projector LayerNorm/GEMM/GELU/GEMM. The 5120-wide input is unique to ViP-LLaVA and easy to validate independently.
- Llama RMSNorm, RoPE, causal attention with KV cache, and SwiGLU MLP. These dominate text prefill/decode.
- Masked scatter as indexed copy. It is small but required for correctness and easier to compile than general `masked_scatter`.

Medium priority:

- Last-token-only logits via `logits_to_keep` to avoid full-sequence LM head on decode or sampling-only paths.
- Fused Llama Q/K/V projections where weight layout permits, with RoPE immediately after projection.
- CLIP MLP GELU epilogue fusion.
- Attention backend comparison: eager vs SDPA/Flash for CLIP fixed sequence 577 and Llama prefill/decode.

Lower priority:

- CLIP position interpolation for non-336 images.
- `inputs_embeds`-only placeholder equality path.
- Multi-GPU/tensor-parallel plans from delegated Llama.
- Training-only dropout, gradient checkpointing, and loss paths.

## 11. Runtime staging plan

Stage 1: parse native `VipLlavaConfig`, normalize legacy 7B compatibility keys, load CLIP/projector/Llama weights, and reject non-native `mucai/*` remote-code configs.

Stage 2: implement image preprocessing contract outside the compiled graph and validate CLIP patch embedding plus one CLIP encoder layer.

Stage 3: run full CLIP vision tower with `output_hidden_states=True`; validate selected hidden-state taps and CLS-drop/concat.

Stage 4: implement projector parity for `[N,576,5120] -> [N,576,H_text]`.

Stage 5: implement placeholder stitch using `input_ids` and indexed copy; require `input_ids` for first multimodal path and reject `inputs_embeds`-only multimodal calls.

Stage 6: connect to Llama prefill and compare logits with image input for short prompts.

Stage 7: implement cached decode and pixel guard so vision/projector are skipped after cached prefill.

Stage 8: enable optimized attention/GEMM/fusion passes and last-token-only logits.

Stage 9: broaden only after parity: variable image interpolation, historical configs, GQA Llama-3 variants as separate model-family audits.

## 12. Parity and validation plan

- Config tests: 13B native config should produce projector input width 5120; 7B legacy config should either normalize to `vision_feature_layers` or be rejected with a clear compatibility error.
- Image processor parity: one PIL image through HF processor vs DinoML data pipeline, compare `pixel_values` shape `[1,3,336,336]` and fp32 values.
- Patch embedding parity: random image through CLIP Conv2d+flatten+transpose vs DinoML lowering.
- CLIP layer parity: one encoder block, then full 24-layer hidden-state tuple for fp32 and fp16.
- Feature selection parity: compare `[-2,-5,-8,-11,6]` CLS-drop concat and repeated-index tiny test config.
- Projector parity: random `[N,576,5120]` into LayerNorm/GELU MLP; tolerances fp32 `1e-5`/`1e-4`, fp16 `5e-3`/`5e-2` depending on backend.
- Stitch parity: prompts with one image, multiple batch rows, and deliberate mismatch. Validate row-major replacement order and exact error on count mismatch.
- Prefill logits parity: fixed prompt/image, compare logits for the prompt tail. Use `logits_to_keep=1` and full logits variants.
- Decode parity: one cached decode step should match HF logits and should not consume `pixel_values`.
- End-to-end smoke: reproduce a short answer shape/token-prefix parity against `llava-hf/vip-llava-7b-hf` or 13B at fp16 tolerance; exact text can vary with sampling, so use greedy decode for token parity.

## 13. Performance probes

- CPU preprocessing throughput: images/sec for resize/crop/normalize and token expansion.
- CLIP vision throughput: batch sweep for `[N,3,336,336]`, report encoder latency separately from projector.
- Feature pyramid/projector cost: measure selected layer concat and `Linear(5120 -> H_text)` bandwidth/GEMM time.
- Prefill throughput: sequence length sweep including 576 image tokens plus text prompt length.
- Decode throughput: tokens/sec with KV cache, batch sweep, with and without last-token-only logits.
- LM head cost: full sequence logits vs `logits_to_keep=1`.
- KV cache memory: native 7B and 13B per token, per batch, with MHA cache shapes.
- Attention backend comparison: CLIP fixed seq 577 and Llama prefill/decode under eager/SDPA/Flash candidates.
- End-to-end request latency split: preprocessing, vision/projector, stitch, prefill, decode.

Any measured observations should be labeled separately; this report includes source/config-derived probes only.

## 14. Skip/defer list

- Training, losses, dropout behavior, gradient checkpointing.
- `inputs_embeds`-only multimodal placeholder detection.
- Variable-resolution CLIP position interpolation.
- Historical non-native `mucai` configs and `model_type="llava_llama"` variants.
- Multi-GPU tensor parallel and pipeline parallel.
- Quantization-specific kernels; 4-bit loading appears in HF tests but is not native graph behavior.
- Beam search and advanced generation controllers beyond standard greedy/sampling hooks.
- Broader remote-code behavior from original ViP-LLaVA repositories.

## 15. Final implementation checklist

- [ ] Parse `VipLlavaConfig` and delegated CLIP/Llama configs.
- [ ] Add compatibility decision for legacy 7B `vision_feature_layer` and ignored `projector_layernorm` fields.
- [ ] Load token embeddings, CLIP vision weights, projector weights, Llama decoder weights, and LM head with tied-weight alias awareness.
- [ ] Implement CLIP image preprocessing contract or require preprocessed `pixel_values`.
- [ ] Lower CLIP patch embedding and encoder blocks.
- [ ] Capture CLIP hidden states needed by `vision_feature_layers`.
- [ ] Implement CLS-drop and multi-layer feature concat.
- [ ] Implement projector LayerNorm + GEMM + GELU + GEMM.
- [ ] Implement image-token placeholder count check.
- [ ] Implement indexed-copy replacement for multimodal embedding stitch.
- [ ] Implement Llama prefill with RoPE and cache update.
- [ ] Implement cached decode without rerunning vision/projector.
- [ ] Preserve `logits_to_keep` LM-head slicing.
- [ ] Add patch embedding, feature-selection, projector, stitch, prefill, and decode parity tests.
- [ ] Benchmark preprocessing, vision/projector, prefill, decode, LM head, and KV cache memory.
