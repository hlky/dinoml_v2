# SigLIP Transformers Family Audit

Primary target: image-text dual encoder contrastive inference for `SiglipModel`, with separately stageable `get_image_features` and `get_text_features`.

## 1. Source basis

```text
Transformers commit/version:
  b75feb2af64c3e29cbbc1bd859958c5432cc7ed4
Model id:
  siglip family; representative checkpoints listed below
Config source:
  Hugging Face config.json, preprocessor_config.json, tokenizer_config.json, special_tokens_map.json
Source files inspected:
  X:/H/transformers/src/transformers/models/siglip/modeling_siglip.py
  X:/H/transformers/src/transformers/models/siglip/configuration_siglip.py
  X:/H/transformers/src/transformers/models/siglip/processing_siglip.py
  X:/H/transformers/src/transformers/models/siglip/image_processing_siglip.py
  X:/H/transformers/src/transformers/models/siglip/image_processing_pil_siglip.py
  X:/H/transformers/src/transformers/models/siglip/tokenization_siglip.py
  X:/H/transformers/src/transformers/models/siglip/convert_siglip_to_hf.py
  X:/H/transformers/src/transformers/image_processing_backends.py
Any missing files or assumptions:
  No remote-code files are required for the inspected official SigLIP v1 checkpoints. The local in-library source is the authoritative implementation for this report. SigLIP2 checkpoint names appear in the converter but are not the primary scope here.
```

Pinned source URLs:

- `modeling_siglip.py`: https://github.com/huggingface/transformers/blob/b75feb2af64c3e29cbbc1bd859958c5432cc7ed4/src/transformers/models/siglip/modeling_siglip.py
- `configuration_siglip.py`: https://github.com/huggingface/transformers/blob/b75feb2af64c3e29cbbc1bd859958c5432cc7ed4/src/transformers/models/siglip/configuration_siglip.py
- `image_processing_siglip.py`: https://github.com/huggingface/transformers/blob/b75feb2af64c3e29cbbc1bd859958c5432cc7ed4/src/transformers/models/siglip/image_processing_siglip.py
- `tokenization_siglip.py`: https://github.com/huggingface/transformers/blob/b75feb2af64c3e29cbbc1bd859958c5432cc7ed4/src/transformers/models/siglip/tokenization_siglip.py
- `convert_siglip_to_hf.py`: https://github.com/huggingface/transformers/blob/b75feb2af64c3e29cbbc1bd859958c5432cc7ed4/src/transformers/models/siglip/convert_siglip_to_hf.py

Representative HF configs fetched:

- `hf-internal-testing/tiny-random-SiglipModel`: https://huggingface.co/hf-internal-testing/tiny-random-SiglipModel/resolve/main/config.json
- `google/siglip-base-patch16-224`: https://huggingface.co/google/siglip-base-patch16-224/resolve/main/config.json
- `google/siglip-base-patch16-256`: https://huggingface.co/google/siglip-base-patch16-256/resolve/main/config.json
- `google/siglip-base-patch16-384`: https://huggingface.co/google/siglip-base-patch16-384/resolve/main/config.json
- `google/siglip-large-patch16-384`: https://huggingface.co/google/siglip-large-patch16-384/resolve/main/config.json
- `google/siglip-so400m-patch14-384`: https://huggingface.co/google/siglip-so400m-patch14-384/resolve/main/config.json

Processor/tokenizer configs were fetched from the same repos using `preprocessor_config.json`, `tokenizer_config.json`, and `special_tokens_map.json`.

## 2. High-level architecture

SigLIP is a CLIP-like dual encoder, but the source differs in important runtime details. The text branch is a bidirectional Transformer encoder, not a causal CLIP text encoder. The vision branch is a ViT-style patch encoder without a CLS token; it pools with a learned one-token multihead-attention pooling head. Both branches produce projected features through biased linear heads. Full `SiglipModel.forward` L2-normalizes both projected features, computes a text-by-image dot product matrix, applies learned `exp(logit_scale)` and learned additive `logit_bias`, and returns both matrix orientations. End-user probabilities are normally `sigmoid(logits_per_image)`.

Dataflow:

```text
images -> CPU/GPU image processor -> pixel_values[N,3,H,W]
       -> NCHW Conv2d patch embedding + learned patch positions -> vision encoder
       -> post LayerNorm -> MAP pooling head -> image_embeds

text -> SentencePiece tokenizer -> input_ids[N,S], optional attention_mask[N,S]
     -> token + learned position embeddings -> bidirectional text encoder
     -> final LayerNorm -> last-token pool -> biased text head -> text_embeds

image_embeds, text_embeds -> L2 normalize -> matmul -> exp(logit_scale) scale + logit_bias -> logits
```

Stage decomposition:

- CPU/data-pipeline: image RGB conversion, resize, rescale, normalize; SentencePiece tokenization, canonicalization/lowercasing, EOS insertion, padding/truncation.
- Independently cacheable encoders: image branch through `get_image_features`; text branch through `get_text_features`.
- Contrastive head: row-wise L2 norm, GEMM, scalar exponential scale, scalar bias add, transpose. This is small and can run after feature caches are built.
- No prefill/decode/KV-cache stage exists. Attention is encoder-style full-sequence attention.

Other heads:

- `SiglipTextModel` and `SiglipVisionModel`: required/useful for staged parity.
- `SiglipForImageClassification`: optional/deferred for primary contrastive inference. It reuses the vision encoder and mean-pools all post-LayerNorm patch tokens before a classifier.
- Training sigmoid loss: deferred for inference.

## 3. Important config dimensions

Source defaults from `configuration_siglip.py`:

| Field | Text default | Vision default | Notes |
| --- | ---: | ---: | --- |
| `vocab_size` | 32000 | n/a | SentencePiece token embedding rows. |
| `hidden_size` | 768 | 768 | Must divide `num_attention_heads`. |
| `intermediate_size` | 3072 | 3072 | Ungated MLP width. |
| `projection_size` | defaults to text hidden | n/a | Text head output width; converter sets it to vision hidden if asymmetric. |
| `num_hidden_layers` | 12 | 12 | Separate layer counts but normally symmetric in v1 SigLIP. |
| `num_attention_heads` | 12 | 12 | MHA, no GQA/MQA. |
| `head_dim` | 64 | 64 | Derived for base defaults. |
| `max_position_embeddings` | 64 | n/a | Text sequence cap. |
| `image_size` | n/a | 224 | Square processor configs in inspected checkpoints. |
| `patch_size` | n/a | 16 | Conv2d kernel and stride. |
| `num_channels` | n/a | 3 | Processor converts to RGB by default. |
| `hidden_act` | `gelu_pytorch_tanh` | `gelu_pytorch_tanh` | Transformers activation registry. |
| `layer_norm_eps` | 1e-6 | 1e-6 | PyTorch LayerNorm. |
| `attention_dropout` | 0.0 | 0.0 | Dropout disabled in inference. |
| Cache support | none | none | No KV cache or recurrent state. |

Representative checkpoint sweep:

| Checkpoint | Text shape | Vision shape | Projection / score head | Image / patch | Vision tokens | Processor | Source of facts |
| --- | --- | --- | --- | --- | ---: | --- | --- |
| `hf-internal-testing/tiny-random-SiglipModel` | 2 layers, H=32, heads=4, MLP=37, max text=512 | 2 layers, H=32, heads=4, MLP=37 | text head 32 -> 32, learned scale+bias | 30 / 2 | 225 | size 30, mean/std 0.5 | `config.json`, preprocessor config |
| `google/siglip-base-patch16-224` | 12 layers by default, H=768, heads=12, MLP=3072, vocab=32000, max text=64 by default | 12 layers by default, H=768, heads=12, MLP=3072 | text head 768 -> 768, learned scale+bias | 224 / 16 | 196 | size 224, bicubic, mean/std 0.5 | `config.json` plus source defaults |
| `google/siglip-base-patch16-256` | source default base text | source default base vision except image=256 | text head 768 -> 768, learned scale+bias | 256 / 16 | 256 | size 256, bicubic, mean/std 0.5 | `config.json` plus source defaults |
| `google/siglip-base-patch16-384` | source default base text | source default base vision except image=384 | text head 768 -> 768, learned scale+bias | 384 / 16 | 576 | size 384, bicubic, mean/std 0.5 | `config.json` plus source defaults |
| `google/siglip-large-patch16-384` | 24 layers, H=1024, heads=16, MLP=4096 | 24 layers, H=1024, heads=16, MLP=4096 | text head 1024 -> 1024, learned scale+bias | 384 / 16 | 576 | size 384, bicubic, mean/std 0.5 | `config.json`, preprocessor config |
| `google/siglip-so400m-patch14-384` | 27 layers, H=1152, heads=16, MLP=4304 | 27 layers, H=1152, heads=16, MLP=4304 | text head 1152 -> 1152, learned scale+bias | 384 / 14 | 729 | size 384, bicubic, mean/std 0.5 | `config.json`, preprocessor config |

Effective defaults and omitted fields:

- Several official configs are intentionally sparse. Missing text fields default to `hidden_size=768`, `intermediate_size=3072`, `num_hidden_layers=12`, `num_attention_heads=12`, `max_position_embeddings=64`, `hidden_act=gelu_pytorch_tanh`, `layer_norm_eps=1e-6`, `attention_dropout=0.0`, `pad_token_id=1`, `bos_token_id=49406`, `eos_token_id=49407`, and `projection_size=hidden_size`.
- Missing vision fields default to `hidden_size=768`, `intermediate_size=3072`, `num_hidden_layers=12`, `num_attention_heads=12`, `num_channels=3`, `image_size=224`, `patch_size=16`, `hidden_act=gelu_pytorch_tanh`, `layer_norm_eps=1e-6`, and `attention_dropout=0.0`.
- Processor configs for official v1 checkpoints use `image_mean=[0.5,0.5,0.5]`, `image_std=[0.5,0.5,0.5]`, `rescale_factor=1/255`, `resample=3` in the fetched JSON, and square `size`.
- Tokenizer configs set `model_max_length=64`, `do_lower_case=true`, `pad_token="</s>"`, `eos_token="</s>"`, `unk_token="<unk>"`, and `model_input_names=["input_ids"]`. The tokenizer class default includes `attention_mask`, but the official converter explicitly avoids returning it for original SigLIP.

## 3a. Family variation traps

- SigLIP is CLIP-like, but it is not a CLIP implementation with only a different loss. Runtime differences from the audited CLIP report:
  - text attention is bidirectional, not causal;
  - text pooling is `last_hidden_state[:, -1, :]`, not EOT/EOS gather;
  - vision has no CLS token and no pre-LayerNorm before the encoder;
  - vision pooling uses a learned MAP head with `torch.nn.MultiheadAttention`, LayerNorm, and MLP;
  - projection heads are biased linears, not CLIP's bias-free `text_projection`/`visual_projection`;
  - similarity uses learned additive `logit_bias` in addition to `exp(logit_scale)`;
  - sigmoid probabilities are downstream behavior, not softmax over candidate labels.
- Official tokenizer configs may omit `attention_mask`. If no mask is passed, the text encoder is fully bidirectional over all positions, including padding tokens. The source then pools the last position, which may itself be padding. For parity with official examples, use `padding="max_length"` and do not assume EOS pooling.
- `SiglipVisionEmbeddings` uses `nn.Conv2d(..., padding="valid")` with default bias present. This differs from CLIP's patch embedding, which is bias-free in the audited CLIP source.
- Vision tokens exclude a class token. Position embedding length is exactly `(image_size // patch_size) ** 2`.
- `google/siglip-so400m-patch14-384` has `384 // 14 == 27`; the source Conv2d floor behavior yields 27x27 patches and leaves edge pixels outside the non-overlap windows. Do not require exact divisibility unless a stricter DinoML admission policy intentionally rejects this checkpoint.
- `image_size` and `patch_size` types allow lists/tuples in config annotations, but the current source computes `(self.image_size // self.patch_size) ** 2` and uses scalar `height // self.patch_size`. First integration should require scalar integer values.
- The MAP pooling head uses PyTorch `MultiheadAttention`, whose packed `in_proj_weight` layout is `[Q; K; V]` rows and `in_proj_bias` is `[bq; bk; bv]`. The converter packs original Q/K/V this way.
- Source `_attn_implementation` dispatch supports eager, SDPA, FlashAttention, and FlexAttention through `ALL_ATTENTION_FUNCTIONS`; Dynamo lowering should first match eager-visible semantics.
- No RoPE, ALiBi, relative bias, sliding window, GQA/MQA, MoE, cross-attention, varlen metadata, generation cache, or token-type IDs are required.
- NCHW/NHWC trap: model entry `pixel_values` is NCHW and patch embedding is NCHW Conv2d. NHWC is only a guarded local optimization around patch extraction/projection.
- Axis-sensitive ops needing no-layout-translation guards include patch Conv2d channel axis, flatten spatial axes, interpolation's NCHW bicubic path, text/pooling sequence axes, mean pooling in `SiglipForImageClassification`, and L2 normalization over the last embedding axis.

## 4. Operator coverage checklist

Tensor/layout ops:

- Text `input_ids.view(-1, S)` from arbitrary leading batch dims.
- Embedding lookup for token, text position, and vision position tables.
- NCHW Conv2d output flatten: `[B, Dv, Gh, Gw] -> [B, Dv, Gh*Gw] -> [B, Gh*Gw, Dv]`.
- Add learned position embeddings with broadcasting.
- Attention reshape/transposes: `[B, T, D] -> [B, H, T, Dh]`, then output transpose/contiguous/reshape back.
- MAP probe repeat: `[1,1,Dv] -> [B,1,Dv]`.
- Last-token text pool: `last_hidden_state[:, -1, :]`.
- Vision MAP output slice: `[B,1,Dv] -> [B,Dv]`.
- L2 norm over last dimension.
- Transpose logits matrix for `logits_per_image`.
- Optional bicubic interpolation path for vision position embeddings.

Neural network primitives:

- NCHW `Conv2d(Cin=3, Cout=Dv, kernel=P, stride=P, padding=valid, bias=True)`.
- Dense GEMMs for encoder Q, K, V, output projection. All `SiglipAttention` linears have bias.
- Dense GEMMs for MLP `hidden -> intermediate -> hidden`, both with bias.
- Text projection head: `Linear(Dt -> text_projection_size, bias=True)`.
- Vision MAP pooling head:
  - learned probe query `[B,1,Dv]`;
  - packed MHA QKV projection with input query/probe and key/value hidden states;
  - output projection, LayerNorm, MLP, residual add, slice.
- LayerNorm over last dimension with epsilon 1e-6.
- `gelu_pytorch_tanh`.
- Residual adds.
- L2 normalization, contrastive GEMM, `exp(logit_scale)`, scalar multiply, scalar bias add.

Attention primitives:

- Standard MHA self-attention for text and vision encoders.
- Cross-attention-like MAP pooling call: query is learned probe, key/value are vision sequence.
- Eager math: `softmax((Q @ K^T) * head_dim^-0.5 + mask, dim=-1, dtype=float32).to(query.dtype) @ V`.
- Text branch: bidirectional additive mask from `create_bidirectional_mask` only when `attention_mask` is supplied or required by backend.
- Vision branch: no mask in normal source path.
- MAP pooling head: source uses `torch.nn.MultiheadAttention(..., batch_first=True)` and returns attention output only.

Preprocessing-coupled ops:

- Image: RGB conversion, resize to configured square `size`, rescale by `1/255`, normalize by 0.5 mean/std, output channel-first `pixel_values`.
- Text: SentencePiece, lowercasing, punctuation removal in tokenizer canonicalization, whitespace collapse, EOS appending, right padding/truncation by tokenizer settings.

No required generation/cache ops, multimodal embedding scatter, packed sequence metadata, or distributed/tensor-parallel ops for the primary target.

## 5. Layer/block breakdown

Vision embeddings:

```text
pixel_values: [B, 3, H, W] NCHW
patch = Conv2d(3 -> Dv, kernel=P, stride=P, padding=valid, bias=True)(pixel_values.to(weight_dtype))
patch = flatten spatial then transpose: [B, Dv, Gh, Gw] -> [B, Gh*Gw, Dv]
x = patch + learned_position[0 : Gh*Gw]
```

Vision encoder:

```text
repeat Nv layers:
  r = x
  x = LayerNorm(x)
  q,k,v = Linear(Dv -> Dv, bias=True)(x)
  x = noncausal MHA(q,k,v, mask=None)
  x = r + Linear(Dv -> Dv, bias=True)(x)
  r = x
  x = LayerNorm(x)
  x = r + Linear(Iv -> Dv, bias=True)(gelu_pytorch_tanh(Linear(Dv -> Iv, bias=True)(x)))
x = post_layernorm(x)
```

Vision MAP pooling:

```text
probe = learned_probe.repeat(B, 1, 1)        # [B,1,Dv]
pooled = MultiheadAttention(query=probe, key=x, value=x, batch_first=True)[0]
r = pooled
pooled = LayerNorm(pooled)
pooled = r + MLP(pooled)
image_features = pooled[:, 0, :]             # [B,Dv]
```

Text embeddings:

```text
input_ids: [B, S]
position_ids default: [1, S] = arange(S)
x = token_embedding[input_ids] + position_embedding[position_ids]  # [B,S,Dt]
```

Text encoder:

```text
mask = create_bidirectional_mask(config, inputs_embeds=x, attention_mask=attention_mask)
repeat Nt layers:
  r = x
  x = LayerNorm(x)
  q,k,v = Linear(Dt -> Dt, bias=True)(x)
  x = bidirectional MHA(q,k,v, mask)
  x = r + Linear(Dt -> Dt, bias=True)(x)
  r = x
  x = LayerNorm(x)
  x = r + Linear(It -> Dt, bias=True)(gelu_pytorch_tanh(Linear(Dt -> It, bias=True)(x)))
x = final_layer_norm(x)
text_features = Linear(Dt -> projection_size, bias=True)(x[:, -1, :])
```

Contrastive head:

```text
image_embeds = image_features / norm(image_features, p=2, dim=-1, keepdim=True)
text_embeds = text_features / norm(text_features, p=2, dim=-1, keepdim=True)
logits_per_text = text_embeds @ image_embeds.T
logits_per_text = logits_per_text * exp(logit_scale) + logit_bias
logits_per_image = logits_per_text.T
probs = sigmoid(logits_per_image)            # user-facing examples, not returned by forward
```

## 6. Attention requirements

Encoder text attention:

- Bidirectional self-attention, not causal.
- MHA only. Base text shape is `[B, 12, S<=64, 64]`; Large is `[B, 16, S<=64, 64]`; SO400M is `[B, 16, S<=64, 72]`.
- Optional additive bidirectional padding mask. Official tokenizer configs list only `input_ids`, so first parity should test both no-mask official path and explicit mask path.
- No KV cache. There are no before/after cache tensors.
- No RoPE or position bias; learned absolute positions are added before attention.
- Eager softmax is computed in fp32 and cast back to query dtype. Dropout is `0.0` in inference.

Vision encoder attention:

- Noncausal self-attention.
- MHA only. Base 224 has `[B, 12, 196, 64]`; Base 384 has `[B, 12, 576, 64]`; SO400M has `[B, 16, 729, 72]`.
- No attention mask in the normal image path.
- No CLS token. All tokens are patch tokens.

Vision MAP pooling attention:

- Single-query attention from learned probe to the full vision token sequence.
- Query length is 1; key/value length is `Gh * Gw`.
- Uses PyTorch `nn.MultiheadAttention` with packed QKV weights. This is structurally different from the encoder `SiglipAttention` class and should be audited as its own lowering pattern.
- No mask, no cache, no causal flag.

FlashAttention/SDPA compatibility:

- Source advertises backend support through Transformers attention interfaces. DinoML fused attention can target noncausal MHA for vision, bidirectional masked MHA for text, and query-length-1 MAP attention, but should preserve fp32 softmax and additive-mask ordering before enabling approximate kernels.

## 7. Position encoding and custom math

Text positions are learned absolute embeddings indexed by default `position_ids[:, :S]`. Vision positions are learned absolute embeddings over patch tokens only; there is no CLS row.

Optional vision position interpolation:

```python
def interpolate_siglip_vision_pos(position_table, embeddings, height, width, patch_size):
    # position_table: [num_positions, dim], no CLS token
    num_positions = position_table.shape[0]
    dim = embeddings.shape[-1]
    new_h = height // patch_size
    new_w = width // patch_size
    old = int(num_positions ** 0.5)
    pos = position_table.unsqueeze(0).reshape(1, old, old, dim)
    pos = pos.permute(0, 3, 1, 2)
    pos = bicubic_interpolate(pos, size=(new_h, new_w), align_corners=False)
    return pos.permute(0, 2, 3, 1).view(1, -1, dim)
```

This can be precomputed for fixed alternate image sizes. It depends on runtime `height` and `width` only when `interpolate_pos_encoding=True` or when tracing dynamic shapes. First integration can require configured image size and skip interpolation.

Custom math:

- `gelu_pytorch_tanh` from Transformers activation registry.
- L2 norm uses `Tensor.norm(p=2, dim=-1, keepdim=True)`. Unlike the CLIP report's source math, this is the PyTorch norm call, not an explicit `_get_vector_norm` helper.
- Logit scale is a learned scalar parameter stored in log space and applied as `exp(logit_scale)`.
- Logit bias is a learned scalar added after scaling. This is required for parity.
- Training-only sigmoid loss uses labels encoded as `+1` for diagonal pairs and `-1` for off-diagonal pairs with `logsigmoid(sign * logits)`.

## 8. Preprocessing and input packing

Image processor contract:

- Source class defaults: bicubic resize, ImageNet standard mean/std defaults in class definition, resize/rescale/normalize/RGB conversion enabled, `default_to_square=False`.
- Official v1 SigLIP preprocessor configs override mean/std to `[0.5,0.5,0.5]`, use `rescale_factor=1/255`, and square `size={"height": image_size, "width": image_size}`.
- The Torchvision backend converts PIL/NumPy inputs to channel-first tensors, converts channel-last inputs to channel-first, and returns `pixel_values` as `[B,3,H,W]`.
- The model casts `pixel_values` to the patch embedding weight dtype before Conv2d.
- GPU runtime graph can start at already-processed `pixel_values`. Image decoding and resize/normalize can stay in the CPU/data pipeline for Stage 1.

Text tokenizer contract:

- `SiglipTokenizer` is SentencePiece-based. It lowercases by default, removes punctuation in `canonicalize_text`, collapses whitespace, strips, and appends EOS if absent.
- Single sequence format is `X </s>`. Pair format is `A </s> B </s>`.
- Official tokenizer configs set `model_max_length=64`, `pad_token="</s>"`, `eos_token="</s>"`, `unk_token="<unk>"`, and `model_input_names=["input_ids"]`.
- Token type IDs are all zeros if requested, but they are not used by the model.
- Important runtime effect: since the text model pools `last_hidden_state[:, -1, :]`, padding strategy changes the pooled token. Official examples use `padding="max_length"` because this is how the model was trained.

There are no modality placeholder tokens, image/text embedding stitch, grid metadata, packed patch rows, or cu-seqlens-style descriptors.

## 9. Graph rewrite / lowering opportunities

### Rewrite: non-overlap Conv2d patch embedding -> GEMM

Source pattern:

```text
Conv2d(Cin=3, Cout=Dv, kernel=P, stride=P, padding=valid, bias=True)
flatten(2).transpose(1, 2)
```

Replacement:

```text
WindowFlatten NCHW patches [B, Gh, Gw, 3*P*P]
-> MatMul(weight_flat.T) [3*P*P -> Dv]
-> BiasAdd
-> Reshape [B, Gh*Gw, Dv]
```

Preconditions:

- `kernel_size == stride == patch_size`.
- `padding == valid`, `dilation == 1`, `groups == 1`.
- Bias must be included.
- Output grid uses PyTorch Conv2d floor formula. Do not require `H % P == 0` if supporting SO400M patch14 at 384.
- Flatten order must match PyTorch Conv2d NCHW receptive-field order.

Weight transform:

```python
w = conv.weight.reshape(out_channels, in_channels * patch_h * patch_w)
b = conv.bias
```

Layout constraints:

- Faithful source layout is NCHW. NHWC optimization may use an NHWC patch extractor, but must transform window flatten order or weight layout to preserve NCHW Conv2d semantics.
- Downstream consumer requires `[B, Gh*Gw, Dv]`.

Failure cases:

- Non-scalar list/tuple image or patch config until DinoML admits that shape contract.
- Dynamic image sizes without position interpolation.
- External callers expecting direct NCHW `pixel_values` must not be silently switched to NHWC.

Parity test sketch:

- Compare Conv2d+flatten+transpose against WindowFlatten+GEMM for Base 224, Base 384, and SO400M 384/14.
- Include fp32 and fp16 weights, batch > 1, and nonzero bias.

### Rewrite: encoder packed QKV projection

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

- Same input tensor and dtype for Q/K/V.
- All three projections have bias.
- Preserve split order `q, k, v`.
- No source-observability requirement for separate projection modules.

Weight transform:

```python
Wqkv = concat([Wq, Wk, Wv], dim=0)
bqkv = concat([bq, bk, bv], dim=0)
```

Failure cases:

- Debug/export modes that require named separate Q/K/V outputs.
- MAP pooling head already stores packed QKV in PyTorch MHA layout; do not repack it as if it were three separate `SiglipAttention` modules.

Parity test sketch:

- One text and one vision `SiglipEncoderLayer`, eager backend, random hidden states, compare packed and unpacked Q/K/V plus final attention output.

### Rewrite: MAP pooling head lowering

Source pattern:

```text
probe.repeat(B,1,1)
torch.nn.MultiheadAttention(query=probe, key=x, value=x, batch_first=True)
LayerNorm + MLP residual
slice token 0
```

Replacement:

```text
Repeat/expand probe -> packed-QKV attention with Tq=1, Tk=Tv=Gh*Gw
-> output projection -> LayerNorm -> MLP residual -> squeeze sequence
```

Preconditions:

- `batch_first=True`.
- Packed weight rows are `[Q; K; V]`.
- Query length is exactly 1.
- No attention mask.

Failure cases:

- If `vision_use_head=False`, `SiglipVisionModel` returns `pooler_output=None`; full `SiglipModel` should reject that combination for contrastive inference unless a separate pooling policy is supplied.

Parity test sketch:

- Compare PyTorch `nn.MultiheadAttention` MAP head against DinoML packed attention for Base and SO400M hidden/head sizes.

### Rewrite: L2 normalize + sigmoid-logit GEMM

Source pattern:

```text
x = x / norm(x, p=2, dim=-1, keepdim=True)
y = y / norm(y, p=2, dim=-1, keepdim=True)
logits = (text @ image.T) * exp(logit_scale) + logit_bias
```

Replacement:

- Fused row-wise norm kernel for image/text embeddings, followed by GEMM and scalar epilogue multiply-add.

Preconditions:

- Last dimension equals shared embedding dimension.
- Source has no epsilon in normalization; do not add one unless a guarded approximate mode is selected.
- `logit_scale` and `logit_bias` are scalar parameters.

Parity test sketch:

- Compare logits for random projected features against source math, including unequal image/text batch sizes. Avoid zero vectors for first pass.

### Layout rewrite: NCHW image ingress -> local NHWC patch region

Opportunity:

- Processor/model source contract is NCHW. A local layout pass can convert only the patch embedding region to NHWC/channel-last for better memory behavior.

Required guards:

- Region includes only processor output or explicit transpose, patch extraction/Conv2d, bias add, flatten, and immediate projection to sequence tokens.
- Axis rewrite: source channel axis `dim=1` becomes last channel in NHWC.
- Weight transform preserves NCHW Conv2d kernel order.
- Position embedding and Transformer encoder consume sequence-major `[B,tokens,hidden]` and are layout-neutral after patch projection.

No-layout-translation guards:

- Positional interpolation path uses NCHW around bicubic interpolate.
- Public `pixel_values` input remains NCHW unless DinoML exposes a separate NHWC contract.
- Image classification mean pool is over sequence dimension `dim=1`, not channel.

## 10. Kernel fusion candidates

Highest priority:

- LayerNorm over `[B,T,D]`: appears before every attention and MLP plus final/post norms and MAP head norm.
- Dense GEMM epilogues with bias and residual add for attention output and MLP second projection.
- Encoder packed QKV projection for both text and vision branches.
- Conv patch embedding lowered to GEMM or specialized patch kernel, especially for 576-token and 729-token vision configs.
- Fused attention for noncausal encoder attention and bidirectional masked text attention.
- MAP pooling attention with `Tq=1`, because it is unique to SigLIP versus CLIP and sits on the critical image feature tail.

Medium priority:

- `gelu_pytorch_tanh` fused into MLP.
- Row-wise L2 normalization plus contrastive GEMM scalar multiply-add.
- Text last-token pool plus biased projection head.
- Position-add fusion after patch/text embeddings.

Lower priority:

- Bicubic positional interpolation.
- Image classification mean-pool/classifier head.
- Training sigmoid loss.
- Full image processor inside DinoML runtime.

## 11. Runtime staging plan

Stage 1: config, processor metadata, tokenizer metadata, and weights.

- Parse nested `SiglipConfig`, `SiglipTextConfig`, and `SiglipVisionConfig`.
- Apply source defaults for sparse official configs.
- Load text encoder, vision encoder, MAP pooling head, text head, `logit_scale`, and `logit_bias`.
- Keep preprocessing outside DinoML and accept `pixel_values` plus `input_ids` and optional `attention_mask`.

Stage 2: independent text encoder parity.

- Implement token/position embeddings, bidirectional mask handling, MHA, LayerNorm, MLP, final norm, last-token pool, and biased text head.
- Validate `get_text_features` before contrastive logits.

Stage 3: independent vision encoder parity.

- Implement NCHW patch embedding with bias, learned patch positions, noncausal encoder, post LayerNorm, and MAP pooling head.
- Validate `get_image_features` for fixed image sizes first.

Stage 4: contrastive head.

- Add L2 normalization, text-by-image GEMM, `exp(logit_scale)`, `logit_bias`, and transpose.
- Validate logits and downstream `sigmoid` probabilities for unequal image/text batch sizes.

Stage 5: optimized kernels.

- Add guarded patch Conv2d-to-GEMM, packed QKV, fused attention, LayerNorm/GEMM epilogues, and MAP head specialization.

Stage 6: optional extensions.

- Positional interpolation for non-default image sizes.
- `SiglipForImageClassification`.
- Processor-in-runtime experiments.

Initially stubbable:

- `return_loss`, hidden-state/attention recording, gradient checkpointing, output tuple variants, image classification head, and positional interpolation.

## 12. Parity and validation plan

Unit parity:

- `gelu_pytorch_tanh` against Transformers activation registry.
- L2 norm and contrastive scalar scale+bias math.
- Text last-token pooling with and without padding.
- Bidirectional additive mask values for no-mask, all-ones mask, and padded mask.
- NCHW patch Conv2d+flatten+transpose against rewritten patch GEMM, including nonzero bias.
- MAP pooling head against PyTorch `nn.MultiheadAttention` packed weights.

Layer parity:

- One text `SiglipEncoderLayer` with bidirectional mask, fp32 tolerance around `1e-5` to `1e-4`.
- One vision `SiglipEncoderLayer` with no mask.
- Full text encoder for Base prompts with `S=64`.
- Full vision encoder for Base 224, Base 384, Large 384, and SO400M 384.

End-to-end parity:

- `get_text_features` for 2-4 prompts using official tokenizer settings.
- `get_image_features` for one image and batched images after HF processor.
- Full `SiglipModel.forward` logits for unequal image/text batch sizes, verifying `logits_per_text`, `logits_per_image`, and `sigmoid(logits_per_image)`.
- Checkpoint sweep: tiny-random, Base 224, Base 256 or 384, Large 384, and SO400M 384.

Suggested tolerances:

- fp32: `atol=1e-4`, `rtol=1e-4` for layer/logit parity.
- fp16/bf16 optimized kernels: start with `atol=2e-2`, `rtol=2e-2` for full logits, then tighten per kernel.
- Attention backend parity should compare eager math before enabling FlashAttention-style kernels.

No DinoML tests were run for this docs-only audit.

## 13. Performance probes

- Processor throughput: images/sec for resize/rescale/normalize on CPU versus optional GPU preprocessing.
- Patch embedding throughput: Conv2d source path versus patch-GEMM rewrite for patch16 and patch14.
- Vision encoder throughput: batch sweep for 196, 256, 576, and 729 vision tokens.
- MAP pooling head latency and throughput with `Tq=1`, separated from the main vision encoder.
- Text encoder throughput: prompt batch sweep for `S=64`, plus tiny fixture `S=512` if supporting generic configs.
- Attention backend comparison: eager, SDPA-like, FlashAttention-like for bidirectional text and noncausal vision.
- Projection and contrastive head: text batch by image batch matrix sizes, including many-text/few-image zero-shot classification.
- End-to-end retrieval: separate image feature caching, text feature caching, logits-only recompute, and sigmoid postprocessing.
- Memory probes: activation memory by vision token count and batch size; no KV cache memory probe is needed.

## 14. Skip/defer list

- Training sigmoid loss and gradient checkpointing.
- Hidden state and attention output capture.
- Beam search, sampling, decode loops, and KV cache.
- Positional interpolation for non-configured image sizes.
- `SiglipForImageClassification`.
- SigLIP2 variants and Gemma tokenizer behavior from the converter.
- Quantization and multi-GPU tensor parallel.
- Full processor implementation inside DinoML runtime; first integration can consume already-preprocessed tensors.

## 15. Final implementation checklist

- [ ] Parse nested SigLIP text/vision/top-level config and apply source defaults.
- [ ] Load text encoder, vision encoder, MAP pooling head, text head, `logit_scale`, and `logit_bias`.
- [ ] Accept processor outputs: `pixel_values[B,3,H,W]`, `input_ids[B,S]`, optional `attention_mask[B,S]`.
- [ ] Preserve official tokenizer behavior enough to validate EOS/pad/max-length and last-token pooling.
- [ ] Implement NCHW biased vision patch embedding and shape/grid checks.
- [ ] Implement learned absolute patch positions without CLS token.
- [ ] Implement text token/position embeddings and max-position guard.
- [ ] Implement bidirectional text mask with optional padding mask.
- [ ] Implement MHA with fp32 softmax parity and noncausal/bidirectional modes.
- [ ] Implement LayerNorm eps 1e-6, `gelu_pytorch_tanh` MLP, and residual adds.
- [ ] Implement vision post LayerNorm and MAP pooling head with packed `[Q;K;V]` weights.
- [ ] Implement text last-token pool and biased projection head.
- [ ] Implement L2 feature normalization, contrastive matmul, `exp(logit_scale)` scale, `logit_bias`, transpose, and sigmoid parity helper.
- [ ] Add one-layer text, vision, and MAP-head parity tests.
- [ ] Add `get_text_features`, `get_image_features`, and full logits parity tests.
- [ ] Add checkpoint sweep coverage for tiny-random, Base 224/384, Large 384, and SO400M 384.
- [ ] Add guarded Conv2d patch embedding to GEMM rewrite with bias and floor-grid handling.
- [ ] Add packed QKV rewrite with separate handling for encoder attention versus MAP pooling.
- [ ] Add benchmarks separating preprocessing, patch embedding, vision encoder, MAP pooling, text encoder, and logits head.
