# CLIPSeg Transformers Family Audit

Primary target: `CLIPSegForImageSegmentation` zero-shot/one-shot image segmentation with text prompts, visual prompts, or caller-provided conditional embeddings. `CLIPSegModel` contrastive dual-encoder inference is a useful staged subtarget but is not the final product path.

## 1. Source basis

```text
Transformers commit/version:
  Local checkout transformers at b75feb2af64c3e29cbbc1bd859958c5432cc7ed4.

Model id:
  Primary production checkpoint: CIDAS/clipseg-rd64-refined.
  Additional checkpoint configs: CIDAS/clipseg-rd64, CIDAS/clipseg-rd16,
  hf-internal-testing/tiny-random-CLIPSegModel.

Config source:
  Hugging Face config.json, preprocessor_config.json, tokenizer_config.json,
  model API metadata, and local Transformers source defaults.

Source files inspected:
  transformers/src/transformers/models/clipseg/modeling_clipseg.py
  transformers/src/transformers/models/clipseg/modular_clipseg.py
  transformers/src/transformers/models/clipseg/configuration_clipseg.py
  transformers/src/transformers/models/clipseg/processing_clipseg.py
  transformers/src/transformers/models/clipseg/convert_clipseg_original_pytorch_to_hf.py
  transformers/tests/models/clipseg/test_modeling_clipseg.py
  transformers/tests/models/clipseg/test_processing_clipseg.py
  Prior reports: agents/plans/transformers/clip/report.md and
  agents/plans/transformers/vit/report.md.

Any missing files or assumptions:
  There is no CLIPSeg-specific image processor implementation in the model
  directory. The official processor combines a CLIP tokenizer with an image
  processor configured from preprocessor_config.json. `modeling_clipseg.py` and
  `configuration_clipseg.py` are generated from `modular_clipseg.py`; future
  source edits should inspect the modular file, while runtime imports use the
  generated files.
```

Pinned source URLs:

- `modeling_clipseg.py`: https://github.com/huggingface/transformers/blob/b75feb2af64c3e29cbbc1bd859958c5432cc7ed4/src/transformers/models/clipseg/modeling_clipseg.py
- `modular_clipseg.py`: https://github.com/huggingface/transformers/blob/b75feb2af64c3e29cbbc1bd859958c5432cc7ed4/src/transformers/models/clipseg/modular_clipseg.py
- `configuration_clipseg.py`: https://github.com/huggingface/transformers/blob/b75feb2af64c3e29cbbc1bd859958c5432cc7ed4/src/transformers/models/clipseg/configuration_clipseg.py
- `processing_clipseg.py`: https://github.com/huggingface/transformers/blob/b75feb2af64c3e29cbbc1bd859958c5432cc7ed4/src/transformers/models/clipseg/processing_clipseg.py

Representative HF config URLs:

- https://huggingface.co/CIDAS/clipseg-rd64-refined/resolve/main/config.json
- https://huggingface.co/CIDAS/clipseg-rd64-refined/resolve/main/preprocessor_config.json
- https://huggingface.co/CIDAS/clipseg-rd64-refined/resolve/main/tokenizer_config.json
- https://huggingface.co/CIDAS/clipseg-rd64/resolve/main/config.json
- https://huggingface.co/CIDAS/clipseg-rd16/resolve/main/config.json
- https://huggingface.co/hf-internal-testing/tiny-random-CLIPSegModel/resolve/main/config.json

## 2. High-level architecture

CLIPSeg is a CLIP-style dual encoder plus a lightweight Transformer segmentation decoder. The query image always runs through the CLIP vision encoder with hidden-state capture. The prompt condition is one of:

- text prompt -> CLIP text encoder -> text projection -> conditional embedding `[B, projection_dim]`;
- visual prompt image -> CLIP vision encoder -> visual projection -> conditional embedding `[B, projection_dim]`;
- caller-provided `conditional_embeddings` with shape `[B, projection_dim]`.

Dataflow:

```text
query image processor -> pixel_values[B,3,H,W]
  -> CLIPSeg vision patch encoder with hidden states
  -> extract configured vision layers
  -> reduce each extracted layer to reduce_dim
  -> FiLM condition from prompt embedding
  -> decoder transformer layers
  -> remove CLS, reshape patch grid
  -> ConvTranspose2d decoder head
  -> mask logits[B,Hmask,Wmask]

text or visual prompt -> CLIP text/vision branch -> projection -> conditional embedding
```

Stage decomposition:

- CPU/data pipeline: image decode, resize to processor `size`, rescale, normalize, CLIP tokenization, padding/truncation.
- Cacheable prompt branch: text/visual prompt embeddings can be computed once and reused for many query images, provided batch alignment is preserved.
- Query vision branch: CLIP vision encoder must return hidden states because the decoder consumes `extract_layers`.
- Decoder/head: CLIPSeg-specific part; independently testable if supplied extracted activations and conditional embeddings.
- Postprocessing: source model returns raw mask logits. End-to-end applications usually apply `sigmoid`, resize/crop to original image size outside the model, and threshold, but those steps are not implemented as a CLIPSeg model method in the inspected source.

Other heads:

- `CLIPSegModel`: required as a staged subtarget because segmentation condition embeddings use `get_text_features` and `get_image_features`.
- `CLIPSegTextModel` and `CLIPSegVisionModel`: required for staged encoder parity.
- Contrastive loss and BCE segmentation loss: deferred for inference.
- Generation, prefill/decode, and KV cache: not applicable.

## 3. Important config dimensions

Source defaults from `configuration_clipseg.py`:

| Field | Default | Notes |
| --- | ---: | --- |
| text hidden / layers / heads | 512 / 12 / 8 | CLIP-style causal text encoder, head dim 64. |
| text intermediate | 2048 | Ungated MLP with `quick_gelu`. |
| text max positions / vocab | 77 / 49408 | CLIP tokenizer contract. |
| vision hidden / layers / heads | 768 / 12 / 12 | ViT-style image encoder, head dim 64. |
| vision intermediate | 3072 | Ungated MLP with `quick_gelu`. |
| vision image_size / patch_size | 224 / 32 | Defaults differ from official CLIPSeg checkpoints. |
| projection_dim | 512 | Shared CLIP embedding dimension. |
| extract_layers | `[3, 6, 9]` | Source indexes `hidden_states[i + 1]`; hidden state 0 is embeddings. |
| reduce_dim | 64 | Decoder token width after reducing vision activations. |
| decoder heads | 4 | Decoder head dim is `reduce_dim / decoder_num_attention_heads`. |
| decoder intermediate | 2048 | Decoder MLP width. |
| conditional_layer | 0 | FiLM is applied at decoder iteration `i == conditional_layer`. |
| use_complex_transposed_convolution | false | Refined checkpoint sets true. |
| logit_scale_init_value | 2.6592 | Used by contrastive `CLIPSegModel`, not by segmentation decoder logits. |
| dtype | checkpoint-dependent | CIDAS configs say `torch_dtype="float32"`. |
| cache support | none | No KV cache or autoregressive decode. |

Representative checkpoint sweep:

| Checkpoint | Architecture | Text | Vision | Processor input | Decoder | Head | Params source |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `CIDAS/clipseg-rd64-refined` | `CLIPSegForImageSegmentation` | H=512, L=12, heads=8, S=77 | H=768, L=12, heads=12, image=224, patch=16 | 352x352, ImageNet mean/std | `reduce_dim=64`, layers from `[3,6,9]` | complex Conv2d + two ConvTranspose2d stages | HF API: ~150.748M F32 + 274 I64 |
| `CIDAS/clipseg-rd64` | `CLIPSegForImageSegmentation` | same as refined | same as refined | 352x352 | `reduce_dim=64` | single ConvTranspose2d kernel/stride 16 | HF API: ~150.694M F32 + 274 I64 |
| `CIDAS/clipseg-rd16` | `CLIPSegForImageSegmentation` | same as refined | same as refined | 352x352 | `reduce_dim=16` | single ConvTranspose2d kernel/stride 16 | HF API: ~149.884M F32 + 274 I64 |
| `hf-internal-testing/tiny-random-CLIPSegModel` | `CLIPSegModel` | H=32, L=5, heads=4, S=512, vocab=1024 | H=32, L=5, heads=4, image=30, patch=2 | fixture config; not production segmentation | `reduce_dim=32`, `[1,2,3]` in config | not a segmentation architecture in metadata | HF config/API |

Omitted/effective defaults:

- CIDAS configs include many old generation fields inherited from older config serialization; current source does not use them for CLIPSeg inference.
- CIDAS text configs use old CLIP ids `bos_token_id=0`, `eos_token_id=2`, `pad_token_id=1`. Current config defaults are `49406/49407/1`; source preserves an `eos_token_id == 2` pooling compatibility branch.
- `decoder_hidden_act` is present in config, but generated source hard-sets decoder `hidden_act = "relu"` after copying the vision config. Current DinoML scope should treat ReLU as source behavior for the decoder MLP.

## 3a. Family variation traps

- The official CIDAS processor resizes to 352x352 while `vision_config.image_size` is 224. Since `CLIPSegForImageSegmentation.forward` defaults `interpolate_pos_encoding=True`, the vision position table is interpolated from the configured 14x14 grid to a 22x22 grid for 352px inputs.
- The decoder assumes the non-CLS token count is a square number: `size = int(math.sqrt(output.shape[2]))`, then `view(batch, channels, size, size)`. Non-square image sizes, non-square grids, or image sizes not divisible by patch size need rejection or a separate audited path.
- `use_complex_transposed_convolution` materially changes output head ops. Refined uses `Conv2d(64->64,k=3,pad=1)`, ReLU, `ConvTranspose2d(64->32,k=4,stride=4)`, ReLU, `ConvTranspose2d(32->1,k=4,stride=4)`. Non-refined rd64 uses one `ConvTranspose2d(64->1,k=16,stride=16)`. rd16 uses one `ConvTranspose2d(16->1,k=16,stride=16)`.
- `conditional_layer` indexes the reversed extracted activations loop, not an original CLIP layer number. With default 0, FiLM is applied after reducing original layer 9 and before the first decoder layer.
- FiLM is implemented with an unusual layout boundary: `output.permute(1,0,2)`, multiply/add with `[B, reduce_dim]`, then permute back. Broadcast semantics are equivalent to per-batch, per-channel scale/add over all tokens, but the source operation order matters for graph capture.
- CLIPSeg inherits CLIP encoders but adds a decoder with post-norm blocks. The vision/text encoders are pre-norm; decoder layers apply LayerNorm after attention residual and after MLP residual.
- The text branch uses causal self-attention even though segmentation is not generative. No KV cache is used.
- `CLIPSegProcessor` allows exactly one prompt kind: text or visual prompt. It raises if both are supplied. It may also return only prompt tensors or only image tensors depending on call arguments.
- `conditional_embeddings` bypasses text/visual prompt computation and must be shape-checked against query image batch and `projection_dim`.
- Source mask logits are raw unnormalized scores. Do not bake sigmoid, resizing to original image size, thresholding, or NMS into the model graph unless an application-level postprocessor explicitly requests it.
- Source pixel tensors are NCHW. NHWC/channel-last optimization is a guarded local layout pass around image preprocessing, patch embedding, and conv-transpose head, not the default semantic translation.
- The vision position interpolation path uses NCHW around bicubic interpolate. It needs a no-layout-translation guard or a dedicated layout-aware equivalent.

## 4. Operator coverage checklist

Tensor/layout ops:

- NCHW image tensors `pixel_values[B,3,H,W]` and optional `conditional_pixel_values[B,3,H,W]`.
- CLIP vision patch Conv2d, flatten spatial, transpose to `[B, tokens, hidden]`.
- CLS token expand, concat on sequence axis, learned position add.
- Optional bicubic interpolation of vision positional embeddings from configured grid to runtime grid.
- Capture/select hidden states at `extract_layers`, with source offset `i + 1`.
- Reverse activations tuple for decoder.
- Linear reduce per extracted activation: `[B, 1+Gh*Gw, 768] -> [B, 1+Gh*Gw, reduce_dim]`.
- Add reduced activation to running decoder state.
- FiLM broadcast with permute boundary: `[T,B,R] * [B,R] + [B,R] -> [B,T,R]`.
- Remove CLS, transpose `[B,T,R] -> [B,R,T]`, reshape to `[B,R,Gh,Gw]`.
- ConvTranspose2d output, squeeze channel dimension to logits `[B,Hmask,Wmask]`.
- Text token and position embedding lookups, pooling gather by EOT/EOS position.
- L2 norm and feature matrix transpose for the contrastive subtarget.

Neural network primitives:

- `Conv2d(3 -> 768, kernel=16, stride=16, bias=False)` for official CIDAS checkpoints.
- CLIP encoder linear projections with bias for Q, K, V, output, MLP fc1/fc2.
- Bias-free CLIP projection heads: text `512 -> 512`, vision `768 -> 512`.
- Decoder reduce linears: three `Linear(768 -> reduce_dim, bias=True)`.
- FiLM linears: two `Linear(512 -> reduce_dim, bias=True)`.
- Decoder attention linears and MLPs at width `reduce_dim`, with decoder MLP `reduce_dim -> 2048 -> reduce_dim` and ReLU.
- LayerNorm over last dimension, epsilon 1e-5.
- `quick_gelu` in CLIP encoders, ReLU in decoder, residual adds.
- `ConvTranspose2d` for mask upsampling; refined also needs a regular 3x3 Conv2d.
- Optional BCEWithLogitsLoss is training-only/deferred.

Attention primitives:

- Vision CLIP encoder: noncausal MHA, no normal mask, head shapes `[B, 12, Tv, 64]`.
- Text CLIP encoder: causal MHA with optional additive padding mask, head shapes `[B, 8, S<=77, 64]`.
- Decoder: noncausal MHA over extracted vision tokens, no attention mask, head shapes `[B, decoder_heads, Tv, reduce_dim/heads]`.
- Eager attention math uses `(Q @ K^T) * head_dim^-0.5`, additive mask if present, softmax in fp32, cast back, then `@ V`.
- Source advertises SDPA/Flash/Flex attention support through Transformers backend dispatch, but no cache.

Preprocessing-coupled ops:

- Image processor: resize to 352x352 for official CIDAS configs, rescale by `1/255`, normalize by ImageNet mean/std `[0.485,0.456,0.406]` / `[0.229,0.224,0.225]`, output NCHW.
- Text tokenizer: CLIP tokenizer from OpenAI CLIP, max length 77, lower-casing, BOS/EOS, padding with `<|endoftext|>`.
- Processor output keys:
  - text + images: `input_ids`, `attention_mask`, `pixel_values`;
  - visual prompt + images: `pixel_values`, `conditional_pixel_values`;
  - text only: text encoding;
  - visual prompt only: `conditional_pixel_values`;
  - images only: image processor batch encoding.

No generation/cache ops, multimodal embedding stitch, packed varlen metadata, NMS, or distributed tensor-parallel ops are required for the primary target.

## 5. Layer/block breakdown

Query vision branch:

```text
pixel_values: [B,3,H,W] NCHW
patch = Conv2d(3 -> Dv, kernel=P, stride=P, bias=False)(pixel_values)
patch = patch.flatten(2).transpose(1,2)              # [B,Gh*Gw,Dv]
x = concat(class_embedding.expand(B,1,Dv), patch)
x = x + interpolated_or_static_position_embedding
x = pre_layernorm(x)
repeat 12 CLIP encoder layers:
  r = x
  x = LayerNorm(x)
  q,k,v = Linear(Dv -> Dv, bias=True)(x)
  x = noncausal MHA(q,k,v)
  x = r + Linear(Dv -> Dv, bias=True)(x)
  r = x
  x = LayerNorm(x)
  x = r + Linear(3072 -> Dv)(quick_gelu(Linear(Dv -> 3072)(x)))
pooled = post_layernorm(x[:,0,:])
vision_projected = Linear(Dv -> 512, bias=False)(pooled)
hidden_states captured for decoder
```

Text prompt branch:

```text
input_ids: [B,S], attention_mask: [B,S]
x = token_embedding(input_ids) + position_embedding(arange(S))
mask = causal additive mask merged with attention_mask
repeat 12 CLIP text layers:
  pre-norm causal MHA + residual
  pre-norm quick_gelu MLP + residual
x = final_layer_norm(x)
pool = x[batch, argmax(input_ids)] when eos_token_id == 2
     else first equality match for eos_token_id
text_projected = Linear(512 -> 512, bias=False)(pool)
```

Decoder, repeated `len(extract_layers)` times:

```text
activations = [hidden_states[i + 1] for i in extract_layers][::-1]
output = None
for i, activation in enumerate(activations):
  reduced = Linear(768 -> R)(activation)
  output = reduced if output is None else reduced + output
  if i == conditional_layer:
    mul = Linear(512 -> R)(conditional_embeddings)    # [B,R]
    add = Linear(512 -> R)(conditional_embeddings)    # [B,R]
    output = mul * output.permute(1,0,2) + add
    output = output.permute(1,0,2)
  output = decoder_layer(output, attention_mask=None)

decoder_layer:
  y = MHA(output)
  output = LayerNorm(output + y)
  y = Linear(2048 -> R)(relu(Linear(R -> 2048)(output)))
  output = LayerNorm(output + y)
```

Mask head:

```text
tokens = output[:,1:,:].transpose(1,2)       # [B,R,Gh*Gw]
grid = tokens.view(B,R,Gh,Gw)                # source assumes Gh == Gw
logits = ConvTranspose head(grid).squeeze(1) # [B,Hmask,Wmask]
```

For official 352x352 input with patch 16, `Gh=Gw=22`, token count is 485 including CLS. The final logits are `[B,352,352]`.

## 6. Attention requirements

Text encoder:

- Causal self-attention, MHA only, no GQA/MQA.
- Head count/head dim: 8 heads, 64 dim for CIDAS.
- Sequence length: tokenizer max 77 for official checkpoints.
- Masking: `create_causal_mask` with optional `attention_mask`; eager path adds the mask before softmax.
- No RoPE, ALiBi, relative bias, sliding window, packed varlen, or cache.
- KV cache is not applicable. Cached prompt embeddings are possible at the model-staging level, not through attention cache tensors.

Vision encoder:

- Noncausal self-attention, MHA only, no mask in normal path.
- Official 352px runtime has 485 tokens after position interpolation; configured 224px grid would have 197 tokens.
- No position bias inside attention; absolute positions are added before the encoder.

Decoder:

- Noncausal self-attention over vision tokens after reduction to `reduce_dim`.
- `rd64` and refined: 4 heads, head dim 16.
- `rd16`: 4 heads, head dim 4.
- No attention mask, no cache, no cross-attention. Prompt conditioning is FiLM, not attention.

FlashAttention/SDPA compatibility:

- Encoder and decoder attention can be routed to fused attention kernels if causal flag and mask semantics are preserved.
- Text eager parity depends on fp32 softmax and additive mask order.
- Decoder sequence length is small enough that MLP and conv-transpose head may dominate for low batch sizes.

## 7. Position encoding and custom math

CLIPSeg uses learned absolute text and vision position embeddings. There is no RoPE/ALiBi/relative bias.

Vision interpolation:

```python
def clipseg_interpolate_pos(position_weight, embeddings, height, width, patch_size):
    num_positions = position_weight.shape[0] - 1
    class_pos = position_weight[None, :1]
    patch_pos = position_weight[None, 1:]
    old = int(num_positions ** 0.5)
    new_h = height // patch_size
    new_w = width // patch_size
    patch_pos = patch_pos.reshape(1, old, old, embeddings.shape[-1])
    patch_pos = patch_pos.permute(0, 3, 1, 2)
    patch_pos = bicubic_interpolate(patch_pos, size=(new_h, new_w), align_corners=False)
    patch_pos = patch_pos.permute(0, 2, 3, 1).view(1, -1, embeddings.shape[-1])
    return concat([class_pos, patch_pos], dim=1)
```

For fixed 352x352 CIDAS inference, the interpolated 22x22 position table can be precomputed per checkpoint/resolution bucket. For exact 224x224 inputs, static positions can be used. If `interpolate_pos_encoding=False`, the source raises when runtime height/width differ from `vision_config.image_size`.

FiLM conditioning:

```python
def clipseg_film(output, conditional_embeddings, film_mul, film_add):
    # output: [B,T,R], conditional_embeddings: [B,512]
    scale = film_mul(conditional_embeddings)   # [B,R]
    bias = film_add(conditional_embeddings)    # [B,R]
    output = scale * output.permute(1, 0, 2) + bias
    return output.permute(1, 0, 2)
```

This is equivalent to `output * scale[:, None, :] + bias[:, None, :]`, but matching the source broadcast path is useful for initial parity.

## 8. Preprocessing and input packing

Image processor contract:

- Official CIDAS `preprocessor_config.json` uses `feature_extractor_type="ViTImageProcessor"`, `processor_class="CLIPSegProcessor"`, `size={"height":352,"width":352}`, `do_resize=true`, `do_rescale=true`, `rescale_factor=1/255`, `do_normalize=true`, ImageNet mean/std, and PIL resample id 2.
- Model entry tensors are NCHW float `pixel_values[B,3,352,352]` and optional `conditional_pixel_values[B,3,352,352]`.
- `CLIPSegProcessor` docstring says visual prompt NumPy/Torch inputs should be `(C,H,W)` per image.
- First DinoML integration can start after preprocessing. A later GPU pipeline may fuse resize/rescale/normalize, but this is outside the model graph.

Text processor contract:

- CLIP tokenizer, lowercase, max length 77, `pad_token="<|endoftext|>"`.
- Runtime graph consumes `input_ids[B,S]` and optional `attention_mask[B,S]`; `position_ids` is optional and usually omitted.
- Text and visual prompts are mutually exclusive in `CLIPSegProcessor.__call__`.

Prompt/condition packing:

- Text-prompt segmentation requires `len(input_ids) == pixel_values.shape[0]`. To segment one query image with N prompts, callers duplicate the image N times; HF integration test uses three texts and three copies of the same image, producing logits `[3,352,352]`.
- Visual-prompt segmentation requires `conditional_pixel_values.shape[0] == pixel_values.shape[0]`.
- Precomputed `conditional_embeddings` must have shape `[B, projection_dim]`.

Structured-output postprocessing:

- Source returns raw `logits[B,H,W]`.
- No NMS, boxes, class labels, or variable-length records are produced.
- Application parity commonly needs `sigmoid(logits)`, optional resize/crop back to original image size, and thresholding. Because this is not a source model method in `clipseg`, DinoML should keep it as an application/pipeline layer unless a dedicated postprocessor contract is added.

## 9. Graph rewrite / lowering opportunities

### Rewrite: CLIPSeg patch Conv2d -> WindowFlatten + GEMM

Source pattern:

```text
Conv2d(3 -> 768, kernel=16, stride=16, padding=0, bias=False)
-> flatten(2) -> transpose(1,2)
```

Replacement:

```text
WindowFlatten_NCHW or NHWC -> GEMM(weight_flat.T) -> [B,Gh*Gw,768]
```

Preconditions:

- `kernel_size == stride == patch_size`.
- `padding == 0`, `dilation == 1`, `groups == 1`, `bias is None`.
- Runtime H/W are divisible by patch size.
- For decoder parity, patch grid must be square unless the decoder reshape is generalized under a separate audited change.

Weight transform:

```python
# source NCHW flatten order
w = conv.weight.reshape(out_channels, in_channels * patch_h * patch_w)

# NHWC local optimization flatten order [kh, kw, c]
w_nhwc = conv.weight.permute(0, 2, 3, 1).reshape(out_channels, patch_h * patch_w * in_channels)
```

Failure cases:

- Non-square image sizes with current decoder.
- Dynamic image sizes without position interpolation and head output-size planning.
- External callers expecting NCHW must not silently receive an NHWC API.

Parity test sketch:

- Compare source Conv2d+flatten+transpose against rewrite for 224 and 352 inputs, patch 16, fp32 and reduced precision.

### Rewrite: precompute interpolated position table for fixed 352px

Source pattern:

```text
reshape 14x14 learned patch positions -> bicubic interpolate to 22x22 -> add to embeddings
```

Replacement:

```text
constant position table [1,485,768] for resolution bucket 352x352
```

Preconditions:

- Runtime image size and patch size are known at compile/profile time.
- `interpolate_pos_encoding=True`.
- Position embedding constant is fixed.

Failure cases:

- Arbitrary dynamic image sizes.
- Non-square grids until decoder reshape is redesigned.

Parity test sketch:

- Compare interpolated position table generated offline against source forward for 352x352 and 224x224.

### Rewrite: decoder FiLM broadcast canonicalization

Source pattern:

```text
film_mul(cond) * output.permute(1,0,2) + film_add(cond)
-> permute(1,0,2)
```

Replacement:

```text
output * scale[:,None,:] + bias[:,None,:]
```

Preconditions:

- `conditional_embeddings` shape `[B, projection_dim]`.
- `output` shape `[B,T,reduce_dim]`.
- No source-observable intermediate permuted tensors are requested.

Failure cases:

- Output/hidden-state capture requiring exact intermediate layout.

Parity test sketch:

- Random `output` and condition vectors for rd16 and rd64; compare both forms.

### Rewrite: single ConvTranspose2d head -> layout-aware upsample projection

Source pattern:

```text
ConvTranspose2d(R -> 1, kernel=P, stride=P)(grid).squeeze(1)
```

Replacement opportunity:

```text
Per-patch linear projection R -> P*P -> pixel scatter/reshape
```

Preconditions:

- Non-refined head only.
- `kernel_size == stride == patch_size`, no padding/output padding/dilation/groups.
- Source transposed-conv weight layout is transformed exactly.

Weight transform:

```python
# PyTorch ConvTranspose2d weight is [in_channels, out_channels, kh, kw]
w = deconv.weight[:, 0, :, :].reshape(reduce_dim, patch_h * patch_w)
```

Failure cases:

- Refined complex head has an intervening 3x3 Conv2d and two deconvs; lower it as conv/deconv first.
- Non-square or non-divisible grids.

Parity test sketch:

- rd64 and rd16 non-refined output logits versus source ConvTranspose2d.

### Rewrite: packed QKV projection

Same as CLIP/ViT reports: concatenate Q/K/V projection weights and biases in `[q,k,v]` order for text, vision, and decoder attention. Guard it off when attention/hidden-state capture needs separate source-observable projection outputs.

### Layout rewrite: local NHWC vision/decoder regions

Candidate regions:

- Image processor output through patch extraction can be NHWC internally if the API contract remains NCHW or an explicit NHWC input mode is declared.
- Refined decoder conv/deconv head may benefit from NHWC/channel-last kernels after tokens are reshaped to grid.

No-layout-translation guards:

- Vision position interpolation currently permutes to NCHW for bicubic interpolate.
- Public `pixel_values` and `conditional_pixel_values` contracts are NCHW.
- Axis-sensitive operations include Conv2d channel axis, flatten order, `transpose(1,2)`, CLS removal, `[B,R,T] -> [B,R,Gh,Gw]` view, ConvTranspose2d channel axis, and final `squeeze(1)`.

## 10. Kernel fusion candidates

Highest priority:

- CLIP encoder LayerNorm + GEMM/residual paths inherited from CLIP/ViT; these dominate the two CLIP branches.
- QKV packed projection for text, vision, and decoder attention.
- Fixed-resolution position interpolation precompute for 352x352 CIDAS inference.
- Decoder reduce/add/FiLM canonicalization: three reduced hidden states plus one condition modulation are unique to CLIPSeg and easy to validate.
- ConvTranspose2d head support. Without it, the segmentation target cannot produce mask logits.

Medium priority:

- Patch Conv2d -> GEMM rewrite for patch16 352px inputs.
- Decoder ReLU MLP fusion at small hidden width but large intermediate 2048.
- Refined head Conv2d/ReLU/ConvTranspose/ReLU/ConvTranspose fusion or at least channel-last lowering.
- Prompt embedding cache: text or visual prompt embeddings reused across query images.

Lower priority:

- Full in-runtime image preprocessing.
- Contrastive logits head from `CLIPSegModel` beyond what is needed for prompt embeddings.
- Training BCE/contrastive losses.
- Arbitrary non-square/dynamic image sizes.

## 11. Runtime staging plan

Stage 1: CLIPSeg config and processor metadata.

- Parse top-level CLIPSeg config plus nested text/vision configs.
- Reject or defer unsupported non-square grids, unknown processor layouts, and unexpected decoder head variants.
- Load weights for CLIP encoders, projection heads, decoder reduces, FiLM, decoder layers, and transposed-conv head.

Stage 2: reuse CLIP staged parity.

- Bring up `get_text_features` and `get_image_features` exactly as in the CLIP report.
- Validate text prompt and visual prompt conditional embeddings independently.

Stage 3: query vision hidden-state extraction.

- Run CLIP vision encoder with `output_hidden_states=True`.
- Extract `hidden_states[i + 1]` for configured `extract_layers`.
- Validate 352x352 position interpolation and 485-token hidden states.

Stage 4: decoder-only parity.

- Feed extracted activations and conditional embeddings to decoder.
- Implement reduce/add loop, FiLM, post-norm decoder attention/MLP, CLS removal, grid reshape.
- Validate decoder logits before optimizing encoders.

Stage 5: mask head parity.

- Implement non-refined single ConvTranspose2d first, then refined complex head.
- Match `[B,352,352]` raw logits for CIDAS checkpoints.

Stage 6: optimization.

- Add fixed-resolution interpolated-position constants, packed QKV, patch GEMM rewrite, prompt embedding cache, and conv/deconv layout improvements with guards.

Initially stubbable:

- Losses, hidden-state/attention return payloads beyond what decoder needs, contrastive logits output, arbitrary dynamic image sizes, and in-runtime processor transforms.

## 12. Parity and validation plan

Unit parity:

- Vision position interpolation: 224 static and 352 interpolated paths.
- Text EOS pooling branch for `eos_token_id == 2`.
- `_get_vector_norm` and CLIP projection features if using `CLIPSegModel`.
- FiLM canonical form versus source permute/broadcast form.
- Decoder layer post-norm ordering.
- Single ConvTranspose2d head and refined complex head.

Layer/block parity:

- One CLIP vision layer and one CLIP text layer against source, reusing CLIP report tests.
- One decoder layer at `reduce_dim=64` and `reduce_dim=16`.
- Decoder-only parity with random activations shaped `[B,485,768]` and condition `[B,512]`.

End-to-end parity:

- `CIDAS/clipseg-rd64-refined`: text prompts `["a cat", "a remote", "a blanket"]`, duplicated query image, expect logits shape `[3,352,352]`; HF integration test checks a small logit slice and first conditional/pooled values.
- `CIDAS/clipseg-rd64`: same input shape, non-refined single deconv head.
- `CIDAS/clipseg-rd16`: reduced decoder width and single deconv head.
- Visual prompt path: `pixel_values` plus `conditional_pixel_values`.
- Precomputed conditional embedding path: bypass prompt encoder and verify shape checks.

Suggested tolerances:

- fp32: `rtol=1e-4`, `atol=1e-4` for encoder/decoder logits; interpolation/deconv may need `1e-3` initially to match HF test precedent.
- fp16/bf16 optimized kernels: start with `rtol=2e-2`, `atol=2e-2`, then tighten per fused kernel.

No DinoML tests were run for this docs-only audit.

## 13. Performance probes

- Processor throughput: 352px image resize/rescale/normalize and CLIP tokenization separately from runtime.
- Prompt embedding cache probe: many query images per text prompt and many prompts per query image.
- Query vision encoder throughput at 485 tokens, batch sweep.
- Decoder-only throughput at token count 485 for rd16 and rd64.
- Refined vs non-refined mask head latency: single deconv versus Conv2d + two deconvs.
- Attention backend comparison for text causal S<=77, vision noncausal T=485, decoder noncausal T=485.
- End-to-end segmentation images/sec for text prompts and visual prompts.
- Memory probes for hidden-state capture: all hidden states versus only `extract_layers` materialized.

## 14. Skip/defer list

- Training losses and gradient checkpointing.
- Generation, beam search, sampling, and KV cache.
- NMS, boxes, class-label postprocessing, and variable-length segmentation records.
- In-model sigmoid/threshold/original-size resizing unless a pipeline contract is added.
- Arbitrary non-square image sizes and non-square patch grids.
- Full GPU image preprocessing.
- Hidden-state/attention output capture beyond decoder-required activations.
- Quantization and multi-GPU tensor parallel.
- Remote-code or community variants that do not use the native `clipseg` source.

## 15. Final implementation checklist

- [ ] Parse `CLIPSegConfig`, nested text/vision configs, processor config, and tokenizer metadata.
- [ ] Load CLIP text/vision encoder weights, projection heads, decoder reduce/FiLM/layer/head weights.
- [ ] Accept processor tensors: `pixel_values[B,3,H,W]`, `input_ids[B,S]`, `attention_mask[B,S]`, `conditional_pixel_values[B,3,H,W]`, or `conditional_embeddings[B,projection_dim]`.
- [ ] Reuse CLIP text encoder parity, including causal mask and old EOS argmax pooling.
- [ ] Reuse CLIP vision encoder parity, including patch embedding and CLS pool.
- [ ] Implement/interpolate vision position embeddings for 352x352 official CIDAS inputs.
- [ ] Capture and extract `hidden_states[i + 1]` for `extract_layers`.
- [ ] Implement text, visual, and precomputed conditional embedding paths with batch/shape checks.
- [ ] Implement decoder reduce/add loop and FiLM conditioning.
- [ ] Implement CLIPSeg decoder post-norm attention/MLP layers with ReLU.
- [ ] Implement CLS removal, square-grid reshape, and final mask head.
- [ ] Add support for non-refined single ConvTranspose2d head.
- [ ] Add support for refined Conv2d/ReLU/ConvTranspose/ReLU/ConvTranspose head.
- [ ] Return raw mask logits `[B,H,W]`; keep sigmoid/threshold/resize postprocessing outside the core graph.
- [ ] Add decoder-only parity tests for rd16 and rd64.
- [ ] Add end-to-end parity for `CIDAS/clipseg-rd64-refined`, `CIDAS/clipseg-rd64`, and `CIDAS/clipseg-rd16`.
- [ ] Add visual-prompt and precomputed-conditional-embedding parity tests.
- [ ] Add guarded patch Conv2d-to-GEMM rewrite.
- [ ] Add guarded fixed-resolution position-table precompute.
- [ ] Add packed QKV rewrite for encoder and decoder attention.
- [ ] Benchmark processor, prompt branch, query vision encoder, decoder, mask head, and end-to-end segmentation separately.
