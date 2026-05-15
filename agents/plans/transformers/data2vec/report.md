# DinoML Transformers Audit: data2vec

## 1. Source basis

```text
Transformers commit/version: b75feb2af64c3e29cbbc1bd859958c5432cc7ed4
Model id: facebook/data2vec-audio-base, facebook/data2vec-audio-base-960h, facebook/data2vec-audio-large-960h, facebook/data2vec-text-base, facebook/data2vec-vision-base, facebook/data2vec-vision-large, facebook/data2vec-vision-base-ft1k, facebook/data2vec-vision-large-ft1k
Config source: local configuration_*.py plus public HF config/preprocessor/tokenizer JSON snapshots in _sources/hf_configs/
Source files inspected:
- transformers/src/transformers/models/data2vec/configuration_data2vec_audio.py
- transformers/src/transformers/models/data2vec/configuration_data2vec_text.py
- transformers/src/transformers/models/data2vec/configuration_data2vec_vision.py
- transformers/src/transformers/models/data2vec/modeling_data2vec_audio.py
- transformers/src/transformers/models/data2vec/modeling_data2vec_text.py
- transformers/src/transformers/models/data2vec/modeling_data2vec_vision.py
- transformers/src/transformers/models/data2vec/modular_data2vec_audio.py
- transformers/src/transformers/models/data2vec/modular_data2vec_text.py
Any missing files or assumptions: no gated/401/403 configs were encountered for the sampled official checkpoints. Audio and text have modular source files; generated modeling files are the direct runtime basis, while modular files are useful for future upstream source edits. Vision is copied/adapted from BEiT-style code and has no modular data2vec vision file in this directory.
```

Source URLs:

- `https://github.com/huggingface/transformers/blob/b75feb2af64c3e29cbbc1bd859958c5432cc7ed4/src/transformers/models/data2vec/modeling_data2vec_audio.py`
- `https://github.com/huggingface/transformers/blob/b75feb2af64c3e29cbbc1bd859958c5432cc7ed4/src/transformers/models/data2vec/modeling_data2vec_text.py`
- `https://github.com/huggingface/transformers/blob/b75feb2af64c3e29cbbc1bd859958c5432cc7ed4/src/transformers/models/data2vec/modeling_data2vec_vision.py`

## 2. High-level architecture

Data2Vec is a shared family name over three separate runtime surfaces:

- `data2vec-audio`: raw waveform encoder, Wav2Vec2-like 1D convolutional feature extractor, convolutional positional embedding, bidirectional Transformer encoder, optional task heads for CTC ASR, sequence classification, frame classification, and x-vector speaker embeddings.
- `data2vec-text`: RoBERTa/BERT-style text encoder with absolute learned positions, token type embeddings, bidirectional masked-LM/classification heads, plus optional decoder/cross-attention/cache branches if `is_decoder=True`.
- `data2vec-vision`: BEiT/ViT-like image encoder with Conv2d patch embedding, CLS token, optional mask token, optional per-layer or shared relative position bias, layer scale, mean/CLS pooling, image classification head, and optional UPerNet-style semantic segmentation head.

Dataflow:

```text
audio waveform + attention_mask -> Conv1d feature encoder -> feature projection -> SpecAugment mask if enabled -> conv positional embedding -> encoder -> CTC/classification/x-vector logits
token ids + token_type/position/attention masks -> embeddings -> text encoder -> pool/LM/task head -> logits
image processor -> pixel_values[N,C,H,W] -> patch Conv2d -> CLS/mask/position embeddings -> vision encoder -> pool/classifier or reshape hidden states -> FPN/UPer head -> logits
```

First useful DinoML targets should be staged independently:

- Audio CTC ASR: base encoder plus `Linear(hidden -> vocab)` and processor-driven CTC decode outside the core graph.
- Text masked LM / embeddings: bidirectional encoder, tied LM head, no cache.
- Vision image classification: patch Conv2d and encoder, then pooling/classifier.
- Vision segmentation and text causal/cross-attention should be later targets because they add FPN/upsampling or generation cache surfaces not needed for the common base checkpoints.

## 3. Important config dimensions

Representative config sweep:

| Checkpoint | Modality | Hidden | Layers | Heads | Head dim | Intermediate | Vocab/labels | Position/input shape | Operator-significant fields |
|---|---:|---:|---:|---:|---:|---:|---:|---|---|
| `facebook/data2vec-audio-base` | audio feature extraction | 768 | 12 | 12 | 64 | 3072 | source default vocab 32, checkpoint omits vocab | raw 16 kHz waveform | Conv1d dims `1->512` then six `512->512`; strides `5,2,2,2,2,2,2`; kernels `10,3,3,3,3,2,2`; total stride 320 |
| `facebook/data2vec-audio-base-960h` | audio CTC | 768 | 12 | 12 | 64 | 3072 | vocab 32 | raw 16 kHz waveform | same conv frontend; CTC head `Linear(768 -> 32)` |
| `facebook/data2vec-audio-large-960h` | audio CTC | 1024 | 24 | 16 | 64 | 4096 | vocab 32 | raw 16 kHz waveform | same conv frontend; Transformer and CTC head widen to 1024 |
| `facebook/data2vec-text-base` | text encoder | 768 | 12 | 12 | 64 | 3072 | vocab 50265 | max positions 514 | RoBERTa-like pad id 1, bos 0, eos 2, tied LM decoder |
| `facebook/data2vec-vision-base` | vision pretrain/features | 768 | 12 | 12 | 64 | 3072 | vocab 8192 in config, classifier labels absent/derived | 224 image, 16 patch, 196 patches + CLS | shared relative position bias `true`, per-layer rel bias `false`, mean pooling `false`, layer scale 0.1 |
| `facebook/data2vec-vision-base-ft1k` | image classification | 768 | 12 | 12 | 64 | 3072 | ImageNet labels | 224 image, 16 patch | per-layer relative position bias `true`, shared rel bias `false`, mean pooling `true` |
| `facebook/data2vec-vision-large` | vision pretrain/features | 1024 | 24 | 16 | 64 | 4096 | vocab 8192 in config | 224 image, 16 patch | shared relative position bias, CLS pooling |
| `facebook/data2vec-vision-large-ft1k` | image classification | 1024 | 24 | 16 | 64 | 4096 | ImageNet labels | 224 image, 16 patch | per-layer relative position bias, mean pooling |

Effective source defaults worth preserving:

| Field | Audio default | Text default | Vision default |
|---|---:|---:|---:|
| `layer_norm_eps` | `1e-5` | `1e-12` | `1e-12` |
| attention backend support | eager/SDPA/Flash/Flex via generic attention registry | eager/SDPA/Flash/Flex | eager/SDPA only |
| cache support | none in audio model | only if `is_decoder=True`; default checkpoint is encoder-only | none |
| activation | GELU | GELU | GELU |
| projection biases | audio/text q/k/v/out all biased | text q/k/v/out all biased | vision q/value/out biased, key bias disabled |

## 3a. Family variation traps

- Do not collapse Data2Vec to one operator surface: audio, text, and vision are separate model types with separate configs and preprocessing.
- Audio configs use raw waveform, not log-mel features. The conv stack reduces length by repeated `floor((L - kernel) / stride) + 1`; attention masks must be downsampled with the source `_get_feature_vector_attention_mask` rule.
- Audio has axis-sensitive layout changes: feature Conv1d runs `N,C,T`, Transformer runs `N,T,C`, and positional Conv1d temporarily returns to `N,C,T`. These should be direct-translation regions or guarded fusion regions, not blind NHWC-style axis rewrites.
- Text `facebook/data2vec-text-base` is encoder-only by default, but source implements `Data2VecTextForCausalLM` with dynamic self-attention KV cache and optional encoder-decoder cross-attention cache when `is_decoder=True` and `add_cross_attention=True`.
- Text uses RoBERTa position construction: non-pad tokens get cumsum positions beginning at `padding_idx + 1`, with pad positions left at pad id. Padding id is 1 in the sampled checkpoint.
- Vision patch inputs are source NCHW `pixel_values`. NHWC/channel-last is a candidate internal lowering for local Conv2d/BatchNorm/upsample regions only; public model and processor contracts remain NCHW.
- Vision configs differ between pretrain and fine-tuned checkpoints: pretrain sampled configs use shared relative position bias and CLS pooling; ft1k configs use per-layer relative position bias and mean pooling.
- Vision configs carry `vocab_size=8192` from BEiT/data2vec pretraining heritage, but the inspected current in-library vision modeling file does not implement a masked-image-modeling codebook head. Treat that field as ignored for this report unless a separate remote-code/pretraining head target is introduced.
- Vision segmentation source assumes `patch_resolution = config.image_size // config.patch_size` when reshaping hidden states. Dynamic `interpolate_pos_encoding=True` may support encoder resolution changes, but segmentation reshape is config-size-coupled and should be guarded.
- Vision relative bias table has three special distances for CLS-to-token, token-to-CLS, and CLS-to-CLS.
- Vision key projection has `bias=False` while query/value/out projections have bias; fused QKV rewrites must preserve this mixed-bias layout.
- Dropout, stochastic depth, layerdrop, SpecAugment, losses, and gradient checkpointing are training-path behavior and can be disabled for first inference parity.

## 4. Operator coverage checklist

### Tensor/layout ops

- Audio: unsqueeze waveform to `N,1,T`; Conv1d output length arithmetic; transpose `N,C,T <-> N,T,C`; mask downsampling with cumsum, indexed write, flip/cumsum/flip; boolean mask fill; stack/weighted sum for optional weighted layer sum.
- Text: token/position/type embedding lookup; cumsum over non-pad tokens; gather default token type ids by position ids; reshape/flatten multiple-choice batch; split QA logits along last dim.
- Vision: NCHW Conv2d patch projection; flatten spatial to sequence; transpose `N,C,H,W -> N,HW,C`; concat CLS; optional masked-token blend; sequence-to-map reshape for segmentation; bilinear/bicubic interpolate; concat channels; adaptive average pool; ConvTranspose2d; MaxPool2d.

### Neural network primitives

- Audio frontend: `Conv1d(1 -> 512, k=10, s=5, bias=False)`, then `Conv1d(512 -> 512, k=3/3/3/3/2/2, s=2, bias=False)`, each with LayerNorm over channel after transpose and GELU.
- Audio projection: `LayerNorm(512) -> Linear(512 -> hidden)`.
- Audio positional conv: repeated `num_conv_pos_embeddings=5` layers of grouped `Conv1d(hidden -> hidden, k=19, padding=9, groups=16) -> trim if even kernel -> affine-free LayerNorm(hidden) -> GELU`.
- Audio Transformer: per layer `Linear(hidden -> hidden)` q/k/v/out, `Linear(hidden -> intermediate)`, GELU, `Linear(intermediate -> hidden)`, LayerNorms.
- Audio optional adapter: `Linear(hidden -> output_hidden)` if needed, then `Conv1d(output_hidden -> 2*output_hidden, k=3, stride=2, padding=1) -> GLU(dim=channel)`.
- Audio task heads: CTC `Linear(output_hidden_or_hidden -> vocab)`; sequence classifier `Linear(hidden -> classifier_proj_size=256) -> mean pool -> Linear(256 -> labels)`; frame classifier `Linear(hidden -> labels)`; x-vector TDNN via Conv1d-equivalent linear kernels, stats pooling, `Linear(3000 -> 512) -> Linear(512 -> 512)`.
- Text: embeddings, LayerNorm, dropout, MHA projections, post-attention residual LayerNorm, MLP, pooler `Linear(hidden -> hidden) + tanh`, LM head `Linear(hidden -> hidden) -> GELU -> LayerNorm -> Linear(hidden -> vocab)`, classifiers.
- Vision: patch `Conv2d(3 -> hidden, k=16, s=16)`; LayerNorm; MHA projections with key bias disabled; MLP; layer-scale elementwise multiply; pooler; segmentation Conv2d/BatchNorm/GELU, ConvTranspose2d, MaxPool2d, AdaptiveAvgPool2d.

### Attention primitives

- Audio encoder noncausal self-attention, MHA, dense masks, no KV cache.
- Text encoder noncausal MHA by default; optional causal self-attention and optional cross-attention if configured as decoder.
- Vision encoder noncausal MHA over `[CLS] + patches`, optional additive relative position bias, SDPA path when attention outputs are not requested.

### Position/relative-bias ops

- Audio convolutional positional embedding, grouped Conv1d with affine-free LayerNorm.
- Text learned absolute position embeddings with pad-aware generated positions.
- Vision optional absolute position table interpolation and relative position bias generation/interpolation.

### Generation/cache ops

- Required only for `Data2VecTextForCausalLM` when `is_decoder=True`; not required for sampled text base masked-LM/encoder target.
- Cache ABI: per layer self-attention keys/values shaped `[batch, heads, cached_seq, head_dim]`; optional `EncoderDecoderCache` also stores cross-attention K/V and per-layer `is_updated`.

### Preprocessing-coupled ops

- Audio Wav2Vec2 feature extractor config: mono 16 kHz waveform, right padding with 0.0, normalize waveform, return attention mask.
- Text tokenizer config: RoBERTa-like vocab/merges and special tokens; runtime graph consumes token ids, token type ids, attention mask, and optional position ids.
- Vision BeitFeatureExtractor config: resize to 224, no center crop, normalize with mean/std `[0.5,0.5,0.5]`, output NCHW pixel values.

## 5. Layer/block breakdown

Audio base path:

```text
input_values: [B, samples]
x = input_values[:, None]                                      # [B,1,T]
for conv layer i in 0..6:
  x = Conv1d(Cin -> 512, kernel=conv_kernel[i], stride=conv_stride[i], bias=conv_bias)(x)
  x = transpose to [B,T',512]
  x = LayerNorm(512)(x)
  x = transpose to [B,512,T']
  x = GELU(x)
x = transpose to [B,T_feat,512]
x_norm = LayerNorm(512)(x)
h = Linear(512 -> hidden)(x_norm)
h = optional SpecAugment mask replacement in training or if mask_time_indices provided
h = encoder(h, downsampled_attention_mask)
```

Audio encoder layer, repeated `num_hidden_layers`:

```text
pos = repeated grouped Conv1d positional embedding over [B,T,hidden]
h = LayerNorm(h + pos)
for each layer:
  attn_residual = h
  q,k,v = Linear(hidden -> hidden, bias=True)(h)
  h_attn = dense noncausal MHA(q,k,v, additive mask)
  h = attn_residual + Dropout(Linear(hidden -> hidden)(h_attn))
  h = LayerNorm(h)
  h = h + Dropout(Linear(intermediate -> hidden)(GELU(Linear(hidden -> intermediate)(h))))
  h = final LayerNorm(h)
```

Text encoder layer, repeated `num_hidden_layers`:

```text
emb = word_embedding(input_ids) + token_type_embedding(token_type_ids) + position_embedding(position_ids)
h = LayerNorm(emb)
q,k,v = Linear(hidden -> hidden, bias=True)(h)
if decoder cache: append/update K,V by layer
h_attn = MHA(q,k,v, bidirectional or causal mask)
h = LayerNorm(Linear(hidden -> hidden)(h_attn) + residual)
if decoder cross-attention and encoder_hidden_states:
  q = Linear(hidden -> hidden)(h)
  k,v = Linear(hidden -> hidden)(encoder_hidden_states), cached after first update
  h = LayerNorm(Linear(hidden -> hidden)(CrossAttention(q,k,v)) + residual)
h = LayerNorm(Linear(intermediate -> hidden)(GELU(Linear(hidden -> intermediate)(h))) + residual)
```

Vision encoder layer, repeated `num_hidden_layers`:

```text
patch = Conv2d(3 -> hidden, kernel=patch_size, stride=patch_size)(pixel_values[N,C,H,W])
h = flatten+transpose to [B, num_patches, hidden]
h = optional masked-token blend
h = concat(cls_token, h)
h = optional absolute_pos_interp + h
for each layer:
  a = LayerNorm(h)
  q = Linear(hidden -> hidden, bias=True)(a)
  k = Linear(hidden -> hidden, bias=False)(a)
  v = Linear(hidden -> hidden, bias=True)(a)
  scores = q @ k.T / sqrt(head_dim)
  scores += optional per-layer and/or shared relative_position_bias
  attn = softmax(scores)
  h = h + DropPath(layer_scale_1 * Linear(hidden -> hidden)(attn @ v))
  m = Linear(hidden -> intermediate)(LayerNorm(h))
  m = GELU(m)
  m = Linear(intermediate -> hidden)(m)
  h = h + DropPath(layer_scale_2 * m)
```

Vision segmentation path:

```text
features = selected hidden_states at out_indices, excluding CLS
maps = x[:,1:,:].permute(0,2,1).reshape(B, hidden, image_size/patch_size, image_size/patch_size)
maps = [ConvTranspose2d x2 twice, ConvTranspose2d x2, Identity, MaxPool2d x2]
logits = UPerHead(PSP + FPN Conv2d/BN/GELU + bilinear upsample + classifier Conv2d)
aux = optional FCNHead
```

## 6. Attention requirements

Audio:

- Noncausal self-attention only; `Data2VecAudioAttention` accepts `key_value_states` but encoder path does not use cross-attention or cache.
- MHA with `heads=12/16`, `head_dim=64`; q/k/v/out all `hidden -> hidden` with bias.
- Masking uses additive bidirectional masks from downsampled attention mask. The source zeros padded hidden states before constructing the mask.
- SDPA/Flash/Flex can be admitted through the generic attention registry; eager fallback materializes `[B, heads, T, T]` scores and is likely too slow for long audio.
- No autoregressive KV cache.

Text:

- Default checkpoint is noncausal self-attention, MHA, `12 x 64`.
- Source supports causal self-attention if `config.is_decoder=True`, with dynamic cache; cross-attention only when `add_cross_attention=True` and encoder states are passed.
- Masking uses `create_bidirectional_mask` for encoder, `create_causal_mask` for decoder, and bidirectional encoder mask for cross-attention.
- q/k/v/out all biased. No GQA/MQA.
- Flash/SDPA/Flex are marked supported; parity must preserve scaling `head_dim**-0.5`, additive mask before softmax, dropout after softmax in eager path.

Vision:

- Noncausal self-attention over `[CLS] + patch tokens`; no packed/varlen support, no sliding window, no KV cache.
- MHA with `heads=12/16`, `head_dim=64`.
- Query/value/output projections have bias; key projection has no bias.
- Attention scores are dense `[B, heads, 1+Hpatch*Wpatch, 1+Hpatch*Wpatch]`; relative bias is additive before softmax.
- SDPA path is available but returns no attention weights when `output_attentions=True`; eager path is required if attention tensors are requested.

## 7. Position encoding and custom math

Audio positional conv can be represented as:

```python
def data2vec_audio_pos_conv(x_btc, layers):
    y = x_btc.transpose(1, 2)  # B,C,T
    for conv, norm in layers:
        y = conv(y)            # grouped Conv1d, padding=kernel//2
        if conv.kernel_size[0] % 2 == 0:
            y = y[:, :, :-1]
        y = gelu(norm(y.transpose(1, 2)).transpose(1, 2))
    return y.transpose(1, 2)
```

Text position ids:

```python
def data2vec_text_position_ids(input_ids, padding_idx=1, past=0):
    mask = (input_ids != padding_idx).to(int)
    pos = (cumsum(mask, dim=1) + past) * mask
    return pos.long() + padding_idx
```

Vision relative bias:

```python
def data2vec_vision_relative_bias(table, window_h, window_w, num_heads):
    # Build (1 + window_h*window_w)^2 index with three special CLS cases.
    # Gather table rows, reshape to [tokens, tokens, heads],
    # then permute to [1, heads, tokens, tokens] for score addition.
    return gathered_bias.permute(2, 0, 1).unsqueeze(0)
```

Precomputable:

- Text position table and token type default buffer.
- Audio conv output length formulas for static waveform lengths.
- Vision relative position index for fixed patch grids, absolute position interpolation for fixed target size.

Dynamic:

- Audio downsampled attention mask depends on per-sample waveform lengths.
- Text position ids depend on padding and cache length.
- Vision relative bias interpolation depends on runtime resolution when `interpolate_pos_encoding=True`.

## 8. Preprocessing and input packing

Audio preprocessing:

- `Wav2Vec2FeatureExtractor`, `sampling_rate=16000`, `feature_size=1`, right padding, padding value `0.0`, `do_normalize=true`, `return_attention_mask=true`.
- CPU/data pipeline should decode/resample audio to mono 16 kHz float waveform and normalize/pad. The GPU graph starts from `input_values[B, samples]` and optional `attention_mask[B, samples]`.
- The model does not split long audio in source; any chunking/streaming policy would be a DinoML scheduler feature and must preserve CTC timestamp/reassembly semantics externally.
- CTC decoding/tokenizer is needed for end-to-end ASR for `*-960h` checkpoints, but greedy/beam decode is outside the neural graph.

Text preprocessing:

- `facebook/data2vec-text-base` ships RoBERTa-style tokenizer files; text graph consumes `input_ids`, `attention_mask`, optional `token_type_ids`, optional `position_ids`.
- Default token type ids are zeros gathered by generated position ids; segment ids still enter the graph if supplied.
- Padding side/tokenizer details affect position ids because pad tokens retain `padding_idx`.

Vision preprocessing:

- `BeitFeatureExtractor`: resize to 224, no center crop in sampled configs, normalize using mean/std `[0.5,0.5,0.5]`, emit `pixel_values[B,3,224,224]`.
- Model source expects NCHW. A layout pass may use NHWC/channel-last inside local Conv2d/BN/interpolate regions but must preserve source axes at graph boundaries and sequence reshapes.
- Segmentation postprocessing should resize logits to target/original size and argmax outside the model graph when matching processor APIs; model forward itself returns logits.

## 9. Graph rewrite / lowering opportunities

### Rewrite: audio Conv1d feature stack to batched GEMM/im2col

Source pattern:

```text
Conv1d -> transpose -> LayerNorm(channel) -> transpose -> GELU
```

Replacement:

```text
WindowExtract1D -> MatMul(weight.T) -> optional BiasAdd -> LayerNorm(last_dim) -> GELU
```

Preconditions:

- Inference only, static kernel/stride/dilation/groups from config.
- `groups == 1` for feature extractor Conv1d.
- Preserve source output length formula exactly.

Shape equations:

- `T_out = floor((T_in - kernel) / stride) + 1`.
- First layer windows are `[B,T_out,1*k]`; later layers `[B,T_out,512*k]`.

Weight transform:

```python
w = conv.weight.reshape(out_channels, in_channels * kernel)
```

Layout constraints:

- Source Conv1d is `N,C,T`; rewritten GEMM can produce `N,T,C` and eliminate immediate transposes only if the following LayerNorm sees channel as last dim and the next Conv1d lowering accepts `N,T,C` windows.

Failure cases:

- Grouped positional Conv1d is a different pattern; adapter Conv1d uses GLU and stride/padding and needs separate guards.

Parity test sketch:

- Random waveform lengths covering kernel boundaries; compare feature extractor outputs and downsampled masks.

### Rewrite: vision non-overlap patch Conv2d -> Linear

Source pattern:

```text
Conv2d(3 -> hidden, kernel=patch, stride=patch) -> flatten(2) -> transpose(1,2)
```

Replacement:

```text
WindowFlatten(NCHW or guarded NHWC) -> MatMul(weight_flat.T) -> BiasAdd -> [B,num_patches,hidden]
```

Preconditions:

- `kernel_size == stride == patch_size`.
- `padding == 0`, `dilation == 1`, `groups == 1`.
- Input H/W divisible by patch H/W.
- Preserve source NCHW pixel-value contract unless entering a fully-contained layout region.

Weight transform:

```python
w = conv.weight.reshape(hidden, num_channels * patch_h * patch_w)
```

Failure cases:

- Dynamic image sizes with segmentation source using config-derived patch resolution need explicit guards.

### Rewrite: vision relative bias precompute

Source pattern:

```text
generate_relative_position_index -> table gather -> reshape/permute -> add to attention scores
```

Replacement:

```text
precomputed_bias[resolution, heads, tokens, tokens] -> Add(scores)
```

Preconditions:

- Fixed patch grid and no runtime relative-bias interpolation.
- Bias table weights unchanged.

Failure cases:

- `interpolate_pos_encoding=True` or arbitrary runtime image sizes.

### Rewrite: x-vector TDNN Linear storage -> Conv1d provider op

Source pattern:

```text
Linear(in_dim * kernel -> out_dim) called by reshaping weight to Conv1d
```

Replacement:

```text
Conv1d(in_dim -> out_dim, kernel=tdnn_kernel[i], dilation=tdnn_dilation[i])
```

Weight transform:

```python
w = linear.weight.view(out_dim, kernel, in_dim).transpose(1, 2)
```

Preconditions:

- No PEFT/LoRA applied to TDNN linear weights.

### Layout guard: no blind NCHW/NHWC translation

Audio and vision include many axis-sensitive ops:

- Audio LayerNorm over channel after `transpose(-2,-1)`, GLU `dim=1`, mask reductions over time, CTC `log_softmax(dim=-1)`.
- Vision concat CLS along sequence dim, concat FPN maps along channel dim, BatchNorm2d channel dim, bilinear interpolate spatial dims, segmentation reshape from token order.

A conceptual `no_layout_translation()` guard should protect whole semantic regions unless every consumer and axis attribute is rewritten together. Safe local NHWC regions include patch Conv2d lowering and segmentation Conv2d/BatchNorm/interpolate fusions whose inputs/outputs are explicitly converted.

## 10. Kernel fusion candidates

Highest priority:

- Audio Conv1d + LayerNorm + GELU frontend. It dominates raw waveform preprocessing in GPU inference and has repeated small channel-normalized conv stages.
- Dense encoder attention for audio/text/vision. Use SDPA/Flash where masks and relative bias are compatible; eager score materialization is expensive.
- Vision patch Conv2d -> GEMM, plus flatten/transpose elimination into sequence layout.
- LayerNorm + residual patterns for text and vision; these are repeated in every layer.

Medium priority:

- Audio grouped positional Conv1d + trim + affine-free LayerNorm + GELU.
- Vision relative bias precompute/gather fusion for fixed 14x14 grids.
- Vision layer-scale multiply + residual add + DropPath identity removal for inference.
- UPerNet segmentation Conv2d/BatchNorm/GELU and bilinear upsample chains.
- Text LM head dense + GELU + LayerNorm + decoder, with last-token-only logits for optional causal LM.

Lower priority:

- Audio x-vector TDNN/stat-pooling kernels, because x-vector is a less common first target.
- Multiple-choice flatten/unflatten and QA split/squeeze heads.
- Training-only SpecAugment, layerdrop, stochastic depth, CTC loss, AMSoftmax loss.

## 11. Runtime staging plan

1. Parse three config classes and route by `model_type`: `data2vec-audio`, `data2vec-text`, `data2vec-vision`.
2. Load weights and run one-block parity for each modality with dropout/layerdrop disabled.
3. Audio Stage A: implement Conv1d feature extractor, projection, downsampled attention mask, encoder, and CTC logits for base/base-960h.
4. Text Stage A: implement encoder-only `Data2VecTextModel`, masked LM head, and sequence classification.
5. Vision Stage A: implement patch embedding, encoder with relative bias modes, pooling, and image classification.
6. Add optimized attention backends and fixed-resolution relative-bias precompute.
7. Add optional surfaces: audio sequence/frame classification, x-vector, text decoder/cache, vision semantic segmentation.
8. Add graph rewrites/fusions with explicit layout guards and parity tests.

Stubbable initially:

- Training losses and masking randomness.
- Beam/CTC decoding.
- Text causal LM/cross-attention if first target is encoder/masked LM.
- Vision segmentation if first target is classification.
- Output attentions for optimized SDPA vision path.

## 12. Parity and validation plan

- Audio custom ops: Conv1d output-length and attention-mask downsampling tests for waveform lengths around stride/kernel boundaries; grouped positional conv parity; optional adapter GLU parity.
- Audio single-layer parity: feature extractor only, projection only, encoder layer, then full CTC logits. Recommended tolerances: fp32 `atol=1e-5`, fp16/bf16 `atol=1e-2` after full encoder.
- Text custom ops: pad-aware position id generation with left/right padding edge cases, default token type id gather, tied LM decoder alias.
- Text parity: embeddings, one encoder layer, masked LM logits, sequence classifier; optional decoder one-token cache update if admitted.
- Vision custom ops: patch Conv2d lowering, absolute position interpolation, relative bias index/gather/interpolation, layer-scale residual.
- Vision parity: patch embeddings, one encoder layer with shared and per-layer relative bias configs, image classification logits, segmentation reshape/FPN/UPer logits if enabled.
- End-to-end: compare ASR logits before CTC decode, text masked-token logits, image classification logits, and segmentation logits before postprocess.

## 13. Performance probes

- Audio feature extractor throughput by waveform length and batch size.
- Audio encoder-only throughput after conv frontend, including attention mask density sweep.
- Audio CTC logits throughput and memory use for long clips.
- Text encoder throughput by batch/sequence length for bidirectional attention.
- Optional text causal LM prefill/decode tokens/sec if decoder mode is supported.
- Vision patch embedding throughput with Conv2d vs GEMM lowering.
- Vision encoder throughput for 224 and larger images with/without relative-bias interpolation.
- Vision segmentation head throughput split into FPN/PPM/interpolate/classifier.
- Attention backend comparison: eager vs SDPA/Flash/Flex where source marks support.
- Layout-pass probe: NCHW baseline vs guarded channel-last Conv2d/BN/interpolate regions, with axis-rewrite correctness checks.

## 14. Skip/defer list

- Training, losses, gradient checkpointing, SpecAugment random mask generation, layerdrop, stochastic depth.
- CTC beam search and tokenizer decode for first neural graph parity.
- Text causal LM and encoder-decoder cross-attention cache unless a decoder checkpoint is explicitly targeted.
- Audio x-vector AMSoftmax loss and PEFT/LoRA TDNN warning behavior.
- Vision semantic segmentation if first milestone is image classification.
- Multi-GPU/tensor parallel and quantization; no source-coupled packed/quantized weight format is present in sampled configs.

## 15. Final implementation checklist

- [ ] Parse and route `data2vec-audio`, `data2vec-text`, and `data2vec-vision` configs separately.
- [ ] Load representative audio/text/vision weights and preserve tied text LM decoder aliases.
- [ ] Implement audio Conv1d feature extractor and exact feature-length/mask downsampling.
- [ ] Implement audio grouped convolutional positional embedding.
- [ ] Implement encoder MHA/MLP/LayerNorm blocks for audio, text, and vision.
- [ ] Implement text pad-aware position id generation and token type embedding path.
- [ ] Implement optional text decoder KV cache behind `is_decoder=True`.
- [ ] Implement vision patch Conv2d path, CLS/mask token insertion, pooling modes.
- [ ] Implement vision shared and per-layer relative position bias.
- [ ] Add guarded patch Conv2d -> Linear rewrite.
- [ ] Add guarded audio Conv1d frontend lowering/fusion.
- [ ] Add fixed-grid vision relative-bias precompute.
- [ ] Add parity tests for audio CTC logits, text masked LM logits, and vision classifier logits.
- [ ] Add optional parity for vision segmentation logits and audio x-vector embeddings.
- [ ] Benchmark frontend, encoder, attention backend, and layout-pass variants separately.
