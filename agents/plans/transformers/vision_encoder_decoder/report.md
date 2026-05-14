# vision_encoder_decoder DinoML wrapper/composition audit

Primary runtime target: inference for generic image-to-text generation through `VisionEncoderDecoderModel`, with a separately audited image encoder and separately audited causal text decoder. This report treats `vision_encoder_decoder` as a composition/runtime ABI family, not as the owner of a fixed neural body.

## 1. Source basis

```text
Transformers commit/version:
  b75feb2af64c3e29cbbc1bd859958c5432cc7ed4
Model id:
  Generic wrapper family. Representative public checkpoints:
  microsoft/trocr-base-handwritten, microsoft/trocr-small-printed,
  nlpconnect/vit-gpt2-image-captioning, ydshieh/vit-gpt2-coco-en.
Config source:
  Official Hugging Face config.json plus preprocessor/tokenizer configs where accessible.
Source files inspected:
  X:/H/transformers/src/transformers/models/vision_encoder_decoder/configuration_vision_encoder_decoder.py
  X:/H/transformers/src/transformers/models/vision_encoder_decoder/modeling_vision_encoder_decoder.py
  X:/H/transformers/src/transformers/models/vision_encoder_decoder/__init__.py
  Supporting delegated-source checks:
  X:/H/transformers/src/transformers/models/vit/modeling_vit.py
  X:/H/transformers/src/transformers/models/vit/image_processing_vit.py
  X:/H/transformers/src/transformers/models/trocr/modeling_trocr.py
  X:/H/transformers/src/transformers/models/trocr/processing_trocr.py
  X:/H/transformers/src/transformers/models/gpt2/modeling_gpt2.py
Any missing files or assumptions:
  No family-local processor file exists. Processor ABI is inherited from the paired encoder/decoder processors, for example TrOCRProcessor or ViTImageProcessor plus GPT2Tokenizer. No sampled config required remote code. This report does not re-own ViT, DeiT, TrOCR, or GPT-2 operator coverage; it composes their separate audits.
```

Pinned source URLs:

- https://github.com/huggingface/transformers/blob/b75feb2af64c3e29cbbc1bd859958c5432cc7ed4/src/transformers/models/vision_encoder_decoder/modeling_vision_encoder_decoder.py
- https://github.com/huggingface/transformers/blob/b75feb2af64c3e29cbbc1bd859958c5432cc7ed4/src/transformers/models/vision_encoder_decoder/configuration_vision_encoder_decoder.py
- https://github.com/huggingface/transformers/blob/b75feb2af64c3e29cbbc1bd859958c5432cc7ed4/src/transformers/models/trocr/modeling_trocr.py
- https://github.com/huggingface/transformers/blob/b75feb2af64c3e29cbbc1bd859958c5432cc7ed4/src/transformers/models/gpt2/modeling_gpt2.py
- https://github.com/huggingface/transformers/blob/b75feb2af64c3e29cbbc1bd859958c5432cc7ed4/src/transformers/models/vit/modeling_vit.py

Representative HF config URLs:

- https://huggingface.co/microsoft/trocr-base-handwritten/raw/main/config.json
- https://huggingface.co/microsoft/trocr-small-printed/raw/main/config.json
- https://huggingface.co/nlpconnect/vit-gpt2-image-captioning/raw/main/config.json
- https://huggingface.co/ydshieh/vit-gpt2-coco-en/raw/main/config.json

Local notes:

- `agents/plans/transformers/vision_encoder_decoder/_sources/source_notes.md`
- `agents/plans/transformers/vision_encoder_decoder/_sources/config_sweep.md`

## 2. High-level architecture

`vision_encoder_decoder` is a two-model composition wrapper:

```text
CPU image preprocessing -> image encoder -> encoder hidden states
decoder input ids / labels -> causal text decoder with cross-attention -> logits -> generation controller
```

Stage decomposition:

- CPU/data pipeline: image load/resize/rescale/normalize and text tokenization are processor-owned, not family-local. The wrapper's model input is `pixel_values`.
- Image encoder: instantiated with `AutoModel` from nested `config.encoder`. It returns `encoder_outputs[0]`, usually `[batch, image_tokens, encoder_hidden]`.
- Optional wrapper bridge: if `encoder.hidden_size != decoder.hidden_size` and decoder `cross_attention_hidden_size` is absent, the wrapper applies `enc_to_dec_proj: Linear(encoder_hidden -> decoder_hidden)` to all encoder tokens.
- Decoder: instantiated with `AutoModelForCausalLM` from nested `config.decoder`, with `is_decoder=True` and `add_cross_attention=True` ensured by config construction helpers.
- Generation/cache: inherited from `GenerationMixin`. Encoder outputs can be computed once and expanded/reused by generic generation code; decoder cache behavior is owned by the decoder family.

Independently stageable pieces:

- Processor parity and `pixel_values` ABI.
- Encoder-only parity and cacheable `encoder_last_hidden_state`.
- Optional bridge projection parity.
- Decoder prefill with supplied `encoder_hidden_states`.
- Decode loop and cache behavior through the delegated decoder.

## 3. Important config dimensions

The wrapper has few intrinsic dimensions; most fields are nested encoder/decoder fields.

| Field | Source owner | Runtime significance |
|---|---|---|
| `model_type="vision-encoder-decoder"` | wrapper config | Dispatches this composition class. |
| `encoder.model_type` | nested config | Selects delegated encoder family and processor shape contract. |
| `decoder.model_type` | nested config | Selects delegated causal LM family, cache ABI, and tokenizer/generation quirks. |
| `encoder.hidden_size` | nested encoder | Width of `encoder_outputs[0]`. Required by wrapper projection check. |
| `decoder.hidden_size` / equivalent | nested decoder | Width expected by decoder self-attention and sometimes cross-attention query. |
| `decoder.cross_attention_hidden_size` | nested decoder when present | If present, must equal `encoder.hidden_size`; disables wrapper `enc_to_dec_proj`. |
| `decoder.is_decoder` | nested decoder | Must be true for causal/cross-attention decoder semantics. |
| `decoder.add_cross_attention` | nested decoder | Must be true for decoder families that condition on encoder hidden states. |
| `tie_word_embeddings` | top-level and nested decoder | Wrapper forces top-level false; decoder-internal LM-head tying remains delegated. |
| `decoder_start_token_id`, `pad_token_id`, `eos_token_id` | top-level/generation config | Used by label shifting and generation start/stop behavior. |
| `use_cache` | nested decoder / call arg | Forwarded directly to decoder. Wrapper does not implement cache internals. |

Representative checkpoint sweep:

| Model id | Encoder | Encoder dims | Decoder | Decoder dims | Image ABI | Projection/adaptation |
|---|---|---:|---|---:|---|---|
| `microsoft/trocr-base-handwritten` | ViT | H=768, L=12, A=12, patch16, image 384 | TrOCR | H=1024, L=12, A=16, FFN=4096, vocab 50265 | `pixel_values[B,3,384,384]`, 577 encoder tokens | `cross_attention_hidden_size=768`; no wrapper projection. |
| `microsoft/trocr-small-printed` | DeiT | H=384, L=12, A=6, patch16, image 384 | TrOCR | H=256, L=6, A=8, FFN=1024, vocab 64044 | `pixel_values[B,3,384,384]`, 577 encoder tokens | `cross_attention_hidden_size=384`; no wrapper projection. |
| `nlpconnect/vit-gpt2-image-captioning` | ViT | H=768, L=12, A=12, patch16, image 224 | GPT-2 LM | H=768, L=12, A=12, vocab 50257 | `pixel_values[B,3,224,224]`, 197 encoder tokens | widths match; no projection. |
| `ydshieh/vit-gpt2-coco-en` | ViT | H=768, L=12, A=12, patch16, image 224 | GPT-2 LM | H=768, L=12, A=12, vocab 50257 | `pixel_values[B,3,224,224]`, 197 encoder tokens | widths match; no projection. |

Processor sweep:

| Model id | Image processor | Size | Mean/std | Tokenizer |
|---|---|---:|---|---|
| `microsoft/trocr-base-handwritten` | `ViTImageProcessor` | 384 | `[0.5]*3 / [0.5]*3` | `RobertaTokenizer` |
| `microsoft/trocr-small-printed` | `DeiTImageProcessor` | 384 | `[0.5]*3 / [0.5]*3` | `XLMRobertaTokenizer` |
| `nlpconnect/vit-gpt2-image-captioning` | legacy `ViTFeatureExtractor` | 224 | `[0.5]*3 / [0.5]*3` | `GPT2Tokenizer` |
| `ydshieh/vit-gpt2-coco-en` | legacy `ViTFeatureExtractor` | 224 | `[0.5]*3 / [0.5]*3` | `GPT2Tokenizer` |

## 3a. Family variation traps

- This family is not a fixed architecture. Operator coverage must be admitted by exact `(encoder_family, decoder_family)` pairs, not by `vision-encoder-decoder` alone.
- The wrapper only composes `AutoModel` encoders and `AutoModelForCausalLM` decoders. It does not own ViT, DeiT, TrOCR, GPT-2, Swin, BEiT, or any other nested body.
- `cross_attention_hidden_size` changes adaptation ownership. If it is present, decoder cross-attention K/V projections consume encoder width directly; wrapper projection must be absent. If it is absent and widths differ, wrapper-owned `enc_to_dec_proj` is required.
- The wrapper rejects encoders with output embeddings / LM heads. DinoML admission should reject checkpoints whose resolved encoder class is a head model rather than a base feature model.
- `encoder_attention_mask` is hardcoded to `None` in the wrapper forward. Do not infer image padding masks at the wrapper level unless a delegated encoder/processor family explicitly owns them.
- Top-level `tie_word_embeddings` is forced false, but decoder models such as GPT-2 or TrOCR can still tie their own LM head to input embeddings. Preserve decoder-internal aliases.
- Sampled TrOCR configs serialize decoder `use_cache=false` while GPT-2 captioning configs set `use_cache=true`; cache enablement is checkpoint/call dependent.
- Processor ABI varies by paired model. There is no `vision_encoder_decoder` processor file.
- Vision source tensors are channel-first NCHW at the encoder boundary. NHWC/channel-last is only a guarded optimization inside separately audited vision regions.
- GPT-2 decoder weights use `Conv1D` storage convention in its own family; the wrapper must not assume normal `Linear[out,in]` layout for delegated decoder weights.
- TrOCR cross-attention can have `decoder.hidden_size != encoder.hidden_size` without wrapper projection because the decoder owns K/V input width through `cross_attention_hidden_size`.

## 4. Operator coverage checklist

Wrapper-owned tensor/control ops:

- Required `pixel_values` input validation/presence when `encoder_outputs` are not supplied.
- Call delegated encoder with `pixel_values`, `output_attentions`, `output_hidden_states`, `return_dict`, and non-`decoder_` kwargs.
- Normalize tuple `encoder_outputs` into `BaseModelOutput`.
- Select `encoder_hidden_states = encoder_outputs[0]`.
- Optional dense projection: `Linear(Henc -> Hdec)` over `[B,Senc,Henc]`.
- Label shift helper for training/eval-with-labels: allocate same shape, copy labels right, insert `decoder_start_token_id`, replace `-100` with `pad_token_id`.
- Call delegated decoder with `input_ids`, `attention_mask`, `encoder_hidden_states`, `encoder_attention_mask=None`, `inputs_embeds`, `past_key_values`, and `use_cache`.
- Return/repackage `Seq2SeqLMOutput`.

Delegated neural primitives:

- Image encoder primitives are owned by nested encoder audits. For ViT/DeiT examples: NCHW patch Conv2d, flatten/transpose, CLS/absolute position add, noncausal MHA, LayerNorm, GELU MLP.
- Decoder primitives are owned by nested decoder audits. For TrOCR/GPT-2 examples: token/position embeddings, causal self-attention, encoder-decoder cross-attention, MLP, LayerNorm, LM projection.

Attention/cache primitives:

- Wrapper requires decoder support for cross-attention over `encoder_hidden_states`.
- Wrapper forwards `past_key_values` and returns `decoder_outputs.past_key_values`; cache tensor shapes and update semantics are decoder-owned.
- Generic generation creates and expands `encoder_outputs` for encoder-decoder models; beam/cache reorder remains delegated to `GenerationMixin` plus cache classes.

Preprocessing-coupled ops:

- Image resize/rescale/normalize and tokenizer decode are processor-owned.
- No placeholder-token scatter, modality token IDs, packed image-token descriptors, or `cu_seqlens` are used by this wrapper.

Graph/ABI metadata:

- Admission must record the resolved encoder class, decoder class, nested config snapshots, processor config, whether wrapper bridge projection exists, and decoder cache type.

## 5. Layer/block breakdown

Wrapper forward:

```text
if encoder_outputs is None:
  require pixel_values
  encoder_outputs = encoder(pixel_values, ...)
elif tuple:
  encoder_outputs = BaseModelOutput(*encoder_outputs)

encoder_hidden_states = encoder_outputs[0]  # [B,Senc,Henc]

if Henc != Hdec and decoder.cross_attention_hidden_size is None:
  encoder_hidden_states = enc_to_dec_proj(encoder_hidden_states)  # [B,Senc,Hdec]

encoder_attention_mask = None

if labels is not None and no decoder inputs:
  decoder_input_ids = shift_tokens_right(labels, pad_token_id, decoder_start_token_id)

decoder_outputs = decoder(
  input_ids=decoder_input_ids,
  attention_mask=decoder_attention_mask,
  encoder_hidden_states=encoder_hidden_states,
  encoder_attention_mask=None,
  inputs_embeds=decoder_inputs_embeds,
  past_key_values=past_key_values,
  use_cache=use_cache,
)

logits = decoder_outputs.logits
```

Representative ViT/DeiT encoder body, delegated:

```text
pixel_values [B,3,H,W]
-> patch Conv2d(kernel=patch,stride=patch)
-> flatten spatial + transpose to [B,Senc,Henc]
-> CLS + absolute positions
-> bidirectional transformer encoder
-> encoder_last_hidden_state [B,Senc,Henc]
```

Representative TrOCR/GPT-2 decoder body, delegated:

```text
decoder_input_ids [B,T]
-> token + position embeddings
-> repeated causal self-attention
-> repeated cross-attention over encoder_hidden_states
-> MLP/residual/norm stack
-> LM projection [B,T,vocab]
```

## 6. Attention requirements

Wrapper-level:

- No attention math is implemented in the wrapper.
- The wrapper requires the decoder to accept `encoder_hidden_states` and, for generation, `past_key_values`.
- `encoder_attention_mask` is always `None` in the wrapper source. Cross-attention is normally over all encoder tokens.
- Attention implementation flags `_supports_flash_attn=True` and `_supports_sdpa=True` are advertised on the wrapper, but actual compatibility depends on both delegated models.

Representative delegated attention contracts:

- ViT/DeiT encoder: noncausal self-attention, MHA, no KV cache, no RoPE, no cross-attention.
- TrOCR decoder: causal self-attention plus rectangular encoder-decoder cross-attention. `EncoderDecoderCache` can carry self K/V and cross K/V; cross K/V can be reused after first update.
- GPT-2 decoder with `add_cross_attention=True`: causal self-attention and cross-attention; cross-attention uses separate query projection and a packed K/V projection from encoder hidden states. Cache can wrap self and cross caches through `EncoderDecoderCache`.

Admission guidance:

- First integration should admit exact delegated pairs with known cache ABI: `vit+gpt2`, `vit+trocr`, and `deit+trocr`.
- Reject or route to fallback if the decoder config does not actually create cross-attention layers despite top-level composition.
- Reject configs with unknown decoder cache types until the decoder family audit owns them.

## 7. Position encoding and custom math

Wrapper-owned label shift:

```python
def shift_tokens_right(labels, pad_token_id, decoder_start_token_id):
    shifted = zeros_like(labels)
    shifted[:, 1:] = labels[:, :-1]
    shifted[:, 0] = decoder_start_token_id
    shifted = where(shifted == -100, pad_token_id, shifted)
    return shifted
```

Wrapper-owned projection rule:

```python
if encoder.hidden_size != decoder.hidden_size and decoder.cross_attention_hidden_size is None:
    encoder_hidden_states = Linear(encoder.hidden_size, decoder.hidden_size)(encoder_hidden_states)
```

Delegated position math:

- ViT/DeiT: learned absolute image positions plus optional bicubic interpolation. Precompute per fixed resolution in the encoder family.
- TrOCR: learned or sinusoidal text positions with decoder-specific offset and padding behavior.
- GPT-2: learned absolute text positions from `past_seen_tokens`.

Precomputable:

- Processor-derived fixed image resolution and encoder position tables.
- Encoder outputs per image.
- Cross-attention K/V per decoder layer when the decoder cache contract supports it.

Dynamic:

- Decoder position IDs and self-attention causal masks depend on current decode length and cache length.
- Optional wrapper projection is per image/token sequence but uses fixed weights.

## 8. Preprocessing and input packing

CPU/data pipeline:

- The wrapper does not define a processor. Use the checkpoint processor or composed encoder image processor plus decoder tokenizer.
- Sampled processors output `pixel_values` in channel-first layout `[B,C,H,W]`.
- TrOCR processors can also return `labels` when text is provided; the wrapper converts labels to decoder inputs only if no decoder inputs/embeds are supplied.
- ViT-GPT2 captioning configs use GPT-2 tokenizer special IDs; TrOCR uses RoBERTa/XLM-R style tokenizer IDs.

GPU/runtime ABI:

- `pixel_values` enters the encoder exactly as the delegated encoder expects.
- `encoder_outputs` may be supplied directly, allowing DinoML to stage encoder and decoder as separate compiled artifacts.
- `decoder_input_ids` or `decoder_inputs_embeds` enter the decoder; wrapper does not embed or stitch image tokens into text embeddings.
- No image placeholder IDs, masked scatter, token type IDs, grid metadata, or packed variable-length descriptors are present in family source.

Generation-controller behavior:

- `GenerationMixin` handles encoder-output preparation for encoder-decoder models.
- Start token and stop token rules come from top-level/generation config and tokenizer. DinoML should record them as ABI metadata, not neural operators.

## 9. Graph rewrite / lowering opportunities

### Rewrite: staged encoder output reuse

Source pattern:

```text
generate(pixel_values) -> wrapper encoder call -> repeated decoder calls
```

Replacement:

```text
EncoderArtifact(pixel_values) -> encoder_last_hidden_state
DecoderArtifact(decoder_input_ids, encoder_last_hidden_state, cache) -> logits/cache
```

Preconditions:

- Encoder has no generation-step-dependent inputs.
- Processor output and encoder config are fixed for the request.
- Decoder receives the same `encoder_hidden_states` values as source.

Shape equations:

- `pixel_values[B,C,H,W] -> encoder_hidden_states[B,Senc,Henc]`.
- If wrapper bridge exists, `encoder_hidden_states[B,Senc,Henc] -> [B,Senc,Hdec]`.

Failure cases:

- Delegated encoder uses stochastic layers in training mode; inference must run eval behavior.
- Unknown encoder outputs tuple order.

Parity test sketch: compare wrapper logits using direct `pixel_values` and precomputed `encoder_outputs`.

### Rewrite: wrapper bridge projection as standalone provider-backed GEMM

Source pattern:

```text
encoder_hidden_states [B,S,Henc] -> nn.Linear(Henc,Hdec)
```

Replacement:

```text
Flatten [B*S,Henc] -> GEMM_RCR_Bias -> reshape [B,S,Hdec]
```

Preconditions:

- `Henc != Hdec`.
- `decoder.cross_attention_hidden_size is None`.
- Bridge parameters exist in checkpoint under wrapper ownership.

Weight transform:

```python
w = enc_to_dec_proj.weight  # [Hdec, Henc]
b = enc_to_dec_proj.bias
```

Failure cases:

- If decoder `cross_attention_hidden_size` is present, do not insert bridge.
- If decoder cross-attention accepts a distinct width, the decoder family owns K/V projection shape.

Parity test sketch: random `[B,S,Henc]` bridge projection compared against HF wrapper.

### Rewrite: cross-attention K/V precompute after encoder

Source pattern:

```text
decoder layer cross-attn projects encoder_hidden_states during decode/cache setup
```

Replacement:

```text
After encoder/bridge: per-layer CrossKVProject -> decoder cross-cache
```

Preconditions:

- Delegated decoder cache ABI has explicit cross-attention cache and stable projected K/V layout.
- Encoder hidden states are immutable across generated tokens.
- Beam expansion/reorder updates cache batch dimension consistently.

Failure cases:

- Unknown decoder family or remote-code cache.
- Cross-attention mask or source changes per token.

Parity test sketch: two-token decode with source `EncoderDecoderCache` versus precomputed cross K/V.

### Rewrite: image patch Conv2d -> GEMM in delegated encoder

This is not wrapper-owned, but common for sampled ViT/DeiT encoders.

Preconditions:

- Encoder patch embedding has `kernel_size == stride`, `padding == 0`, `dilation == 1`, `groups == 1`.
- Input height/width divisible by patch size.
- Token flatten order matches source.

Layout guard:

- Keep wrapper boundary NCHW-faithful unless the whole processor-to-patch region is controlled. NHWC/channel-last is a local vision optimization.

## 10. Kernel fusion candidates

Highest priority:

- Composition split: compile encoder, optional wrapper projection, and decoder as staged artifacts with explicit ABI between them. This unlocks encoder-output caching and isolates delegated family coverage.
- Decoder cross-attention cache/precompute for admitted decoder families. Image encoder states are fixed during text decode, so projected cross K/V reuse is high value.
- Optional wrapper bridge GEMM when present. It is small but central to correctness for mismatched hidden widths without `cross_attention_hidden_size`.
- Last-token-only LM projection in delegated decoder generation.

Medium priority:

- ViT/DeiT patch Conv2d-to-GEMM rewrite inside admitted encoder families.
- Packed self-attention QKV and fused attention in delegated decoders.
- Processor-to-encoder channel-last vision island for admitted ViT/DeiT encoders.

Lower priority:

- Training-only label shifting and loss. Keep the shift helper available for parity tests, but do not optimize it first.
- Output attentions/hidden states tuple reconstruction.
- Broad AutoModel pair support. Prefer allowlisted pair-by-pair admission.

## 11. Runtime staging plan

1. Parse `VisionEncoderDecoderConfig` and preserve nested encoder/decoder config snapshots.
2. Resolve and admit an exact encoder/decoder pair, initially `vit+gpt2`, `vit+trocr`, and `deit+trocr`.
3. Parse processor/tokenizer/generation metadata for the selected checkpoint.
4. Load weights into encoder, decoder, and optional wrapper bridge; preserve decoder-internal weight aliases.
5. Run encoder-only parity and record `encoder_last_hidden_state` ABI.
6. Run wrapper bridge parity if present; otherwise assert bridge absence matches source rules.
7. Run decoder prefill parity with supplied `encoder_hidden_states`.
8. Compose end-to-end wrapper parity for logits from `pixel_values + decoder_input_ids`.
9. Add decode loop with delegated cache ABI; then add cross-K/V precompute where allowed.
10. Add graph rewrites/fusions inside the admitted encoder/decoder families after functional parity is stable.

Initially stub:

- Training loss and `CrossEntropyLoss`.
- Output hidden states/attentions unless a parity test needs them.
- Beam search/cache reorder beyond greedy decode.
- Unknown AutoModel combinations and remote-code variants.

## 12. Parity and validation plan

- Config admission tests:
  - Reject missing nested encoder/decoder config.
  - Reject encoder with output embeddings.
  - Verify decoder `is_decoder` and `add_cross_attention`.
  - Verify bridge insertion/absence for width and `cross_attention_hidden_size` combinations.
- Wrapper unit parity:
  - `shift_tokens_right` with normal labels, `-100`, missing pad/start token errors.
  - Tuple `encoder_outputs` conversion to first hidden state.
  - Optional bridge projection values for random tensors.
- Delegated pair parity:
  - ViT-GPT2 prefill logits for `[B,3,224,224]` and short decoder prompt.
  - TrOCR base handwritten prefill logits for `[B,3,384,384]`.
  - TrOCR small printed shape/value smoke to catch Henc/Hdec mismatch with `cross_attention_hidden_size`.
- Cache parity:
  - Decode two or more tokens with decoder cache enabled where config/call permits.
  - Compare full prefill last-token logits to incremental decode logits.
  - Cross-cache reuse parity for GPT-2 and TrOCR separately.
- End-to-end parity:
  - Processor output shape and numeric normalization.
  - Greedy generated token IDs and decoded text for one image per admitted checkpoint.
- Tolerances:
  - fp32 hidden/logit parity: `rtol=1e-5`, `atol=1e-5`.
  - fp16/bf16 optimized paths: start with `rtol=1e-2`, `atol=1e-2`, plus exact greedy token parity where deterministic.

## 13. Performance probes

- Processor throughput: resize/rescale/normalize per image.
- Encoder-only throughput by admitted encoder family and image size.
- Optional bridge projection time and memory bandwidth for `[B,Senc,Henc]`.
- Decoder prefill throughput with fixed encoder states.
- Decode tokens/sec with cache disabled vs enabled.
- Cross-attention K/V precompute cost and memory per layer.
- End-to-end image-to-text requests/sec split into processor, encoder, bridge, prefill, decode, and logits.
- Batch-size sweep for encoder and decoder separately.
- Target-length sweep for decoder cache memory and tokens/sec.
- Attention backend comparison for delegated decoder self-attention and rectangular cross-attention.
- Last-token-only logits versus full-sequence logits projection.

## 14. Skip/defer list

- Training loss, gradient checkpointing, and dropout behavior.
- Broad arbitrary AutoModel/AutoModelForCausalLM combinations.
- Remote-code checkpoints.
- Beam search, sampling processors, and cache reorder for the first greedy target.
- Output attentions and hidden-state collection.
- Processor implementation inside DinoML runtime; keep image/token preprocessing in CPU/data pipeline first.
- Quantization and packed weight formats unless introduced by a delegated family audit.
- NHWC/channel-last translation at the wrapper boundary. Only apply layout rewrites inside admitted vision encoder regions.

## 15. Final implementation checklist

- [ ] Parse `VisionEncoderDecoderConfig` with nested encoder/decoder configs.
- [ ] Record checkpoint processor/tokenizer/generation ABI.
- [ ] Add pair-level admission for `vit+gpt2`, `vit+trocr`, and `deit+trocr`.
- [ ] Reject encoders with output embeddings / LM heads.
- [ ] Validate decoder `is_decoder` and `add_cross_attention`.
- [ ] Load encoder, decoder, and optional `enc_to_dec_proj` weights.
- [ ] Preserve decoder-internal tied embedding/LM-head aliases.
- [ ] Implement wrapper label-shift helper for parity and optional training-surface tests.
- [ ] Implement or compose encoder artifact returning `[B,Senc,Henc]`.
- [ ] Implement bridge projection GEMM only under source conditions.
- [ ] Implement decoder artifact accepting `encoder_hidden_states`.
- [ ] Add precomputed `encoder_outputs` wrapper parity test.
- [ ] Add cache ABI parity for each admitted decoder family.
- [ ] Add cross-attention K/V precompute optimization behind decoder-family guards.
- [ ] Add end-to-end greedy generation parity for one TrOCR and one ViT-GPT2 checkpoint.
- [ ] Benchmark processor, encoder, bridge, prefill, decode, cross-cache memory, and logits separately.

