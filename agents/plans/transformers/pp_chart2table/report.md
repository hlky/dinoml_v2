# PP-Chart2Table (`pp_chart2table`) DinoML audit

## 1. Source basis

```text
Transformers commit/version: b75feb2af64c3e29cbbc1bd859958c5432cc7ed4
Model id: PaddlePaddle/PP-Chart2Table_safetensors
Config source: Hugging Face config.json snapshot plus native Transformers config defaults
Source files inspected:
  X:/H/transformers/src/transformers/models/pp_chart2table/configuration_pp_chart2table.py
  X:/H/transformers/src/transformers/models/pp_chart2table/modular_pp_chart2table.py
  X:/H/transformers/src/transformers/models/pp_chart2table/image_processing_pp_chart2table.py
  X:/H/transformers/src/transformers/models/pp_chart2table/image_processing_pil_pp_chart2table.py
  X:/H/transformers/src/transformers/models/pp_chart2table/processing_pp_chart2table.py
  X:/H/transformers/src/transformers/models/got_ocr2/modeling_got_ocr2.py
  X:/H/transformers/src/transformers/models/got_ocr2/configuration_got_ocr2.py
  X:/H/transformers/src/transformers/models/got_ocr2/processing_got_ocr2.py
  X:/H/transformers/src/transformers/models/got_ocr2/image_processing_got_ocr2.py
  X:/H/transformers/src/transformers/models/qwen2/modeling_qwen2.py
  X:/H/transformers/src/transformers/models/qwen2/configuration_qwen2.py
Any missing files or assumptions:
  Native Transformers has no modeling_pp_chart2table.py file. AutoModelForVision2Seq maps pp_chart2table to GotOcr2ForConditionalGeneration.
  Generated pp_chart2table files are produced from modular_pp_chart2table.py; future source edits should inspect/edit the modular file first.
  PaddlePaddle/PP-Chart2Table is older remote-code/custom-code GOT format, not native pp_chart2table. It is useful as historical comparison only.
  No gated representative links were found; both PaddlePaddle repos queried successfully.
```

Snapshots saved beside this report:

- `PaddlePaddle__PP-Chart2Table_safetensors.config.json`
- `PaddlePaddle__PP-Chart2Table_safetensors.preprocessor_config.json`
- `PaddlePaddle__PP-Chart2Table_safetensors.tokenizer_config.json`
- `PaddlePaddle__PP-Chart2Table_safetensors.repo.json`
- `PaddlePaddle__PP-Chart2Table.config.json`
- `PaddlePaddle__PP-Chart2Table.tokenizer_config.json`
- `PaddlePaddle__PP-Chart2Table.repo.json`
- `stepfun-ai__GOT-OCR-2.0-hf.config.json`
- `stepfun-ai__GOT-OCR-2.0-hf.preprocessor_config.json`
- `stepfun-ai__GOT-OCR-2.0-hf.tokenizer_config.json`
- `stepfun-ai__GOT-OCR-2.0-hf.repo.json`

Primary links:

- [PaddlePaddle/PP-Chart2Table_safetensors](https://huggingface.co/PaddlePaddle/PP-Chart2Table_safetensors)
- [PaddlePaddle/PP-Chart2Table](https://huggingface.co/PaddlePaddle/PP-Chart2Table)
- [stepfun-ai/GOT-OCR-2.0-hf](https://huggingface.co/stepfun-ai/GOT-OCR-2.0-hf)

## 2. High-level architecture

Primary DinoML target: image-to-text chart parsing with native `GotOcr2ForConditionalGeneration` body loaded from a `pp_chart2table` config.

Architecture:

```text
CPU image preprocessing + tokenizer prompt
  -> pixel_values [B_img, 3, 1024, 1024]
  -> SAM-like vision encoder [B_img, 256, 64, 64]
  -> two stride-2 conv projector + Linear [B_img, 256, 1024]
  -> replace <imgpad> token embeddings
  -> Qwen2 causal decoder prefill/decode
  -> tied LM logits / generation
```

Stage decomposition:

- CPU/data pipeline: resize/rescale/normalize chart images, tokenizer prompt with `<img>`, repeated `<imgpad>`, `</img>`, chart query text, and assistant prefix.
- Independently cacheable image stage: `pixel_values -> image_features` can be computed once per image/patch group before text decoder prefill.
- Prefix construction: token embedding lookup followed by bounded image-token row replacement. The native source uses `masked_scatter`, but the processor/chat template creates a contiguous run of image placeholder tokens.
- Prefill: Qwen2 causal decoder over full prompt with image embeddings already stitched into `inputs_embeds`.
- Decode: Qwen2 autoregressive decode with KV cache. Pixel values are passed only on the first generation iteration when cache is enabled.

## 3. Important config dimensions

Representative native checkpoint: `PaddlePaddle/PP-Chart2Table_safetensors` (`config.json`; dtype from config).

| Field | Value | Provenance / note |
| --- | ---: | --- |
| `model_type` | `pp_chart2table` | config.json |
| architecture | `GotOcr2ForConditionalGeneration` | config.json/native AutoModel mapping |
| dtype | `bfloat16` | config.json |
| `image_token_index` | 151859 | config.json/source default |
| `image_seq_length` | 256 | config.json; source default is 576 but modeling checks actual feature count |
| tokenizer image token | `<imgpad>` id 151859 | tokenizer config / processor |
| vision image size | 1024 x 1024 | config.json/source default |
| vision patch size | 16 x 16 | config.json/source default |
| vision patch grid | 64 x 64 | inferred from image/patch size |
| vision hidden size | 768 | config.json/source default |
| vision layers | 12 | config.json/source default |
| vision heads / head dim | 12 / 64 | config.json + source inference |
| vision MLP dim | 3072 | config.json/source default |
| vision output channels | 256 | config.json/source default |
| vision global layers | 2, 5, 8, 11 | config.json/source default |
| vision local window size | 14 | config.json/source default |
| text model | Qwen2 | config text_config / source default |
| text hidden size | 1024 | config.json/source default |
| text layers | 24 | config.json/source default |
| text heads / KV heads / head dim | 16 / 16 / 64 | config.json + Qwen2 source |
| text MLP intermediate | 2816 | config.json/source default |
| text activation | SiLU gated MLP | config.json/Qwen2 source |
| vocab size | 151860 | config.json/tokenizer |
| max positions | 32768 | config.json/source default |
| RoPE theta | 1000000.0 | config.json/source default |
| sliding window | disabled | `use_sliding_window=false`; `sliding_window` field ignored by Qwen2 post-init when disabled |
| cache support | enabled | config.json/Qwen2 source |
| LM head tie | tied | config.json and `_tied_weights_keys` |

Representative checkpoint sweep:

| Repo / basis | Scope | Native? | Key differences |
| --- | --- | --- | --- |
| `PaddlePaddle/PP-Chart2Table_safetensors` | target | yes, `model_type=pp_chart2table` | 1024 image, 256 image tokens, bfloat16, native GotOcr2 body, repo still advertises custom-code auto_map names |
| `PaddlePaddle/PP-Chart2Table` | historical Paddle/GOT repo | no | `model_type=GOT`, remote `modeling_GOT.GOTQwenForCausalLM`, PaddleOCR library, no native `pp_chart2table` body |
| `stepfun-ai/GOT-OCR-2.0-hf` | native GOT-OCR baseline | yes, `model_type=got_ocr2` | same Qwen2 dims but sparse/minimal vision config in saved config; native source fills vision defaults |
| native `PPChart2TableConfig()` default | source default | yes | Qwen2 dims match target; `image_seq_length=576` default conflicts with 1024/two-stride projector feature count of 256 and should not drive stitch length |

## 3a. Family variation traps

- `pp_chart2table` does not own a separate modeling file in native Transformers. DinoML should route it through the audited `got_ocr2` body plus Qwen2 decoder.
- The target repo `auto_map` names `modeling_pp_chart2table.PPChart2TableForConditionalGeneration`, but native Transformers maps the family to `GotOcr2ForConditionalGeneration`. Treat remote auto_map as repo metadata, not the native source basis.
- Older `PaddlePaddle/PP-Chart2Table` is `model_type=GOT` and requires custom remote code. Reject or separately audit that path; do not load it as native `pp_chart2table`.
- `image_seq_length` is config metadata, but native source validates placeholder count against actual projected image features. For 1024 image size and two stride-2 projector convs, the runtime image feature length is 16 x 16 = 256.
- The native PP processor is minimal and requires both `images` and `text`; target repo chat template supplies a contiguous 256-token image placeholder region. If callers pass arbitrary text with scattered `<imgpad>` tokens, source still accepts it if counts match because it uses boolean `masked_scatter`.
- Vision source uses channel-last hidden states `[B,H,W,C]` inside transformer blocks, but patch embedding and neck/projector convolutions use NCHW. Layout passes need explicit guard boundaries.
- Vision attention alternates windowed local attention (most layers) and global attention (layers 2,5,8,11). Window size 14 pads a 64x64 grid to 70x70, producing 25 windows per image.
- Qwen2 config has `num_key_value_heads == num_attention_heads` for this checkpoint, so no GQA repeat is needed today. Qwen2 source still supports GQA; future configs with fewer KV heads need repeat/broadcast support.
- Qwen2 source supports sliding attention through `layer_types`, but target config disables sliding windows. Historical `sliding_window=32768` does not matter unless `use_sliding_window=true`.
- Qwen2 has biased Q/K/V projections and biasless O/MLP projections.
- Vision QKV is a single packed Linear output in all-Q/all-K/all-V split order after reshape, not per-head interleaved weight storage.
- Tied token embedding/LM head is a logical alias for the target. Lowering must preserve the tie or intentionally materialize one shared constant.
- Repo preprocessor fields such as `original_image_size`, `channel_first`, `normalize_order`, `do_to_chw`, and `keep_keys` are Paddle/processor metadata. Native PP image processor source only defines resize/rescale/normalize defaults; verify runtime processor behavior before admitting those fields as graph ABI.

## 4. Operator coverage checklist

Tensor/layout ops:

- NCHW image input, Conv2d patch embedding, permute NCHW -> NHWC.
- NHWC LayerNorm, residual add, reshape, view, permute, flatten, contiguous.
- Window partition/unpartition: pad NHWC spatial axes, reshape, permute, slice/crop.
- NCHW neck/projector convolutions and channel-first LayerNorm implemented as permute -> LayerNorm -> permute.
- Token embedding lookup, boolean mask construction, expand, masked scatter or guarded indexed row copy.
- Last-token/logits slicing via `logits_to_keep`.

Neural primitives:

- Conv2d(3 -> 768, kernel=16, stride=16, bias=True) patch embedding.
- Vision Linear(768 -> 2304, bias=True) packed QKV.
- Vision Linear(768 -> 768, bias=True) output projection.
- Vision MLP Linear(768 -> 3072, bias=True), GELU, Linear(3072 -> 768, bias=True).
- Neck Conv2d(768 -> 256, kernel=1, bias=False), LayerNorm(C=256), Conv2d(256 -> 256, kernel=3, padding=1, bias=False), LayerNorm(C=256).
- Projector Conv2d(256 -> 512, kernel=3, stride=2, padding=1, bias=False), Conv2d(512 -> 1024, kernel=3, stride=2, padding=1, bias=False), Linear(1024 -> 1024, bias=True).
- Qwen2 token embedding `[151860,1024]`, decoder RMSNorm, Linear Q/K/V/O, SwiGLU MLP, final RMSNorm, tied LM head Linear(1024 -> 151860, bias=False).

Attention primitives:

- Vision dense noncausal MHA over local windows and full 64x64 global grid, with decomposed relative position bias.
- Text causal self-attention with RoPE and KV cache. Target is MHA; Qwen2 source supports GQA.
- Causal mask addition, optional sliding mask path should be rejected/deferred for target unless config enables it.

Position/custom math:

- Vision absolute position add `[1,64,64,768]`.
- Vision decomposed relative bias from learned `rel_pos_h/rel_pos_w`, linear interpolation, integer gather, two einsums, bias add.
- Qwen2 RoPE cos/sin generated in fp32 then cast to hidden dtype.

Generation/cache ops:

- DynamicCache-style per-layer K/V update for Qwen2.
- Cache reorder for generation/beam can come from Transformers generation; first DinoML target can use greedy/single-beam.
- Pixel values should be omitted after first cached iteration.

Preprocessing-coupled ops:

- Resize to 1024x1024, rescale by 1/255, CLIP-like mean/std normalize, RGB conversion, tokenizer prompt construction, special token IDs.
- Optional GOT-OCR tiling processor from `got_ocr2` is not used by native PP image processor defaults, but remains relevant if routing through `GotOcr2Processor`.

Scatter/indexed update:

- Required source behavior: boolean `masked_scatter` into `inputs_embeds`.
- DinoML optimization: reject arbitrary masks initially and admit contiguous `<imgpad>` token runs with exact feature count, lowered to indexed row copy.

## 5. Layer/block breakdown

Vision patch/encoder:

```text
pixel_values [B,3,1024,1024]
  -> Conv2d(3,768,k=16,s=16) [B,768,64,64]
  -> permute [B,64,64,768]
  -> add abs_pos [1,64,64,768]
```

Vision block, repeated 12 times:

```text
residual = x
x = LayerNorm(768) over NHWC channel
if layer not in {2,5,8,11}: x = window_partition(x, window=14, pad H/W to multiples)
qkv = Linear(768 -> 2304, bias=True)
q,k,v = reshape/split to [B_or_windows*12, H*W, 64]
scores = (q * 1/sqrt(64)) @ k.T
scores += decomposed_rel_pos(q, rel_pos_h, rel_pos_w)
p = softmax(scores, fp32 accumulation).to(dtype)
attn = p @ v
x = Linear(768 -> 768, bias=True)
if windowed: x = window_unpartition(...), crop back to original H/W
x = residual + x
x = x + Linear(3072 -> 768)(GELU(Linear(768 -> 3072)(LayerNorm(x))))
```

Vision neck/projector:

```text
x [B,64,64,768] -> permute [B,768,64,64]
x = Conv2d(768 -> 256,k=1,bias=False) -> channel-first LayerNorm
x = Conv2d(256 -> 256,k=3,pad=1,bias=False) -> channel-first LayerNorm
x = Conv2d(256 -> 512,k=3,s=2,pad=1,bias=False)  # [B,512,32,32]
x = Conv2d(512 -> 1024,k=3,s=2,pad=1,bias=False) # [B,1024,16,16]
x = flatten spatial -> [B,256,1024]
x = Linear(1024 -> 1024,bias=True)
```

Qwen2 decoder block, repeated 24 times:

```text
residual = x
x = RMSNorm(x)
q = Linear(1024 -> 1024,bias=True).view(B,T,16,64).transpose(1,2)
k = Linear(1024 -> 1024,bias=True).view(B,T,16,64).transpose(1,2)
v = Linear(1024 -> 1024,bias=True).view(B,T,16,64).transpose(1,2)
q,k = RoPE(q,k,position_ids)
k,v = cache.update(k,v,layer_idx) if cache enabled
attn = causal_attention(q,k,v,mask,scale=1/sqrt(64))
x = residual + Linear(1024 -> 1024,bias=False)(attn)
residual = x
x = RMSNorm(x)
x = residual + Linear(2816 -> 1024,bias=False)(SiLU(Linear(1024 -> 2816,bias=False)(x)) * Linear(1024 -> 2816,bias=False)(x))
```

## 6. Attention requirements

Vision attention:

- Noncausal self-attention.
- MHA: 12 heads, head_dim 64.
- Query/key/value width: 768 total each.
- Local layers use windowed attention over 14x14 windows. For 64x64, pad to 70x70, 5x5 windows, 25 windows/image, each window length 196.
- Global layers use full 64x64 attention length 4096 per image.
- Masking: no semantic attention mask; padding is removed after window unpartition.
- Relative bias: decomposed height/width learned tables; global table length 127 per axis, local table length 27 per axis.
- No KV cache in vision encoder.
- Source flags `_supports_flash_attn=False`, `_supports_sdpa=False` for GotOcr2 wrapper; eager/math parity is the source basis.

Text attention:

- Causal self-attention.
- Target checkpoint: MHA, 16 query heads, 16 KV heads, head_dim 64. Qwen2 source supports GQA through `num_key_value_heads < num_attention_heads`.
- Q/K/V projections are separate Linear ops, Q/K/V bias enabled, O projection bias disabled.
- RoPE is applied before cache update, so cached keys are post-RoPE.
- Masking uses Transformers causal mask creation; target does not require sliding attention.
- KV cache shape before repeat: `[B, num_key_value_heads, T_cache, head_dim]`; target `[B,16,T_cache,64]`.
- Eager fallback repeats KV only if GQA is enabled. Target has repeat factor 1.
- FlashAttention/SDPA compatibility comes from Qwen2 attention backend abstraction, but GotOcr2 wrapper disables flash/sdpa flags. First DinoML integration should use explicit causal attention parity and then admit optimized kernels with Qwen2 tests.

## 7. Position encoding and custom math

Vision decomposed relative position bias:

```python
def got_rel_pos(q_size, k_size, rel_pos):
    max_rel_dist = 2 * max(q_size, k_size) - 1
    rel = interpolate_1d(rel_pos, size=max_rel_dist)
    q = arange(q_size)[:, None] * max(k_size / q_size, 1.0)
    k = arange(k_size)[None, :] * max(q_size / k_size, 1.0)
    idx = (q - k) + (k_size - 1) * max(q_size / k_size, 1.0)
    return rel[idx.long()]

def got_decomposed_rel_pos(query, rel_h, rel_w, H, W):
    rh = got_rel_pos(H, H, rel_h)
    rw = got_rel_pos(W, W, rel_w)
    q = query.reshape(batch_heads, H, W, head_dim)
    bh = einsum("bhwc,hkc->bhwk", q, rh)
    bw = einsum("bhwc,wkc->bhwk", q, rw)
    return bh[:, :, :, :, None] + bw[:, :, :, None, :]
```

Qwen2 RoPE:

```python
inv_freq = 1.0 / (rope_theta ** (arange(0, head_dim, 2) / head_dim))
freqs = (inv_freq[None, :, None] @ position_ids[:, None, :]).transpose(1, 2)
emb = cat([freqs, freqs], dim=-1)
cos, sin = emb.cos(), emb.sin()
q = q * cos[:, None] + rotate_half(q) * sin[:, None]
k = k * cos[:, None] + rotate_half(k) * sin[:, None]
```

Precomputable: RoPE inv_freq for default RoPE, static vision abs position, static local/global rel-pos tables. Dynamic: RoPE cos/sin depend on `position_ids`; vision relative gather depends on runtime H/W if image size is ever admitted dynamically.

## 8. Preprocessing and input packing

PPChart2Table native image processor defaults:

- Resize to 1024x1024 with resample value 3.
- Rescale by 1/255, normalize with mean `[0.48145466, 0.4578275, 0.40821073]` and std `[0.26862954, 0.26130258, 0.27577711]`.
- The checkpoint preprocessor snapshot includes RGB/HWC-to-CHW metadata; model source expects NCHW `pixel_values`.

Text/token ABI:

- Special tokens from checkpoint tokenizer: `<img>` id 151857, `</img>` id 151858, `<imgpad>` id 151859.
- Target repo chat template uses 16 x 16 = 256 `<imgpad>` tokens and the literal chart instruction "Chart to table".
- Native PP processor requires both `images` and `text`. For an end-to-end DinoML integration, prompt construction can live in the CPU/tokenizer pipeline.

Image embedding stitch:

- Source computes `inputs_embeds = embedding(input_ids)`.
- `image_features` shape for target: `[B_img, 256, 1024]`.
- Placeholder mask is `input_ids == image_token_id`, expanded to embedding width.
- Source validates `masked inputs_embeds` element count equals `image_features.numel()`.
- Source `masked_scatter` flattens replacement in row-major order over the expanded boolean mask. With the checkpoint chat template this is equivalent to copying image feature rows into a contiguous `<imgpad>` span.

First DinoML admission policy:

- Require exactly one contiguous image-token span per image item, length equal to projected feature length.
- Reject arbitrary scattered image masks, multiple spans, mismatched counts, or text-supplied `inputs_embeds` image-token detection until separately audited.
- Permit caching `image_features` independently from decoder KV cache.

## 9. Graph rewrite / lowering opportunities

### Rewrite: patch Conv2d -> Linear/GEMM

Source pattern: `Conv2d(3 -> 768, kernel=16, stride=16, padding=0)`.

Replacement:

```text
NCHW WindowFlatten patches [B,64,64,3*16*16] -> MatMul(W.T) + bias -> NHWC [B,64,64,768]
```

Preconditions:

- `kernel_size == stride == patch_size`, `padding == 0`, `dilation == 1`, `groups == 1`.
- Runtime image size exactly 1024x1024 or divisible static admitted shape.
- Flatten order must match PyTorch Conv2d NCHW kernel order `[out,in,kh,kw]`.

Failure cases: dynamic image sizes without transformed abs-pos and rel-pos parity, non-NCHW inputs, non-divisible dimensions.

Parity test sketch: compare patch embedding output for random fp32/fp16 input against PyTorch Conv2d before and after abs-pos add.

### Rewrite: placeholder masked_scatter -> indexed row copy

Source pattern: boolean expanded mask over `inputs_embeds`.

Replacement:

```text
inputs_embeds[:, token_start:token_start+N, :] = image_features.reshape(B, N, H)
```

Preconditions:

- `input_ids` contain one contiguous run of image token id 151859 per sample.
- Run length equals `image_features.shape[1]`.
- Feature width equals text hidden size.
- Batch/image grouping is one-to-one or explicitly packed.

Failure cases: arbitrary scattered `<imgpad>`, multiple pages/patch groups without clear packing metadata, caller-provided `inputs_embeds` path.

### Rewrite: vision window partition as structured view/copy

Source pattern: NHWC pad, reshape, permute, contiguous, attention, inverse reshape/permute, crop.

Replacement: specialized window-attention packing kernel or metadata view plus local attention launch.

Preconditions:

- NHWC hidden layout inside vision blocks.
- Static `window_size=14`.
- Pad sizes derived from H/W and removed after inverse partition.

Failure cases: layout translation across neck/projector boundaries without axis rewrites, output attentions requiring exact dense attention matrix layout.

### Rewrite: Qwen2 MLP -> fused SwiGLU GEMM

Source pattern:

```text
down_proj(silu(gate_proj(x)) * up_proj(x))
```

Replacement: fused dual GEMM + SiLU multiply + down GEMM or CUTLASS epilogue where supported.

Preconditions: biasless gate/up/down, hidden=1024, intermediate=2816 for target, activation exactly SiLU.

### Layout rewrite opportunities and guards

- Safe candidate: keep NCHW through patch Conv2d and neck/projector Conv2d; keep NHWC inside vision transformer blocks. Fuse adjacent permutes where producer/consumer are controlled.
- Guarded axis rewrites: LayerNorm over `dim=-1` in NHWC vision blocks versus channel-first LayerNorm in neck implemented by permutes; reductions/softmax always over last attention dimension; window partition pads spatial axes in NHWC.
- No-layout-translation guard around placeholder stitch and Qwen2 decoder token sequence: sequence axis and hidden axis must remain `[B,T,H]`.

## 10. Kernel fusion candidates

Highest priority:

- Qwen2 RMSNorm, Q/K/V projection, RoPE, causal attention, O projection, and SwiGLU MLP. These dominate decode/prefill.
- Vision global attention layers over 4096 tokens. Even only four layers can be expensive.
- Placeholder row-copy stitch. Avoid general boolean scatter in the runtime.
- Conv projector chain from `[B,256,64,64]` to `[B,256,1024]`; important for image prefix cost.

Medium priority:

- Vision local window attention with relative position bias.
- Vision LayerNorm + Linear/MLP blocks in NHWC.
- Patch Conv2d lowered to GEMM.
- Last-token-only logits via `logits_to_keep=1` for decode.

Lower priority:

- Output attentions materialization.
- Training loss path.
- GOT-OCR optional multi-page/crop-to-patches processor behavior.
- Beam search cache reorder beyond single-beam greedy/chart extraction.

## 11. Runtime staging plan

Stage 1: Config/tokenizer admission.

- Parse `pp_chart2table` config.
- Reject `model_type=GOT` remote-code checkpoints for this native path.
- Validate target dims and token ids.

Stage 2: Vision encoder parity.

- Load vision weights and run patch embedding, one local layer, one global layer, neck.
- Keep PyTorch-axis faithful layout first.

Stage 3: Projector and image-feature cache.

- Implement two conv projector + flatten/permute + Linear.
- Expose/cache `[B,256,1024]` image features.

Stage 4: Text-only Qwen2 prefill/decode.

- Reuse separate Qwen2 audit/implementation where possible.
- Validate text-only prompt and tied LM head.

Stage 5: Multimodal stitch and prefill.

- Implement guarded indexed row copy for contiguous `<imgpad>` spans.
- Run full prompt prefill logits parity.

Stage 6: Cached decode.

- Ensure pixel values are omitted after first cached iteration.
- Validate token-by-token greedy output.

Stage 7: Optimization.

- Add attention kernels, conv-to-GEMM rewrites, row-copy fusion, last-token logits, and layout cleanup.

Stubbable initially: image processor CPU pipeline, tokenizer prompt construction, beam search, output formatting, attentions/hidden-states outputs.

## 12. Parity and validation plan

- Config loading: assert effective native config for target checkpoint, including Qwen2 nested defaults.
- Processor ABI: fixed image -> `pixel_values` shape/dtype and prompt token count equals 256 placeholders.
- Patch embedding parity: random fp32 and bf16/fp16 inputs against PyTorch.
- Vision relative-position parity: unit test `get_rel_pos` and decomposed bias for local 14x14 and global 64x64.
- Single vision layer parity: local layer and global layer separately.
- Vision encoder + projector parity: compare `[B,256,1024]`.
- Stitch parity: compare source `masked_scatter` with guarded row-copy for contiguous spans; negative tests for mismatch/scattered tokens.
- Qwen2 text block parity: one block, N blocks, final norm/logits.
- Prefill logits parity: full chart prompt with image features.
- Decode parity: one-token and multi-token greedy decode with KV cache.
- End-to-end parity: same image and "Chart to table" prompt, compare decoded table text under deterministic greedy generation.

Suggested tolerances:

- fp32: max abs `1e-4` for block outputs, logits `2e-4`.
- bf16/fp16: max abs `2e-2` for long blocks/logits, plus relative tolerance checks; use stricter per-op checks where accumulation is fp32.

## 13. Performance probes

- CPU preprocessing throughput: images/sec resize/rescale/normalize at 1024.
- Vision encoder throughput: batch sweep B=1,2,4 for image features only.
- Local vs global vision attention timing: isolate windowed layers and global layers.
- Projector throughput and memory bandwidth.
- Multimodal prefill throughput: prompt length including 256 image tokens.
- Decode tokens/sec with and without last-token logits.
- KV cache memory: 24 layers x 2 tensors x `[B,16,T,64]` x dtype.
- Image-feature cache memory: `[B,256,1024]` x dtype.
- Attention backend comparison: eager/BMM-softmax-BMM vs optimized causal attention for Qwen2 and optimized local/global attention for vision.
- Layout pass probe: NCHW/NHWC permute cost around vision neck/projector.

## 14. Skip/defer list

- Training and loss.
- Gradient checkpointing.
- Paddle remote-code `model_type=GOT` path.
- Arbitrary `masked_scatter` image-token layouts.
- GOT-OCR multi-page and crop-to-patches packing unless product needs it.
- Sliding-window Qwen2 layers; target config disables them.
- GQA/MQA Qwen2 variants; target has full MHA.
- Beam search and cache reorder.
- Output attentions/hidden states.
- Quantized/FP8 community variants.
- Multi-GPU tensor parallel plans.

## 15. Final implementation checklist

- [ ] Parse native `pp_chart2table` config and nested Qwen2/vision config.
- [ ] Reject or route remote-code `model_type=GOT` checkpoints separately.
- [ ] Load tied embedding/LM-head weights as one logical parameter.
- [ ] Implement PP image preprocessing ABI or require precomputed `pixel_values`.
- [ ] Implement vision patch Conv2d and absolute position add.
- [ ] Implement vision window partition/unpartition.
- [ ] Implement vision local/global MHA with decomposed relative position bias.
- [ ] Implement vision neck Conv2d + channel-first LayerNorm.
- [ ] Implement projector Conv2d/flatten/Linear and expose image-feature cache.
- [ ] Implement Qwen2 decoder with RoPE and KV cache.
- [ ] Implement guarded image-token indexed row copy.
- [ ] Add negative guards for placeholder count/layout mismatch.
- [ ] Add prefill logits parity test.
- [ ] Add cached decode parity test.
- [ ] Add end-to-end chart prompt parity test.
- [ ] Benchmark preprocessing, vision encoder, projector, prefill, and decode separately.
