# GLM-ASR source notes

Audit date: 2026-05-13

Transformers source checkout: `X:/H/transformers`
Pinned commit: `b75feb2af64c3e29cbbc1bd859958c5432cc7ed4`

## Local source files inspected

- `src/transformers/models/glmasr/configuration_glmasr.py`
- `src/transformers/models/glmasr/modeling_glmasr.py`
- `src/transformers/models/glmasr/modular_glmasr.py`
- `src/transformers/models/glmasr/processing_glmasr.py`
- `src/transformers/models/glmasr/convert_glmasr_weights_to_hf.py`
- `docs/source/en/model_doc/glmasr.md`
- Neighbor/composed files:
  - `src/transformers/models/llama/modeling_llama.py`
  - `src/transformers/models/llama/configuration_llama.py`
  - `src/transformers/models/whisper/feature_extraction_whisper.py`
  - `src/transformers/models/audioflamingo3/modeling_audioflamingo3.py`
  - `src/transformers/models/audioflamingo3/processing_audioflamingo3.py`

`modeling_glmasr.py` and `processing_glmasr.py` are generated from
`modular_glmasr.py`; the generated files are the direct runtime basis in this
checkout, while future Transformers source edits should go through the modular
file.

## Hub configs and metadata inspected

- Current official model: https://huggingface.co/zai-org/GLM-ASR-Nano-2512
  - API metadata SHA: `61ba4e0b3309b6656edea3e93e419f7bd5c61957`
  - Public, not gated.
  - Hub metadata reports `safetensors.parameters.BF16 = 2257843200`.
  - License tag from Hub metadata: `mit`.
- Current config:
  - https://huggingface.co/zai-org/GLM-ASR-Nano-2512/resolve/main/config.json
  - Native `model_type="glmasr"` with `audio_config` and `text_config`.
- Current processor config:
  - https://huggingface.co/zai-org/GLM-ASR-Nano-2512/resolve/main/processor_config.json
  - Contains nested Whisper feature extractor settings.
- Current generation config:
  - https://huggingface.co/zai-org/GLM-ASR-Nano-2512/resolve/main/generation_config.json
- Current tokenizer config:
  - https://huggingface.co/zai-org/GLM-ASR-Nano-2512/resolve/main/tokenizer_config.json
- Current chat template:
  - https://huggingface.co/zai-org/GLM-ASR-Nano-2512/resolve/main/chat_template.jinja
- Mirror config:
  - https://huggingface.co/eustlb/GLM-ASR-Nano-2512/resolve/main/config.json
  - Same operator-significant fields as current official config, but
    `architectures` spelling is historical (`GlmasrForConditionalGeneration`).
- Historical official config:
  - https://huggingface.co/zai-org/GLM-ASR-Nano-2512/resolve/fdc39709f86b00cdce879c04d967c2146ce4053c/config.json
  - Uses remote-code-style fields: `lm_config`, `whisper_config`,
    `adapter_type`, `merge_factor`, `use_rope`, `max_whisper_length`,
    `max_length`, `mlp_adapter_act`, and `attn_implementation`.
  - These are useful admission traps, but the current native source basis reads
    `text_config`, `audio_config`, `audio_token_id`, and
    `projector_hidden_act`.

## Important extracted facts

- Primary target is ASR generation: raw audio/text prompt -> log-mel features
  and expanded audio placeholders -> GLM-ASR audio encoder -> projector -> audio
  embeddings stitched into Llama inputs -> Llama causal LM generate.
- Processor defaults: `return_tensors="pt"`, text padding enabled, common
  `padding_side="left"`, audio `sampling_rate=16000`,
  `return_attention_mask=True`, `padding="max_length"`.
- Whisper feature extractor settings from processor config:
  - `feature_size=128`, `sampling_rate=16000`, `chunk_length=30`,
    `n_samples=480000`, `nb_max_frames=3000`, `hop_length=160`,
    `n_fft=400`, `dither=0.0`, `padding_value=0.0`.
- Audio chunking: processor splits each user sample into 30 second windows,
  floors `max_audio_len // chunk_length` to 21 windows for the default
  `max_audio_len=655`, and flattens all windows across the batch.
- Audio encoder input shape from processor is `[flat_windows, 128, 3000]`.
- Audio encoder convs:
  - `Conv1d(128 -> 1280, kernel_size=3, stride=1, padding=1)` + GELU.
  - `Conv1d(1280 -> 1280, kernel_size=3, stride=2, padding=1)` + GELU.
  - Transpose to `[B, T/2, 1280]`.
- Audio encoder block repeated 32 times:
  - LayerNorm, noncausal self-attention, residual.
  - LayerNorm, Linear 1280 -> 5120, GELU, Linear 5120 -> 1280, residual.
- Audio attention:
  - 20 query heads, 20 KV heads, head dim 64.
  - q bias true, k bias false, v bias true, o bias true.
  - partial RoPE factor 0.5, so only 32 dims/head are rotated.
  - Source passes `attention_mask=None`; `input_features_mask` is used later for
    pooling/valid feature selection, not encoder attention masking.
- Audio pooling/projector:
  - Hidden `[flat_windows, 1500, 1280]` is reshaped to
    `[original_batch, -1, 5120]`, effectively grouping 4 encoder frames.
  - Projector is Linear 5120 -> 4096, GELU, Linear 4096 -> 2048.
  - Valid projected rows are flattened across batch with a boolean mask derived
    from original feature lengths.
- Text LM is native Llama from `text_config`:
  - 28 layers, hidden 2048, intermediate 6144, 16 query heads, 4 KV heads,
    head dim 128, causal, use cache true.
  - RMSNorm eps `1e-5`, SiLU SwiGLU MLP, no attention/MLP bias in current
    config, vocab size 59264.
  - KV cache is Llama `Cache`/`DynamicCache` per layer, before KV repeat, after
    RoPE, shaped logically `[B, 4, S, 128]` for keys and values.
- Audio placeholder token:
  - Processor `audio_token="<|pad|>"`; current config `audio_token_id=59260`.
  - Chat template inserts `<|begin_of_audio|><|pad|><|end_of_audio|><|user|>`.
  - Processor expands the single audio token into the exact computed number of
    audio feature tokens before tokenization.
- Stitch path:
  - Model builds token embeddings, computes flattened projected audio features,
    checks `audio token count * hidden_size == audio_features.numel()`, then
    uses `inputs_embeds.masked_scatter(...)`.
  - DinoML can lower this as guarded row-copy from flattened audio features into
    placeholder positions; no need to admit general boolean scatter for the
    first ASR path if processor invariants are enforced.
