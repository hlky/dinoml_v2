# Voxtral Realtime Source Notes

Scope: `voxtral_realtime` only. No DinoML code edits, imports, tests, or model execution were performed.

## Local source basis

- Transformers checkout: `X:/H/transformers`
- Commit inspected: `b75feb2af64c3e29cbbc1bd859958c5432cc7ed4`
- Family directory: `X:/H/transformers/src/transformers/models/voxtral_realtime`
- Generated file warning: `modeling_voxtral_realtime.py` is generated from `modular_voxtral_realtime.py`; future source edits should target the modular file.

Files inspected:

- `configuration_voxtral_realtime.py`
- `modeling_voxtral_realtime.py`
- `modular_voxtral_realtime.py`
- `feature_extraction_voxtral_realtime.py`
- `processing_voxtral_realtime.py`
- `convert_voxtral_realtime_weights_to_hf.py`
- Neighbor/composed references: `src/transformers/models/voxtral/*`, `auto/modeling_auto.py`, `auto/processing_auto.py`, `auto/feature_extraction_auto.py`, `auto/tokenization_auto.py`

## Key local source observations

- `VoxtralRealtimeForConditionalGeneration` composes `AutoModel.from_config(config.audio_config)`, `VoxtralRealtimeTextForCausalLM(config.text_config)`, a two-layer multimodal projector, and `VoxtralRealtimeTimeEmbedding`.
- Audio frontend in the model is stateful:
  - `VoxtralRealtimeCausalConv1d` uses left causal padding when no cache is supplied.
  - `VoxtralRealtimeConv1dPaddingCache` stores per-conv left context under keys `conv1` and `conv2`.
  - The cache layer mutates a fixed tensor with `copy_` and marks static address outside torchdynamo compilation.
- Audio encoder is causal transformer-style, not classic bidirectional Whisper:
  - causal or sliding-window causal mask selected from `config.sliding_window`
  - RoPE on Q/K
  - optional `past_key_values` for streaming audio encoder reuse
- Text decoder is Llama/Mistral-like:
  - token embedding -> repeated decoder layers -> RMSNorm -> tied LM head
  - GQA in production config: `num_attention_heads=32`, `num_key_value_heads=8`, `head_dim=128`
  - sliding-window causal attention with default window 8192
  - Ada RMS conditioning: each decoder layer multiplies post-attention normalized hidden states by `1 + ada_rms_norm(t_cond)`, where `t_cond` is sinusoidal embedding of `num_delay_tokens`.
- Audio/text coupling:
  - Offline/non-generator path computes audio hidden states, reshapes groups of `downsample_factor=4`, projects to text hidden size, then adds projected audio embeddings directly to text token embeddings.
  - Generation path can precompute only the causal conv embedder from `input_features` into `encoder_inputs_embeds`, then slices those embeddings by text decode position using `start_idx = past_seen_tokens * downsample_factor`.
  - A generator-valued `input_features` is treated as streaming input; generation stops when that generator is exhausted.
- Cache ABI:
  - Text decoder cache: standard `past_key_values`.
  - Audio encoder cache: `encoder_past_key_values`.
  - Causal conv padding cache: `padding_cache`.
  - `cache_implementation` only supports `static` and `offloaded_static` for the audio encoder cache; other cache implementations raise.
- Placeholder/gated gap:
  - `get_placeholder_mask` references `self.config.audio_token_id`, but `VoxtralRealtimeConfig` does not declare `audio_token_id` and the official `config.json` inspected does not contain it.
  - The main `forward` path does not call `get_placeholder_mask`; it performs additive audio/text embedding coupling. DinoML should not admit masked-scatter placeholder lowering for this family without a separate source/config justification.
- Feature extractor:
  - raw waveform -> Torch STFT -> power magnitudes -> Slaney mel filter bank -> log10 clamp -> global max clamp -> affine normalization.
  - Defaults: sampling rate 16000, mono, feature size 128, `n_fft=400`, `win_length=400`, `hop_length=160`, `global_log_mel_max=1.5`.
  - Output model tensor is `input_features` shaped `[B, 128, T_mel]`; optional attention mask is sampled from padded waveform mask at `win_length - 1 :: hop_length`.
- Processor:
  - Requires `mistral-common` tokenizer backend.
  - Validates feature extractor STFT/mel parameters against `mistral_common` audio config.
  - First streaming chunk produces text tokens and audio features; subsequent chunks produce only audio features.
  - `center=True` for first chunk and `center=False` after the first chunk.
  - Chunk helpers derive first chunk and per-chunk sample counts from `num_delay_tokens`, `audio_length_per_tok`, `hop_length`, and `win_length`.

## Representative config sources

Official native Transformers config:

- `https://huggingface.co/mistralai/Voxtral-Mini-4B-Realtime-2602/raw/main/config.json`
- `https://huggingface.co/mistralai/Voxtral-Mini-4B-Realtime-2602/raw/main/generation_config.json`
- `https://huggingface.co/mistralai/Voxtral-Mini-4B-Realtime-2602/raw/main/processor_config.json`
- HF API metadata: `https://huggingface.co/api/models/mistralai/Voxtral-Mini-4B-Realtime-2602`
- HF metadata reports `gated=false`, `library_name=vllm`, `pipeline_tag=automatic-speech-recognition`, Apache-2.0 license, and safetensors BF16 parameter count `4,429,679,360`.

Debug/native-style config:

- `https://huggingface.co/onnx-internal-testing/tiny-random-VoxtralRealtimeForConditionalGeneration/raw/main/config.json`
- Native schema with tiny dimensions and `dtype=float32`; useful for parser/admission shape tests, not quality.

Mirrors / non-native schema:

- `https://huggingface.co/RedHatAI/Voxtral-Mini-4B-Realtime-2602/raw/main/config.json`
  - Same native schema/dimensions as official config at time inspected.
- `https://huggingface.co/mlx-community/Voxtral-Mini-4B-Realtime-2602-4bit/raw/main/config.json`
  - MLX schema, not directly loadable by `VoxtralRealtimeConfig` without conversion/adaptation.
  - Contains quantization metadata `{bits: 4, group_size: 64, mode: affine}` and explicit audio encoding args.

Unavailable/missing files checked:

- `preprocessor_config.json`: 404 for the official repo; the repo uses `processor_config.json`.
- `tokenizer_config.json`: 404 for the official repo; tokenizer artifacts include `tekken.json`.
- `chat_template.json`: 404 for the official repo.
- `audio_encoder.json`: 404 for the official repo.

## Source URLs

- Transformers model docs: `https://huggingface.co/docs/transformers/model_doc/voxtral_realtime`
- Official model: `https://huggingface.co/mistralai/Voxtral-Mini-4B-Realtime-2602`
- Tiny random model: `https://huggingface.co/onnx-internal-testing/tiny-random-VoxtralRealtimeForConditionalGeneration`
- MLX 4-bit mirror: `https://huggingface.co/mlx-community/Voxtral-Mini-4B-Realtime-2602-4bit`
- RedHat mirror: `https://huggingface.co/RedHatAI/Voxtral-Mini-4B-Realtime-2602`
