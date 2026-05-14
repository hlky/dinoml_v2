# VibeVoice ASR Source Notes

Audit target: `vibevoice_asr`

Transformers source basis:

- Local checkout: `X:/H/transformers`
- Commit inspected: `b75feb2af64c3e29cbbc1bd859958c5432cc7ed4`
- Family source files:
  - `src/transformers/models/vibevoice_asr/configuration_vibevoice_asr.py`
  - `src/transformers/models/vibevoice_asr/modeling_vibevoice_asr.py`
  - `src/transformers/models/vibevoice_asr/modular_vibevoice_asr.py`
  - `src/transformers/models/vibevoice_asr/processing_vibevoice_asr.py`
  - `src/transformers/models/vibevoice_asr/convert_vibevoice_asr_to_hf.py`
- Delegated source sampled:
  - `src/transformers/models/vibevoice_acoustic_tokenizer/configuration_vibevoice_acoustic_tokenizer.py`
  - `src/transformers/models/vibevoice_acoustic_tokenizer/modeling_vibevoice_acoustic_tokenizer.py`
  - `src/transformers/models/vibevoice_acoustic_tokenizer/feature_extraction_vibevoice_acoustic_tokenizer.py`
  - `src/transformers/models/qwen2/configuration_qwen2.py`
  - `src/transformers/models/qwen2/modeling_qwen2.py`

Authoritative source note:

- `modeling_vibevoice_asr.py` and `configuration_vibevoice_asr.py` are generated from `modular_vibevoice_asr.py`. Runtime import uses the generated files; future Transformers source edits should start from the modular file.
- The generated file is self-contained enough for runtime inspection, but the modular source shows intended inheritance from `AudioFlamingo3ForConditionalGeneration`.

Official Hub files read:

- `https://huggingface.co/microsoft/VibeVoice-ASR-HF/resolve/main/config.json`
- `https://huggingface.co/microsoft/VibeVoice-ASR-HF/resolve/main/processor_config.json`
- `https://huggingface.co/microsoft/VibeVoice-ASR-HF/resolve/main/generation_config.json`
- `https://huggingface.co/microsoft/VibeVoice-ASR-HF/resolve/main/tokenizer_config.json`

Hub file gaps:

- `https://huggingface.co/microsoft/VibeVoice-ASR-HF/resolve/main/preprocessor_config.json` returned 404. Processor facts come from `processor_config.json` plus `VibeVoiceAcousticTokenizerFeatureExtractor` source defaults.

Representative config facts:

- `microsoft/VibeVoice-ASR-HF` uses native `VibeVoiceAsrForConditionalGeneration`, dtype `bfloat16`, nested acoustic and semantic `vibevoice_acoustic_tokenizer_encoder` configs, and a nested `qwen2` text config.
- Acoustic encoder hidden size is 64. Semantic encoder hidden size is 128. Both use channels 1, `num_filters=32`, depths `[3,3,3,3,3,3,8]`, downsampling ratios `[2,2,4,5,5,8]`, kernel size 7, GELU FFN, and `vae_std=0.625`.
- Effective acoustic hop length is `2*2*4*5*5*8 = 3200` waveform samples.
- Text config: hidden 3584, layers 28, query heads 28, KV heads 4, head dim inferred 128, intermediate 18944, vocab 152064, RoPE theta 1000000.0, max positions 131072, all layer types `full_attention`, `use_cache=true`.
- Generation config: greedy by default (`do_sample=false`), `eos_token_id=151643`, `pad_token_id=151655`, `max_new_tokens=32768`, `use_cache=true`.
- Processor config: 24 kHz audio, normalization to target dB FS -25, right audio padding, audio placeholder tokens `<|box_start|>`, `<|object_ref_start|>`, `<|object_ref_end|>`, and duration placeholder `<|AUDIO_DURATION|>`.

Variation notes:

- Search found older/non-native derivatives such as `microsoft/VibeVoice-ASR` and MLX/4-bit/5-bit mirrors with config keys like `acoustic_tokenizer_config`, `semantic_tokenizer_config`, `decoder_config`, `diffusion_head_config`, and architecture names such as `VibeVoiceForASRTraining`. Those do not match the pinned native `VibeVoiceAsrConfig` field names and should be routed to a separate remote-code/export audit rather than admitted as this native family.
- Quantized MLX/bitsandbytes mirrors are loading/provider concerns, not source-defined neural graph changes for this native audit.
