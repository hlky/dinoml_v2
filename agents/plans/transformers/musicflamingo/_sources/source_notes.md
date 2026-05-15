# MusicFlamingo Source Notes

Audit date: 2026-05-13

## Source basis

- Transformers checkout: `transformers`
- Commit: `b75feb2af64c3e29cbbc1bd859958c5432cc7ed4`
- Report spec: `H:/dinoml_v2/agents/plans/transformers/PROMPT.md`
- Family source:
  - `src/transformers/models/musicflamingo/configuration_musicflamingo.py`
  - `src/transformers/models/musicflamingo/modeling_musicflamingo.py`
  - `src/transformers/models/musicflamingo/processing_musicflamingo.py`
  - `src/transformers/models/musicflamingo/modular_musicflamingo.py`
  - `src/transformers/models/musicflamingo/convert_musicflamingo_to_hf.py`
  - `docs/source/en/model_doc/musicflamingo.md`
- Neighbor/composed source:
  - `src/transformers/models/audioflamingo3/configuration_audioflamingo3.py`
  - `src/transformers/models/audioflamingo3/modeling_audioflamingo3.py`
  - `src/transformers/models/audioflamingo3/processing_audioflamingo3.py`
  - `src/transformers/models/qwen2/configuration_qwen2.py`
  - `src/transformers/models/qwen2/modeling_qwen2.py`

No model imports, tests, or model execution were run.

## Representative HF configs

- [nvidia/music-flamingo-2601-hf config.json](https://huggingface.co/nvidia/music-flamingo-2601-hf/raw/main/config.json), repository SHA from HF API: `6b5be086d52f65a1e204cb0faf70bf54e2741ecd`
- [nvidia/music-flamingo-2601-hf processor_config.json](https://huggingface.co/nvidia/music-flamingo-2601-hf/raw/main/processor_config.json)
- [nvidia/music-flamingo-2601-hf tokenizer_config.json](https://huggingface.co/nvidia/music-flamingo-2601-hf/raw/main/tokenizer_config.json)
- [nvidia/music-flamingo-2601-hf generation_config.json](https://huggingface.co/nvidia/music-flamingo-2601-hf/raw/main/generation_config.json)
- [nvidia/music-flamingo-think-2601-hf config.json](https://huggingface.co/nvidia/music-flamingo-think-2601-hf/raw/main/config.json), repository SHA from HF API: `cbd8dec3066752db700a473d8869b8759e7437b8`
- [nvidia/music-flamingo-think-2601-hf processor_config.json](https://huggingface.co/nvidia/music-flamingo-think-2601-hf/raw/main/processor_config.json)
- [nvidia/music-flamingo-think-2601-hf tokenizer_config.json](https://huggingface.co/nvidia/music-flamingo-think-2601-hf/raw/main/tokenizer_config.json)
- [nvidia/music-flamingo-think-2601-hf generation_config.json](https://huggingface.co/nvidia/music-flamingo-think-2601-hf/raw/main/generation_config.json)

Both public MusicFlamingo HF repos report `gated: false` via the Hugging Face API. `preprocessor_config.json` returned 404 for both; the feature extractor is embedded under `processor_config.json`.

## Runtime-significant config sweep

| Field | music-flamingo-2601-hf | music-flamingo-think-2601-hf |
|---|---:|---:|
| architecture | `MusicFlamingoForConditionalGeneration` | same |
| dtype | `bfloat16` | same |
| HF safetensors params | 8,267,215,360 BF16 | same |
| model type | `musicflamingo` | same |
| audio encoder | `audioflamingo3_encoder` | same |
| audio hidden/layers/heads | 1280 / 32 / 20 | same |
| audio FFN | 5120 | same |
| mel bins | 128 | same |
| audio max source positions | 1500 | same |
| text model | `qwen2` | same |
| text hidden/layers/heads/KV heads | 3584 / 28 / 28 / 4 | same |
| Qwen2 head dim | inferred 128 | same |
| text FFN | 18944 | same |
| vocab size | 151672 | same |
| text max positions | 32768 | same |
| `use_cache` | false | false |
| `use_sliding_window` / `sliding_window` | false / null | false / null |
| projector | 1280 -> 3584 -> 3584, GELU, bias | same |
| MusicFlamingo RoTE params | default, theta 1200, partial 0.2 | same |
| Whisper extractor | 16 kHz, 30 s, 128 mel, hop 160, n_fft 400 | same |
| max audio length | 1200 s | same |
| generation max_new_tokens | 2048 | same |

## Source-derived notes

- `MusicFlamingoForConditionalGeneration` composes `AutoModel.from_config(audio_config)`, `AutoModelForCausalLM.from_config(text_config)`, `MusicFlamingoMultiModalProjector`, and `MusicFlamingoRotaryEmbedding`.
- Audio preprocessing splits each waveform into 30-second windows, pads each to Whisper feature extractor length, then expands one `<sound>` marker into `<|sound_bos|>` plus one `<sound>` per post-pool audio frame plus `<|sound_eos|>`.
- For a full 30-second window at 16 kHz, Whisper features are 3000 frames. Audio tower conv2 reduces to 1500 frames and avg-pool reduces to 750 projected audio frames. A 20-minute sample is capped at 40 windows, giving up to 30,000 `<sound>` placeholders plus boundary tokens.
- Audio tower layout starts as `(batch_windows, 128, mel_frames)`, uses Conv1d over time, permutes to `(batch_windows, seq, hidden)` for transformer attention, then permutes back for AvgPool1d.
- The audio encoder uses dense bidirectional self-attention, no KV cache, and a custom attention math order where Q is scaled before the backend call and `scaling=1.0` is passed to the attention implementation.
- MusicFlamingo-specific RoTE is applied after the audio tower and before the multimodal projector. It uses timestamp-derived axial rotary factors, converts hidden states to float64 for rotation, then casts back.
- The LM is ordinary Qwen2 causal LM with GQA: Q heads 28, KV heads 4, inferred head_dim 128. Cache tensors store post-RoPE keys and values as `(batch, 4, seq, 128)` per layer before repeat-to-28 for eager attention.
- The checkpoint config sets Qwen2 `use_cache=false`. Generation can still receive `use_cache=True` from caller/generation config paths, but default source config is no-cache.

## Gaps and cautions

- Public HF model repos are not gated, but `convert_musicflamingo_to_hf.py` says the original source repository `nvidia/music-flamingo-2601` is private and required to recompute expected outputs from original components.
- Source config defaults differ from checkpoint IDs: `MusicFlamingoConfig.audio_token_id` defaults to `151669`, while public MusicFlamingo checkpoints use `151667`. Do not rely on source defaults for token IDs.
- The converter script passes `audio_rotary_dim=256`, which is not a declared field in the inspected `MusicFlamingoConfig`; the public config instead represents audio rotation via `rope_parameters.partial_rotary_factor=0.2`, yielding a 256-dimension rotated prefix of audio hidden size 1280. Treat the public config as the runtime basis.
- The converter constructs `text_config.model_max_length=8192`, while public MusicFlamingo configs use `32768`.
- HF API `transformersInfo.auto_model` reports `AutoModelForSeq2SeqLM`, but the inspected source architecture is `MusicFlamingoForConditionalGeneration` wrapping a Qwen2 causal LM. Treat the source and `architectures` field as authoritative for runtime ABI.
- No discrete codec/codebook/vocoder path exists in this source family; audio is represented as continuous log-mel features and projected hidden states.
