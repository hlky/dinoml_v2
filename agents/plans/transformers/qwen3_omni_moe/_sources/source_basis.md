# qwen3_omni_moe source basis notes

Date: 2026-05-13

## Local Transformers checkout

- Path: `transformers`
- Commit: `b75feb2af64c3e29cbbc1bd859958c5432cc7ed4`
- Last commit summary: `b75feb2af6 fix(minicpmv4_6): skip invalid failing tests (#45836)`
- Family path: `transformers/src/transformers/models/qwen3_omni_moe`

## Files inspected

- `configuration_qwen3_omni_moe.py`
- `processing_qwen3_omni_moe.py`
- `modeling_qwen3_omni_moe.py`
- `modular_qwen3_omni_moe.py` was identified as the authoritative modular source because the generated files carry the Transformers warning that edits should be made there. The detailed audit used the generated `modeling_*.py` and `processing_*.py` for the exact runtime classes currently imported by Transformers.

## Official Hugging Face configs fetched

All configs were fetched from official `Qwen/*` repositories and were not gated.

| Model id | HF revision | Last modified from HF API | Notes |
|---|---:|---|---|
| `Qwen/Qwen3-Omni-30B-A3B-Instruct` | `26291f793822fb6be9555850f06dfe95f2d7e695` | 2025-09-22 | Full any-to-any config: thinker, talker, code2wav, `enable_audio_output=true`. |
| `Qwen/Qwen3-Omni-30B-A3B-Thinking` | `2f443cfc4c54b14a815c0e2bb9a9d6cbcd9a748b` | 2025-09-22 | Thinker-only text output, `enable_audio_output=false`. |
| `Qwen/Qwen3-Omni-30B-A3B-Captioner` | `a2bd106cbf527db5676e79662674da22b0545ec0` | 2025-09-22 | Thinker-only audio-caption target, `enable_audio_output=false`. |

## Processor files checked

The three official repos expose the same `preprocessor_config.json` shape:

```json
{
  "feature_extractor_type": "WhisperFeatureExtractor",
  "feature_size": 128,
  "hop_length": 160,
  "n_fft": 400,
  "n_samples": 4800000,
  "sampling_rate": 16000,
  "image_processor_type": "Qwen2VLImageProcessor",
  "patch_size": 16,
  "temporal_patch_size": 2,
  "merge_size": 2,
  "min_pixels": 3136,
  "max_pixels": 12845056,
  "nb_max_frames": 30000,
  "processor_class": "Qwen3OmniMoeProcessor"
}
```

`processor_config.json` and `video_preprocessor_config.json` were not present in the checked repos. `tokenizer_config.json` and `chat_template.json` were present.
