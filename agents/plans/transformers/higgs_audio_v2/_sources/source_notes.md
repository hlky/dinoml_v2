# higgs_audio_v2 source notes

## Local source basis

- Transformers checkout: `transformers`
- Commit inspected: `b75feb2af64c3e29cbbc1bd859958c5432cc7ed4`
- Family source directory: `transformers/src/transformers/models/higgs_audio_v2`
- Authoritative generated-file note: `modeling_higgs_audio_v2.py` and `configuration_higgs_audio_v2.py` both state they are generated from `modular_higgs_audio_v2.py`; future source edits should inspect/update the modular file first.

## Files inspected

- `src/transformers/models/higgs_audio_v2/configuration_higgs_audio_v2.py`
- `src/transformers/models/higgs_audio_v2/modeling_higgs_audio_v2.py`
- `src/transformers/models/higgs_audio_v2/generation_higgs_audio_v2.py`
- `src/transformers/models/higgs_audio_v2/processing_higgs_audio_v2.py`
- `src/transformers/models/higgs_audio_v2/modular_higgs_audio_v2.py` was identified as the generator source.
- Coupled codec/tokenizer source:
  - `src/transformers/models/higgs_audio_v2_tokenizer/configuration_higgs_audio_v2_tokenizer.py`
  - `src/transformers/models/higgs_audio_v2_tokenizer/modeling_higgs_audio_v2_tokenizer.py`
  - `src/transformers/models/dac/feature_extraction_dac.py`

## Hugging Face configs inspected

Fetched with `Invoke-WebRequest -UseBasicParsing` from Hugging Face raw URLs, no model imports or DinoML tests.

- `https://huggingface.co/bosonai/higgs-audio-v2-generation-3B-base/raw/main/config.json`
- `https://huggingface.co/bosonai/higgs-audio-v2-generation-3B-base/raw/main/processor_config.json`
- `https://huggingface.co/bosonai/higgs-audio-v2-generation-3B-base/raw/main/generation_config.json`
- `https://huggingface.co/bosonai/higgs-audio-v2-generation-3B-base/raw/main/tokenizer_config.json`
- `https://huggingface.co/bosonai/higgs-audio-v2-tokenizer/raw/main/config.json`
- `https://huggingface.co/bosonai/higgs-audio-v2-tokenizer/raw/main/preprocessor_config.json`

## Representative checkpoint availability

- Production generation checkpoint found: `bosonai/higgs-audio-v2-generation-3B-base`.
- Coupled audio tokenizer checkpoint found: `bosonai/higgs-audio-v2-tokenizer`.
- No separate small/debug `higgs_audio_v2` checkpoint was found from the quick HF/source search. The report uses source defaults as the "debug/default" column and labels them as source defaults, not checkpoint facts.
- The in-source examples also mention `eustlb/higgs-audio-v2-generation-3B-base` and `eustlb/higgs-audio-v2-tokenizer`; the accessible official/raw configs used for concrete dimensions were the `bosonai/*` repos above.

## Source-derived coupling highlights

- The generator is a Llama-like causal decoder with GQA, Llama-3 RoPE defaults, RMSNorm, SwiGLU MLPs, a separate audio embedding table, and dual text/audio FFNs selected by `audio_token_mask`.
- End-to-end waveform parity depends on `HiggsAudioV2Processor`, `DacFeatureExtractor`, and `HiggsAudioV2TokenizerModel`.
- The processor expands one text placeholder `<|AUDIO_OUT|>` into a sequence of `<|AUDIO_OUT|>` plus delay tokens, builds delayed multi-codebook audio IDs, and later reverts the delay pattern before codec decode.
- The generation mixin returns audio code sequences, not text-token sequences, and supports greedy/sample only. Temperature/top-k/top-p operate after logits are reshaped per codebook.
- `HiggsAudioV2TokenizerModel` is a separately stageable neural codec. It uses HuBERT semantic features, DAC acoustic encoder/decoder, semantic Conv1d/ConvTranspose1d blocks, RVQ nearest-codebook search, and torchaudio resampling.

## Gaps and caution notes

- `HiggsAudioV2Model.get_placeholder_mask` documentation says it checks placeholder/audio length equality, but the inspected implementation only builds the boolean mask. The processor/generation code carries the stricter count/order contract.
- `configuration_higgs_audio_v2.py` defaults `codebook_size=1024`; the production generation config sets `codebook_size=1026`, apparently reserving stream BOS/EOS IDs inside the per-codebook prediction range. DinoML should not hard-code 1024 for generator heads.
- The tokenizer config uses `codebook_size=1024` for real codec code indices; the generator config uses 1026 for logits/embeddings to include stream control IDs 1024 and 1025.
- The tokenizer model has a source TODO noting a padding difference from Boson original code in `_extract_semantic_features`; treat tokenizer parity as its own audit before claiming waveform parity.
- No code tests, imports, or commits were run for this audit.
