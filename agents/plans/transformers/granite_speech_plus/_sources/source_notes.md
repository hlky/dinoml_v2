# granite_speech_plus source notes

Audit date: 2026-05-13

## Local source basis

- Transformers checkout: `X:/H/transformers`
- Inspected commit: `b75feb2af64c3e29cbbc1bd859958c5432cc7ed4`
- `git rev-parse HEAD` at inspection time matched the requested commit.
- Directory tree hashes at the inspected commit:
  - `src/transformers/models/granite_speech_plus`: `b3ecb44d5536b35783e98198af6644a478686ef1`
  - `src/transformers/models/granite_speech`: `847b98a1153e3ad2edc885f4974953a89454dce7`
  - `src/transformers/models/granite`: `c5209aa0ec40cb3470457fa671853fa156caa628`
  - `src/transformers/models/blip_2`: `50440e1c942e8326b197209e28f12dafc7489d6b`

## Files inspected

- `X:/H/transformers/src/transformers/models/granite_speech_plus/configuration_granite_speech_plus.py`
- `X:/H/transformers/src/transformers/models/granite_speech_plus/modeling_granite_speech_plus.py`
- `X:/H/transformers/src/transformers/models/granite_speech_plus/modular_granite_speech_plus.py`
- `X:/H/transformers/src/transformers/models/granite_speech_plus/__init__.py`
- `X:/H/transformers/src/transformers/models/granite_speech/feature_extraction_granite_speech.py`
- `X:/H/transformers/src/transformers/models/granite_speech/processing_granite_speech.py`
- `X:/H/transformers/src/transformers/models/granite_speech/modeling_granite_speech.py`
- `X:/H/transformers/src/transformers/models/granite_speech/configuration_granite_speech.py`
- `X:/H/transformers/src/transformers/models/granite/modeling_granite.py`
- `X:/H/transformers/src/transformers/models/granite/configuration_granite.py`
- `X:/H/transformers/src/transformers/models/blip_2/modeling_blip_2.py`
- `X:/H/transformers/src/transformers/models/blip_2/configuration_blip_2.py`

## Hugging Face files inspected

Primary checkpoint:

- `https://huggingface.co/ibm-granite/granite-speech-4.1-2b-plus`
- `https://huggingface.co/ibm-granite/granite-speech-4.1-2b-plus/raw/main/config.json`
- `https://huggingface.co/ibm-granite/granite-speech-4.1-2b-plus/raw/main/processor_config.json`
- `https://huggingface.co/ibm-granite/granite-speech-4.1-2b-plus/raw/main/generation_config.json`
- `https://huggingface.co/api/models/ibm-granite/granite-speech-4.1-2b-plus`

Related config sweep:

- `https://huggingface.co/ibm-granite/granite-speech-4.1-2b/raw/main/config.json`
- `https://huggingface.co/ibm-granite/granite-4.0-1b-speech/raw/main/config.json`
- `https://huggingface.co/ibm-granite/granite-speech-3.3-2b/raw/main/config.json`
- `https://huggingface.co/ibm-granite/granite-speech-3.3-8b/raw/main/config.json`
- `https://huggingface.co/ibm-granite/granite-speech-4.1-2b/raw/main/preprocessor_config.json`
- `https://huggingface.co/ibm-granite/granite-4.0-1b-speech/raw/main/preprocessor_config.json`

## Source gaps and gates

- Only one official in-library `model_type="granite_speech_plus"` checkpoint was found: `ibm-granite/granite-speech-4.1-2b-plus`.
- Nearby official checkpoints are `granite_speech`, not `granite_speech_plus`; they are useful for variation warnings but are out of scope for Plus-specific `cat_hidden_layers` behavior.
- `preprocessor_config.json` for the Plus checkpoint returned 404; the model uses `processor_config.json` containing nested `audio_processor` settings instead.
- The checkpoint is public (`gated=false` in the HF model API result).
- No imports, model execution, safetensors loading, or DinoML tests were run.
