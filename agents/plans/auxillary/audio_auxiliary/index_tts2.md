# IndexTTS2

## Coverage

- Diffusers: not covered.
- Transformers: partially overlaps through Qwen/Wav2Vec2-BERT style submodels, but IndexTTS2 as a pipeline is third-party.
- Third-party/UI: Wan2GP TTS pipeline.

## Runtime Contract

Wan2GP's `IndexTTS2Pipeline` resolves multiple checkpoint folders: IndexTTS2 root, BigVGAN 22 kHz vocoder, Qwen emotion model, and W2V-BERT. It supports legacy, CUDA graph, and vLLM-like LM decoder engine modes, with optional FlashAttention2/vLLM kernels.

## Operators

- Text frontend and tokenizer.
- Language/acoustic token generation model.
- Wav2Vec2-BERT/Qwen emotion auxiliary encoders.
- BigVGAN waveform decoder.
- Chunking/splitting policies for long text/audio.

## DinoML Notes

Do not model IndexTTS2 as one graph initially. Split into text frontend, LM/acoustic token generator, codec/vocoder, and runtime engine selection. BigVGAN can reuse the vocoder report.

## Sources

- `H:/uis/deepbeepmeep/Wan2GP/models/TTS/index_tts2/pipeline.py`
- `H:/uis/deepbeepmeep/Wan2GP/models/TTS/index_tts2/infer_v2.py`

