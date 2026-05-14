# Qwen3 TTS, Kokoro, HeartMuLa, and KugelAudio References

## Coverage

- Diffusers: not covered as these named TTS/helper stacks.
- Transformers: Qwen-family text models are covered separately, but Qwen3 TTS as a product pipeline is not just a base decoder.
- Third-party/UI: listed in Wan2GP/auxiliary planning as candidate TTS/audio surfaces.

## Runtime Contract

These are TTS or audio-generation references that may combine text normalization, tokenizer/model generation, acoustic codec tokens, vocoder decode, and voice/style controls.

## Operators

Unknown until exact implementation is selected. Expected categories include LLM/decoder generation, audio token/codebook handling, Conv1d vocoder or diffusion decoder, and text/audio preprocessing.

## DinoML Notes

Keep as candidate reports, not admitted support. Each needs a separate source-backed audit before implementation planning.

## Sources

- `agents/plans/auxiliary/audio_auxiliary.md`

