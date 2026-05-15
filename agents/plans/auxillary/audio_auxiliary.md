# Audio Auxiliary Models

## Why This Matters

Wan2GP and Comfy expose audio as a first-class model surface: soundtracks,
text-to-speech, voice conversion, speaker diarization, speech/audio encoders,
audio VAEs, vocoders, and audio-conditioned video. This sits mostly outside the
diffusion image model path.

## Model and Feature Families

- Audio encoders: Whisper, Wav2Vec2, patched Wav2Vec2 variants.
- Audio/video generation: MMAudio, LTX-2 audio-video transformer blocks.
- Vocoders and audio VAEs: BigVGAN, Oobleck-style audio VAE, causal audio VAE.
- TTS and music/audio diffusion: IndexTTS2, Chatterbox, ACE-Step,
  ACE-Step1.5, Qwen3 TTS, Kokoro, HeartMuLa/KugelAudio references.
- Voice and speech utilities: SeedVC, Pyannote diarization, audio separation,
  SpeechBrain, OpenVoice-style modules, codec/tokenizer models.

## Packages

`openai-whisper`, `speechbrain`, `pyannote.audio`, `audio-separator`, `librosa`,
`soundfile`, `pyloudnorm`, `torchaudio`, `s3tokenizer`, `conformer`,
`phonemizer-fork`, plus local BigVGAN/codec/vocoder implementations.

## Code Anchors

- `Comfy-Org/ComfyUI/comfy/audio_encoders/whisper.py`.
- `Comfy-Org/ComfyUI/comfy/audio_encoders/wav2vec2.py`.
- `Comfy-Org/ComfyUI/comfy_extras/nodes_audio_encoder.py:8`
  audio encoder nodes.
- `Comfy-Org/ComfyUI/comfy_extras/nodes_lt_audio.py:9`
  Lightricks audio helpers.
- `Comfy-Org/ComfyUI/comfy/ldm/audio/autoencoder.py`.
- `Comfy-Org/ComfyUI/comfy/ldm/mmaudio/vae/bigvgan.py`.
- `Comfy-Org/ComfyUI/comfy/ldm/lightricks/vae/audio_vae.py`.
- `deepbeepmeep/Wan2GP/models/wan/fantasytalking/model.py:7`
  `AudioProjModel`.
- `deepbeepmeep/Wan2GP/models/wan/fantasytalking/model.py:19`
  audio cross-attention processor.
- `deepbeepmeep/Wan2GP/models/wan/fantasytalking/infer.py:10`
  `parse_audio`.
- `deepbeepmeep/Wan2GP/models/wan/multitalk/multitalk_model.py:353`
  MultiTalk `AudioProjModel`.
- `deepbeepmeep/Wan2GP/models/wan/multitalk/wav2vec2.py:9`
  patched Wav2Vec2 model.
- `deepbeepmeep/Wan2GP/models/ltx2/ltx_core/model/transformer/transformer.py:58`
  `BasicAVTransformerBlock`.
- `deepbeepmeep/Wan2GP/models/ltx2/ltx_core/conditioning/types/latent_cond.py:47`
  `AudioConditionByLatent`.
- `deepbeepmeep/Wan2GP/models/ltx2/scenema_audio.py:938`
  `ScenemaAudioPipeline`.
- `deepbeepmeep/Wan2GP/postprocessing/mmaudio/model/networks.py:27`
  `MMAudio`.
- `deepbeepmeep/Wan2GP/preprocessing/speakers_separator.py:36`
  Pyannote pipeline.
- `deepbeepmeep/Wan2GP/postprocessing/seedvc/api.py:210`
  SeedVC inference.
- `deepbeepmeep/Wan2GP/models/TTS/index_tts2_handler.py:178`
  IndexTTS2 handler.
- `deepbeepmeep/Wan2GP/models/TTS/index_tts2/pipeline.py:64`
  `IndexTTS2Pipeline`.
- `deepbeepmeep/Wan2GP/models/TTS/chatterbox/pipeline.py:12`
  `ChatterboxPipeline`.
- `deepbeepmeep/Wan2GP/models/TTS/ace_step/pipeline_ace_step.py:92`
  `ACEStepPipeline`.
- `deepbeepmeep/Wan2GP/models/TTS/ace_step15/pipeline_ace_step15.py:101`
  `ACEStep15Pipeline`.

## DinoML Gap

Very high if UI scope includes audio. DinoML would need audio tensor specs,
sample-rate metadata, mel/spectrogram preprocessing, codec tokens, vocoder
contracts, long waveform chunking, time-aligned conditioning, and potentially
separate provider paths for 1D convolution/audio attention workloads.

