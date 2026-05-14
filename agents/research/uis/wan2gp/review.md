# deepbeepmeep/Wan2GP

## Source

- UI clone: `H:/uis/deepbeepmeep/Wan2GP`

## Summary

Wan2GP is the richest source for video/audio/runtime gaps. It is less a generic
image UI and more a low-VRAM video-generation workbench with many system
postprocesses, preprocessors, audio tools, quantized formats, and kernel
switches. Its README and model tree are direct evidence that DinoML needs
video/audio auxiliary contracts beyond diffusers/transformers model execution.

## Model Support

- Main video models: Wan and derived models, Hunyuan Video, LTXV, LTX-2,
  Cosmos/world-style models via related projects, Vista4D, and VACE ControlNet
  workflows.
- Main image models: Flux, Qwen, Z-Image, LongCat, Kandinsky, HiDreamO1 and
  related image/editing models.
- Audio/TTS models: Qwen3 TTS, Chatterbox, HearMula, Omnivoice, ScenemeAI,
  IndexTTS2, MMAudio, LTX video-to-audio, BigVGAN vocoders, Wav2Vec2-style
  semantic encoders.
- Agent/VLM helpers: Deepy uses Qwen3VL-style checkpoints for prompt/tool
  assistance and can generate image/video/audio or inspect/transcribe media.
- Formats/runtime: int8, fp8, GGUF, NVFP4, Nunchaku, mmgp offload, optimum
  quanto, Triton, SageAttention, SpargeAttention, Lightx2v and related kernels.
- Adapters/control: LoRA, system LoRAs, LTX Ic LoRAs, outpaint/refocus/ungrade/
  uncompress control LoRAs, VACE ControlNet, pose/depth/flow extraction.

## Feature Surface

- Wan, Hunyuan Video, Flux, Qwen, Z-Image, LongCat, Kandinsky, LTXV/LTX-2,
  Qwen3 TTS, Chatterbox, HearMula and related model families.
- Low-VRAM execution with model offload and residency profiles.
- Quantized checkpoint formats: int8, fp8, GGUF, NVFP4, Nunchaku.
- Integrated video tools: mask editor, prompt enhancer, temporal/spatial
  generation, MMAudio, video browser, pose/depth/flow extractors, motion
  designer.
- FlashVSR video upsampling/postprocessing.
- SAM3/Magic Mask image and video masks with temporal consistency.
- LTX video-to-audio and TTS/dialogue workflows.
- Sliding windows, phased generation, outpainting, injected frames, camera or
  motion transfer, pose alignment.

## Auxiliary Model Families

- FlashVSR spatial/video upsampler.
- SAM3 image/video segmentation and tracking.
- Pose, depth, and optical flow extractors.
- MMAudio and video-to-audio models.
- BigVGAN/BigVGAN-v2 vocoders.
- IndexTTS2, Omnivoice, Chatterbox, HearMula, Qwen3 TTS and related TTS stacks.
- Wav2Vec2-style semantic/audio encoders.
- Wan/LTX/Hunyuan/Cosmos/Mochi/Qwen video model support.
- GGUF, NVFP4, Nunchaku, scaled fp8, optimum-quanto, mmgp offload.
- SageAttention, SpargeAttention, Triton and other kernel acceleration paths.

## Packages and Loaders

- `mmgp` for offload/profiles/quant router behavior.
- Local `shared.qtypes.*` handlers for quantized formats.
- Local kernel hooks under `shared.kernels`.
- Local model implementations under `models/wan`, `models/hyvideo`,
  `models/TTS`, and related trees.

## Code Anchors

- `H:/uis/deepbeepmeep/Wan2GP/README.md:8` lists supported video/image/audio
  model families.
- `H:/uis/deepbeepmeep/Wan2GP/README.md:14` lists quantized checkpoint formats.
- `H:/uis/deepbeepmeep/Wan2GP/README.md:16` lists integrated video tools.
- `H:/uis/deepbeepmeep/Wan2GP/README.md:40`
  describes Omnivoice TTS.
- `H:/uis/deepbeepmeep/Wan2GP/README.md:42`
  describes ScenemeAI LTX-derived TTS.
- `H:/uis/deepbeepmeep/Wan2GP/README.md:48` announces FlashVSR video upsampler.
- `H:/uis/deepbeepmeep/Wan2GP/README.md:65`
  describes Vista4D.
- `H:/uis/deepbeepmeep/Wan2GP/README.md:71` describes Magic Mask powered by
  SAM3.
- `H:/uis/deepbeepmeep/Wan2GP/README.md:73` describes video mask generator with
  SAM3 support.
- `H:/uis/deepbeepmeep/Wan2GP/README.md:75`
  describes LTX-2 video-to-audio.
- `H:/uis/deepbeepmeep/Wan2GP/README.md:111` describes human motion transfer
  with pose alignment.
- `H:/uis/deepbeepmeep/Wan2GP/README.md:153`
  describes LTX 2 Id LoRA/talking-head workflow.
- `H:/uis/deepbeepmeep/Wan2GP/README.md:167`
  describes Deepy image/video/audio generation and media tools.
- `H:/uis/deepbeepmeep/Wan2GP/README.md:203`
  describes Deepy Qwen3VL/GGUF requirements.
- `H:/uis/deepbeepmeep/Wan2GP/README.md:406`
  links VACE ControlNet docs.
- `H:/uis/deepbeepmeep/Wan2GP/wgp.py:36` imports `mmgp` offload/profile/quant
  router helpers.
- `H:/uis/deepbeepmeep/Wan2GP/wgp.py:166` registers scaled fp8 qtype handler.
- `H:/uis/deepbeepmeep/Wan2GP/wgp.py:167` registers NVFP4 qtype handler.
- `H:/uis/deepbeepmeep/Wan2GP/wgp.py:168` registers Nunchaku int4 qtype
  handler.
- `H:/uis/deepbeepmeep/Wan2GP/wgp.py:169` registers Nunchaku fp4 qtype handler.
- `H:/uis/deepbeepmeep/Wan2GP/wgp.py:170` registers GGUF qtype handler.
- `H:/uis/deepbeepmeep/Wan2GP/wgp.py:177` imports quanto int8 kernel injection.
- `H:/uis/deepbeepmeep/Wan2GP/models/wan/ovi/modules/mmaudio/ext/bigvgan/bigvgan.py:12`
  defines BigVGAN wrapper.
- `H:/uis/deepbeepmeep/Wan2GP/models/wan/ovi/modules/mmaudio/ext/bigvgan/models.py:169`
  defines BigVGAN vocoder.
- `H:/uis/deepbeepmeep/Wan2GP/models/wan/ovi/modules/mmaudio/ext/autoencoder/autoencoder.py:12`
  connects MMAudio autoencoder to BigVGAN.
- `H:/uis/deepbeepmeep/Wan2GP/models/TTS/index_tts2/pipeline.py:64`
  defines IndexTTS2 pipeline.
- `H:/uis/deepbeepmeep/Wan2GP/models/TTS/index_tts2/pipeline.py:148`
  requires BigVGAN assets for IndexTTS2.
- `H:/uis/deepbeepmeep/Wan2GP/models/TTS/index_tts2/utils/maskgct/models/codec/facodec/facodec_trainer.py:246`
  references Wav2Vec2 phoneme/semantic features.

## DinoML Gaps

- Multi-modal model support spanning video, image/editing, audio/TTS, VLM/agent
  helpers, LoRA/control variants, qtypes, offload profiles, and custom kernels.
- Video-first contracts: temporal shape metadata, sliding window state, phase
  schedules, control-video injection, temporal masks, and video postprocess.
- Audio-first contracts: TTS reference audio, vocoder loading, semantic/audio
  encoders, video-to-audio synchronization, and dialogue generation.
- Provider/runtime contracts for quantized qtypes, residency/offload, sparse
  attention, and custom kernels.

## Further Exploration Additions

- Wan-derived families are much broader than generic Wan: Alpha/transparent
  RGBA video, Animate, Chrono Edit, Fun InP, Phantom, ReCamMaster, Stand-In,
  Lynx, Ditto, Multitalk/Infinitalk/Fantasy talking, Ovi, MoCha, SCAIL,
  SteadyDancer, WanMove, SVI2Pro, Lucy/Kiwi edit.
  Anchors: `H:/uis/deepbeepmeep/Wan2GP/models/wan/wan_handler.py:53`,
  `H:/uis/deepbeepmeep/Wan2GP/defaults/alpha.json:6`,
  `H:/uis/deepbeepmeep/Wan2GP/defaults/multitalk.json:11`.
- Hunyuan 1.5 coverage includes t2v/i2v 480p/720p, Lightx2v, dedicated
  upsamplers, avatar, custom-audio, and custom-edit.
  Anchors:
  `H:/uis/deepbeepmeep/Wan2GP/defaults/hunyuan_1_5_480_t2v_lightx2v.json:5`,
  `H:/uis/deepbeepmeep/Wan2GP/defaults/hunyuan_avatar.json:6`.
- LTX-2 support includes ID-LoRA, union control, outpaint, HDR IC-LoRA, audio
  VAE, video-to-audio, audio-window slicing, video length not limited by audio,
  and frame injection.
  Anchors: `H:/uis/deepbeepmeep/Wan2GP/models/ltx2/ltx2_handler.py:18`,
  `H:/uis/deepbeepmeep/Wan2GP/models/ltx2/ltx2_handler.py:476`.
- Process Full Video plugin supports chunked full-video processing,
  continuation/resume metadata, outpaint, detailer, HDR conversion, refocus,
  ungrade, uncompress, and FlashVSR system handling.
  Anchors:
  `H:/uis/deepbeepmeep/Wan2GP/plugins/wan2gp-process-full-video/settings/FlashVSR Upscale.json:2`,
  `H:/uis/deepbeepmeep/Wan2GP/plugins/wan2gp-process-full-video/process_runner.py:287`.
- Audio/music model coverage includes ACE-Step v1/v1.5/v1.5 XL, KugelAudio,
  YuE, HeartMuLa RL, Qwen3 TTS variants, and OmniVoice/Whisper dependency.
  Anchors: `H:/uis/deepbeepmeep/Wan2GP/defaults/ace_step_v1.json:5`,
  `H:/uis/deepbeepmeep/Wan2GP/defaults/ace_step_v1_5_xl.json:5`,
  `H:/uis/deepbeepmeep/Wan2GP/models/TTS/yue_handler.py:6`.
- Flux coverage includes Flux Chroma/Radiance, SRPO, UMO, USO, DreamOmni2,
  and Flux.2 Dev/Klein/NVFP4.
  Anchors: `H:/uis/deepbeepmeep/Wan2GP/defaults/flux_chroma.json:5`,
  `H:/uis/deepbeepmeep/Wan2GP/defaults/flux2_dev_nvfp4.json:5`.
- Kandinsky should be tracked as Kandinsky 5 Lite/Pro T2V/I2V with MagCache/
  sparse config handling.
  Anchor: `H:/uis/deepbeepmeep/Wan2GP/models/kandinsky5/kandinsky_handler.py:81`.
- LongCat and Magi Human are distinct model families, including LongCat video/
  avatar handlers and Magi Human base/distill/SR 1080p talking-head pipeline.
  Anchors: `H:/uis/deepbeepmeep/Wan2GP/models/longcat/longcat_handler.py:9`,
  `H:/uis/deepbeepmeep/Wan2GP/models/magi_human/magi_human_handler.py:32`.
- Preprocess/mask stack includes MatAnyone, Magic Mask negative masks, SAM3.1
  assets, Depth Anything v3, NLFPose/SCAIL, and pose alignment.
  Anchors: `H:/uis/deepbeepmeep/Wan2GP/plugins/wan2gp-video-mask-creator/plugin.py:10`,
  `H:/uis/deepbeepmeep/Wan2GP/shared/magic_mask.py:20`,
  `H:/uis/deepbeepmeep/Wan2GP/models/wan/wan_handler.py:829`.
- Runtime/kernel rows should include GGUF llama.cpp CUDA linear/embedding fast
  paths, GGUF qtypes Q2_K through Q8_0, Nunchaku SVDQ/AWQ kernels, NVFP4
  validation, Sage3/radial attention, MPS early support, and torch compile
  hooks.
  Anchors: `H:/uis/deepbeepmeep/Wan2GP/shared/qtypes/gguf.py:787`,
  `H:/uis/deepbeepmeep/Wan2GP/shared/qtypes/nunchaku_int4.py:113`,
  `H:/uis/deepbeepmeep/Wan2GP/shared/attention.py:193`.
- Plugin ecosystem includes Gallery, LoRA Manager/Merger/Organizer/Multipliers
  UI, Multi-Angle Prompt Helper, Queue Editor, downloads/models/config/plugin
  manager/guides/motion designer/process full video/video mask creator.
  Anchors: `H:/uis/deepbeepmeep/Wan2GP/plugins.json:3`,
  `H:/uis/deepbeepmeep/Wan2GP/plugins.json:57`,
  `H:/uis/deepbeepmeep/Wan2GP/shared/utils/plugins.py:199`.
