# Diffusers LTX2 full-audit report

## 1. Source basis

```text
Diffusers commit/version: b3a515080752a3ba7ca92161e25530c7f280f629
Model id(s): Lightricks/LTX-2; Lightricks/LTX-2.3; Lightricks/LTX-2.3-fp8; Lightricks/LTX-2.3-nvfp4; dg845/LTX-2.3-Diffusers; CalamitousFelicitousness/LTX-2.3-dev-Diffusers; OzzyGT/tiny_LTX2
Primary target: non-deprecated ltx2 text-to-video-plus-audio base pipeline
Runtime scope: base LTX2Pipeline from pipeline __call__; related variants inventoried separately
```

Config sources checked:

| Source | Files checked | Notes |
| --- | --- | --- |
| `H:/configs/Lightricks/LTX-2/model_index.json` | local model index | Confirms `LTX2Pipeline`, `LTX2VideoTransformer3DModel`, `AutoencoderKLLTX2Video`, `AutoencoderKLLTX2Audio`, `LTX2TextConnectors`, `LTX2Vocoder`, `FlowMatchEulerDiscreteScheduler`, Gemma3 text encoder. Local cache did not contain component configs. |
| Hugging Face `Lightricks/LTX-2` | `model_index.json`, `transformer/config.json`, `vae/config.json`, `audio_vae/config.json`, `connectors/config.json`, `scheduler/scheduler_config.json`, `vocoder/config.json`, text encoder/tokenizer configs, safetensors metadata | Official Diffusers component configs are accessible and are the primary production basis. |
| Hugging Face `Lightricks/LTX-2.3`, `LTX-2.3-fp8`, `LTX-2.3-nvfp4` | repo file lists and root safetensors metadata | Official repos expose root safetensors/README/LICENSE/image files but no Diffusers `model_index.json` or component config directories. Local `H:/configs/Lightricks/LTX-2.3*` model-index files are empty placeholders. |
| Hugging Face `dg845/LTX-2.3-Diffusers` | Diffusers component configs and safetensors metadata | Open mirror used only as labeled evidence for LTX-2.3 Diffusers-format component shapes. |
| Hugging Face `CalamitousFelicitousness/LTX-2.3-dev-Diffusers` | Diffusers component configs and safetensors metadata | Open mirror used as labeled evidence; has a config caveat: transformer `caption_channels=4096` while connectors use `caption_channels=3840`. |
| Hugging Face `OzzyGT/tiny_LTX2` | Diffusers component configs and safetensors metadata | Tiny/debug shape source, useful for parser and smoke tests but not production parity. |

Pipeline files inspected:

- `X:/H/diffusers/src/diffusers/pipelines/ltx2/pipeline_ltx2.py`: `LTX2Pipeline` at line 185, `_pack_latents` at line 530, `_unpack_audio_latents` at line 627, `prepare_latents` at line 645, `prepare_audio_latents` at line 692, `__call__` at line 810.
- `X:/H/diffusers/src/diffusers/pipelines/ltx2/pipeline_ltx2_image2video.py`: `LTX2ImageToVideoPipeline` variant.
- `X:/H/diffusers/src/diffusers/pipelines/ltx2/pipeline_ltx2_condition.py`: `LTX2VideoCondition` and `LTX2ConditionPipeline` variant.
- `X:/H/diffusers/src/diffusers/pipelines/ltx2/pipeline_ltx2_latent_upsample.py` and `latent_upsampler.py`: separate upsampling variant.
- `X:/H/diffusers/src/diffusers/pipelines/ltx2/pipeline_output.py`, `connectors.py`, `vocoder.py`, `export_utils.py`, `utils.py`.

Model files inspected:

- `X:/H/diffusers/src/diffusers/models/transformers/transformer_ltx2.py`: `LTX2AudioVideoAttnProcessor` at line 145, `LTX2Attention` at line 330, `LTX2VideoTransformerBlock` at line 412, `LTX2AudioVideoRotaryPosEmbed` at line 795, `LTX2VideoTransformer3DModel` at line 1062.
- `X:/H/diffusers/src/diffusers/models/autoencoders/autoencoder_kl_ltx2.py`: `AutoencoderKLLTX2Video` at line 1025, `encode` at line 1240, `decode` at line 1292.
- `X:/H/diffusers/src/diffusers/models/autoencoders/autoencoder_kl_ltx2_audio.py`: `AutoencoderKLLTX2Audio` at line 668, `encode` at line 760, `decode` at line 776.
- Shared support: `attention_dispatch.py`, `embeddings.py`, `normalization.py`, `attention.py`, loader mixins in `loaders/lora_pipeline.py`.

Scheduler/processors/helpers inspected:

- `X:/H/diffusers/src/diffusers/schedulers/scheduling_flow_match_euler_discrete.py`
- `VideoProcessor` use through `diffusers.video_processor`
- LTX2 local helper functions for prompt enhancement, latent packing, velocity/x0 conversion, vocoder, and connectors.

External component configs inspected:

- Gemma3 text encoder config from `Lightricks/LTX-2/text_encoder/config.json`: text hidden size 3840, 48 layers, 16 heads, head dim 256, intermediate size 15360, vocab size 262208.
- Tokenizer and processor files from `Lightricks/LTX-2`: Gemma tokenizer/processor metadata for prompt tokenization and optional prompt enhancement.

Missing files or assumptions:

- No gated-config blocker was hit for official LTX-2. Official LTX-2 component configs were fetched with `huggingface_hub`.
- Official LTX-2.3, LTX-2.3-fp8, and LTX-2.3-nvfp4 do not list Diffusers component configs. This is an absence of files, not an authentication failure observed during this audit.
- LTX-2.3 Diffusers-format facts below are from mirrors and are labeled as such; they should not be treated as official checkpoint contracts until official component configs are published or authenticated access exposes them.
- Safetensors metadata was used for weight packaging and approximate size/precision evidence only; operator shape facts come from component configs and source.

## 2. Pipeline and component graph

Base dataflow:

```text
prompt strings
  -> Gemma tokenizer + Gemma3ForConditionalGeneration hidden-state stack
  -> LTX2TextConnectors: video prompt stream + audio prompt stream + mask
  -> initialize packed video latents and packed audio latents
  -> denoising loop:
       LTX2VideoTransformer3DModel(video tokens, audio tokens, text streams, coords, timesteps)
       -> video/audio velocity predictions
       -> optional CFG/STG/modality-isolation guidance in x0 space
       -> FlowMatchEulerDiscreteScheduler step for video
       -> independent copied FlowMatchEulerDiscreteScheduler step for audio
  -> unpack video tokens to [B, C, F, H, W]
  -> unpack audio tokens to [B, C, T, M]
  -> AutoencoderKLLTX2Video decode -> VideoProcessor postprocess
  -> AutoencoderKLLTX2Audio decode -> mel spectrogram -> LTX2Vocoder or LTX2VocoderWithBWE
  -> LTX2PipelineOutput(frames, audio)
```

Required base components:

| Component | Class/file | Role |
| --- | --- | --- |
| `text_encoder` | `Gemma3ForConditionalGeneration` from Transformers | Prompt hidden states; pipeline stacks all hidden states and flattens layer and feature dimensions before connectors. |
| `tokenizer` | Gemma tokenizer | Prompt and negative prompt tokenization, left padding, max sequence length default 1024. |
| `connectors` | `LTX2TextConnectors`, `pipelines/ltx2/connectors.py` | Converts stacked Gemma features into video and audio prompt-conditioning streams. |
| `transformer` | `LTX2VideoTransformer3DModel`, `models/transformers/transformer_ltx2.py` | Joint video/audio denoiser returning video and audio velocity tokens. |
| `scheduler` | `FlowMatchEulerDiscreteScheduler` | Flow-match Euler denoising. Pipeline deep-copies it for audio so video and audio step indices are independent. |
| `vae` | `AutoencoderKLLTX2Video` | Video KL autoencoder; base T2V path decodes generated video latents. |
| `audio_vae` | `AutoencoderKLLTX2Audio` | Latent audio spectrogram codec; base path decodes generated audio latents to a mel-like spectrogram. |
| `vocoder` | `LTX2Vocoder` or `LTX2VocoderWithBWE`, `pipelines/ltx2/vocoder.py` | Mel spectrogram to waveform; LTX-2 uses 24 kHz, LTX-2.3 mirror uses 48 kHz BWE. |
| `video_processor` | `VideoProcessor` constructed in pipeline | Video postprocessing with VAE spatial scale factor. |

Pipeline metadata:

- `model_cpu_offload_seq = "text_encoder->connectors->transformer->vae->audio_vae->vocoder"`.
- `_optional_components = ["processor"]`; the runtime-critical components above are registered in `__init__`.
- Independently cacheable stages: tokenized prompt inputs, raw Gemma hidden-state stack, connector prompt streams and masks, initial latents, precomputed video/audio RoPE coordinates, scheduler timesteps/sigmas, decoded VAE/audio latents for variant reuse.

Separate candidate reports:

| Candidate | Class/file anchors | Delta from base |
| --- | --- | --- |
| `ltx2_image2video` | `LTX2ImageToVideoPipeline`, `pipeline_ltx2_image2video.py` | Adds image preprocessing, VAE encode of the first frame, packed conditioning mask, and per-step first-frame preservation. |
| `ltx2_condition` | `LTX2VideoCondition`, `LTX2ConditionPipeline`, `pipeline_ltx2_condition.py` | Adds arbitrary image/video condition objects, VAE encode, conditioning strengths, token masks, and token-wise timestep/blend behavior. |
| `ltx2_latent_upsample` | `pipeline_ltx2_latent_upsample.py`, `latent_upsampler.py` | Separate latent upsampler pipeline/model with its own tensor contracts. |
| `ltx2_lora_connectors` | `LTX2LoraLoaderMixin`, `loaders/lora_pipeline.py` | LoRA/runtime adapter mutation for `transformer` and `connectors`; includes non-Diffusers LTX2 LoRA conversion. |
| `ltx2_vocoder_bwe` | `LTX2VocoderWithBWE`, `vocoder.py` | 16 kHz input mel path plus bandwidth extension and 48 kHz waveform output. |
| `ltx2_video_vae` | `AutoencoderKLLTX2Video`, `autoencoder_kl_ltx2.py` | Codec-specific conv/tiling/temporal decode optimization target. |
| `ltx2_audio_vae` | `AutoencoderKLLTX2Audio`, `autoencoder_kl_ltx2_audio.py` | Audio codec-specific 2D causal conv/down/up path. |
| `ltx2_single_file_quantized_variants` | `FromSingleFileMixin`, official root LTX-2.3/fp8/nvfp4 repos | Separate import/config reconstruction problem; official 2.3 repos do not expose Diffusers component configs. |

Extension inventory:

- LoRA/runtime adapters: supported through `LTX2LoraLoaderMixin`; loadable modules are `transformer` and `connectors`.
- Textual inversion: no LTX2 pipeline inheritance or target-specific path observed.
- IP-Adapter: no LTX2-specific IP-Adapter pipeline path observed.
- ControlNet: no LTX2 ControlNet pipeline observed.
- T2I-Adapter: no LTX2 T2I-Adapter pipeline observed.
- GLIGEN: no LTX2 GLIGEN path observed.
- img2img: no generic LTX2 img2img class observed; image-to-video is present and separate.
- inpaint: no LTX2 inpaint pipeline observed.
- depth2img: no LTX2 depth pipeline observed.
- upscaling: latent upsample pipeline is present and should be reviewed separately.

## 3. Important config dimensions

Representative checkpoint/component sweep:

| Repo | Basis | Transformer | Text/connectors | Video VAE | Audio codec/vocoder | Scheduler |
| --- | --- | --- | --- | --- | --- | --- |
| `Lightricks/LTX-2` | Official Diffusers configs | 48 layers; video heads 32 x 128 = 4096; audio heads 32 x 64 = 2048; in/out channels 128; patch 1x1; qk norm `rms_norm_across_heads`; RoPE `split`; audio rate/hop 16000/160 | Gemma3 text hidden 3840, 48 layers; connector caption channels 3840; video/audio connector layers 2, heads 30, head dim 128, 128 learnable registers | `latent_channels=128`; spatial compression 32; temporal compression 8; patch 4 x 1; encoder causal true, decoder causal false in config; scaling factor 1.0 | `AutoencoderKLLTX2Audio`: 2 input channels, latent 8, 64 mel bins, 16 kHz, hop 160. `LTX2Vocoder`: in 128, out 2, hidden 1024, 24 kHz | FlowMatch Euler; dynamic shifting true; base shift 0.95; max shift 2.05; shift terminal 0.1; default pipeline sigmas are linear from 1.0 to 1/steps |
| `dg845/LTX-2.3-Diffusers` | Mirror Diffusers configs | Same base width/depth as LTX-2; adds config flags `audio_cross_attn_mod=True`, `audio_gated_attn=True` | Connector layers 8; video heads 32 x 128; audio heads 32 x 64; 128 registers | `latent_channels=128`; spatial 32; temporal 8; different decoder channel/up type mix from LTX-2 | `LTX2VocoderWithBWE`: in 128, out 2, hidden 1536, input 16 kHz, output 48 kHz, hop 80 | Same FlowMatch dynamic-shift values |
| `CalamitousFelicitousness/LTX-2.3-dev-Diffusers` | Mirror Diffusers configs | 48 layers; video 4096, audio 2048; RoPE `interleaved`; `audio_gated_attn=True`; transformer `caption_channels=4096` | Connectors still report `caption_channels=3840`, 8 layers; treat as mirror/dev inconsistency | `latent_channels=128`; spatial 32; temporal 8 | Model index names `LTX2Vocoder`; vocoder config is sparse and reports sample rate 44100/hop 512 | Same FlowMatch dynamic-shift values |
| `OzzyGT/tiny_LTX2` | Tiny/debug mirror | 2 layers; video heads 2 x 32; audio heads 2 x 32; in/out channels 4; caption/cross dims 64; RoPE `interleaved` | Connector caption 64; 1 layer; 8 registers | `latent_channels=4`; spatial 32; temporal 8 | Audio VAE still latent 8/mel 64; tiny vocoder in 128, hidden 32, 24 kHz | FlowMatch Euler; dynamic shifting false |

Source and config defaults that affect parity:

- Base `__call__` defaults: `height=512`, `width=768`, `num_frames=121`, `frame_rate=24`, `num_inference_steps=40`, `guidance_scale=4.0`, `stg_scale=0.0`, `modality_scale=1.0`, `guidance_rescale=0.0`, `max_sequence_length=1024`.
- Pipeline defaults audio guidance fields with `audio_guidance_scale = audio_guidance_scale or guidance_scale`, and similarly for audio STG/modality/rescale. A user-supplied `0.0` audio value inherits the video value.
- LTX-2 production default latent shape at 512 x 768 x 121: video latent map `[B, 128, 16, 16, 24]`; packed video tokens `[B, 6144, 128]`.
- Audio default for 121 frames at 24 fps: duration 5.0417 s; latent frames are `round(num_frames / frame_rate * 16000 / 160 / 4) = 126`; latent mel bins are `64 / 4 = 16`; packed audio tokens `[B, 126, 128]`.
- Scheduler family required for first parity: `FlowMatchEulerDiscreteScheduler` with dynamic shifting and terminal shift. Other scheduler swaps are not part of the sampled model-index contracts.
- Prediction target: transformer outputs flow/velocity tokens. Pipeline converts velocity to x0 for CFG/STG/modality deltas, then converts back to velocity before scheduler step.
- Safetensors metadata indicates large sharded production weights: official LTX-2 has sharded transformer/text encoder plus codec weights and root monolithic safetensors. This is a loading/staging concern, not an operator difference.

## 3a. Family variation traps

- LTX2 is a coupled video-plus-audio denoiser, not a video-only DiT. Every transformer block has video self-attention, audio self-attention, video/text cross-attention, audio/text cross-attention, audio-to-video cross-attention, video-to-audio cross-attention, and separate video/audio feed-forward paths.
- LTX-2 and LTX-2.3 mirrors share the same main transformer width but change connector depth, prompt/cross-attention modulation, audio gating, RoPE style, VAE decoder details, and vocoder type.
- Official LTX-2.3/fp8/nvfp4 repos do not expose Diffusers component configs. Do not promote mirror-specific 2.3 dimensions to official facts without a config source.
- CalamitousFelicitousness 2.3-dev has a connector/transformer caption-channel mismatch; use it only as a variation signal.
- Pipeline-level latent packing and VAE-internal patchify use different reshape/permute orders. Treat VAE patchify/depatchify as codec-local and guard it from generic "same as transformer packing" rewrites.
- Source layouts are NCDHW for video VAE maps `[B,C,F,H,W]`, packed `[B,S,D]` for transformer tokens, NCHW-like audio maps `[B,C,T,M]`, and Conv1d audio `[B,C,T]` in vocoder. Any channel-last pass needs explicit axis rewrites and local layout guards.
- Prompt embeddings before connectors are not just the last hidden state. The pipeline stacks all Gemma hidden states and flattens layer and feature dimensions.
- The pipeline duplicates the scheduler object for audio. A compiled runtime must not share one mutable scheduler step index across modalities.
- Dynamic scheduler shifting requires `mu`; base pipeline computes it from scheduler config max image sequence settings, not directly from the actual packed sequence length.
- LTX-2.3 mirror uses `LTX2VocoderWithBWE`, adding STFT/mel extraction, interpolation, bandwidth extension, skip upsampling, clamping, and output cropping.
- I2V and condition variants add VAE encode and token masks; they are not just extra optional arguments on the base T2V contract.
- STG and modality isolation add extra transformer forwards when enabled. Base defaults disable them, but the public `__call__` exposes them.

## 4. Runtime tensor contract

CPU/data-pipeline inputs:

- `prompt`, `negative_prompt`: string or list of strings.
- Optional prompt-enhancement path uses processor/chat-template/generation before normal tokenization.
- Tokenization: max length defaults to 1024; Gemma padding side is left; missing pad token is set to EOS.
- `prompt_embeds` may be supplied externally. If not supplied, text encoder returns hidden states, pipeline stacks them as `[B, S_text, hidden, layers_plus_embedding]` and flattens to `[B, S_text, hidden * num_hidden_states]`.
- Negative prompt defaults to the empty string when CFG is active and `negative_prompt_embeds` is not supplied.

Connector boundary:

- Input: stacked Gemma prompt embeddings and attention mask.
- Output: `connector_prompt_embeds` for video attention, `connector_audio_prompt_embeds` for audio attention, and `connector_attention_mask`.
- Production LTX-2 transformer expects video cross-attention dim 4096 and audio cross-attention dim 2048. Connector sequence length includes text tokens plus optional learnable registers when configured.
- Connector outputs, attention mask, and RoPE coordinates are reusable across denoising steps for fixed prompt/shape.

Video latent contract:

- Source latent map layout: `[B, C_v, F_lat, H_lat, W_lat]`, with C_v from transformer `in_channels`.
- LTX-2 production defaults: C_v=128, F_lat=16, H_lat=16, W_lat=24 at 121 frames, 512 x 768, compression 8 x 32.
- Random initialization uses `randn_tensor` in float32, then latents are packed and cast as needed for model inputs.
- Normalization before packing uses VAE `latents_mean`, `latents_std`, and `scaling_factor` over channel axis.
- `_pack_latents` source order:

```text
[B,C,F,H,W]
  -> reshape [B,C,F/pt,pt,H/p,p,W/p,p]
  -> permute [B,F/pt,H/p,W/p,C,pt,p,p]
  -> flatten patch/features -> [B, S_video, C*pt*p*p]
```

- With transformer patch 1 x 1, packed dimension remains 128.
- `_unpack_latents` is the inverse and returns `[B,C,F,H,W]`.

Audio latent contract:

- Audio latent map layout: `[B, C_a, T_lat, M_lat]`.
- Production defaults: audio VAE latent C_a=8, mel bins 64, mel compression 4, so M_lat=16.
- Latent frames use `round(num_frames / frame_rate * sample_rate / hop_length / temporal_compression_ratio)`. LTX-2 default is `round(121 / 24 * 16000 / 160 / 4) = 126`.
- `_pack_audio_latents` default path transposes and flattens `[B,C,T,M] -> [B,T,C*M]`; with C=8 and M=16, feature dim is 128.
- `_unpack_audio_latents` returns `[B,C,T,M]`.

Denoiser step inputs and outputs:

| Tensor | Shape/layout | Notes |
| --- | --- | --- |
| `hidden_states` | `[B_or_2B, S_video, 128]` for production | Packed video latents; duplicated for CFG. |
| `audio_hidden_states` | `[B_or_2B, S_audio, 128]` | Packed audio latents; duplicated for CFG. |
| `encoder_hidden_states` | `[B_or_2B, S_conn, 4096]` | Video prompt stream after connectors. |
| `audio_encoder_hidden_states` | `[B_or_2B, S_conn, 2048]` | Audio prompt stream after connectors. |
| `encoder_attention_mask` | `[B_or_2B, S_conn]` | Converted to additive attention bias inside transformer. |
| `timestep`, `sigma` | `[B_or_2B]` or broadcastable tensor | `sigma` is used by LTX-2.3 prompt/cross modulation. |
| `video_coords` | `[B_or_2B, 3, S_video, 2]` | Prepared once from frames, latent H/W, and fps. |
| `audio_coords` | `[B_or_2B, 1, S_audio, 2]` | Prepared once from audio latent frames and audio rate/hop/compression. |
| outputs | video `[B_or_2B, S_video, 128]`, audio `[B_or_2B, S_audio, 128]` | Flow/velocity predictions. |

Decode and postprocess:

- Video decode input: denormalized `[B,128,F_lat,H_lat,W_lat]`; output sample `[B,3,F,H,W]`; `VideoProcessor.postprocess_video` returns PIL/numpy/torch frames depending `output_type`.
- If `vae.config.timestep_conditioning` is true, decode can mix latents with random noise and pass a decode timestep. Sampled LTX2 configs set timestep conditioning false.
- Audio decode input: denormalized `[B,8,T_lat,16]`; audio VAE output is a mel spectrogram for the vocoder.
- Vocoder output: waveform `[B,2,num_samples]`, 24 kHz for `LTX2Vocoder`, 48 kHz for `LTX2VocoderWithBWE`.
- If `output_type == "latent"`, the pipeline returns denormalized video latents and audio latents rather than decoded media.

GPU/runtime versus host:

- Host first: tokenization, optional prompt enhancement generation, scheduler object setup, shape validation, output object construction.
- GPU first: connector transformer, main transformer, VAE decode, audio VAE decode, vocoder, CFG/STG/guidance arithmetic, scheduler step arithmetic.
- Precomputable per request: prompt streams, masks, video/audio coords, timestep/sigma tables, latent shape constants.

## 5. Operator coverage checklist

Tensor/layout ops:

- `reshape`, `view`, `permute`, `transpose`, `flatten`, `unflatten`, `repeat`, `repeat_interleave`, `chunk`, `cat`, `stack`, `pad`, `crop`, mask expansion, scalar/tensor broadcasting.
- NCDHW video map packing/unpacking and NCHW-like audio map packing/unpacking.
- Axis-sensitive reductions for guidance rescale and norms.

Convolution/downsample/upsample ops:

- Video VAE: causal and non-causal `Conv3d`, residual blocks, spatial/temporal/spatiotemporal downsample and upsample, patchify/depatchify with Conv3d around `[B,C,F,H,W]`.
- Audio VAE: causal `Conv2d`, `GroupNorm` or PixelNorm paths, avg-pool/stride Conv2d downsample, nearest-interpolate upsample plus Conv2d, optional attention block over `H*W`.
- Vocoder: Conv1d, ConvTranspose1d, residual dilated Conv1d, SnakeBeta activation, LeakyReLU, optional STFT/mel and interpolation for BWE.

GEMM/linear ops:

- Main transformer input/output projections: video `128 -> 4096 -> 128`, audio `128 -> 2048 -> 128` in production.
- Per-attention Q/K/V/out projections with bias; cross-attention projections where query dim and context dim differ.
- Feed-forward MLPs with approximate GELU.
- Connector projection and 1D transformer layers; Gemma hidden-stack projection into video/audio prompt widths.
- Ada/time/modulation projections and gate tables.

Attention primitives:

- Dense non-causal video self-attention, audio self-attention, video/text cross-attention, audio/text cross-attention, audio-to-video attention, video-to-audio attention.
- Prompt cross-attention uses additive mask derived from text mask.
- QK RMSNorm across flattened heads, RoPE on Q/K, optional per-head gates.
- Connector 1D attention with RoPE.
- Audio VAE local attention block as Conv2d Q/K/V plus BMM softmax.

Normalization and adaptive conditioning:

- RMSNorm in transformer attention and VAE blocks.
- LayerNorm in transformer final layers.
- AdaLayerNormSingle/time-conditioned scale/shift/gates.
- GroupNorm and PixelNorm in audio VAE.
- Channel-wise VAE latent mean/std and scaling-factor arithmetic.

Position/timestep/guidance embeddings:

- LTX2 video/audio/cross RoPE coordinate preparation.
- Sinusoidal/rotary frequency generation.
- Timestep/sigma embeddings for video and audio streams.
- CFG, STG, modality-isolation, and guidance-rescale math in x0 space.

Scheduler and guidance arithmetic:

- FlowMatch Euler `set_timesteps`, sigma shifting, terminal shift stretch, mutable step index, scalar `dt = sigma_next - sigma`.
- Velocity/x0 conversions and separate video/audio scheduler states.
- Optional extra transformer passes for STG and modality isolation.

VAE/postprocessing/audio-specific ops:

- Video processor resize/postprocess contracts.
- Audio latent-to-mel decode and vocoder waveform synthesis.
- BWE STFT/mel path, interpolation, skip/residual addition, clamp/crop for 48 kHz variants.

## 6. Denoiser/model breakdown

`LTX2VideoTransformer3DModel` forward path:

1. Convert prompt mask to additive attention bias.
2. Prepare video/audio coordinates if not supplied; compute video RoPE, audio RoPE, and cross-attention RoPE.
3. Project packed video tokens with `proj_in` and audio tokens with `audio_proj_in`.
4. Compute timestep/sigma embeddings. LTX-2.3 mirror configs use audio/prompt cross-attention modulation and gated attention flags.
5. Optionally project prompt embeddings inside the transformer for LTX2.0-style configs; LTX2.3-style configs move more prompt projection into connectors.
6. Run `num_layers` transformer blocks.
7. Apply output norm/modulation and linear projections back to video/audio latent-channel token dims.

`LTX2VideoTransformerBlock`:

- Video self path: RMSNorm/adaptive scale-shift -> `LTX2Attention` with video RoPE -> gated residual.
- Audio self path: separate RMSNorm/adaptive scale-shift -> `LTX2Attention` with audio RoPE -> gated residual.
- Video/text cross path: norm and optional cross-attn AdaLN modulation -> attention over connector video prompt states with prompt mask -> residual gate.
- Audio/text cross path: same pattern with audio prompt states and audio width.
- Cross-modal paths: audio-to-video and video-to-audio attention, controlled by `isolate_modalities`; optional scale/shift/gate tables.
- Feed-forward paths: independent video and audio `FeedForward` modules, approximate GELU, gated residual.

Attention implementation:

- `LTX2Attention` requires `qk_norm="rms_norm_across_heads"` for sampled configs.
- Q, K, V are independent Linear layers with bias; output is Linear plus dropout.
- Optional `to_gate_logits` produces per-head gates multiplied as `2 * sigmoid(gate_logits)`.
- `LTX2AudioVideoAttnProcessor` applies Q/K/V projection, Q/K RMSNorm, RoPE, unflatten to `[B,S,H,D]`, calls `dispatch_attention_fn`, flattens, applies output projection, optional gating.
- `LTX2PerturbedAttnProcessor` is the STG processor; it can replace perturbed attention output with V-projection-derived values for selected blocks/tokens.

Video VAE:

- Encoder input `[B,3,F,H,W]` is patchified with patch size 4 and temporal patch size 1, using a codec-local permute order, then Conv3d/down blocks/mid/norm/Conv3d produce posterior moments.
- Source config registers latent mean/std; production configs use scaling factor 1.0 and explicit compression 32 x 8.
- Decoder maps `[B,128,F_lat,H_lat,W_lat]` through Conv3d/mid/up blocks/norm/Conv3d and inverse patch reshape to `[B,3,F,H,W]`.
- Tiling and temporal tiling paths exist and are codec optimization candidates, not first-slice base requirements.

Audio VAE and vocoder:

- Audio VAE uses 2D latent spectrogram maps `[B,C,T,M]`, causal Conv2d, resnet/down/up blocks, optional attention, and KL posterior.
- Base pipeline only needs decode for generated audio latents, but encode must be implemented for condition/continuation variants and codec parity.
- Vocoder maps generated mel spectrograms to stereo waveform with Conv1d/ConvTranspose1d/residual blocks. BWE variant adds a low-rate vocoder, STFT/mel extraction, bandwidth-extension vocoder, skip upsample, clamp, and crop.

## 7. Attention requirements

Required attention variants:

| Variant | Query/context | Heads | Mask | Positional work | Notes |
| --- | --- | --- | --- | --- | --- |
| Video self | video tokens -> video tokens | LTX-2: 32 heads x 128 | none | video RoPE, 3D coords | Dense non-causal. |
| Audio self | audio tokens -> audio tokens | LTX-2: 32 heads x 64 | none | audio RoPE, time coords | Dense non-causal. |
| Video/text cross | video tokens -> connector video prompt | video heads/width | additive prompt mask | cross RoPE | Context dim 4096 in production. |
| Audio/text cross | audio tokens -> connector audio prompt | audio heads/width | additive prompt mask | cross RoPE | Context dim 2048 in production. |
| Audio-to-video | video queries over audio context | video/audio block-specific dims | none | cross-modal RoPE | Disabled only under modality-isolation extra pass. |
| Video-to-audio | audio queries over video context | video/audio block-specific dims | none | cross-modal RoPE | Disabled only under modality-isolation extra pass. |
| Connector attention | text/register sequence | config-dependent | connector mask | 1D RoPE | Smaller but required before denoising. |
| Audio VAE attention | spectrogram positions | Conv2d q/k/v | none | none | BMM over flattened spatial/time positions. |

Backend dispatch:

- The primary implementation for LTX2 attention is `attention_dispatch.py` through `dispatch_attention_fn`, not classic `attention_processor.py`.
- Native parity path uses PyTorch scaled-dot-product attention after explicit Q/K RMSNorm and RoPE.
- Diffusers registry also has flash, flash varlen, xFormers, flex, sage, and newer flash backends behind constraints.
- A Dinoml flash-style provider is feasible for self and cross-modal attention under CUDA fp16/bf16 with head dims 128 and 64, no dropout, and dense non-causal masks.
- Prompt cross-attention needs additive mask support or a guarded fallback; arbitrary additive masks are the highest-risk flash condition.
- QK RMSNorm, RoPE, and gate projection are explicit pre/post attention work. A fused attention kernel can consume normalized/rotary Q/K only if parity tests cover split and interleaved RoPE variants.
- Source projections are not fused QKV modules. Fusing Q/K/V is an optimization with a weight-layout transform, not a required source operator.

## 8. Scheduler and denoising-loop contract

Scheduler setup:

- Source scheduler: `FlowMatchEulerDiscreteScheduler`.
- Production scheduler config: `num_train_timesteps=1000`, `shift=1.0`, `use_dynamic_shifting=True`, `base_shift=0.95`, `max_shift=2.05`, `shift_terminal=0.1`, `invert_sigmas=False`, `stochastic_sampling=False`, `time_shift_type="exponential"`.
- Base pipeline default `sigmas`: `np.linspace(1.0, 1.0 / num_inference_steps, num_inference_steps)`.
- Dynamic shift `mu` is computed with `calculate_shift(...)` from scheduler config values.
- Pipeline calls `retrieve_timesteps` for video scheduler and an independent deep-copied audio scheduler.

Loop-side graph work:

1. Duplicate video/audio latents and prompt streams for CFG when guidance is active.
2. Run transformer once for conditional and unconditional batches.
3. Convert video/audio velocity predictions to x0 using current scheduler sigma.
4. Compute CFG delta: `(guidance_scale - 1) * (x0_cond - x0_uncond)` separately for video and audio, with audio scales defaulting to video scales.
5. If STG is enabled, run an extra perturbed transformer pass and add STG x0 delta.
6. If modality isolation is enabled, run an extra transformer pass with `isolate_modalities=True` and add modality delta.
7. Optional `guidance_rescale` matches standard-deviation statistics over non-batch dims.
8. Convert guided x0 back to velocity.
9. Call video scheduler `step` and copied audio scheduler `step`.

Scheduler step math:

- Non-stochastic FlowMatch Euler uses `prev_sample = sample + (sigma_next - sigma) * model_output`.
- Scheduler owns mutable `step_index`; this is why audio uses a copy.
- `per_token_timesteps` exists in scheduler source but is not used by the base pipeline.

Initial Dinoml slice:

- Implement host-controlled `FlowMatchEulerDiscreteScheduler` with dynamic shifting, terminal shift, custom sigmas, mutable step index, and scalar step math.
- Compile guidance arithmetic and velocity/x0 conversions as small elementwise kernels after one-step parity is stable.

## 9. Position, timestep, and custom math

Prompt hidden-state stack:

```text
hidden_states = stack(all_gemma_hidden_states, dim=-1)
prompt_embeds = flatten(hidden_states, dims=(feature, layer))
```

This means Gemma3 hidden size 3840 and 49 hidden states imply a 188160-wide pre-connector feature vector for production LTX-2.

LTX2 RoPE:

- `LTX2AudioVideoRotaryPosEmbed` supports `rope_type="split"` and `rope_type="interleaved"`.
- Video coords have shape `[B,3,S_video,2]` for frame, height, and width axes. Frame coords include causal offset/scale behavior and fps conversion.
- Audio coords have shape `[B,1,S_audio,2]` and convert latent frame indices to seconds using sample rate, hop length, and audio compression.
- Cross-attention RoPE uses sliced coordinate dimensions from video/audio coords.
- Coordinates can be precomputed for fixed batch, frame count, latent H/W, fps, and audio length.

Timestep and modulation:

- `LTX2AdaLayerNormSingle` produces time embeddings, scale/shift/gate values, and optional cross-attention prompt modulation.
- LTX-2.3 mirror configs turn on audio cross-attention modulation and audio gated attention; LTX-2 production has fewer active modulation flags.
- Timestep embeddings depend on current scheduler timestep/sigma and cannot be fully precomputed across arbitrary custom timesteps, but the sigma table can.

Custom math:

- Velocity/x0 conversion is scheduler-sigma dependent and must match pipeline helpers.
- Guidance rescale computes per-sample standard deviations over all non-batch axes and rescales CFG predictions.
- SnakeBeta in the vocoder is a model-specific activation and likely needs a direct kernel or composition parity test.
- VAE/video latent normalization and denormalization are channel-wise and axis-sensitive under layout translation.

## 10. Preprocessing and input packing

Prompt processing:

- Base prompt path is CPU tokenization plus GPU/accelerator Gemma forward unless prompt embeds are supplied.
- Negative prompt uses empty string by default under CFG.
- Prompt and negative prompt embeds are concatenated along batch after encoding. Connector outputs and masks are repeated/concatenated to match CFG batch.
- Optional prompt enhancement is a separate Gemma generation path and should not block first-slice parity with supplied prompts or prompt embeds.

Video preprocessing/packing:

- Base T2V has no image/video input preprocessing before denoising.
- Video latent maps are initialized directly in VAE latent space, normalized, then packed to transformer tokens.
- Decode path unpacks tokens, denormalizes, then calls VAE decode and `VideoProcessor.postprocess_video`.

Audio preprocessing/packing:

- Base T2V has no source audio feature extraction before denoising.
- Audio latents are initialized as latent spectrogram maps `[B,8,T,16]`, normalized, then packed to `[B,T,128]`.
- Decode path unpacks, denormalizes, audio-VAE-decodes to mel, then vocodes to stereo waveform.

Variant-coupled preprocessing:

- Image-to-video preprocesses an image with `VideoProcessor`, VAE-encodes the first frame, packs a conditioning mask, and preserves/blends the conditioned latent across steps.
- Condition pipeline preprocesses image/video conditions, encodes them through VAE, applies condition strength masks, and passes token-wise conditioning into denoising.
- These variant paths should not inflate the base first implementation slice, but their VAE encode and mask contracts should be reviewed before claiming family coverage.

## 11. Graph rewrite / lowering opportunities

Video latent pack/unpack canonicalization:

- Source pattern: NCDHW reshape/permute/flatten for `_pack_latents`, inverse for `_unpack_latents`.
- Replacement: metadata-only view plus layout-aware tokenization kernel when `patch_size=patch_size_t=1`; general patch kernel for other patch sizes.
- Preconditions: contiguous source layout or explicit copy; known patch divisibility; transformer patch axes match config.
- Failure cases: VAE-internal patchify uses a different order; do not share weight/layout transforms with VAE patchify.
- Parity sketch: random `[B,C,F,H,W]` tensors, all sampled patch sizes, pack then unpack exact equality.

Audio latent pack/unpack canonicalization:

- Source pattern: `[B,C,T,M] -> transpose(1,2).flatten(2,3)`.
- Replacement: reshape/stride view when contiguous and M is static; fused normalization-plus-pack kernel.
- Preconditions: audio transformer feature dim equals `C*M` unless explicit patch sizes are passed.
- Failure cases: future configs with audio patch size not equal to mel latent width require generic path.
- Parity sketch: random audio maps and configured mel compression values.

QKV plus QK norm plus RoPE lowering:

- Source pattern: separate Q/K/V Linear with bias -> RMSNorm(Q,K) -> RoPE -> dispatch attention -> output Linear -> optional head gate.
- Replacement: fused projection and pre-attention transform feeding native/flash attention.
- Preconditions: static heads/head dim, `rms_norm_across_heads`, supported RoPE type, dtype supported by backend, prompt mask either supported or fallback.
- Weight transform: concatenate Q/K/V weights and biases in source projection order.
- Failure cases: additive prompt masks, interleaved-vs-split RoPE mismatch, gated attention placement, STG perturbed processor.
- Parity sketch: one attention module with random context/masks under fp32 and bf16/fp16 tolerances.

Flow guidance elementwise fusion:

- Source pattern: velocity -> x0, CFG/STG/modality deltas, optional guidance rescale, x0 -> velocity, scheduler step.
- Replacement: small fused elementwise kernels per modality with host scheduler state.
- Preconditions: scalar sigma per modality and no per-token scheduler path.
- Failure cases: enabled STG/modality isolation changes number of transformer forwards but not elementwise formula.
- Parity sketch: fixed sigma, random latents/preds, all guidance scale combinations including zero audio values.

Video VAE codec-local patchify and conv lowering:

- Source pattern: VAE patchify/depatchify around NCDHW Conv3d/resnet/down/up blocks.
- Replacement: guarded codec-local lowering, possible NDHWC optimized conv region.
- Preconditions: all consumers inside VAE region accept translated layout and axes for norm/concat/down/up are rewritten.
- Failure cases: tiling/temporal slicing boundaries, causal temporal padding, latent mean/std channel axis.
- Parity sketch: VAE encode/decode random tensors with tiling disabled first.

Vocoder ConvTranspose/Snake lowering:

- Source pattern: Conv1d -> ConvTranspose1d upsampling stack -> residual dilated Conv1d with SnakeBeta/LeakyReLU -> final Conv1d.
- Replacement: Conv1d/ConvTranspose1d kernels plus fused activation/residual where profitable.
- Preconditions: fixed kernel/stride/dilation from config and waveform layout `[B,C,T]`.
- Failure cases: BWE variant adds STFT/mel/interpolation and output crop.
- Parity sketch: mel spectrogram random input, waveform output tolerance for fp32/fp16.

## 12. Kernel fusion candidates

Highest priority:

- Q/K/V projection + QK RMSNorm + RoPE + attention dispatch for LTX2 transformer attention. It dominates 48-layer production denoising and is exercised six times per block.
- AdaLayerNorm/time modulation + residual gates in transformer blocks. These are repeated around every attention and FFN subpath.
- GEMM + approximate GELU FFN fusion for video/audio transformer and connectors.
- Flow guidance/scheduler elementwise fusion for video and audio tokens; cheap but repeated every step and easy to validate.

Medium priority:

- Video latent pack/unpack plus normalize/denormalize fusion. Reduces memory traffic at pipeline/model boundaries.
- Connector 1D transformer attention/FFN fusion. Smaller than the main transformer but prompt-side latency is visible and reusable.
- Video VAE Conv3d/resnet/down/up fusion. Needed for full media decode parity and later VAE optimization.
- Audio VAE Conv2d/resnet/up fusion and audio map pack/unpack.
- Vocoder ConvTranspose1d + residual activation fusion, especially for 48 kHz BWE variants.

Lower priority:

- Prompt enhancement generation path; separate text-generation workload.
- STG perturbed attention specialization; default disabled.
- Modality-isolation extra pass specialization; default scale is 1.0.
- VAE tiling/temporal tiling optimization; important for memory but can follow non-tiled parity.
- BWE STFT/mel fusion; only required for LTX-2.3-style vocoder variant.

## 13. Runtime staging plan

1. Parse component configs and model index for `Lightricks/LTX-2` and `OzzyGT/tiny_LTX2`; load or stub weights without touching official 2.3 mirrors for first parity.
2. Implement tensor contracts for video/audio latent normalization and pack/unpack; add exact random round-trip tests.
3. Bring up connector output ingestion as an external input first: one `LTX2VideoTransformer3DModel` block with supplied video/audio prompt embeddings, masks, coords, and latents.
4. Validate one full tiny transformer forward, then production-shape one-step forward with random weights or loaded weights as available.
5. Implement host-side FlowMatch Euler dynamic-shift scheduler and one-step denoising-loop parity with externally supplied connector embeddings.
6. Add CFG velocity/x0 guidance and optional guidance rescale parity; keep STG and modality isolation disabled initially.
7. Add `LTX2TextConnectors` after denoiser parity; Gemma text encoder can remain external/cached for the first runtime slice.
8. Add video VAE decode and audio VAE decode as separate codec stages. Encode is required for I2V/condition candidates but not for base T2V decode-only smoke.
9. Add `LTX2Vocoder` 24 kHz path. Treat `LTX2VocoderWithBWE` as a follow-up.
10. Run short deterministic end-to-end base pipeline smoke with few steps; then add optimized attention/norm/GEMM kernels.
11. Review and stage `ltx2_image2video`, `ltx2_condition`, latent upsampling, LoRA/connectors, and 2.3 single-file/quantized import as separate candidates.

What can be stubbed initially:

- Gemma text encoder and tokenizer, if connector prompt embeddings are supplied.
- VAE/video/audio decode for denoiser-only parity.
- Vocoder, if audio latents or mel spectrogram parity is the first audio milestone.
- Prompt enhancement, STG, modality isolation, tiling, BWE, and variants.

## 14. Parity and validation plan

- Config parser tests: load `Lightricks/LTX-2` and `OzzyGT/tiny_LTX2`; assert transformer/audio/VAE/scheduler dimensions and compression values.
- Latent pack tests: random video latents `[B,C,F,H,W]`, sampled patch sizes, pack/unpack exact or bitwise equality in fp32.
- Audio pack tests: random `[B,8,T,16]`, pack/unpack exact equality.
- RoPE tests: compare `prepare_video_coords`, `prepare_audio_coords`, split/interleaved RoPE outputs against Diffusers for default and tiny shapes.
- Attention module parity: one `LTX2Attention` with self and cross modes, prompt mask present/absent, fp32 tight tolerance, bf16/fp16 relaxed tolerance.
- Transformer block parity: one block with random hidden states, prompt streams, masks, timesteps/sigmas, and both RoPE types where configs expose them.
- Full denoiser parity: tiny full transformer one forward; then production one-step shape smoke.
- Scheduler parity: `set_timesteps` with default sigmas, custom timesteps, dynamic shift, terminal shift, and `step` output/step-index behavior for copied video/audio schedulers.
- Guidance parity: CFG enabled/disabled, audio scale inheritance, guidance rescale, velocity/x0 round trip.
- VAE parity: video VAE decode first, then encode/decode; audio VAE decode first, then encode/decode. Keep tiling disabled until non-tiled parity passes.
- Vocoder parity: 24 kHz `LTX2Vocoder` random mel smoke; BWE 48 kHz variant separately.
- End-to-end smoke: tiny pipeline with deterministic generator and 1-2 steps, then LTX-2 production-shape memory smoke.

Suggested tolerances:

- fp32 custom math/layout: exact for pure pack/unpack; around `1e-5` absolute/relative for attention and scheduler arithmetic.
- bf16/fp16 attention/conv: use backend-specific tolerances and compare both final tensors and max/mean error.
- Media decode parity should compare latent/mel/waveform tensors before postprocess formatting to avoid PIL/numpy conversion noise.

## 15. Performance probes

- Gemma text encoder throughput and memory by prompt length; separate from connector throughput.
- Connector latency by sequence length and register count; LTX-2 2-layer versus 2.3 mirror 8-layer configs.
- One transformer step latency and VRAM by batch, CFG on/off, video token count, audio token count, and dtype.
- Attention backend comparison: native SDPA versus candidate flash for self/cross-modal attention; prompt cross-attention fallback cost with masks.
- Guidance overhead: baseline CFG only versus STG and modality-isolation extra forwards.
- Scheduler/guidance CPU versus GPU overhead over 40 steps.
- Video VAE decode throughput by latent grid and frame count, with and without tiling.
- Audio VAE decode plus vocoder throughput; 24 kHz vocoder versus 48 kHz BWE.
- Full denoising loop by step count and default 512 x 768 x 121 shape.
- Offload/load timing following `text_encoder->connectors->transformer->vae->audio_vae->vocoder`.
- Workspace/temp memory probes for pack/unpack, QKV projections, attention logits, VAE Conv3d, and vocoder upsampling.

## 16. Scope boundary and separate candidates

Separate candidate reports, not ignored:

- `ltx2_image2video`: `LTX2ImageToVideoPipeline` in `pipeline_ltx2_image2video.py`; adds image processor, VAE encode, first-frame conditioning mask, and post-step latent preservation.
- `ltx2_condition`: `LTX2VideoCondition` and `LTX2ConditionPipeline` in `pipeline_ltx2_condition.py`; adds arbitrary image/video conditions, strengths, token masks, and token-wise denoising behavior.
- `ltx2_latent_upsample`: `pipeline_ltx2_latent_upsample.py` and `latent_upsampler.py`; separate latent-resolution model and pipeline.
- `ltx2_lora_connectors`: `LTX2LoraLoaderMixin` in `loaders/lora_pipeline.py`; runtime mutation and conversion for transformer/connectors LoRA.
- `ltx2_video_vae`: `AutoencoderKLLTX2Video`; VAE encode/decode, tiling, temporal tiling, and Conv3d optimization.
- `ltx2_audio_vae`: `AutoencoderKLLTX2Audio`; audio latent codec encode/decode and Conv2d/attention optimization.
- `ltx2_vocoder_bwe`: `LTX2VocoderWithBWE`; 48 kHz vocoder with STFT/mel and bandwidth extension.
- `ltx2_single_file_quantized`: official LTX-2.3/fp8/nvfp4 root safetensors; separate config reconstruction/loading problem because official component configs are absent.
- Textual inversion, IP-Adapter, ControlNet, T2I-Adapter, GLIGEN, generic img2img, inpaint, and depth2img were not observed as LTX2 family-specific pipelines, but should be rechecked if new Diffusers files appear.

Genuinely out of scope for this audit:

- Multi-GPU and context-parallel paths.
- Callback mutation and interactive interrupt handling.
- XLA, NPU, MPS, Flax, and ONNX branches.
- Safety checker and NSFW filtering.
- Training, losses, dropout semantics beyond inference module definitions, and gradient checkpointing.
- Implementing Dinoml operators or running Dinoml tests.
- Staging or committing repository changes.

## 17. Final implementation checklist

- [ ] Parse LTX2 model index and component configs for official LTX-2 and tiny LTX2.
- [ ] Load or externally supply transformer, connector, VAE, audio VAE, and vocoder weights.
- [ ] Implement video latent normalize/denormalize and pack/unpack.
- [ ] Implement audio latent normalize/denormalize and pack/unpack.
- [ ] Implement LTX2 video/audio RoPE coordinate generation and split/interleaved application.
- [ ] Implement `LTX2Attention`: Linear Q/K/V, QK RMSNorm, RoPE, dense attention, output projection, optional per-head gate.
- [ ] Implement one `LTX2VideoTransformerBlock` with video/audio self, text cross, cross-modal attention, FFNs, adaptive norm/gates.
- [ ] Implement full `LTX2VideoTransformer3DModel` forward for tiny then production configs.
- [ ] Implement `FlowMatchEulerDiscreteScheduler` dynamic shifting, terminal shift, custom sigmas, copied audio scheduler state, and scalar step.
- [ ] Implement velocity/x0 conversion, CFG, guidance rescale, and audio guidance-scale inheritance parity.
- [ ] Add connector model parity or accept cached connector prompt embeddings for the first denoiser milestone.
- [ ] Add video VAE decode parity, then encode parity for variants.
- [ ] Add audio VAE decode parity, then encode parity for variants.
- [ ] Add 24 kHz `LTX2Vocoder` parity; defer BWE to its own candidate.
- [ ] Add random tensor tests for all custom layout and math helpers.
- [ ] Add one-block, one-step, scheduler, and tiny end-to-end parity tests.
- [ ] Benchmark transformer attention backend, full denoising step, VAE decode, audio VAE/vocoder, and memory/offload behavior.
- [ ] Open separate audits for image-to-video, condition, latent upsample, LoRA/connectors, BWE, video/audio codecs, and official 2.3 single-file/quantized loading.
