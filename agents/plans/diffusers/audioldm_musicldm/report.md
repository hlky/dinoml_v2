# AudioLDM / AudioLDM2 / MusicLDM Diffusers Audit

Candidate slug: `audioldm_musicldm`

## 1. Source basis

```text
Diffusers commit/version:
  Local checkout X:/H/diffusers at b3a515080752a3ba7ca92161e25530c7f280f629.

Model id(s):
  cvssp/audioldm
  cvssp/audioldm-s-full-v2
  cvssp/audioldm-m-full
  cvssp/audioldm-l-full
  cvssp/audioldm2
  cvssp/audioldm2-large
  cvssp/audioldm2-music
  ucsd-reach/musicldm
  anhnct/audioldm2_gigaspeech as an open TTS/VITS AudioLDM2 shape variant

Config sources:
  Local H:/configs contains model_index.json for the listed cvssp and
  ucsd-reach repos, but not component configs.
  Official Hugging Face raw component configs were inspected without writing
  them into H:/configs:
    model_index.json
    unet/config.json
    vae/config.json
    scheduler/scheduler_config.json
    vocoder/config.json
    text_encoder/config.json
    text_encoder_2/config.json where present
    language_model/config.json where present
    projection_model/config.json where present
    tokenizer*/tokenizer_config.json
    feature_extractor/preprocessor_config.json where present

Pipeline files inspected:
  X:/H/diffusers/src/diffusers/pipelines/deprecated/audioldm/pipeline_audioldm.py
  X:/H/diffusers/src/diffusers/pipelines/deprecated/musicldm/pipeline_musicldm.py
  X:/H/diffusers/src/diffusers/pipelines/audioldm2/pipeline_audioldm2.py
  X:/H/diffusers/src/diffusers/pipelines/audioldm2/__init__.py

Model files inspected:
  X:/H/diffusers/src/diffusers/pipelines/audioldm2/modeling_audioldm2.py
  X:/H/diffusers/src/diffusers/models/unets/unet_2d_condition.py
  X:/H/diffusers/src/diffusers/models/unets/unet_2d_blocks.py
  X:/H/diffusers/src/diffusers/models/transformers/transformer_2d.py
  X:/H/diffusers/src/diffusers/models/autoencoders/autoencoder_kl.py
  X:/H/diffusers/src/diffusers/models/resnet.py
  X:/H/diffusers/src/diffusers/models/attention.py
  X:/H/diffusers/src/diffusers/models/attention_processor.py
  X:/H/diffusers/src/diffusers/models/embeddings.py

Scheduler/processors/helpers inspected:
  X:/H/diffusers/src/diffusers/schedulers/scheduling_ddim.py
  X:/H/diffusers/src/diffusers/schedulers/scheduling_utils.py
  Transformers SpeechT5 HiFi-GAN source from installed transformers package.

External component configs inspected:
  CLAP text/model configs, T5 encoder configs, GPT2 configs, VITS configs,
  SpeechT5HifiGan vocoder configs, and CLAP feature-extractor configs.

Any missing files or assumptions:
  No gated config blockers were encountered. Component paths that 404 are
  absent by pipeline design, such as AudioLDM v1 having no text_encoder_2,
  language_model, projection_model, or feature_extractor.
```

## 2. Pipeline and component graph

AudioLDM v1 and MusicLDM are deprecated Diffusers pipelines. They remain useful
as first Dinoml candidates because they are narrow: text embedding to a
spectrogram UNet, VAE decode, and HiFi-GAN waveform synthesis. AudioLDM2 adds a
dual text-encoder, projection, and GPT2 hidden-state generation stage, plus a
custom UNet that can attend to two conditioning streams.

```text
prompt / optional transcription
  -> tokenizer(s) and CLAP / T5 / VITS encoders
  -> AudioLDM2 projection + GPT2 hidden-state generation when applicable
  -> spectrogram latent initialization
  -> denoising loop: UNet + CFG + DDIM/Karras-compatible scheduler
  -> AutoencoderKL decode to log-mel spectrogram
  -> SpeechT5HifiGan vocoder to waveform
  -> crop and optional CLAP scoring/reorder
```

Required base components:

| Pipeline | Components |
| --- | --- |
| `AudioLDMPipeline` | `AutoencoderKL`, `ClapTextModelWithProjection`, `RobertaTokenizerFast`, `UNet2DConditionModel`, `DDIMScheduler` by config, `SpeechT5HifiGan` |
| `MusicLDMPipeline` | `AutoencoderKL`, `ClapModel`, `RobertaTokenizerFast`, optional `ClapFeatureExtractor` for scoring, `UNet2DConditionModel`, `DDIMScheduler`, `SpeechT5HifiGan` |
| `AudioLDM2Pipeline` | `AutoencoderKL`, `ClapModel`, `T5EncoderModel` or `VitsModel`, `AudioLDM2ProjectionModel`, `GPT2Model`, tokenizers, optional CLAP feature extractor for scoring, `AudioLDM2UNet2DConditionModel`, `DDIMScheduler`, `SpeechT5HifiGan` |

Cacheable stages are prompt embeddings, AudioLDM2 generated prompt embeddings,
attention masks, scheduler timesteps, initial noise latents, VAE latents, and
decoded mel spectrograms before vocoding.

Separate candidate reports:

- LoRA/runtime adapter mutation: generic `UNet2DConditionLoadersMixin` and
  `AttentionMixin` surfaces exist in UNet classes, but these audio pipelines do
  not expose LoRA loader mixins in their pipeline inheritance. Treat as generic
  UNet adapter work, not an audio first slice.
- Textual inversion: no pipeline textual-inversion mixin was found for these
  audio pipelines.
- IP-Adapter, ControlNet, T2I-Adapter, GLIGEN, img2img, inpaint, depth2img,
  upscaling: no family-local pipeline variants or required side-input contracts
  were found. GLIGEN hooks exist inside shared `BasicTransformerBlock` through
  `cross_attention_kwargs`, but are inactive for the selected configs.
- `audioldm2_tts_vits`: `anhnct/audioldm2_gigaspeech` changes the second
  encoder from T5 to VITS and requires transcription; it deserves a separate
  variant report after the music/sound-effect path.
- `speecht5_hifigan_vocoder`: shared Conv1d/ConvTranspose1d vocoder island
  deserves a codec-focused report because it is used by all three pipelines.

## 3. Important config dimensions

Representative checkpoint sweep:

| Repo | Pipeline | Latent sample | UNet width | Conditioning | Scheduler |
| --- | --- | ---: | --- | --- | --- |
| `cvssp/audioldm` | AudioLDM | 128 | `[128,256,384,640]` | CLAP projection as class embedding, 512 dim | DDIM epsilon |
| `cvssp/audioldm-m-full` | AudioLDM | 128 | `[192,384,576,960]` | same | DDIM epsilon |
| `cvssp/audioldm-l-full` | AudioLDM | 128 | `[256,512,768,1280]` | same | DDIM epsilon |
| `ucsd-reach/musicldm` | MusicLDM | 256 | `[128,256,384,640]` | CLAP projection as class embedding, 512 dim | DDIM epsilon |
| `cvssp/audioldm2` | AudioLDM2 | 256 | `[128,256,384,640]` | GPT2 768 + T5 1024 streams | DDIM epsilon |
| `cvssp/audioldm2-large` | AudioLDM2 | 256 | `[128,256,384,640]` | GPT2 768 + T5 1024 plus extra self-attn slots | DDIM epsilon |
| `cvssp/audioldm2-music` | AudioLDM2 | 256 | `[128,256,384,640]` | GPT2 768 + T5 1024 streams | DDIM epsilon |
| `anhnct/audioldm2_gigaspeech` | AudioLDM2 TTS | 262 | `[128,256,384,640]` | GPT2 768 + VITS 192 streams | DDIM epsilon |

Shared spectrogram/audio dimensions:

| Field | AudioLDM v1 | AudioLDM2/MusicLDM | Source |
| --- | ---: | ---: | --- |
| latent channels | 8 | 8 | VAE/UNet configs |
| VAE input/output channels | 1 log-mel image channel | 1 | VAE config |
| VAE block channels | `[128,256,512]` | `[128,256,512]` | VAE config |
| VAE scale factor | 4 | 4 | pipeline from 3 VAE channel blocks |
| vocoder mel bins | 64 | 64 | vocoder config |
| vocoder sample rate | 16000 Hz | 16000 Hz | vocoder config |
| vocoder upsample rates | `[5,4,2,2,2]` | `[5,4,2,2,2]` | vocoder config |
| vocoder upsample factor | 320 samples/frame | 320 samples/frame | config product |
| default audio length | about 10.24 s for sample_size 128 | about 20.48 s for sample_size 256 | inferred from pipeline formula |

Text and language dimensions:

| Component | Key dimensions | Source |
| --- | --- | --- |
| AudioLDM `ClapTextModelWithProjection` | hidden 768, heads 12, layers 12, projection 512, max positions 514 | `text_encoder/config.json` |
| MusicLDM `ClapModel` | hidden 768, projection 512, text layers 16 in sampled config | `text_encoder/config.json` |
| AudioLDM2 `ClapModel` | projection 512, hidden 768 | `text_encoder/config.json` |
| AudioLDM2 `T5EncoderModel` | `d_model=1024`, layers 24, heads 16 | `text_encoder_2/config.json` |
| AudioLDM2 GPT2 | `n_embd=768`, layers 12, heads 12 | `language_model/config.json` |
| AudioLDM2 projection | CLAP 512 and T5 1024 to language model 768 | `projection_model/config.json` |
| TTS variant VITS | hidden 192, layers 6, heads 2 | `text_encoder_2/config.json` |

Scheduler:

| Field | Value |
| --- | --- |
| sampled default | `DDIMScheduler` |
| `num_train_timesteps` | 1000 |
| beta schedule | `scaled_linear`, `beta_start=0.0015`, `beta_end=0.0195` |
| prediction type | `epsilon` |
| `clip_sample` | `false` |
| `steps_offset` | `1` |
| source pipeline type | `KarrasDiffusionSchedulers` compatible surface |
| first Dinoml scheduler slice | DDIM epsilon, `eta=0`, `clip_sample=false`, `steps_offset=1` |

## 3a. Family variation traps

- These pipelines use 2D spectrogram latents, not rank-3 waveform latents. The
  UNet tensors are NCHW `[batch, 8, latent_time, 16]`, where the width is fixed
  by `vocoder.model_in_dim / vae_scale_factor = 64 / 4 = 16`.
- Audio length changes the latent height. Source pads height upward to a
  multiple of the VAE scale factor before denoising, then crops waveform output
  to the requested number of samples.
- AudioLDM v1 and MusicLDM do not use UNet cross-attention states in the
  pipeline call. They pass `encoder_hidden_states=None` and put CLAP text
  features in `class_labels`; their UNet config uses `class_embed_type =
  simple_projection` and `projection_class_embeddings_input_dim = 512`.
- AudioLDM2 uses a custom UNet with multiple attention modules per ResNet layer.
  `cross_attention_dim` is a per-block list such as `[None, 768, 1024]`: `None`
  means an extra self-attention layer, 768 attends GPT2 generated embeddings,
  and 1024 attends prompt embeddings from T5.
- `cvssp/audioldm2-large` adds a fourth `None` attention slot per cross-attn
  block. Do not assume exactly three attention modules per layer.
- TTS AudioLDM2 swaps T5 for VITS, adds learned positional embeddings in the
  projection model, requires `transcription`, and changes sample size to 262.
- VAE scaling factors differ by family: about `0.92279` for AudioLDM,
  `1.07532` for MusicLDM, and `0.41109` for AudioLDM2.
- SpeechT5HifiGan expects `[B, time, mel_bins]`, while VAE decode returns
  `[B, 1, time, mel_bins]`; the pipeline squeezes channel dimension before
  vocoder.
- VAE slicing exists through inherited helpers. Keep first slice no slicing.
- The current source marks AudioLDM and MusicLDM as deprecated with last
  supported Diffusers version `0.33.1`; source edits should treat the local
  deprecated files as authoritative for pipeline behavior.

## 4. Runtime tensor contract

Pipeline inputs:

- `prompt: str | list[str]`, optional negative prompt, `num_waveforms_per_prompt`.
- AudioLDM2 optionally accepts `transcription` for VITS/TTS, `max_new_tokens`,
  precomputed `prompt_embeds`, `generated_prompt_embeds`, and masks.
- `audio_length_in_s` controls spectrogram height. If absent, source computes
  `unet.sample_size * vae_scale_factor * 320 / 16000`.

Pre-denoising tensors:

- AudioLDM/MusicLDM CLAP prompt features: `[B, 512]`, L2-normalized in
  AudioLDM v1 and returned from `get_text_features` in MusicLDM. CFG batches
  concatenate negative and positive features.
- AudioLDM2 prompt embeddings after second encoder: T5 path `[B, seq, 1024]`;
  CLAP path `[B, 1, 512]`; projection output `[B, projected_seq, 768]`;
  generated GPT2 hidden states `[B, max_new_tokens, 768]`.
- AudioLDM2 default `generate_language_model` uses `max_new_tokens=8` unless
  caller or GPT2 config overrides it. TTS examples require 512 generated tokens.

Latents:

- Source latent layout is NCHW. Shape is
  `[B_eff, 8, ceil(height/4), 16]`.
- `height = int(audio_length_in_s / 0.02)` because the vocoder upsample factor
  is 320 samples at 16 kHz.
- For 10.24 s, height is 512 and latent height is 128. For 20.48 s, height is
  1024 and latent height is 256.
- Initial latents are Gaussian noise multiplied by `scheduler.init_noise_sigma`.

Denoiser step:

- AudioLDM/MusicLDM UNet call:
  `sample [B_eff,8,H,16]`, scalar timestep, `encoder_hidden_states=None`,
  `class_labels [B_eff,512]`.
- AudioLDM2 UNet call:
  `sample [B_eff,8,H,16]`, scalar timestep,
  `encoder_hidden_states=generated_prompt_embeds [B_eff,Lg,768]`,
  `encoder_hidden_states_1=prompt_embeds [B_eff,Lt,1024 or 192]`,
  `encoder_attention_mask_1 [B_eff,Lt]`.
- CFG doubles batch for one UNet call, chunks outputs, then computes
  `uncond + guidance_scale * (text - uncond)`.

Decode/postprocess:

- VAE decode consumes scaled latents `latents / vae.config.scaling_factor`.
- VAE output mel spectrogram is `[B, 1, time, 64]`.
- Pipeline squeezes to `[B, time, 64]` and calls `SpeechT5HifiGan`.
- Vocoder transposes to `[B,64,time]`, runs Conv1d/ConvTranspose1d residual
  stack, returns waveform `[B, time * 320]`.
- Output is cropped to `int(audio_length_in_s * 16000)` and optionally
  converted to NumPy.

CPU/data-pipeline work includes tokenization, librosa resampling for scoring,
CLAP feature extraction for scoring, output crop, and NumPy conversion. GPU
runtime work includes encoders if admitted, projection/GPT2 generation, UNet,
scheduler arithmetic, VAE decode, and vocoder.

## 5. Operator coverage checklist

Tensor/layout ops:

- NCHW spectrogram latent tensors, NLC text tokens, NCT vocoder tensors.
- `cat`, `chunk`, `repeat`, `view`, `unsqueeze`, `squeeze`, `transpose`,
  `permute`, crop/slice, ceil-to-multiple shape arithmetic, mask-to-bias.

Convolution/downsample/upsample ops:

- UNet/VAE `Conv2d`, `GroupNorm`, SiLU, ResNet residuals, downsample and
  upsample blocks on source NCHW.
- Representative first convs:
  `Conv2d(8 -> 128, 3x3, padding=1)` for small AudioLDM/MusicLDM/AudioLDM2;
  `Conv2d(8 -> 192/256, 3x3)` for medium/large AudioLDM.
- VAE `Conv2d(1 -> 128, 3x3)` encoder/decode family, latent channels 8, output
  `Conv2d(128 -> 1, 3x3)`.
- SpeechT5HifiGan `Conv1d(64 -> 1024, kernel=7)`,
  transposed Conv1d stages with channels `1024->512->256->128->64->32`,
  kernels `[16,16,8,4,4]`, strides `[5,4,2,2,2]`, and residual Conv1d
  kernels `[3,7,11]` with dilations `[1,3,5]`.

GEMM/linear ops:

- CLAP, T5, GPT2, and VITS encoder stacks if included.
- AudioLDM/MusicLDM simple class projection: `Linear(512 -> time_embed_dim)`.
- AudioLDM2 projection: `Linear(512 -> 768)` and `Linear(1024 -> 768)` or
  `Linear(192 -> 768)` for VITS.
- GPT2 autoregressive hidden-state generation over input embeddings.

Attention primitives:

- Shared `BasicTransformerBlock` self-attention over flattened spectrogram
  tokens inside `Transformer2DModel`.
- Cross-attention to CLAP/T5/GPT2 condition streams depending on pipeline.
- AudioLDM2 extra self-attention slots where `cross_attention_dim=None`.
- Attention masks are additive large-negative biases after conversion from
  `[B, key_tokens]`.

Normalization and adaptive conditioning:

- GroupNorm in UNet/VAE/resnet blocks; LayerNorm in transformer attention
  blocks; timestep sinusoidal embeddings; class embedding addition for v1 and
  MusicLDM.

Scheduler and guidance arithmetic:

- DDIM set_timesteps, `scale_model_input`, epsilon prediction step, optional
  `eta` path, CFG pointwise arithmetic.

Audio-specific ops:

- HiFi-GAN ConvTranspose1d, Conv1d dilations, LeakyReLU, residual averaging,
  final `tanh`.
- Optional CLAP scoring path: librosa resample, CLAP audio feature extraction,
  logits sort, `index_select`.

## 6. Denoiser/model breakdown

AudioLDM/MusicLDM use shared `UNet2DConditionModel`:

```text
time embedding: Timesteps -> TimestepEmbedding
class embedding: simple Linear(512 -> time_embed_dim), add to timestep embedding
conv_in
down blocks:
  DownBlock2D or CrossAttnDownBlock2D
  ResnetBlock2D: GroupNorm -> SiLU -> Conv2d -> time add -> GroupNorm -> SiLU -> Conv2d -> residual
  Transformer2DModel in cross-attn blocks, but pipeline passes no encoder states
mid block with attention
up blocks with skip concat, ResNet, attention, upsample
GroupNorm -> SiLU -> conv_out
```

For the selected configs the v1/Music first slice should preserve the
class-label conditioning path and can avoid real cross-attention conditioning
unless source attention blocks self-attend when `encoder_hidden_states=None`.

AudioLDM2 `AudioLDM2UNet2DConditionModel` follows the same UNet skeleton but
replaces cross-attention blocks with custom local classes that instantiate one
`Transformer2DModel` per configured cross-attention dimension. Within each
ResNet layer:

```text
ResnetBlock2D
for each configured cross_attention_dim:
  None -> Transformer2DModel double-self-attention / self-attention slot
  dim 768 -> attention to GPT2 generated prompt embeddings
  dim 1024 or 192 -> attention to T5/VITS prompt embeddings with mask
optional downsample or upsample
```

`AudioLDM2ProjectionModel`:

```text
CLAP features [B,1,512] -> Linear(512,768) -> add SOS/EOS
T5/VITS states [B,L,1024 or 192] -> optional learned position for VITS -> Linear(...,768) -> add SOS/EOS
concat projected token streams and masks
GPT2 consumes inputs_embeds and autoregressively appends hidden states
```

## 7. Attention requirements

The relevant implementation path is `Attention` plus `AttnProcessor2_0` in
`attention_processor.py` when PyTorch SDPA is available and `scale_qk` is true;
fallback is eager `AttnProcessor`. `attention_dispatch.py` is not the primary
path for this target.

Attention types:

- Spectrogram self-attention over flattened NCHW feature maps in
  `Transformer2DModel`.
- Cross-attention to generated GPT2 embeddings in AudioLDM2, feature dim 768.
- Cross-attention to T5 embeddings in AudioLDM2, feature dim 1024, with
  additive attention mask.
- Cross-attention to VITS embeddings in the TTS variant, feature dim 192.
- Extra self-attention slots in AudioLDM2 where `cross_attention_dim=None`.

Head counts follow the Diffusers compatibility rule: sampled configs set
`attention_head_dim=8` and `num_attention_heads=null`, and source treats the
effective number of heads as 8. Per block hidden widths are `[128,256,384,640]`,
so head dims are `[16,32,48,80]` in the small configs.

Flash-style constraints:

- First parity is dense, non-causal SDPA/eager attention.
- A Dinoml flash-style provider is valid for no-dropout inference with additive
  masks broadcastable to `[B, heads, query, key]`, but it must support different
  query/key lengths for AudioLDM2 cross-attention.
- No RoPE, QK norm, varlen packing, or GQA is active in these UNets.
- AudioLDM2-large can have more attention calls per block than base; kernels
  cannot assume one self-attn plus one cross-attn only.
- Fused projections via `FusedAttnProcessor2_0` are source-supported generally,
  but not required by sampled configs for first parity.

## 8. Scheduler and denoising-loop contract

All sampled official configs use DDIM epsilon prediction:

```text
scheduler.set_timesteps(num_inference_steps, device)
latents = randn(shape) * scheduler.init_noise_sigma
for t in timesteps:
  model_input = cat([latents, latents]) if guidance_scale > 1 else latents
  model_input = scheduler.scale_model_input(model_input, t)
  noise_pred = unet(model_input, t, conditioning)
  if CFG: noise_pred = uncond + guidance_scale * (text - uncond)
  latents = scheduler.step(noise_pred, t, latents, eta=eta, generator=generator).prev_sample
```

DDIM fields required for first parity are `num_train_timesteps=1000`,
`scaled_linear` beta table from 0.0015 to 0.0195, `prediction_type=epsilon`,
`steps_offset=1`, and `clip_sample=false`. The source pipeline accepts
`KarrasDiffusionSchedulers`, so LMS/PNDM-style swaps are compatible separate
candidates, but first Dinoml parity should use the checkpoint default DDIM
epsilon path.

Keep `set_timesteps`, schedule validation, and optional eta/generator handling
as host-visible state initially. CFG and DDIM pointwise maps are good compiled
kernel candidates once the scheduler artifact names DDIM epsilon explicitly.

## 9. Position, timestep, and custom math

- UNet timestep conditioning uses standard Diffusers sinusoidal `Timesteps`
  followed by `TimestepEmbedding`; source returns f32 time features and casts to
  sample dtype before the MLP.
- AudioLDM/MusicLDM class conditioning is a learned projection of CLAP text
  features into the UNet time embedding and added to timestep embedding.
- AudioLDM2 projection adds learned SOS/EOS token parameters around each text
  stream. The VITS variant optionally adds a learned positional embedding with
  shape `[1, text_encoder_1_dim, max_seq_length]` before projection.
- GPT2 hidden-state generation uses autoregressive cache state and concatenates
  the last hidden state token for `max_new_tokens` iterations.
- HiFi-GAN custom math is simple but audio-specific: LeakyReLU, average of
  multiple residual blocks per upsample stage, and final `tanh`.

Precomputable per request: CLAP/T5/VITS embeddings, AudioLDM2 projected prompt
embeddings, generated GPT2 hidden states, masks, scheduler tables, and fixed
VAE/vocoder shape buckets. Dynamic: audio length, batch/waveform count,
guidance scale, timestep, and TTS transcription length.

## 10. Preprocessing and input packing

AudioLDM v1:

- Roberta tokenizer pads/truncates to `tokenizer.model_max_length`.
- CLAP text encoder returns `text_embeds [B,512]`; pipeline L2 normalizes over
  feature dim.
- CFG negative prompt is encoded separately, duplicated for waveform count, and
  concatenated along batch.

MusicLDM:

- Roberta tokenizer and CLAP `get_text_features` produce `[B,512]`.
- If multiple waveforms per prompt are requested, generated waveforms may be
  scored by CLAP audio/text similarity. This is postprocess ranking, not part
  of denoising.

AudioLDM2:

- CLAP prompt path is compressed to a single `[B,1,512]` token.
- T5 path emits `[B,L,1024]`; VITS path emits `[B,L,192]` from transcription
  and patches an end token into phoneme ids.
- Projection inserts SOS/EOS around both streams and concatenates masks.
- GPT2 generates `[B,max_new_tokens,768]` hidden states from projected inputs.
- Denoiser receives GPT2 generated states as `encoder_hidden_states` and T5/VITS
  prompt states as `encoder_hidden_states_1`.

Spectrogram packing:

- Pipeline computes log-mel image height from audio length and vocoder frame
  duration. Width is fixed to 64 mel bins before VAE, 16 in latent space.
- VAE decode returns mel spectrogram in image layout `[B,1,T,64]`.
- Vocoder consumes `[B,T,64]` and internally transposes to `[B,64,T]`.

## 11. Graph rewrite / lowering opportunities

1. Spectrogram NHWC local layout region
   - Source pattern: UNet and VAE use NCHW Conv2d/GroupNorm/attention maps.
   - Replacement: guarded NHWC/channel-last provider region for Conv2d-heavy
     blocks.
   - Preconditions: region includes all axis-sensitive ops; GroupNorm channel
     axis rewritten from `dim=1` semantics; attention flatten order preserves
     source `height,width` token order; skip concatenation channel axis
     rewritten.
   - Failure cases: crossing VAE/vocoder boundary or public latent ABI without
     explicit layout conversion.
   - Test: one down block and one up block parity for latent shape
     `[1,8,128,16]` and `[1,8,256,16]`.

2. VAE decode-only island
   - Source pattern: `latents / scaling_factor -> AutoencoderKL.decode`.
   - Replacement: dedicated VAE decode graph with constant scaling folded into
     input or first conv when legal.
   - Preconditions: decode-only, no VAE slicing, fixed scaling factor per
     checkpoint, no latent observer before decode.
   - Failure cases: encode path admission or mixed checkpoints in one artifact.
   - Test: decode parity for each family scaling factor.

3. HiFi-GAN NTC/NCT transpose sink
   - Source pattern: `[B,T,64] -> transpose -> Conv1d stack -> squeeze`.
   - Replacement: accept NTC spectrogram at boundary and run an internal NCT or
     channel-last Conv1d provider with explicit transpose elimination.
   - Preconditions: vocoder owns the whole Conv1d stack; ConvTranspose1d output
     length formula matches PyTorch; weight norm materialized if present.
   - Failure cases: external code observing intermediate NCT tensors.
   - Test: vocoder parity for short and long mel lengths.

4. CFG plus DDIM step fusion
   - Source pattern: chunk + linear interpolation + scheduler step.
   - Replacement: fused pointwise kernel over `[B,8,H,16]`.
   - Preconditions: DDIM epsilon, `clip_sample=false`, known eta branch, same
     dtype and shape for unconditional/conditional outputs.
   - Failure cases: scheduler swaps, guidance rescale not present here but
     possible in other families.
   - Test: one-step DDIM parity with fixed model outputs.

5. AudioLDM2 projection packing
   - Source pattern: two Linear projections + SOS/EOS concat + mask concat.
   - Replacement: static request-side packing kernel or host graph.
   - Preconditions: fixed hidden dims and sequence lengths, no VITS learned
     positional variant unless separately admitted.
   - Failure cases: TTS VITS max sequence or custom generated embeddings.
   - Test: projection output and mask parity for CLAP+T5 prompts.

## 12. Kernel fusion candidates

Highest priority:

- Conv2d + GroupNorm + SiLU in UNet/VAE ResNet blocks. This is the main
  spectrogram denoiser and decoder workload.
- Dense SDPA/eager attention over flattened spectrogram feature maps, including
  AudioLDM2 cross-attention to 768 and 1024 dim streams.
- SpeechT5HifiGan ConvTranspose1d + residual Conv1d + LeakyReLU stages. This
  is the audio-specific post-VAE cost shared by all candidates.
- CFG and DDIM epsilon scheduler pointwise fusion over latent maps.

Medium priority:

- AudioLDM/MusicLDM class embedding projection and timestep embedding add.
- AudioLDM2 projection model plus GPT2 prompt generation boundary.
- VAE decode-only graph with scaling-factor fold.
- Attention mask conversion and bias broadcast for AudioLDM2 T5/VITS masks.

Lower priority:

- CLAP/T5/VITS text encoder compilation; accept cached embeddings first.
- Automatic CLAP waveform scoring and sorting.
- VAE slicing and attention slicing.
- Scheduler swaps beyond DDIM.

## 13. Runtime staging plan

Stage 1: parse component configs for `ucsd-reach/musicldm` or
`cvssp/audioldm-s-full-v2`, including UNet, VAE, DDIM, CLAP projection, and
vocoder metadata.

Stage 2: denoiser-only parity with externally supplied CLAP `[B,512]`
class-label embeddings and latents `[B,8,128 or 256,16]`. Keep scheduler and
text encoder in Python.

Stage 3: DDIM epsilon scheduler parity and short denoising loop with PyTorch
VAE/vocoder still external.

Stage 4: VAE decode-only island from latents to `[B,1,T,64]` mel spectrogram.

Stage 5: SpeechT5HifiGan vocoder island from `[B,T,64]` to waveform.

Stage 6: integrate CLAP prompt embedding cache ABI; compile CLAP later only if
needed.

Stage 7: AudioLDM2 base path with external `prompt_embeds`,
`generated_prompt_embeds`, and masks, then add projection/GPT2 generation.

Stage 8: separate TTS/VITS AudioLDM2 variant.

First Dinoml admission recommendation: start with MusicLDM or AudioLDM small
denoiser plus DDIM and decode boundaries, not AudioLDM2 full prompt generation.
It exercises the same spectrogram latent/VAE/vocoder contract with fewer
conditioning surfaces.

## 14. Parity and validation plan

- Config parsing tests for `cvssp/audioldm-s-full-v2`,
  `ucsd-reach/musicldm`, `cvssp/audioldm2`, and `cvssp/audioldm2-large`.
- Random tensor tests for timestep embeddings, class-label projection,
  mask-to-bias conversion, and AudioLDM2 projection SOS/EOS packing.
- Single `ResnetBlock2D` parity and one `Transformer2DModel` parity at channel
  widths 128, 256, 384, and 640.
- UNet one-step parity at `[1,8,128,16]` and `[1,8,256,16]`, fixed timestep,
  fixed embeddings, fp32 first.
- AudioLDM2 custom block parity with generated states `[1,8,768]` and prompt
  states `[1,L,1024]`, plus mask.
- DDIM scheduler table and one-step parity with `steps_offset=1`,
  `clip_sample=false`, `epsilon`.
- VAE decode parity for all three scaling factors.
- SpeechT5HifiGan vocoder parity for mel lengths 128, 512, and 1024.
- End-to-end smoke with small step count and `output_type="latent"`, then
  decode/vocoder smoke.

Suggested tolerances: fp32 custom and scheduler math `rtol=1e-5, atol=1e-6`;
fp16/bf16 UNet/VAE/vocoder initially `rtol=2e-2, atol=2e-2`, tightened per
kernel.

## 15. Performance probes

- One UNet denoiser step by latent height 128, 256, and 262, batch 1 and CFG
  batch 2.
- UNet Conv/ResNet time versus transformer attention time.
- VAE decode throughput for 5 s, 10.24 s, and 20.48 s requests.
- SpeechT5HifiGan vocoder throughput by mel length and ConvTranspose1d stage.
- DDIM/CFG overhead by step count 10 and 200.
- AudioLDM2 prompt path: CLAP + T5 encoding, projection, GPT2 generation for
  `max_new_tokens=8` and 512.
- Memory probes for CFG batch doubling and long AudioLDM2 generated embeddings.
- NCHW versus guarded NHWC Conv2d provider comparison.
- NTC/NCT vocoder layout comparison.

No benchmark measurements were run in this audit.

## 16. Scope boundary and separate candidates

Separate candidate reports or work items:

- `speecht5_hifigan_vocoder`: Conv1d/ConvTranspose1d/LeakyReLU audio synthesis
  island used by AudioLDM, MusicLDM, and AudioLDM2.
- `audioldm2_prompt_generation`: CLAP+T5 projection and GPT2 hidden-state
  generation, including cache state.
- `audioldm2_tts_vits`: VITS/transcription variant with learned positional
  projection and longer generated prompt embeddings.
- `scheduler_karras_audio_swaps`: LMS/PNDM/DDIM-compatible swaps beyond the
  sampled DDIM default.
- `generic_unet_lora_audio`: runtime attention/UNet adapter mutation if audio
  LoRA checkpoints become a target.
- `autoencoderkl_spectrogram_audio`: VAE decode/encode specialization for
  1-channel mel spectrogram images.

Ignored/out of scope for this audit:

- XLA/NPU/MPS/Flax/ONNX branches.
- Training, losses, dropout behavior, and gradient checkpointing.
- Multi-GPU/context parallel.
- Callback mutation and interactive interrupt behavior.
- Safety checker/NSFW filtering; not present.
- Automatic scoring as a first denoising/runtime requirement.

## 17. Final implementation checklist

- [ ] Parse AudioLDM/MusicLDM/AudioLDM2 model_index and component configs.
- [ ] Admit spectrogram latents as NCHW `[B,8,H,16]` with dynamic/bucketed H.
- [ ] Implement DDIM epsilon scheduler parity for scaled-linear beta configs.
- [ ] Implement UNet class-label conditioning for AudioLDM/MusicLDM.
- [ ] Add UNet block parity tests for `[128,256,384,640]` widths.
- [ ] Add AudioLDM2 multi-condition attention block tests.
- [ ] Implement VAE decode-only path for 1-channel mel spectrograms.
- [ ] Implement SpeechT5HifiGan vocoder operator island.
- [ ] Add CFG + DDIM fused pointwise candidate with guarded scheduler metadata.
- [ ] Add CLAP prompt-embedding cache ABI before compiling text encoders.
- [ ] Add AudioLDM2 projection/GPT2 prompt generation as a separate milestone.
- [ ] Benchmark UNet, VAE decode, vocoder, scheduler/CFG, and prompt-generation stages.
