# Diffusers Video Autoencoder Codec Audit

## 1. Source basis

```text
Diffusers commit/version:
  Local checkout diffusers at b3a515080752a3ba7ca92161e25530c7f280f629.

Candidate:
  video_autoencoders: video VAE/autoencoder codecs only, not full video
  pipelines or denoisers.

Source files inspected:
  diffusers/src/diffusers/models/autoencoders/autoencoder_kl_cogvideox.py
  diffusers/src/diffusers/models/autoencoders/autoencoder_kl_wan.py
  diffusers/src/diffusers/models/autoencoders/autoencoder_kl_ltx.py
  diffusers/src/diffusers/models/autoencoders/autoencoder_kl_ltx2.py
  diffusers/src/diffusers/models/autoencoders/autoencoder_kl_ltx2_audio.py
    (listed only to separate audio scope)
  diffusers/src/diffusers/models/autoencoders/autoencoder_kl_mochi.py
  diffusers/src/diffusers/video_processor.py
  diffusers/src/diffusers/image_processor.py

Pipeline files lightly inspected for VAE boundary formulas:
  diffusers/src/diffusers/pipelines/cogvideo/pipeline_cogvideox*.py
  diffusers/src/diffusers/pipelines/wan/pipeline_wan*.py
  diffusers/src/diffusers/pipelines/ltx/pipeline_ltx*.py
  diffusers/src/diffusers/pipelines/mochi/pipeline_mochi.py

Local config cache checked first:
  H:/configs/zai-org/CogVideoX-2b/model_index.json
  H:/configs/zai-org/CogVideoX-5b/model_index.json
  H:/configs/zai-org/CogVideoX1.5-5B/model_index.json
  H:/configs/genmo/mochi-1-preview/model_index.json
  H:/configs/Lightricks/LTX-Video/model_index.json
  H:/configs/Lightricks/LTX-Video-0.9.7-dev/model_index.json
  H:/configs/Lightricks/LTX-Video-0.9.8-13B-distilled/model_index.json
  H:/configs/Wan-AI/Wan2.1-T2V-1.3B-Diffusers/model_index.json
  H:/configs/Wan-AI/Wan2.2-TI2V-5B-Diffusers/model_index.json

Official component configs fetched in-memory from Hugging Face raw URLs:
  zai-org/CogVideoX-2b/vae/config.json
  zai-org/CogVideoX-5b/vae/config.json
  zai-org/CogVideoX1.5-5B/vae/config.json
  genmo/mochi-1-preview/vae/config.json
  Lightricks/LTX-Video/vae/config.json
  Lightricks/LTX-Video-0.9.7-dev/vae/config.json
  Lightricks/LTX-Video-0.9.8-13B-distilled/vae/config.json
  Wan-AI/Wan2.1-T2V-1.3B-Diffusers/vae/config.json
  Wan-AI/Wan2.2-TI2V-5B-Diffusers/vae/config.json

Missing files or assumptions:
  The local cache held model_index.json files for the sampled official repos,
  but not local component config.json files. Official raw component configs were
  accessible without a gated/authenticated retry and were not saved because this
  task's owned write path is this report only. XLA/NPU/MPS/Flax/ONNX,
  safety/NSFW, training/loss/dropout/gradient checkpointing,
  multi-GPU/context parallel, callbacks, and interrupt paths were ignored.
```

## 2. Candidate map

All inspected video codecs use source tensors shaped `[B,C,T,H,W]`
(`NCDHW`/`NCTHW`). Treat `NDHWC` as an optimization only after faithful source
layout parity.

| Family | Class | File | Primary role | First-slice risk |
| --- | --- | --- | --- | --- |
| Wan | `AutoencoderKLWan` | `autoencoder_kl_wan.py` | Wan 2.1/2.2 video VAE, encode for I2V/V2V and decode for T2V | Best first staging target: explicit temporal cache loop, 16-channel common config, no KL quant conv optionality. |
| CogVideoX | `AutoencoderKLCogVideoX` | `autoencoder_kl_cogvideox.py` | CogVideoX video VAE | Good second target; chunked temporal batching and `CogVideoXSafeConv3d` need separate care. |
| LTX 0.9.x | `AutoencoderKLLTXVideo` | `autoencoder_kl_ltx.py` | LTX video VAE with latent patchification and optional timestep-conditioned decoder | Needs deeper individual report before admission. |
| LTX 2.x | `AutoencoderKLLTX2Video` | `autoencoder_kl_ltx2.py` | LTX-2 video VAE variant with runtime causal flag and changed blocks | Separate from LTX 0.9.x; not just a config variant. |
| Mochi | `AutoencoderKLMochi` | `autoencoder_kl_mochi.py` | Mochi 1 preview VAE | Deeper report required; attention-bearing encoder and 15-channel input make it less bounded. |
| LTX2 audio | `AutoencoderKLLTX2Audio` | `autoencoder_kl_ltx2_audio.py` | Audio codec | Out of this video report; should be `audio_autoencoders`. |

## 3. Representative config dimensions

| Repo | VAE class | In/out | Latent | Spatial compression | Temporal compression | Scale / stats |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| `zai-org/CogVideoX-2b` | `AutoencoderKLCogVideoX` | 3/3 | 16 | 8 inferred from 4 blocks | 4 | `scaling_factor=1.15258426`, no mean/std. |
| `zai-org/CogVideoX-5b` | `AutoencoderKLCogVideoX` | 3/3 | 16 | 8 | 4 | `scaling_factor=0.7`, no mean/std. |
| `zai-org/CogVideoX1.5-5B` | `AutoencoderKLCogVideoX` | 3/3 | 16 | 8 | 4 | `scaling_factor=0.7`, no mean/std. |
| `Wan-AI/Wan2.1-T2V-1.3B-Diffusers` | `AutoencoderKLWan` | default 3/3 | `z_dim=16` | 8 | 4 | 16-element `latents_mean/std`; pipelines normalize with reciprocal std. |
| `Wan-AI/Wan2.2-TI2V-5B-Diffusers` | `AutoencoderKLWan` | 12/12 | `z_dim=48` | 16 | 4 | 48-element `latents_mean/std`, `patch_size=2`. |
| `genmo/mochi-1-preview` | `AutoencoderKLMochi` | 15/3 | 12 | `2*2*2=8` | `1*2*3=6` | 12-element `latents_mean/std`, `scaling_factor=1.0`. |
| `Lightricks/LTX-Video` | `AutoencoderKLLTXVideo` | 3/3 | 128 | default `4*2^3=32` | default `1*2^3=8` | zero mean/one std buffers, `scaling_factor=1.0`. |
| `Lightricks/LTX-Video-0.9.7-dev` | `AutoencoderKLLTXVideo` | 3/3 | 128 | config `32` | config `8` | wider encoder, same latent count. |
| `Lightricks/LTX-Video-0.9.8-13B-distilled` | `AutoencoderKLLTXVideo` | 3/3 | 128 | config `32` | config `8` | same codec shape as 0.9.7-dev. |

Config-derived fields above come from official component configs fetched
in-memory, except source defaults explicitly called out as inferred.

## 4. Runtime tensor contracts

Common source boundary:

```text
video sample: [B, C_in, T, H, W]  NCTHW/NCDHW
posterior moments: [B, 2*C_latent, T_lat, H_lat, W_lat]
posterior mode/sample: [B, C_latent, T_lat, H_lat, W_lat]
decoded sample: [B, C_out, T_out, H_out, W_out]
```

Temporal shape formulas:

| Family | Encode frames | Decode frames | Notes |
| --- | --- | --- | --- |
| Wan | `T_lat = (T - 1) // 4 + 1` in pipelines | `T = (T_lat - 1) * 4 + 1` pipeline contract | Source `_decode` decodes one latent frame at a time through causal feature caches. |
| CogVideoX | `T_lat = (T - 1) // 4 + 1` in pipelines | `T = (T_lat - 1) * 4 + 1` pipeline contract | Source chunks sample frames by 8 and latent frames by 2, carrying causal conv cache. |
| LTX | `T_lat = (T - 1) // temporal_compression + 1` in temporal tiling | `T = (T_lat - 1) * temporal_compression + 1` in temporal tiling | Default/config sampled compression is 8; source supports temporal framewise modes. |
| LTX2 | Same formula family as LTX | Same formula family as LTX | Adds runtime `causal` parameter on encode/decode. |
| Mochi | compression product is 6 | default decode drops first `compression-1` upscaled frames | `drop_last_temporal_frames=True` yields `(T_lat - 1) * 6 + 1`. |

Latent scaling/stat formulas at pipeline boundaries:

- CogVideoX: encode multiplies latents by `vae.config.scaling_factor`; decode
  divides by that scalar before `vae.decode`.
- Wan: encode normalizes `(latent - mean) * (1 / std)`; decode restores
  `latent / (1 / std) + mean`, equivalent to `latent * std + mean`.
- LTX: helper normalizes `(latent - mean) * scaling_factor / std`; decode
  restores `latent * std / scaling_factor + mean`.
- Mochi: if mean/std exist, decode does `latents * latents_std /
  scaling_factor + latents_mean`, else scalar unscale.

Posterior contract is the shared Diffusers KL distribution convention: VAE
encode returns moments interpreted by `DiagonalGaussianDistribution`; first
Dinoml parity should use `mode()` unless an image/video conditioning path
requires sampling with explicit RNG.

## 5. Architecture and operator notes by family

### Wan

`AutoencoderKLWan` builds:

```text
Encoder: WanCausalConv3d -> residual/down blocks -> optional WanAttentionBlock
quant_conv: WanCausalConv3d(z_dim*2 -> z_dim*2, 1)
Decoder: post_quant_conv -> residual/up blocks -> WanCausalConv3d out
```

Important details:

- `WanCausalConv3d` subclasses `nn.Conv3d` and rewrites padding to causal time
  padding before the convolution. Feature caches are lists mutated across
  per-frame encode/decode chunks.
- Temporal down/up helpers include `AvgDown3D` and `DupUp3D`, which use
  reshape/permute/mean/repeat patterns rather than only Conv3d stride.
- Common Wan 2.1 VAE has `z_dim=16`, `base_dim=96`, spatial scale 8, temporal
  scale 4. TI2V 5B changes to `z_dim=48`, `in_channels=out_channels=12`,
  `base_dim=160`, `decoder_base_dim=256`, `patch_size=2`, spatial scale 16.
- Source `_decode` applies `post_quant_conv`, then decodes one latent frame at a
  time with `first_chunk=True` for frame zero and shared feature cache after.
- Source `_encode` optionally `patchify` when `patch_size` is set, then encodes
  first frame and subsequent four-frame chunks through the same cache.

First Dinoml Wan slice should admit only the common 16-channel, 3-channel
Wan2.1/2.2 VAE with tiling/slicing disabled, `mode()` posterior, and host-owned
feature-cache state made artifact-visible.

### CogVideoX

`AutoencoderKLCogVideoX` builds causal 3D encoder/decoder blocks over
`CogVideoXCausalConv3d`, with optional `quant_conv`/`post_quant_conv` disabled
in sampled official configs.

Important details:

- `CogVideoXCausalConv3d` uses explicit temporal-left padding and returns
  `(output, conv_cache)`; `pad_mode="replicate"` uses PyTorch padding while
  constant mode concatenates cached previous frames.
- Encoder chunks sample frames with `num_sample_frames_batch_size=8`; decoder
  chunks latent frames with `num_latent_frames_batch_size=2`. These chunk sizes
  are not arbitrary performance knobs; source comments warn the temporal VAE is
  trained around this behavior.
- Spatial tiling defaults are derived from sample config: min tile height/width
  are half of `sample_height/sample_width`; overlap factors are 1/6 height and
  1/5 width.
- `CogVideoXSafeConv3d` is used in parts of the model to avoid OOM, so a
  Dinoml lowering should be ready to see Conv3d operations that may be
  internally sliced/chunked in PyTorch but semantically remain Conv3d.

CogVideoX is a good second codec after Wan because the channel contract is
stable and scalar scaling is simple, but the temporal chunk/cache behavior
needs direct parity tests.

### LTX 0.9.x

`AutoencoderKLLTXVideo` differs from Wan/CogVideoX in two ways that affect
first admission:

- It patchifies at the codec boundary: `patch_size=4`, `patch_size_t=1` in
  sampled configs. Encoder input channels become `in_channels * patch_size**2`.
  Decode performs the inverse unpatchify.
- Spatial and temporal compression are computed as
  `patch_size * 2**sum(spatio_temporal_scaling)` and
  `patch_size_t * 2**sum(spatio_temporal_scaling)`, or config overrides.

Operators include causal `LTXVideoCausalConv3d`, ResNet 3D blocks,
downsampler/upsampler 3D modules, optional decoder timestep conditioning
`temb`, spatial tiling, temporal tiling, and batch slicing. Sampled official
configs use `latent_channels=128`, spatial compression 32, temporal compression
8. That wide latent surface is large enough to deserve its own report before
Dinoml admission.

### LTX2 video

`AutoencoderKLLTX2Video` is not just a renamed LTX 0.9.x class:

- It uses `LTX2VideoCausalConv3d`, where causality is passed at runtime to
  `forward(..., causal=True/False)`.
- Defaults are much wider: encoder block channels `(256,512,1024,2048)`,
  decoder block channels `(256,512,1024)`.
- Down/upsample type tuples distinguish spatial, temporal, and spatiotemporal
  factors.
- Decoder default is causal and uses reflect spatial padding by default.

Treat this as `ltx2_video_autoencoder`, a separate deeper report and admission
candidate from LTX 0.9.x.

### Mochi

`AutoencoderKLMochi` has the most distinctive first-slice traps:

- Official config uses `in_channels=15`, `out_channels=3`, `latent_channels=12`.
  The 15-channel input means encode is coupled to a packed/video condition
  contract rather than plain RGB-only video.
- Encoder and decoder use temporal expansions `(1,2,3)` and spatial expansions
  `(2,2,2)`, giving temporal compression 6 and spatial compression 8.
- `MochiChunkedGroupNorm3D` permutes `[B,C,T,H,W]` to per-frame chunks,
  applies GroupNorm over `B*T` slices, then restores layout.
- Encoder blocks can include attention; source explicitly rejects framewise
  encoding because attention makes intermediate frames dependent.
- Decode can framewise chunk with causal conv cache and then drop the first
  `temporal_compression_ratio - 1` frames by default.

Mochi should get an individual report before admission; it is not the right
first video VAE slice.

## 6. Layout notes and no-layout-translation guards

Faithful source layout is `NCDHW`/`NCTHW` throughout all inspected codecs. A
future `NDHWC` island is plausible for Conv3d-heavy regions, but requires guards
and axis rewrites for:

- Conv3d weights: source `[O,I,kT,kH,kW]`; NDHWC kernels need explicit weight
  transform and stride/padding mapping.
- GroupNorm/RMSNorm/channel stats: channel axis is dim 1 in source and last dim
  in NDHWC.
- Posterior split: moments split/chunk along dim 1.
- Latent mean/std/scaling: broadcast shape is `[1,C,1,1,1]`.
- Temporal caches: cached tensors concatenate/slice along dim 2.
- Tiling/blending: source height dim 3, width dim 4; temporal blending dim 2.
- Patchify/unpatchify in Wan TI2V and LTX: channel/spatial packing order must be
  rewritten, not guessed.
- Attention flattening in Mochi/Wan VAE attention blocks: preserve token order
  or wrap in a `no_layout_translation` region until tested.

Initial graph translation should preserve NCDHW and only introduce layout
translation inside fully controlled Conv3d islands with explicit boundary
conversions.

## 7. Operator coverage checklist

Required for the first Wan/CogVideoX-style slice:

- `Conv3d` with causal temporal padding and cached previous frames.
- `Conv3d` 1x1 quant/post-quant convs.
- 3D residual blocks: GroupNorm/RMSNorm-like channel norm, SiLU/swish, Conv3d,
  residual add, optional shortcut conv.
- Temporal downsample/upsample helpers: mean over reshaped factors,
  repeat/view/permute upsample, strided temporal Conv3d.
- Tensor ops: split/chunk dim 1, cat dim 2 and dim 1, view/reshape/permute,
  repeat/repeat_interleave, clamp for Wan decode output.
- KL posterior: mean/logvar split, clamp/exp/sample/mode from
  `DiagonalGaussianDistribution`.
- Per-channel latent normalization/denormalization over `[B,C,T,H,W]`.
- Spatial tiling and blending later: H/W crop, overlap blend, concat/crop.
- Temporal tiling/chunking later: T crop, conv-cache carry, temporal blend.

Additional operators needed before LTX/Mochi:

- Codec-level patchify/unpatchify for LTX and Wan TI2V.
- Decoder timestep embedding input for LTX decode paths that pass `temb`.
- Runtime causal flag dispatch for LTX2.
- Per-frame chunked GroupNorm and attention blocks for Mochi.
- Conv2d/spatial attention inside Wan VAE `WanAttentionBlock` if configs enable
  attention scales.

## 8. Fusion candidates

Highest priority:

- Causal Conv3d + norm + SiLU/swish + Conv3d residual blocks in Wan/CogVideoX.
- Per-channel latent mean/std/scale pointwise kernels at VAE boundary.
- Temporal cache-aware Conv3d chunk decode for Wan and CogVideoX.
- Conv3d downsample/upsample helpers with adjacent reshape/permute operations.

Medium priority:

- Spatial tiling blend kernels for H/W overlap.
- Temporal tiling blend kernels for LTX/Mochi long-video paths.
- Patchify/unpatchify plus first/last Conv3d in LTX and Wan TI2V.
- VAE attention block QKV/attention/output projection for Mochi and Wan configs
  with attention enabled.

Lower priority:

- Batch slicing as a runtime memory policy; it should remain a host/runtime
  staging decision rather than a new op.
- `CogVideoXSafeConv3d` internal safety chunking as an optimization, after plain
  Conv3d parity.
- LTX decoder noise/timestep conditioning and LTX2 padding-mode variants.

## 9. First Dinoml staging and admission recommendation

Recommended first admission: `wan_vae_decode_16ch_ncthw`.

Scope:

- `AutoencoderKLWan`, `z_dim=16`, `in_channels=3`, `out_channels=3`,
  `scale_factor_spatial=8`, `scale_factor_temporal=4`.
- Decode only first, with external normalized latents.
- Tiling, slicing, framewise policy toggles, TI2V `patch_size=2`, and
  48-channel VAE deferred.
- Preserve NCDHW layout.
- Make feature-cache state explicit in the runtime plan: per-session cache
  buffers for causal convs, reset at decode start, first-frame flag visible.
- Decode boundary includes Wan denormalization:
  `latents = latents * std + mean`.

Why Wan first:

- It is already important to the Wan pipeline target and has a representative
  prior report.
- The common config is smaller and cleaner than LTX 128-channel latents or
  Mochi attention-bearing encoder.
- Its per-frame decode loop is awkward but bounded and artifact-visible, which
  matches current Dinoml runtime priorities.

Second stage: `wan_vae_encode_16ch_ncthw` for I2V/video2video condition
latents, using posterior `mode()` first.

Third stage: `cogvideox_vae_decode_16ch_ncthw`, then encode. This validates a
second causal-conv cache implementation and scalar scaling.

Defer until individual reports:

- `ltx_video_autoencoder`
- `ltx2_video_autoencoder`
- `mochi_video_autoencoder`
- `wan_ti2v_48ch_vae`
- `video_vae_tiling_temporal_chunking_runtime`

## 10. Parity and validation plan

- Shape contract tests for each family:
  `[B,C,T,H,W] -> [B,C_lat,T_lat,H_lat,W_lat] -> [B,C,T_out,H,W]`.
- Wan decode block parity for one latent frame and two latent frames, proving
  cache reset and `first_chunk` behavior.
- Wan full decode parity for synthetic small shapes with `z_dim=16`; compare
  Diffusers `vae.decode` to Dinoml NCDHW.
- Wan denormalization parity with 16-element mean/std.
- Wan encode parity through posterior moments and `mode()`, first frame plus
  one four-frame chunk.
- CogVideoX decode parity with 1, 2, and odd latent-frame counts because source
  chunk batching special-cases remainders.
- CogVideoX scalar scaling parity for 2B and 5B configs.
- LTX patchify/unpatchify standalone parity before full LTX VAE.
- Mochi chunked GroupNorm and temporal-drop decode parity before full Mochi VAE.
- Tiling/chunking parity should compare against Diffusers tiled/chunked output,
  not non-tiled output.

Suggested tolerances: fp32 `rtol=1e-4, atol=1e-5`; fp16/bf16 start with
`rtol=2e-2, atol=2e-2` and tighten after Conv3d/norm providers are stable.

## 11. Performance probes

- Wan VAE decode throughput and memory by latent grid:
  `[B,16,21,60,104]`, `[B,16,21,90,160]`, and small synthetic grids.
- Conv3d/norm/residual block time split for Wan and CogVideoX.
- Per-frame cached decode versus larger temporal chunk decode where source
  supports chunking.
- NCDHW faithful path versus guarded NDHWC Conv3d island.
- Latent denormalization overhead versus fused pointwise kernel.
- Spatial tiled decode memory/latency, after non-tiled parity.
- LTX 128-channel latent decode cost and patchify/unpatchify overhead.
- Mochi attention-bearing VAE block cost and chunked GroupNorm overhead.

## 12. Scope boundary and deeper reports

Separate candidate reports that should happen next:

- `wan_vae_16ch`: focused decode/encode admission report for the common
  Wan2.1/2.2 VAE.
- `wan_ti2v_48ch_vae`: 48-channel, 12-channel sample, `patch_size=2`,
  spatial-scale-16 variant.
- `cogvideox_vae`: scalar-scaled 16-channel causal Conv3d codec with safe conv
  and temporal chunk cache.
- `ltx_video_autoencoder`: LTX 0.9.x patchified 128-channel codec.
- `ltx2_video_autoencoder`: LTX2 runtime-causal, wider codec.
- `mochi_video_autoencoder`: 15-channel input, attention-bearing encoder,
  chunked GroupNorm, temporal drop policy.
- `video_vae_tiling_temporal_chunking_runtime`: memory policy report covering
  spatial tiling, temporal tiling, batch slicing, and feature-cache residency
  across all video codecs.

Ignored/out of scope for this audit:

- Full transformer denoisers, text encoders, schedulers, CFG, and pipeline
  variants except for VAE boundary formulas.
- Multi-GPU/context parallel paths.
- Callback mutation and interactive interrupt.
- XLA, NPU, MPS, Flax, and ONNX variants.
- Safety checker and NSFW filtering.
- Training, losses, dropout behavior, and gradient checkpointing.

## 13. Final implementation checklist

- [ ] Parse `AutoencoderKLWan` common 16-channel configs and reconcile source defaults.
- [ ] Add explicit NCDHW video tensor shape contract for VAE encode/decode.
- [ ] Implement causal Conv3d with artifact-visible feature-cache state.
- [ ] Implement Wan residual/down/up helper ops used by decode.
- [ ] Implement Wan VAE decode with tiling/slicing disabled.
- [ ] Add Wan latent denormalization pointwise kernel.
- [ ] Validate Wan decode parity on small random tensors and representative latent grids.
- [ ] Add Wan encode through posterior `mode()` for I2V/video2video readiness.
- [ ] Add CogVideoX decode report and parity tests for temporal chunk/cache behavior.
- [ ] Add LTX/LTX2/Mochi individual reports before admitting their wider surfaces.
- [ ] Add guarded NDHWC Conv3d island only after faithful NCDHW parity.
