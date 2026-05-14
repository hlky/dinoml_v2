# Transformers Audit Tracker

Purpose: visible coordination state for the model-family report audit under
`agents/plans/transformers/`.

Scope note: the local Transformers checkout currently has hundreds of model
family directories. The audit should run in reviewed batches, prioritizing
families that add new operator, layout, cache, preprocessing, or generation
patterns before near-duplicates.

## Coordinator Rules

- Keep `PROMPT.md` as the single report contract.
- Use subagents for individual family reports with disjoint write paths.
- Review each batch before starting the next one.
- Update `report_review.md` only for prompt/process findings, not for every
  ordinary family detail.
- Do not commit audit docs unless the user explicitly asks.

## Completed Calibration Reports

- [x] `t5`
- [x] `llama`
- [x] `mistral`
- [x] `bert`
- [x] `vit`
- [x] `whisper`

## Batch 1 Completed

- [x] `qwen2` - decoder-only RoPE/GQA/long-context production family
- [x] `roberta` - BERT-like encoder with tokenizer/position differences
- [x] `deberta_v2` - encoder with disentangled relative attention
- [x] `swin` - shifted-window vision transformer with relative bias
- [x] `dinov2` - ViT-like feature encoder with DINOv2-specific variants

## Batch 2 Completed

- [x] `llava`
- [x] `qwen2_vl`
- [x] `gemma3`
- [x] `mixtral`
- [x] `wav2vec2`

## Batch 3 Completed

- [x] `bart`
- [x] `gpt2`
- [x] `bloom`
- [x] `clip`
- [x] `convnext`

## Batch 4 Completed

- [x] `detr`
- [x] `sam`
- [x] `beit`
- [x] `albert`
- [x] `electra`

## Batch 5 Completed

- [x] `gpt_neox`
- [x] `falcon`
- [x] `mpt`
- [x] `opt`
- [x] `blip`

## Batch 6 Completed

- [x] `gptj`
- [x] `phi`
- [x] `phi3`
- [x] `gemma`
- [x] `qwen3_moe`

## Batch 7 Completed

- [x] `clipseg`
- [x] `segformer`
- [x] `mask2former`
- [x] `owlvit`
- [x] `siglip`

## Batch 8 Completed

- [x] `grounding_dino`
- [x] `rt_detr`
- [x] `conditional_detr`
- [x] `yolos`
- [x] `dpt`

## Batch 9 Completed

- [x] `qwen2_audio`
- [x] `seamless_m4t`
- [x] `speecht5`
- [x] `musicgen`
- [x] `clap`

## Batch 10 Completed

- [x] `encodec`
- [x] `mimi`
- [x] `hubert`
- [x] `wavlm`
- [x] `audio_spectrogram_transformer`

## Batch 11 Completed

- [x] `blip_2`
- [x] `instructblip`
- [x] `paligemma`
- [x] `idefics2`
- [x] `kosmos2`

## Batch 12 Completed

- [x] `idefics`
- [x] `idefics3`
- [x] `llava_next`
- [x] `vipllava`
- [x] `chameleon`

## Batch 13 Completed

- [x] `mllama`
- [x] `pixtral`
- [x] `fuyu`
- [x] `git`
- [x] `pix2struct`

## Batch 14 Completed

- [x] `mamba`
- [x] `mamba2`
- [x] `rwkv`
- [x] `jamba`
- [x] `dbrx`

## Batch 15 Completed

- [x] `deepseek_v2`
- [x] `deepseek_v3`
- [x] `qwen2_moe`
- [x] `longformer`
- [x] `big_bird`

## Batch 16 Completed

- [x] `starcoder2`
- [x] `gpt_bigcode`
- [x] `reformer`
- [x] `xlnet`
- [x] `longt5`

## Batch 17 Completed

- [x] `gemma2`
- [x] `qwen3`
- [x] `olmo`
- [x] `olmo2`
- [x] `modernbert`

## Batch 18 Completed

- [x] `llama4`
- [x] `qwen3_next`
- [x] `gpt_oss`
- [x] `granitemoehybrid`
- [x] `glm4_moe`

## Batch 19 Completed

- [x] `cohere`
- [x] `cohere2`
- [x] `granite`
- [x] `granitemoe`
- [x] `glm4`

## Batch 20 Completed

- [x] `qwen2_5_vl`
- [x] `qwen2_5_omni`
- [x] `llava_onevision`
- [x] `llava_next_video`
- [x] `gemma3n`

## Batch 21 Completed

- [x] `aya_vision`
- [x] `cohere2_vision`
- [x] `smolvlm`
- [x] `ovis2`
- [x] `janus`

## Batch 22 Completed

- [x] `video_llava`
- [x] `aria`
- [x] `bridgetower`
- [x] `chinese_clip`
- [x] `altclip`

## Batch 23 Completed

- [x] `code_llama`
- [x] `codegen`
- [x] `biogpt`
- [x] `bitnet`
- [x] `bamba`

## Batch 24 Completed

- [x] `deberta`
- [x] `distilbert`
- [x] `convbert`
- [x] `camembert`
- [x] `xlm_roberta`

## Batch 25 Completed

- [x] `bark`
- [x] `clvp`
- [x] `dia`
- [x] `csm`
- [x] `cohere_asr`

## Batch 26 Completed

- [x] `convnextv2`
- [x] `deit`
- [x] `depth_anything`
- [x] `dinov2_with_registers`
- [x] `mobilevit`

## Batch 27 Completed

- [x] `efficientnet`
- [x] `resnet`
- [x] `poolformer`
- [x] `levit`
- [x] `mobilevitv2`

## Batch 28 Completed

- [x] `regnet`
- [x] `pvt`
- [x] `cvt`
- [x] `focalnet`
- [x] `perceiver`

## Batch 29 Completed

- [x] `mobilenet_v1`
- [x] `mobilenet_v2`
- [x] `bit`
- [x] `data2vec`
- [x] `upernet`

## Batch 30 Completed

- [x] `layoutlm`
- [x] `layoutlmv2`
- [x] `layoutlmv3`
- [x] `donut`
- [x] `trocr`

## Batch 31 Completed

- [x] `table_transformer`
- [x] `deformable_detr`
- [x] `dab_detr`
- [x] `glpn`
- [x] `zoedepth`

## Batch 32 Completed

- [x] `oneformer`
- [x] `maskformer`
- [x] `sam2`
- [x] `sam2_video`
- [x] `depth_pro`

## Batch 33 Completed

- [x] `dinov3_vit`
- [x] `dinov3_convnext`
- [x] `timm_wrapper`
- [x] `siglip2`
- [x] `ijepa`

## Batch 34 Completed

- [x] `align`
- [x] `flava`
- [x] `vilt`
- [x] `visual_bert`
- [x] `lxmert`

## Batch 35 Completed

- [x] `mt5`
- [x] `byt5`
- [x] `umt5`
- [x] `m2m_100`
- [x] `mbart`

## Batch 36 Completed

- [x] `videomae`
- [x] `timesformer`
- [x] `vivit`
- [x] `x_clip`
- [x] `tvp`

## Batch 37 Completed

- [x] `got_ocr2`
- [x] `nougat`
- [x] `markuplm`
- [x] `layoutxlm`
- [x] `udop`

## Batch 38 Completed

- [x] `d_fine`
- [x] `rt_detr_v2`
- [x] `lw_detr`
- [x] `rf_detr`
- [x] `omdet_turbo`

## Batch 39 Completed

- [x] `time_series_transformer`
- [x] `informer`
- [x] `autoformer`
- [x] `patchtst`
- [x] `patchtsmixer`

## Batch 40 Completed

- [x] `glm_ocr`
- [x] `florence2`
- [x] `fast_vlm`
- [x] `deepseek_vl`
- [x] `colpali`

## Batch 41 Completed

- [x] `led`
- [x] `pegasus`
- [x] `pegasus_x`
- [x] `bigbird_pegasus`
- [x] `blenderbot`

## Batch 42 Completed

- [x] `blenderbot_small`
- [x] `marian`
- [x] `fsmt`
- [x] `plbart`
- [x] `mvp`

## Batch 43 Completed

- [x] `bert_generation`
- [x] `encoder_decoder`
- [x] `vision_encoder_decoder`
- [x] `speech_encoder_decoder`
- [x] `vision_text_dual_encoder`

## Batch 44 Completed

- [x] `dpr`
- [x] `bros`
- [x] `lilt`
- [x] `luke`
- [x] `canine`

## Batch 45 Complete

- [x] `mobilebert`
- [x] `mpnet`
- [x] `funnel`
- [x] `fnet`
- [x] `ibert`

## Batch 46 Completed

- [x] `rembert`
- [x] `squeezebert`
- [x] `roberta_prelayernorm`
- [x] `megatron_bert`
- [x] `nomic_bert`

## Batch 47 Completed

- [x] `deepseek_vl_hybrid`
- [x] `qwen3_vl`
- [x] `qwen3_vl_moe`
- [x] `qwen3_omni_moe`
- [x] `phi4_multimodal`

## Batch 48 Complete

- [x] `deepseek_v4`
- [x] `gemma4`
- [x] `glm4v`
- [x] `internvl`
- [x] `minicpmv4_6`

## Batch 49 Complete

- [x] `paddleocr_vl`
- [x] `qianfan_ocr`
- [x] `lighton_ocr`
- [x] `pp_ocrv5_mobile_det`
- [x] `pp_ocrv5_mobile_rec`

## Batch 50 Complete

- [x] `pp_ocrv5_server_det`
- [x] `pp_ocrv5_server_rec`
- [x] `pp_doclayout_v2`
- [x] `pp_doclayout_v3`
- [x] `pp_formulanet`

## Batch 51 Complete

- [x] `sam3`
- [x] `sam3_video`
- [x] `sam3_tracker`
- [x] `sam3_tracker_video`
- [x] `sam3_lite_text`
- [x] `edgetam`
- [x] `edgetam_video`

## Batch 52 Complete

- [x] `qwen3_5`
- [x] `qwen3_5_moe`
- [x] `recurrent_gemma`
- [x] `falcon_h1`
- [x] `olmo3`
- [x] `olmo_hybrid`
- [x] `olmoe`
- [x] `granitemoeshared`
- [x] `longcat_flash`
- [x] `exaone4`
- [x] `exaone_moe`
- [x] `granite_speech`

## Batch 53 Complete

- [x] `granite_speech_plus`
- [x] `higgs_audio_v2`
- [x] `higgs_audio_v2_tokenizer`
- [x] `vibevoice_asr`

## Batch 54 Complete

- [x] `dac`
- [x] `xcodec`
- [x] `falcon_mamba`
- [x] `exaone4_5`
- [x] `ernie4_5`
- [x] `ernie4_5_moe`

## Batch 55 Complete

- [x] `ernie4_5_vl_moe`
- [x] `glm4v_moe`
- [x] `glm46v`
- [x] `glm_moe_dsa`
- [x] `lfm2`
- [x] `lfm2_moe`

## Batch 56 Complete

- [x] `lfm2_vl`
- [x] `voxtral`
- [x] `voxtral_realtime`
- [x] `moonshine`
- [x] `moonshine_streaming`
- [x] `moshi`

## Batch 57 Complete

- [x] `audioflamingo3`
- [x] `musicflamingo`
- [x] `kyutai_speech_to_text`
- [x] `glmasr`
- [x] `parakeet`
- [x] `fastspeech2_conformer`

## Batch 58 Complete

- [x] `efficientloftr`
- [x] `lightglue`
- [x] `superpoint`
- [x] `sam_hq`
- [x] `vitpose`
- [x] `vitmatte`

## Batch 59 Complete

- [x] `vitdet`
- [x] `vitpose_backbone`
- [x] `timm_backbone`
- [x] `hgnet_v2`
- [x] `swinv2`
- [x] `pvt_v2`

## Batch 60 Complete

- [x] `gpt_neo`
- [x] `ctrl`
- [x] `xglm`
- [x] `xlm`
- [x] `xlm_roberta_xl`
- [x] `xmod`

## Batch 61 Complete

- [x] `gpt_neox_japanese`
- [x] `flaubert`
- [x] `roformer`
- [x] `roc_bert`
- [x] `splinter`
- [x] `modernbert_decoder`

## Batch 62 Complete

- [x] `sew`
- [x] `sew_d`
- [x] `unispeech`
- [x] `unispeech_sat`
- [x] `wav2vec2_bert`
- [x] `wav2vec2_conformer`

## Batch 63 Complete

- [x] `speech_to_text`
- [x] `seamless_m4t_v2`
- [x] `musicgen_melody`
- [x] `pop2piano`
- [x] `univnet`
- [x] `vits`

## Batch 64 Complete

- [x] `aimv2`
- [x] `dinat`
- [x] `groupvit`
- [x] `hiera`
- [x] `imagegpt`
- [x] `vit_mae`

## Batch 65 Complete

- [x] `deimv2`
- [x] `eomt`
- [x] `eomt_dinov3`
- [x] `mm_grounding_dino`
- [x] `swiftformer`
- [x] `swin2sr`

## Batch 66 Complete

- [x] `switch_transformers`
- [x] `nllb_moe`
- [x] `nystromformer`
- [x] `mra`
- [x] `prophetnet`
- [x] `tapas`

## Batch 67 Complete

- [x] `afmoe`
- [x] `apertus`
- [x] `arcee`
- [x] `blt`
- [x] `chmv2`
- [x] `colmodernvbert`

## Batch 68 Complete

- [x] `colqwen2`
- [x] `cpmant`
- [x] `cwm`
- [x] `decision_transformer`
- [x] `diffllama`
- [x] `doge`

## Batch 69 Complete

- [x] `dots1`
- [x] `emu3`
- [x] `ernie`
- [x] `esm`
- [x] `eurobert`
- [x] `evolla`

## Batch 70 Complete

- [x] `flex_olmo`
- [x] `gemma4_assistant`
- [x] `glm`
- [x] `glm_image`
- [x] `glm4_moe_lite`
- [x] `granite4_vision`

## Batch 71 Complete

- [x] `helium`
- [x] `hunyuan_v1_dense`
- [x] `hunyuan_v1_moe`
- [x] `hy_v3`
- [x] `hyperclovax`
- [x] `instructblipvideo`

## Batch 72 Complete

- [x] `jais2`
- [x] `jetmoe`
- [x] `jina_embeddings_v3`
- [x] `kosmos2_5`
- [x] `laguna`
- [x] `lasr`

## Batch 73 Complete

- [x] `metaclip_2`
- [x] `mgp_str`
- [x] `minimax`
- [x] `minimax_m2`
- [x] `ministral`
- [x] `ministral3`

## Batch 74 Complete

- [x] `mistral3`
- [x] `mistral4`
- [x] `mlcd`
- [x] `modernvbert`
- [x] `nanochat`
- [x] `nemotron`

## Batch 75 Complete

- [x] `nemotron_h`
- [x] `openai`
- [x] `openai_privacy_filter`
- [x] `owlv2`
- [x] `pe_audio`
- [x] `pe_audio_video`

## Batch 76 Complete

- [x] `pe_video`
- [x] `perception_lm`
- [x] `persimmon`
- [x] `phimoe`
- [x] `pi0`
- [x] `pixio`

## Batch 77 Complete

- [x] `pp_chart2table`
- [x] `pp_lcnet`
- [x] `pp_lcnet_v3`
- [x] `prompt_depth_anything`
- [x] `rag`
- [x] `seed_oss`

## Batch 78 Complete

- [x] `seggpt`
- [x] `shieldgemma2`
- [x] `slanet`
- [x] `slanext`
- [x] `smollm3`
- [x] `solar_open`

## Batch 79 Complete

- [x] `stablelm`
- [x] `superglue`
- [x] `t5gemma`
- [x] `t5gemma2`
- [x] `textnet`
- [x] `timesfm`

## Batch 80 Complete

- [x] `timesfm2_5`
- [x] `uvdoc`
- [x] `vaultgemma`
- [x] `vibevoice_acoustic_tokenizer`
- [x] `video_llama_3`
- [x] `videomt`

## Batch 81 Complete

- [x] `vit_msn`
- [x] `vjepa2`
- [x] `xlstm`
- [x] `yoso`
- [x] `youtu`
- [x] `zamba`
- [x] `zamba2`


## Suggested Next Batches

Further batches should be selected by architecture coverage first, then by
popularity/usage. Near-duplicate families can reuse findings only after a
source/config check confirms the operator surface is actually equivalent.
