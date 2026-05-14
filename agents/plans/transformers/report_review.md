# Transformers Report Review and Prompt Refinement Notes

## Reports reviewed

- `agents/plans/transformers/t5/report.md`
- `agents/plans/transformers/llama/report.md`
- `agents/plans/transformers/mistral/report.md`
- `agents/plans/transformers/bert/report.md`
- `agents/plans/transformers/glm_ocr_dinoml_ops_report.md` (legacy standalone report; findings extracted, file removed)
- `agents/plans/transformers/vit/report.md`
- `agents/plans/transformers/whisper/report.md`
- `agents/plans/transformers/qwen2/report.md`
- `agents/plans/transformers/roberta/report.md`
- `agents/plans/transformers/deberta_v2/report.md`
- `agents/plans/transformers/swin/report.md`
- `agents/plans/transformers/dinov2/report.md`
- `agents/plans/transformers/llava/report.md`
- `agents/plans/transformers/qwen2_vl/report.md`
- `agents/plans/transformers/gemma3/report.md`
- `agents/plans/transformers/mixtral/report.md`
- `agents/plans/transformers/wav2vec2/report.md`
- `agents/plans/transformers/bart/report.md`
- `agents/plans/transformers/gpt2/report.md`
- `agents/plans/transformers/bloom/report.md`
- `agents/plans/transformers/clip/report.md`
- `agents/plans/transformers/convnext/report.md`
- `agents/plans/transformers/detr/report.md`
- `agents/plans/transformers/sam/report.md`
- `agents/plans/transformers/beit/report.md`
- `agents/plans/transformers/albert/report.md`
- `agents/plans/transformers/electra/report.md`
- `agents/plans/transformers/gpt_neox/report.md`
- `agents/plans/transformers/falcon/report.md`
- `agents/plans/transformers/mpt/report.md`
- `agents/plans/transformers/opt/report.md`
- `agents/plans/transformers/blip/report.md`
- `agents/plans/transformers/gptj/report.md`
- `agents/plans/transformers/phi/report.md`
- `agents/plans/transformers/phi3/report.md`
- `agents/plans/transformers/gemma/report.md`
- `agents/plans/transformers/qwen3_moe/report.md`
- `agents/plans/transformers/clipseg/report.md`
- `agents/plans/transformers/segformer/report.md`
- `agents/plans/transformers/mask2former/report.md`
- `agents/plans/transformers/owlvit/report.md`
- `agents/plans/transformers/siglip/report.md`
- `agents/plans/transformers/grounding_dino/report.md`
- `agents/plans/transformers/rt_detr/report.md`
- `agents/plans/transformers/conditional_detr/report.md`
- `agents/plans/transformers/yolos/report.md`
- `agents/plans/transformers/dpt/report.md`
- `agents/plans/transformers/qwen2_audio/report.md`
- `agents/plans/transformers/seamless_m4t/report.md`
- `agents/plans/transformers/speecht5/report.md`
- `agents/plans/transformers/musicgen/report.md`
- `agents/plans/transformers/clap/report.md`
- `agents/plans/transformers/encodec/report.md`
- `agents/plans/transformers/mimi/report.md`
- `agents/plans/transformers/hubert/report.md`
- `agents/plans/transformers/wavlm/report.md`
- `agents/plans/transformers/audio_spectrogram_transformer/report.md`
- `agents/plans/transformers/blip_2/report.md`
- `agents/plans/transformers/instructblip/report.md`
- `agents/plans/transformers/paligemma/report.md`
- `agents/plans/transformers/idefics2/report.md`
- `agents/plans/transformers/kosmos2/report.md`

## Review findings

### T5 report

- The report is useful and mostly source-grounded. It captures the key non-obvious T5 requirements: encoder-decoder structure, T5 RMS-style norm, relative attention bias, shared bias across layers, cross-attention cache reuse, and the fact that original T5 3B/11B have `A * D != H`.
- The later config sweep was valuable and should be required by the prompt. `t5-small` alone would have hidden the high-width original T5 variants and the gated-gelu T5 v1.1/FLAN path.
- Minor wording risk: the dtype row says F32 based on checkpoint/model metadata rather than `config.json`; future reports should separate config fields from model-card or safetensors metadata.
- Good prompt-fit: sections 4, 6, 7, 9, and 10 are concrete enough for op coverage and kernel planning.
- Improvement opportunity: add a short "family variation traps" subsection to each report so future agents do not bury the most important config variability in a large table.

### Llama report

- The report exercises different prompt dimensions than T5: decoder-only cache, RoPE, GQA, `logits_to_keep`, tokenizer/chat-template separation, and tensor-parallel hints from config.
- The config sweep found prompt-critical variation: Llama 2 MHA, Llama 3/TinyLlama GQA, CodeLlama long context, different vocab sizes, and large RoPE theta changes.
- Source compatibility note: current Transformers source uses `rope_parameters`, while many existing configs still expose older `rope_theta`/`rope_scaling` fields. The prompt should ask reports to call out config schema evolution and default normalization.
- Gated official Meta repos can block config access. The prompt should explicitly allow open mirrors for dimension sampling when official gated repos are inaccessible, while labeling that fact.

### Mistral report

- The refined prompt worked better than the first two reports because it forced a representative checkpoint sweep and a "family variation traps" section. That surfaced real integration differences: `sliding_window=4096` in Mistral v0.1, `sliding_window=null` in later 7B instruct configs, 131K vocab/context in Nemo, and per-layer `layer_types` in Ministral.
- The report shows why "same as Llama" is not enough. Mistral uses the same core decoder skeleton, but local/sliding attention and large-vocab variants change attention backend requirements and performance priorities.
- The `layer_types` warning in `MistralConfig` suggests another prompt refinement: when source config warns that a related class should be used for a variant, the report should either exclude that variant from the plain family report or explicitly mark it as requiring a separate family-specific assessment.
- The source is generated from `modular_mistral.py`, so future reports should mention generated/modular source relationships when present.

### BERT report

- BERT exercised an encoder-only, non-generation architecture and showed that cache-focused prompt language needs an explicit "not applicable for primary path" answer rather than forcing generation details everywhere.
- The report surfaced a task/head ambiguity: the same family covers base encoder, masked LM, pretraining NSP, sequence classification, token classification, QA, multiple choice, and optional decoder mode. The prompt should require reports to name the primary runtime target and classify other heads as required, optional, or deferred.
- BERT also emphasized embedding stack details that were less important in decoder LLMs: token type IDs, absolute learned position embeddings, post-tokenizer segment construction, and embedding LayerNorm. The prompt's preprocessing section should explicitly ask for segment/type IDs and special-token layout for text encoders.

### GLM-OCR legacy report

- The legacy GLM-OCR report had strong multimodal/runtime-planning material that the generic prompt did not yet force: processor output tensor contracts, image grid metadata, modality token type IDs, placeholder-token scatter into text embeddings, varlen vision attention with `cu_seqlens`, and separable vision/prefix/prefill/decode stages.
- It showed that multimodal reports should document processor outputs as first-class runtime inputs, not just "preprocessing." For GLM-OCR, `pixel_values` were flattened patch rows and `image_grid_thw` drove vision positions; this sort of shape contract is easy to miss if the report only reads model code.
- It provided a useful pattern for stage decomposition: CPU preprocessing, vision embedding service, prefix builder, text prefill, and text decode. The prompt should ask for this whenever a model has independently cacheable encoders/projectors or expensive prefix construction.
- It formalized layout-sensitive ConvNd-to-GEMM rewrites, including dynamic guards, weight permutation, activation flatten order, and exact failure cases. The generic rewrite prompt now needs to require layout and weight-transform details for conv lowering rather than accepting a vague "conv can become linear" claim.
- It included benchmark-derived throughput interpretation. The prompt should allow prior measurements, but require provenance labeling and separation from source-derived facts.

### ViT report

- ViT validated the new NHWC/layout-elision requirement. The source patch embedding is written as NCHW Conv2d followed by flatten and transpose, but Dinoml should usually lower the same operation as NHWC non-overlap window flatten plus GEMM with a compile-time weight permutation.
- Vision-only models need the prompt to ask for source layout and preferred runtime layout around every vision op. The updated prompt now asks reports to identify NCHW/NCDHW modeling code, channel-last opportunities, and source permutes/transposes that can be eliminated, sunk, or folded.
- ViT also reinforced that processor configs matter even for simple vision models: resize/normalize choices and fixed image-size buckets determine the patch projection shape and whether position interpolation is needed.

### Whisper report

- Whisper validated the need for first-class audio feature-extractor contracts. The model graph consumes fixed log-mel tensors, but the preprocessor controls sampling rate, chunk length, STFT hop/window, mel-bin count, normalization, and the `[B, mel, frames]` layout.
- The checkpoint sweep found a real trap: older Whisper checkpoints use 80 mel bins, while large-v3 and large-v3-turbo use 128; turbo also keeps the 32-layer large encoder but cuts the decoder to 4 layers. Prompt language must not assume symmetric encoder/decoder sizes or source defaults.
- Whisper exposed generation-controller requirements beyond ordinary seq2seq logits: forced language/task/no-timestamp prompt IDs, suppress-token processors, timestamp processors, no-speech/long-form controls, and attention-weight slow paths. Reports should split core graph work from scheduler/generation-controller work.
- It also caught a fused-attention parity detail: Whisper scales Q before calling the backend and passes attention scaling as `1.0`. The prompt now asks reports to document attention math order, casting, masking, and fallback semantics.
- The NHWC/layout guidance still applies, but for audio the source layout is `[B, mel, time]`; any time-major/channel-last lowering should be a local frontend pass with a `no_layout_translation()` guard around unchecked axis-sensitive code.

### Batch 1 subagent reports

- Qwen2, RoBERTa, DeBERTa-v2, Swin, and DINOv2 were produced by parallel subagents and then coordinator-reviewed. The prompt was strong enough for consistent sections and did not need another structural change.
- Qwen2 reinforced existing decoder guidance: post-init config normalization matters, raw JSON fields like `sliding_window` can be inactive, cache tensors should be documented in KV-head shape, and last-token logits must be guarded as a generation-path optimization.
- RoBERTa and DeBERTa-v2 showed that text encoders need tokenizer/position details just as much as decoder LMs need cache details. RoBERTa's padding-aware position IDs and DeBERTa-v2's disentangled relative bias are good examples of non-obvious runtime graph requirements.
- Swin and DINOv2 validated the guarded layout-pass wording: source-faithful NCHW/NHWC semantics should be reported first, while NHWC/channel-last opportunities are described as local rewrites with axis and output-layout constraints.
- Minor review fix applied: reports should avoid claiming dtype from config when it is absent; runtime dtype should come from weights, safetensors metadata, or deployment policy.

### Batch 2 subagent reports

- LLaVA, Qwen2-VL, Gemma3, Mixtral, and Wav2Vec2 were produced by parallel subagents and coordinator-reviewed. The prompt continued to hold without structural changes.
- LLaVA and Qwen2-VL validated the processor tensor-contract requirements. LLaVA depends on placeholder expansion and masked scatter over projected vision tokens, while Qwen2-VL is stricter: the model consumes packed patch rows plus grid metadata, not raw image tensors.
- Qwen2-VL added a clear example of why reports must document packed/varlen metadata: `image_grid_thw`/`video_grid_thw`, `cu_seqlens`, M-RoPE deltas, and placeholder counts are runtime inputs, not incidental preprocessing.
- Gemma3 showed that even one family can contain text-only and multimodal generation surfaces. Its hybrid full/sliding attention, separate local/global RoPE buckets, `1 + weight` RMSNorm, and image-aware masks from `token_type_ids` are all implementation-critical.
- Mixtral surfaced the first MoE report. Future MoE audits should explicitly describe router softmax/top-k normalization, token-to-expert grouping, expert weight layout, route-weighted scatter-add, and whether grouped GEMM/provider work is required.
- Wav2Vec2 usefully contrasts Whisper: raw waveform processor, Conv1d length reduction, feature-mask reduction, positional grouped Conv1d with weight norm, no STFT/mel path, and no autoregressive cache.

### Batch 3 subagent reports

- BART, GPT-2, BLOOM, CLIP, and ConvNeXT were produced by parallel subagents and coordinator-reviewed. The reports were consistent enough to proceed with scaling, with only small prompt refinements needed.
- BART added encoder-decoder generation coverage beyond T5: learned absolute positions with offset 2, post-norm blocks, `EncoderDecoderCache`, static cross-attention cache reuse, and summarization-critical generation controllers such as beams, forced BOS/EOS, length penalties, and no-repeat n-gram constraints.
- GPT-2 and BLOOM showed why reports must document model-specific packed projection layouts. GPT-2 `Conv1D` stores weights as `[in_features, out_features]`; BLOOM QKV is packed per head as `[q,k,v]` groups and ALiBi is built from `attention_mask.cumsum`, so padding policy changes attention math.
- BLOOM also reinforced that optimized backend assumptions must be source-derived. The inspected source uses an eager `baddbmm + mask + fp32 softmax + bmm` path rather than direct SDPA/Flash dispatch, and `pretraining_tp` only changes runtime math under `slow_but_exact=True`.
- CLIP exposed a distinct dual-encoder/contrastive runtime shape: independently cacheable image/text branches, causal text attention without decode cache, EOS/EOT pooling compatibility branches, L2 feature normalization, `exp(logit_scale)`, and both text-by-image and image-by-text logits orientations.
- ConvNeXT validated that some Transformers model directories are non-attention CNNs. These reports should explicitly say when attention/cache/generation sections are not applicable, while still documenting source NCHW, local NHWC islands, channel LayerNorm, depthwise convolution, global pooling axes, and guarded layout-region rewrites.

### Batch 4 subagent reports

- DETR, SAM, BEiT, ALBERT, and ELECTRA were produced by parallel subagents and coordinator-reviewed. The reports broadened the prompt into structured-output vision, promptable segmentation, and shared-weight encoder families.
- DETR showed that object-detection postprocessing is part of inference parity: no-object class handling, no NMS, `cxcywh -> xyxy`, target-size scaling, score filtering, and per-image variable outputs must be documented separately from training-only Hungarian/loss paths.
- SAM added promptable segmentation staging: image-embedding cache, point/box/mask prompt encoder, dynamic `point_batch_size`, two-way mask decoder, hypernetwork mask matmul, and mask postprocess using original and reshaped image sizes. Reports for promptable models should treat cached encoders and repeated prompt batches as first-class runtime plans.
- BEiT reinforced relative-position-bias variation: classification/segmentation use per-layer relative bias, pretraining MIM uses shared relative bias, and higher-resolution segmentation requires interpolation/re-indexing. It also highlighted sequence-to-NCHW reshapes for segmentation heads as guarded layout boundaries.
- ALBERT introduced architectural parameter sharing as an integration issue. The report distinguishes logical layer applications from physical shared `AlbertLayerGroup` modules, factorized `E -> H` embeddings, grouped-layer indexing, and tied MLM decoder weights.
- ELECTRA showed that generator/discriminator variants change both head shape and embedding projection geometry. Reports should cover primary target scope, optional generator support, `embedding_size != hidden_size`, tied generator LM weights, and non-primary decoder/cache paths without letting them dominate the encoder report.

### Batch 5 subagent reports

- GPT-NeoX, Falcon, MPT, OPT, and BLIP were produced by parallel subagents and coordinator-reviewed. The reports confirmed that near-duplicate decoder families still carry important source-specific lowering traps.
- GPT-NeoX reinforced partial-RoPE handling: official configs rotate only a prefix of each head with GPT-NeoX half-rotation, use per-head `[q,k,v]` packed QKV layout, and can switch residual topology with `use_parallel_residual`.
- Falcon surfaced the strongest packed-QKV variability so far: old full-MHA ALiBi, old MQA RoPE, and new-decoder grouped QKV all use different split rules. The report correctly separates HF source-expanded new-decoder cache behavior from DinoML's preferred compact GQA/MQA cache contract.
- MPT showed a native-source versus historical remote-code gap. The native in-library implementation is ALiBi-only full MHA and ignores or does not implement several config-advertised remote-code features such as learned positions, MQA, q/k LayerNorm, prefix-LM masks, sequence-id masks, non-default `alibi_bias_max`, and attention backend flags.
- OPT highlighted learned-position details that differ from GPT-2: padding-aware `cumsum(dim=1)`, position offset `+2`, `opt-350m` post-LN/project-in/project-out exception, and query pre-scaling before attention backend dispatch.
- BLIP added an older multimodal stack with multiple heads: captioning text decoder cross-attends to cached image tokens, VQA stages question encoding before answer decoding, ITM/retrieval has classifier and projection-similarity paths, and generation has BLIP-specific BOS/SEP prompt behavior.

### Batch 6 subagent reports

- GPT-J, Phi, Phi-3, Gemma, and Qwen3-MoE were produced by parallel subagents and coordinator-reviewed. The prompt held without new structural changes.
- GPT-J demonstrated that "partial RoPE" is not one thing: GPT-J uses interleaved even/odd rotation with an absolute `rotary_dim`, separate bias-free Q/K/V linears, mandatory parallel residual, and a bias-bearing LM head.
- Phi contrasted with both GPT-J and Llama-like reports: it uses affine LayerNorm rather than RMSNorm, biased projections and LM head, partial RoPE with possible future GQA, one shared block LayerNorm feeding attention and MLP, and historical remote-code flags that should be translated or rejected.
- Phi-3 surfaced native/remote family splitting and long-context hazards. Native `phi3` uses packed all-Q/all-K/all-V QKV, packed gate/up SwiGLU, MHA or GQA, LongRoPE short/long factor switching, sliding-window masks, and cache invalidation at the original-context boundary. `phi3small` is a separate remote-code target.
- Gemma clarified that Gemma is not Gemma3: no multimodal branch or softcapping, but it does require embedding scaling by `sqrt(hidden_size)`, one-plus RMSNorm `(1 + weight)`, tied embedding/LM head, and MQA/GQA for 2B versus full MHA for 7B.
- Qwen3-MoE is meaningfully different from Mixtral and Qwen2-MoE: Q/K head RMSNorm precedes RoPE, official configs use top-8 routing over 128 experts with `norm_topk_prob=true`, expert gate/up weights are packed 3D tensors, and there is no Qwen2-MoE shared expert path.

### Batch 7 subagent reports

- CLIPSeg, SegFormer, Mask2Former, OWL-ViT, and SigLIP were produced by parallel subagents and coordinator-reviewed. The reports were healthy, and the only prompt update needed was to ask prompt-conditioned models to name their conditioning mechanism and cacheable condition state explicitly.
- CLIPSeg showed that conditioning is not always cross-attention: the source uses CLIP-style prompt embeddings plus FiLM-style decoder modulation, with raw mask logits returned from the model and sigmoid/resize/threshold behavior outside the core graph. It also exposed a processor/config mismatch where common CIDAS checkpoints use 352x352 processor inputs while the vision config starts at 224x224 positional embeddings.
- SegFormer reinforced overlapping patch embeddings, sequence-reduction attention on K/V only, Mix-FFN depthwise convolution, and decode-head multi-scale concat as first-class layout-pass stress tests. Its postprocess is resize plus argmax, separate from the encoder/decode-head graph.
- Mask2Former added the strongest structured-output requirement so far: Swin backbone tokens feed a pixel decoder with multi-scale deformable attention, then a masked transformer decoder with learned queries and class/mask heads. Semantic, instance, and panoptic postprocessing are variable-output Python loops and should be staged outside the compiled graph first.
- OWL-ViT clarified open-vocabulary detection contracts: nested text prompts flatten to text batches, query validity uses source-specific token logic, patch-major logits are `[batch, patches, queries]`, text-query detection lacks NMS, and image-guided detection adds IoU suppression outside the core dual-encoder path.
- SigLIP looks CLIP-like but changes several runtime assumptions: bidirectional text, last-token pooling, no CLS token in vision, MAP pooling implemented with packed `nn.MultiheadAttention`, biased projection heads, learned `logit_bias`, and sigmoid probabilities rather than CLIP's symmetric softmax contract.

### Batch 8 subagent reports

- Grounding DINO, RT-DETR, Conditional DETR, YOLOS, and DPT were produced by parallel subagents and coordinator-reviewed. The prompt continued to produce actionable reports; the main refinement was to require explicit nested-backbone feature contracts for AutoBackbone/timm/composed-family models.
- Grounding DINO showed that text-conditioned detection is neither CLIP similarity nor fixed-class DETR. It requires BERT phrase-block masks, repeated image/text fusion, text cross-attention in the decoder, token-level logits padded to `max_text_len`, tied decoder bbox heads, and grounded phrase extraction with no NMS.
- RT-DETR emphasized hybrid CNN/transformer staging: only selected levels run AIFI attention, FPN/PAN remain convolutional, proposal top-k seeds decoder queries, and multiscale deformable attention fallback uses `grid_sample`. The R101 config also changes `encoder_hidden_dim` while keeping `d_model=256`.
- Conditional DETR exposed a fused-attention hazard: decoder cross-attention concatenates content and positional channels so Q/K head width is doubled while V head width remains normal. Its postprocess uses sigmoid global top-k over `Q*C`, not DETR's softmax/no-object path.
- YOLOS demonstrated a ViT-style detector with learned detection-token suffixes, no consumed `pixel_mask`, interpolated initial and mid-layer position tables, and no NMS. The `small-dwr` checkpoint's `head_dim=55` is a useful shape stress case.
- DPT clarified dense-prediction composition: native ViT, hybrid BiT+ViT, and AutoBackbone variants are materially different. Reassemble/readout modes, ConvTranspose2d factors, SwinV2 no-reassemble behavior, and depth/segmentation postprocess resizing all need explicit guards.

### Batch 9 subagent reports

- Qwen2-Audio, SeamlessM4T, SpeechT5, MusicGen, and CLAP were produced by parallel subagents and coordinator-reviewed. The prompt still held, with one refinement applied: reports must distinguish autoregressive KV caches from cross-attention caches, encoder/projector caches, retrieval branch caches, prompt/audio code caches, and processor metadata caches.
- Qwen2-Audio demonstrated an audio-prefix generation pattern: Whisper log-mel features feed a bidirectional audio encoder, projected audio embeddings replace expanded `<|AUDIO|>` tokens by masked scatter, then an ordinary Qwen2 decoder handles prefill/decode. Audio must be absent from cached decode iterations.
- SeamlessM4T showed a multi-path translation/synthesis runtime: text and speech encoders feed a BART-like text decoder, speech output adds a separate T2U encoder-decoder, and the vocoder uses value-dependent duration expansion. Its feature extractor hands the model stride-packed `[B, frames//2, 160]` audio features rather than raw mels.
- SpeechT5 added pluggable modality prenets/postnets: ASR is closest to standard seq2seq token generation, while TTS/VC use a custom mel loop with speaker embeddings, reduction-factor frames, stop probabilities, postnet, and optional HiFi-GAN. The eval-time speech decoder prenet dropout is a parity hazard.
- MusicGen introduced audio-code generation rather than waveform or text tokens. The key integration risks are delayed codebook pattern masks, mono/stereo codebook interleave, EnCodec RVQ decode, per-codebook LM heads, CFG as controller-level batch duplication, and generation modes restricted to greedy/sampling.
- CLAP broadened dual-encoder coverage into audio-text retrieval: CPU log-mel preprocessing can be stochastic for long clips, fused and unfused checkpoints have different `input_features`/`is_longer` contracts, HTSAT window attention is the audio bottleneck, and branch embeddings should be cached after projection plus L2 normalization.

### Batch 10 subagent reports

- EnCodec, Mimi, HuBERT, WavLM, and Audio Spectrogram Transformer were produced by parallel subagents and coordinator-reviewed. No structural prompt change was needed beyond the cache taxonomy already added during Batch 9.
- EnCodec established the codec-only baseline: no attention, no KV cache, and no token generation. Required runtime work is Conv1d/ConvTranspose1d with exact padding/trim, LSTM wrappers, RVQ distance/GEMM/argmax, bandwidth-to-quantizer selection, and 48 kHz chunked overlap-add.
- Mimi contrasted with EnCodec by adding causal sliding-window transformer stacks and streaming state. It needs both transformer KV caches and Conv1d padding caches, plus RoPE, RVQ semantic/acoustic codebook handling, stride-2 downsample/upsample, and NCL codec layout guards.
- WavLM contrasted with HuBERT/Wav2Vec2 through gated bucketed relative-position bias. The source disables ordinary Flash/SDPA/Flex support, so DinoML fused attention must explicitly handle hidden-state-gated additive bias rather than treating WavLM as plain encoder MHA.
- HuBERT covered the wav2vec-style baseline: raw waveform Conv1d frontend, group-norm versus per-conv LayerNorm feature extractors, positional grouped Conv1d with weight norm, stable pre-norm versus post-norm encoders, CTC heads, and optional weighted-layer-sum classification.
- Audio Spectrogram Transformer covered fixed-shape fbank classification. It should reject mismatched `max_length`, `num_mel_bins`, stride, or position-embedding shapes because current source has no runtime 2D position interpolation; patch Conv2d lowering must preserve frequency/time transpose and token order.

### Batch 11 subagent reports

- BLIP-2, InstructBLIP, PaliGemma, Idefics2, and KOSMOS-2 were produced by parallel subagents and coordinator-reviewed. The prompt held without structural changes; the existing cache taxonomy and composed-backbone requirements were exercised heavily.
- BLIP-2 highlighted a Q-Former bridge whose projected query outputs are cacheable before delegated OPT/T5 prefill. Its important traps are source-defaulted vision/Q-Former configs, checkpoint-specific `<image>` ids, fp32 Q-Former island, periodic query-only cross-attention, and OPT versus T5 cache semantics.
- InstructBLIP is not just BLIP-2 with different weights: the Q-Former consumes instruction text, has query/text split FFNs, and its projected image features depend on both image and instruction prompt. It also splits cleanly into T5 seq2seq and Vicuna/LLaMA causal branches.
- PaliGemma uses SigLIP plus Gemma/Gemma2 with repeated image placeholders and a direct embedding stitch. Official Google configs were gated, so the report correctly labels converter/source-derived dimensions; PaliGemma2 adds softcap, GQA, and full/sliding layer-type handling.
- Idefics2 added variable-resolution vision and image splitting: one logical `<image>` can become five processed image rows and five placeholder runs. The connector is a gated MLP plus Perceiver GQA resampler, and dense parity should precede AWQ text-GEMM lowering.
- KOSMOS-2 uses fixed learned latent queries and a causal text decoder, not a Q-Former. It has a cacheable image projection boundary, sinusoidal text positions with offset 2, tied XLM-R-style text embeddings/LM head, and grounding postprocessing based on textual patch-index tokens rather than detection heads.

### Batch 12 subagent reports

- Idefics, Idefics3, LLaVA-NeXT, ViP-LLaVA, and Chameleon were produced by parallel subagents and coordinator-reviewed. One prompt refinement was applied: tokenized-image/discrete-code families need explicit codebook, vocabulary mapping, placeholder expansion, logits-mask, and cache-boundary coverage.
- Idefics covered the legacy Flamingo-style path: CLIP-like vision tower, optional Perceiver resampler, gated cross-attention every `cross_layer_interval`, decoupled vocabulary, cached image/perceiver embeddings, and text self-attention KV cache with no cross-attention KV cache.
- Idefics3 contrasted with Idefics2 by removing the Perceiver and using a SigLIP-like variable-resolution encoder plus connector pixel-shuffle. The key prompt stressors were processor split/global placeholder expansion, patch-mask unfolding, bucketized vision positions, and stale config fields that native source ignores.
- LLaVA-NeXT broadened LLaVA-style projector coverage with AnyRes patch packing, spatial unpadding, learned image newline insertion, dynamic placeholder counts, and processor/model coupling for `patch_size`, `num_additional_image_tokens`, and feature selection strategy.
- ViP-LLaVA exercised multi-layer vision feature pyramids: selected CLIP hidden states, unconditional CLS drop, channel concat across repeated or negative layer indices, and projector width derived from `len(vision_feature_layers) * vision_hidden_size`. Native 7B legacy keys should be normalized or rejected.
- Chameleon introduced the tokenized-image path: VQ-VAE/VQGAN image tokenizer, 1024 discrete image codes per 512x512 image, code-index to BPE token mapping through `IMGIMG*` vocabulary names, shared token embeddings, and LM-head suppression of raw image-code logits for text generation.

### Batch 13 subagent reports

- Mllama, Pixtral, Fuyu, GIT, and Pix2Struct were produced by parallel subagents and coordinator-reviewed. No new prompt change was needed: the existing conditioning-mechanism, processor-contract, composed-backbone, cache-taxonomy, and layout-guard requirements covered the new patterns.
- Mllama covered a native cross-attention multimodal decoder: image placeholders create dense cross-attention ranges rather than embedding splice points, and decode must reuse image cross-attention K/V while only growing self-attention KV cache.
- Pixtral showed a vision-only family composed through a LLaVA wrapper and Mistral decoder. It adds variable-size image patch packing, block-diagonal vision attention across concatenated images, 2D vision RoPE, and official-params versus converted-HF provenance concerns.
- Fuyu exercised direct patch projection without a vision transformer: NCHW image patchify, `Linear(2700 -> 4096)`, placeholder/newline token streams, Persimmon packed QKV, per-head Q/K LayerNorm, partial RoPE, and `relu2` MLP activation.
- GIT used visual prefix concatenation rather than placeholder scatter or cross-attention. Its decoder is BERT-shaped but causal, with a bidirectional image-prefix block mask and video variants implemented as repeated frame image encodes plus temporal embeddings.
- Pix2Struct covered flattened-patch sequence input: the runtime contract starts from `flattened_patches [B,S,770]`, row/column embeddings, T5-style seq2seq decode, cross-attention K/V cache, and VQA/doc-QA preprocessing where text is rendered into the image.

### Batch 14 subagent reports

- Mamba, Mamba2, RWKV, Jamba, and DBRX were produced by parallel subagents and coordinator-reviewed. One prompt refinement was applied: recurrent/state-space families need explicit state-cache ABIs and hybrid cache manifests rather than generic KV-cache language.
- Mamba established the selective-scan baseline: no attention, conv state `[B,I,K]`, recurrent SSM state `[B,I,N]`, optional `mamba-ssm`/`causal-conv1d` kernels, attention masks as multiplicative hidden masks, and static-address cache mutation concerns.
- Mamba2 added SSD chunk scan, grouped `B/C` state projections, conv cache `[B, conv_dim, K]`, recurrent cache `[B, heads, head_dim, state]`, grouped-state expansion hazards, and rejection of `mamba2attn-*` as a separate hybrid target.
- RWKV covered a different recurrent ABI: five state tensors `[B,H,L]`, fp32 WKV numerator/denominator/max states, no attention mask in the core graph, source one-token fallback behavior, and inference rescale materialization.
- Jamba combined sparse attention layers, Mamba layers, and MoE FFNs. It requires materialized layer schedules, a hybrid cache manifest containing both attention KV and Mamba conv/recurrent states, and independent attention/MoE schedules rather than a single block pattern.
- DBRX refreshed MoE coverage with packed Wqkv plus required QKV clamp, LayerNorm without bias, GQA cache `[B,KVH,T,D]`, flattened `w1/v1/w2` expert weights, top-4 routing over 16 experts, and a pinned-source expert reshape hazard that should be guarded before parity claims.

## Prompt refinements to apply before scaling to all families

1. Require a "representative checkpoint sweep" for every family, not just one model id.

   Suggested prompt addition:

   ```text
   Inspect at least 3-5 representative checkpoint configs when available:
   a small/debug checkpoint, the most common production checkpoint, and any
   known variant that changes operator structure such as GQA, MoE, gated MLP,
   long context, vision/audio branch, or custom position encoding. Use official
   repos when accessible; if gated, use an open mirror and label it.
   ```

2. Add a "family variation traps" section.

   Suggested section:

   ```markdown
   ## Family variation traps

   List config-dependent behavior that invalidates naive assumptions, such as
   hidden_size != num_heads * head_dim, num_key_value_heads < num_attention_heads,
   optional biases, different MLP activations, long-context RoPE variants,
   vocab/tokenizer changes, or encoder/decoder layer count asymmetry.
   ```

3. Distinguish config-derived facts from metadata/model-card facts.

   Suggested style rule:

   ```text
   Label whether dtype, parameter count, license, task, or model size comes from
   config.json, safetensors index/metadata, Hugging Face repo metadata, or an
   inference from source defaults.
   ```

4. Require config default reconciliation.

   Suggested prompt addition:

   ```text
   If a checkpoint config omits fields that the current Transformers config class
   supplies by default, list the omitted fields and the effective defaults.
   ```

5. Ask for cache layout explicitly.

   Suggested prompt addition:

   ```text
   For generation models, state the exact per-layer cache tensor shapes before
   and after any MQA/GQA/repeat expansion, and identify whether cached keys are
   stored before or after position encoding.
   ```

6. Ask for source backend/fallback paths.

   Suggested prompt addition:

   ```text
   Identify the source eager/fallback implementation and the optimized backend
   dispatch path, if any, because runtime parity often depends on reproducing
   fallback semantics before replacing them with fused kernels.
   ```

7. Require generated/modular source notes.

   Suggested prompt addition:

   ```text
   If a modeling file is generated from a modular source file, inspect both when
   practical and state which one is authoritative for future source edits.
   ```

8. Require variant exclusion notes.

   Suggested prompt addition:

   ```text
   If representative checkpoints share a model_type but the config/source warns
   they should use another class for correct behavior, mark them as out-of-scope
   for the current report or create a separate follow-up target.
   ```

9. Require task/head scope.

   Suggested prompt addition:

   ```text
   State the primary runtime target for the report, such as base encoder,
   masked LM, causal LM, seq2seq LM, image classification, or multimodal
   generation. For every other head implemented in the source, mark it as
   required, optional, or deferred for the stated target.
   ```

10. Expand tokenizer-coupled text encoder prompts.

   Suggested prompt addition:

   ```text
   For text encoders, document special-token layout, segment/token type IDs,
   default position IDs, padding side, and which of those enter the GPU graph.
   ```

11. Require multimodal processor tensor contracts.

   Suggested prompt addition:

   ```text
   For multimodal models, inspect processor/preprocessor configs and document
   the exact tensors produced for runtime, including pixel/input feature shapes,
   grid metadata, modality token type IDs, placeholder tokens, packed sequence
   descriptors, and cu_seqlens-style metadata.
   ```

12. Require multi-stage decomposition for multimodal models.

   Suggested prompt addition:

   ```text
   For multimodal or multi-stage models, separate CPU/data-pipeline work,
   independently cacheable encoders/projectors, prefix construction, prefill,
   and decode. Identify which stages can be validated and optimized separately.
   ```

13. Tighten conv-to-GEMM rewrite requirements.

   Suggested prompt addition:

   ```text
   For convolution-to-linear/GEMM rewrites, include layout-aware preconditions,
   activation flatten order, weight flatten/permutation, bias handling, dynamic
   guards, and failure cases. If preprocessing already emits flattened windows,
   describe the specialized pattern separately from the general ConvNd lowering.
   ```

14. Label benchmark provenance.

   Suggested prompt addition:

   ```text
   If benchmark observations or prior measurements are included, label their
   provenance and separate them from source-derived facts. Prefer probes that
   isolate processor, encoder/projector, prefill, decode, logits, and cache
   costs.
   ```

15. Require audio feature-extractor contracts.

   Suggested prompt addition:

   ```text
   For audio models, inspect feature extractor/preprocessor configs and document
   sampling rate, mono/stereo expectations, chunk length, padding/truncation,
   FFT/hop/window settings, mel/bin counts, normalization/clamp math, output
   shape, and whether feature extraction is CPU/data-pipeline or GPU/runtime.
   ```

16. Split generation-controller behavior from the core graph.

   Suggested prompt addition:

   ```text
   For generation-heavy models, document forced decoder IDs, language/task
   prompts, suppress-token processors, timestamp processors, no-speech or
   long-form controls, assistant/speculative paths, and which pieces can be
   stubbed for first integration.
   ```

17. Require attention math-order notes.

   Suggested prompt addition:

   ```text
   Document source-specific attention scaling, casting, masking, softmax, and
   dropout order before proposing fused attention kernels.
   ```

### Batch 15 subagent reports

- DeepSeek V2, DeepSeek V3, Qwen2-MoE, Longformer, and BigBird were produced by parallel subagents and coordinator-reviewed. One prompt refinement was applied: sparse/local/block attention families must document exact pattern metadata, padding/bucket admission, mask conventions, dense fallback thresholds, and output-attention reconstruction separately from hidden-state fast paths.
- DeepSeek V2 covered MLA with complex YaRN RoPE, q_lora/direct Q split, compressed KV plus shared RoPE key, source-expanded cache K `[B,H,T,192]` and V `[B,H,T,128]`, Flash value pad/crop, group-limited top-6 MoE, and ungated shared expert.
- DeepSeek V3 covered native DeepSeek V3 MLA/MoE: q/kv low-rank projections, interleaved RoPE, asymmetric QK/V attention, FP8 metadata as loader concern, 256 routed experts/top-8 with shared expert, and V3.2 `model_type` exclusion.
- Qwen2-MoE refreshed Qwen sparse MoE with top-4 over 60 experts plus a gated shared expert, effective qkv_bias default, sparse/dense layer schedule, MHA official configs but GQA-capable source, and differences from Qwen3-MoE/Mixtral.
- Longformer covered encoder-only local+global sliding attention: pad-to-window, dtype-extreme mask encoding for masked/local/global states, separate global Q/K/V projections, task auto global-mask helpers, and no cache/generation path.
- BigBird covered block-sparse encoder attention: dense fallback threshold at `(5 + 2 * num_random_blocks) * block_size`, block padding/unpadding, first/last global blocks, local+random block plan, eval zero-random behavior, optional dense attention-prob reconstruction, and CausalLM as a separate full-attention follow-up.

### Batch 16 subagent reports

- StarCoder2, GPTBigCode, Reformer, XLNet, and LongT5 were produced by parallel subagents and coordinator-reviewed. Prompt refinements were applied for hidden-state memory caches, deterministic admission of randomized/hash/sort attention, ignored historical config fields, hash/sort/bucket operator categories, and matching performance probes.
- StarCoder2 covered modern code-generation decoder variants with biased projections, LayerNorm, RoPE/GQA, sliding-window cache required by production checkpoints, `rope_parameters` normalization from legacy `rope_theta`, tied-vs-untied LM head differences across sizes, and source-ignored historical config fields.
- GPTBigCode covered StarCoder/SantaCoder-style MQA with `c_attn: H -> H + 2D`, compact K/V cache `[B,1,T,D]`, learned absolute positions, backend-specific causal mask behavior, non-MQA fallback, cross-attention rejection for MQA, and gated/checkpoint-access limitations for representative large configs.
- Reformer covered local and LSH attention, reversible residual streams, axial position embeddings, random rotations, stable bucket sorting/unsorting, multi-hash combine, one-token decode constraints, and `ReformerDynamicCache` storing buckets plus hidden states rather than standard K/V.
- XLNet covered Transformer-XL-style hidden-state `mems`, AC/BD/EF relative attention terms, `rel_shift_bnij`, optional two-stream attention with `target_mapping`, dense `perm_mask` generation flow, layer-input memory update rules, and `mem_len=None` growing behavior.
- LongT5 covered encoder local and transient-global sparse attention, local block padding to `local_radius + 1`, per-block aggregate side K/V and side relative bias, decoder reuse of dense T5 attention/cache semantics, official checkpoint overrides for gated-GELU and untied LM head, and transient-global edge cases such as zero global blocks.

### Batch 17 subagent reports

- Gemma2, Qwen3, OLMo, OLMo2, and ModernBERT were produced by parallel subagents and coordinator-reviewed. One prompt refinement was applied: bounded math and source-specific projection/post-score transforms must be documented with exact placement, dtype, config enablement, and ordering relative to RoPE, cache, masking, softmax, and logits.
- Gemma2 covered alternating sliding/full attention, hybrid cache where sliding layers retain `sliding_window - 1` previous tokens, attention scaling by `query_pre_attn_scalar`, attention tanh softcap before mask addition, final logit softcap, GQA production shapes, and differences from Gemma/Gemma3.
- Qwen3 covered dense decoder Q/K per-head RMSNorm before RoPE/cache update, GQA with 8 KV heads in official dense configs, inactive-but-source-supported sliding-window fields, effective post-init `layer_types`, varying `rope_theta`, dense SwiGLU-only path, and tied-vs-untied LM head variation.
- OLMo covered parameter-free fp32-accumulating LayerNorm, SwiGLU, RoPE, optional `clip_qkv` before reshape/RoPE, tied-vs-untied embedding differences, post-RoPE K cache storage, and explicit separation from OLMo2.
- OLMo2 covered learned RMSNorm, Q/K post-projection RMSNorm, post-sublayer norm placement before residual adds, MHA/GQA split across official versions, bias-free projections, untied LM head, and modular-vs-generated source provenance.
- ModernBERT covered encoder-only full/sliding bidirectional attention schedules derived from `global_attn_every_n_layers`, RoPE despite ignored `position_embedding_type`, packed QKV order, FlashAttention unpad/varlen and packed-position inference, backend-specific `output_attentions` behavior, and tied MLM decoder weights.

### Batch 18 subagent reports

- Llama4, Qwen3-Next, GPT-OSS, GraniteMoEHybrid, and GLM4-MoE were produced by parallel subagents and coordinator-reviewed. Prompt refinements were applied for explicit projection dimensions that differ from hidden size, packed projection split order, and source-coupled quantized/packed weight metadata with dequant/provider fallback planning.
- Llama4 covered early-fusion multimodal MoE: image features inserted into text embeddings with `masked_scatter`, GQA, mixed full/chunked masks, source-defined RoPE/NoPE behavior, Q/K L2 norm, NoPE query temperature scaling, packed expert BMM weights, sigmoid top-k routing, NCHW vision tiling, Unfold+Linear patch embedding, 2D complex RoPE, pixel shuffle, and open-mirror provenance because official configs were gated.
- Qwen3-Next covered a hybrid decoder with 36 gated-delta linear-attention layers plus 12 full GQA layers in default configs, a gated query projection on full attention, partial RoPE, Gemma3-style `(1 + weight)` RMSNorm, fixed conv states `[B,8192,4]`, recurrent states `[B,32,128,128]`, and 512-expert top-10 MoE with a shared expert.
- GPT-OSS covered decoder-only MoE with GQA, alternating sliding/full caches, YaRN half-split RoPE, learned attention sink logits included in softmax normalization, top-4 router selected-softmax, expert gate/up interleaving, official MXFP4 expert-only `_blocks`/`_scales` metadata, and the key trap that attention width can be 4096 while `hidden_size` is 2880.
- GraniteMoEHybrid covered text-only hybrid Mamba2-style state-space plus causal GQA layers, per-layer mixed cache manifests, mostly `position_embedding_type="nope"` production configs, config-dependent MoE where some checkpoints disable routed experts, Granite-specific multipliers, and Mamba scan/decode as the dominant implementation risk.
- GLM4-MoE covered decoder-only GQA MoE with partial RoPE, compact post-RoPE K cache, production Q/K/V biases, Q/K norm differences between full and Air configs, dense-before-MoE schedules, fp32 sigmoid router logits, optional grouped routing, normalized/scaled top-k weights, packed expert gate/up weights, and FP8/compressed-tensors loading as a separate provider concern.

### Batch 19 subagent reports

- Cohere, Cohere2, Granite, GraniteMoE, and GLM4 were produced by parallel subagents and coordinator-reviewed. One prompt refinement was applied: family-specific scalar multipliers such as embedding, attention, residual, and logit scaling must be documented as source-specific math with exact placement.
- Cohere covered parallel residual decoder blocks where one LayerNorm feeds both attention and SwiGLU MLP, config-dependent MHA/GQA, post-RoPE compact KV cache, interleaved even/odd RoPE rather than Llama half-split, required `logit_scale`, tied embeddings, and gated production config provenance.
- Cohere2 covered a separate hybrid sliding/full decoder family with 3-sliding/1-full legacy `sliding_window_pattern=4`, bounded sliding cache layers, parallel residual blocks, biasless LayerNorm, tied embeddings, final `logits * logit_scale`, GPT-J-style interleaved RoPE, and the absence of Cohere Q/K norms.
- Granite covered dense Llama-like causal decoding where parity depends on `embedding_multiplier`, `attention_multiplier`, `residual_multiplier`, and `logits_scaling`; production GQA cache `[B,KVH,T,D]`; mandatory dense RoPE; and rejection of older `model_type=llama` Granite-code checkpoints from the native `granite` path.
- GraniteMoE covered sparse-MoE Granite decoding with the same Granite multipliers, GQA with post-RoPE K cache, top-k over raw router logits followed by selected softmax, dynamic expert counts, sort/regroup, per-expert GEMMs, weighted `index_add`, and explicit separation from `granitemoehybrid` NoPE/Mamba behavior.
- GLM4 covered dense text-only decoder attention with separate Q/K/V projections, GQA, partial interleaved RoPE, packed SwiGLU, post-attention/post-MLP sandwich RMSNorms, no Q/K norm in native GLM4, checkpoint-varying Q/K/V bias and KV-head counts, and compact post-RoPE cache semantics.

### Batch 20 subagent reports

- Qwen2.5-VL, Qwen2.5-Omni, LLaVA-OneVision, LLaVA-NeXT-Video, and Gemma3n were produced by parallel subagents and coordinator-reviewed. Prompt refinements were applied for staged composite models with optional output modalities and codec/diffusion/vocoder generation operator surfaces.
- Qwen2.5-VL covered packed flattened patch rows from the processor, `mm_token_type_ids`, `image_grid_thw`, `video_grid_thw`, `second_per_grid_ts`, M-RoPE position construction, vision `window_index` reorder/reverse with selected full-attention blocks, strict placeholder counts, `masked_scatter` visual stitching, and Qwen2-style biased GQA decoder cache.
- Qwen2.5-Omni covered staged multimodal generation: audio/vision encoders plus thinker text decoder as first target, optional talker decoder for speech-code tokens, optional fp32 token2wav DiT and BigVGAN waveform synthesis, M-RoPE `rope_deltas`, audio chunked packed attention, vision local/full packed attention, and batch-size-1 audio output.
- LLaVA-OneVision covered SigLIP vision plus 2-layer projector plus Qwen2 decoder, AnyRes image packing, placeholder expansion, newline tokens, `masked_scatter` stitching, distinct video path with per-frame projection and 27x27-to-14x14 bilinear pooling, and OneVision-specific differences from LLaVA-NeXT/VipLLaVA.
- LLaVA-NeXT-Video covered CLIP vision plus delegated causal LM, separate image/video placeholder IDs, inherited AnyRes image tiling/unpadding/newline packing, independent frame encoding, default CLS removal, video spatial pooling before projection, strict token/feature count checks, and text-only decode after multimodal prefill.
- Gemma3n covered a distinct multimodal generation family with AltUp, LAUREL, per-layer embeddings, activation sparsity, trailing-layer KV sharing, MobileNet/Timm delegated vision, USM-style audio encoder, SSCP Conv2d projection, local relative audio attention with logit softcap, hard/soft audio/image placeholder ranges, and separate nested-backbone ownership guidance.

### Batch 21 subagent reports

- Aya Vision, Cohere2 Vision, SmolVLM, Ovis2, and Janus were produced by parallel subagents and coordinator-reviewed. No prompt change was needed; existing clauses for delegated submodules, tokenized image/code paths, staged multimodal targets, packed layouts, and indexed embedding stitch covered the new findings.
- Aya Vision covered GotOcr2-style image tiling, structured BOI/EOI/tile tokens, patch-only `masked_scatter`, Aya-specific SigLIP pixel-shuffle SwiGLU projector, different 8B/32B delegated decoder families (`cohere2` vs `cohere`), different image token ids, and Aya-owned LM head without delegated Cohere logit scaling.
- Cohere2 Vision covered SigLIP-style vision plus pixel-shuffle/SwiGLU projector plus Cohere2 decoder, tile-based packing up to 12 crops plus thumbnail, 256 projected embeddings per tile, BOI/EOI/line-break tokens left as text tokens, source use of `siglip_vision_model` despite SigLIP2 card wording, and mirror-specific MLX quantization.
- SmolVLM covered native SmolVLM2-oriented source, historical SmolVLM checkpoints that still route as `idefics3`, vision encoder plus pixel-shuffle connector plus indexed `<image>` stitch, square patch-sequence assumptions, processor-coupled image splitting/packing, delegated Llama cache, and independently cacheable image features.
- Ovis2 covered converted in-library `thisisiron/Ovis2-*` scope, rejection of original remote-code `model_type="ovis"` repos from native Ovis2, local vision tokenizer with patch Conv2d, learned positions, noncausal ViT, hidden-stride packing, visual-vocab softmax plus embedding table matmul, 256 atom tokens per 448 tile, and mixed `masked_scatter`/indexed indicator-token replacement.
- Janus covered native `deepseek-community/Janus-Pro-*` scope, rejection of older remote-code-era configs, image-understanding stitch of 576 vision embeddings, image generation with CFG batch doubling and separate 16384-way VQ-code head, VQ codebook ids rather than tokenizer vocab ids, static cache sizing for 576 generated codes, and NCHW-sensitive vision/VQ paths.

### Batch 22 subagent reports

- Video-LLaVA, Aria, BridgeTower, ChineseCLIP, and AltCLIP were produced by rolling subagents and coordinator-reviewed. No prompt change was needed; existing clauses covered video placeholder packing, learned-query cross-attention projectors, dual-encoder contrastive heads, no-generation/cache absence, and layout guards.
- Video-LLaVA covered separate image/video CLIP towers with shared projector, `[B,F,C,H,W] -> [B*F,C,H,W]` video flattening, video CLS retention versus image CLS drop, `257 * frames` video placeholder counts, strict `masked_scatter`, delegated LLaMA/Vicuna cache, and differences from LLaVA-NeXT-Video.
- Aria covered Idefics3-style vision plus learned-query cross-attention projector plus AriaText MoE decoder, historical remote-code config boundaries, 490/980 crop sizes mapping to 128/256 projected tokens, image `masked_scatter`, and MoE router/top-k/sort/expert GEMM as the dominant text runtime risk.
- BridgeTower covered non-generative dual-stream vision/text fusion with six paired cross-modal layers, `pixel_mask` overwrite to all-ones in the fusion path, stale config keys ignored by source, base/large patch/token differences, ITM/MLM/contrastive heads, and no KV-cache requirement.
- ChineseCLIP covered ViT-backed dual-encoder retrieval, NCHW image preprocessing, CLS-token text pooling from `last_hidden_state[:,0,:]`, projection heads, L2 normalization, `exp(logit_scale)`, `[B_text,B_image]` logits orientation plus transpose, and RN50 as a separate non-native follow-up.
- AltCLIP covered XLM-R/RoBERTa-style bidirectional text encoder plus ViT image encoder, first-token text pooling rather than CLIP EOT pooling, top-level `projection_dim`, m18 wider/deeper vision/projection variation, optional vision position interpolation, NCHW patch embedding, and no generation/KV-cache path.

### Batch 23 subagent reports

- CodeLlama, CodeGen, BioGPT, BitNet, and Bamba were produced by rolling subagents and coordinator-reviewed. No prompt change was needed; existing clauses covered delegated source behavior, tokenizer-side code infill, packed QKV split-order traps, odd attention head counts, source-coupled quantized/packed linear providers, and hybrid state-cache manifests.
- CodeLlama covered lack of distinct model/config implementation at the pinned commit, neural delegation to `llama`, tokenizer-side infill prompt packing, 7B/13B full MHA versus 34B/70B GQA, long-context RoPE config differences, post-RoPE K cache, and clickable gated Meta CodeLlama links.
- CodeGen covered GPT-J-like decoder with parallel residuals, hardcoded `mp_num=4` QKV packing and `query, value, key` split order, partial GPT-J-style RoPE, MHA-only cache `[B,H,T,D]`, and source-ignored historical config fields.
- BioGPT covered decoder-only learned absolute-position LM with offset `+2`, scaled token embeddings, pre-norm blocks, biased Q/K/V/O, ungated GELU MLP, tied bias-free LM head, Moses+BPE biomedical tokenizer coupling, valid non-power-of-two head count `25`, and 401 Microsoft task fine-tune links.
- BitNet covered Llama/Gemma-like decoder with attention-output RMSNorm before `o_proj`, gated `relu2 * up` MLP plus intermediate RMSNorm, GQA `20q/5kv`, quantization injected by Transformers BitNet quantizer rather than modeling source, separate offline/online `autobitlinear`, packed `BitLinear`, and GGUF `i2_s` provider contracts.
- Bamba covered 32-layer hybrid decoder with attention layers `[9,18,27]` and 29 Mamba2-style layers, attention cache `[B,8,T,128]`, Mamba conv state `[B,8448,4]`, recurrent state `[B,128,64,128]`, ignored historical Mamba config flags, partial RoPE over 64/128 head dims, FP8 compressed-tensors provider scope, and Mamba2 mixer/kernel risk.

### Batch 24 subagent reports

- DeBERTa, DistilBERT, ConvBERT, CamemBERT, and XLM-RoBERTa were produced by rolling subagents and coordinator-reviewed. One prompt refinement was applied: dynamic/local convolutional attention with generated kernels must document kernel-generation, local-window extraction, padding/alignment, softmax axis, mask interaction, temporary shapes, and layout guards separately from static ConvNd lowering.
- DeBERTa covered v1 disentangled relative attention with c2p/p2c gather bias, production `relative_attention=true` despite source default false, q/v biases, custom fp32-accumulating `DebertaLayerNorm`, position-biased-input differences, no generation cache, and non-powerful tiny-random config traps.
- DistilBERT covered BERT simplifications: no token type embeddings, no pooler in the base model, learned or optional sinusoidal-initialized position embeddings, post-norm blocks, ignored historical `output_past`/`hidden_act` fields, no cache, and a clickable auth-style `google/distilbert-base-uncased` config gap.
- ConvBERT covered reduced-head dense attention plus dynamic local convolution: effective heads from `num_attention_heads // head_ratio`, half-width attention heads, `SeparableConv1D`, per-token generated convolution kernels, `unfold` local windows, grouped FFN `GroupedLinearLayer`, optional `embedding_size -> hidden_size` projection, and no primary cache despite decoder branches.
- CamemBERT covered generated source from modular RoBERTa inheritance, SentencePiece/Unigram tokenizer coupling, RoBERTa/fairseq padding-aware position ids with 514 rows, production all-zero token type table, primary encoder/MLM task heads, and optional CausalLM/cache as a separate non-primary branch.
- XLM-RoBERTa covered the same RoBERTa-style encoder math with multilingual Unigram tokenizer contracts, 250002-vocab embedding/LM-head cost, production all-zero token type table, masked-position-only logits as a practical rewrite, optional CausalLM/cache separation, and clickable XL/XXL 401 config gaps.

### Batch 25 subagent reports

- Bark, CLVP, Dia, CSM, and Cohere ASR were produced by rolling subagents and coordinator-reviewed. One prompt refinement was applied: audio preprocessors that split or pack one user sample into multiple model examples must document split policy, mapping metadata, and decode/reassembly rules separately from the model graph.
- Bark covered staged text-to-audio generation with semantic, coarse, and fine GPT-like submodels; codebook-specific logits processors; voice history prompt tensors; causal caches for semantic/coarse only; fine bidirectional generation; EnCodec decode as a later composed stage; and no-layout guards around fine codebook axes.
- CLVP covered Tortoise-style text/speech contrastive scoring plus optional conditional speech-token generation: English number normalization, log-mel feature extraction, conditioning Conv1d/GroupNorm/self-attention, partial RoPE over q/k/v in encoders, GPT-2-style decoder `Conv1D` weights, L2-normalized projections, `exp(logit_scale)`, and audio NCL layout guards.
- Dia covered native HF Dia TTS: byte text encoder plus 9-channel DAC-code decoder, multi-channel embedding offset-sum, encoder/decoder projection widths larger than hidden size, decoder GQA self-cache plus cross-cache, default delay pattern `[0,8,9,10,11,12,13,14,15]`, CFG and EOS delay processors, channel-flattened logits, and DAC decode as a separate stage.
- CSM covered gated official `sesame/csm-1b` access, native mirror configs, generated/modular source boundary, two-level backbone/depth-decoder generation, Mimi encode/decode composition, 32-codebook embedding offset-sum, fixed 31-token depth decoder schedule, codebook-position heads, GQA/RoPE caches, and chat-template/audio placeholder expansion.
- Cohere ASR covered gated official `CohereLabs/cohere-transcribe-03-2026`, open ONNX mirror dimensions, Parakeet Fast Conformer encoder composition, energy-based long-audio chunking and reassembly metadata, deterministic dither/preemphasis/STFT/log-mel normalization, decoder prompt tokens for language/punctuation, encoder relative-position attention bias, and decoder self/cross `EncoderDecoderCache`.

### Batch 26 subagent reports

- ConvNeXtV2, DeiT, Depth Anything, DINOv2-with-registers, and MobileViT were produced by rolling subagents and coordinator-reviewed. No prompt change was needed; existing layout, nested-backbone, generated-source, and local layout-translation clauses covered guarded NHWC regions, public NCHW output contracts, register-token ABI, interpolation contracts, and MobileViT patch unfold/fold.
- ConvNeXtV2 covered NCHW model boundaries with NHWC only inside residual blocks, hard-coded internal LayerNorm eps `1e-6`, V2 GRN math, depthwise 7x7 blocks, Conv2d-to-GEMM stem/downsample rewrites, guarded NHWC residual-block optimization, and public NCHW backbone output materialization.
- DeiT covered generated/modular source boundary, native distilled checkpoints versus legacy `model_type="vit"` DeiT-named repos, CLS+distillation token order, `num_patches+2` position tables, bicubic interpolation with two special rows, teacher head averaging, and patch Conv2d-to-GEMM guarded rewrites.
- Depth Anything covered nested DINOv2 `AutoBackbone` with `reshape_hidden_states=False`, reassemble sequence slice/reshape/permute, ConvTranspose2d factors 2/4 and stride-2 Conv2d factor 0.5, top-down fusion with mixed bilinear `align_corners` contracts, relative versus metric depth head, and postprocess bicubic resize.
- DINOv2-with-registers covered generated/modular source boundary, four register tokens inserted after CLS and before patch tokens, positions for CLS+patch only, patch-token slices beginning at `1 + num_register_tokens`, hardcoded bicubic float32 interpolation with `antialias=True`, classifier patch mean excluding registers, optional SwiGLU giant path, and backbone NCHW ABI.
- MobileViT covered a CNN/Transformer hybrid with NCHW-only source contracts, Conv2d+BatchNorm2d+SiLU inverted residual blocks, 2x2 patch-token unfold/fold with bilinear repair for non-divisible feature maps, standard noncausal MHA over `[B*patch_area,Np,H]`, guarded NHWC conv-region opportunities, `output_stride` stride/dilation changes, and DeepLabV3 ASPP segmentation heads.

### Batch 27 subagent reports

- EfficientNet, ResNet, PoolFormer, LeViT, and MobileViTV2 were produced by parallel subagents and coordinator-reviewed. No prompt change was required; existing graph-rewrite, layout-guard, local-kernel, and source-specific math clauses covered the new findings. EfficientNet-Lite configs returned 401 and were recorded with clickable HF links in the report.
- EfficientNet covered MBConv blocks with explicit asymmetric `correct_pad`, source-specific `depthwise_padding` behavior, squeeze-excitation reduce width based on stage input channels, eval drop-connect identity, BN folding, depthwise NHWC kernels, and guarded SE/1x1 GEMM rewrites.
- ResNet covered native classification plus `ResNetBackbone` feature-map outputs, Basic versus Bottleneck residual blocks, stride-placement flags (`downsample_in_first_stage`, `downsample_in_bottleneck`), Conv+BN folding, residual-add-ReLU fusion, global average pool lowering, and shared ConvNext image-processor mapping.
- PoolFormer covered an attention-free MetaFormer with `GroupNorm(1,C)` over full NCHW sample tensors, border-sensitive `AvgPool2d(count_include_pad=False) - x` token mixing, layer-scale vectors, eval DropPath identity, no cache despite stale config fields, and guarded pooling-mixer fusion.
- LeViT covered Conv/BN/Hardswish stem to row-major token sequence, LinearNoBias+BatchNorm1d projection folding, dense image-grid attention with learned gathered 2D relative-bias tables, attention-downsample with strided query gather, teacher/classifier head averaging, and token-order preservation across any NHWC stem optimization.
- MobileViTV2 covered separable linear self-attention over unfolded patch columns `[B,D,patch_area,Npatch]`, no interpolation before unfold, official-shape divisibility assumptions, GroupNorm/Conv2d patch-domain axes, segmentation ASPP reuse, and a fused unfold + 1x1 qkv + linear-attention + fold candidate guarded by channel-axis and edge-semantics checks.

### Batch 28 subagent reports

- RegNet, PVT, CvT, FocalNet, and Perceiver were produced by parallel subagents and coordinator-reviewed. One prompt refinement was applied: query-driven cross-attention decoders and latent-attention models must document query construction, rectangular attention lengths, q/k width, value width, mask side, and cache absence separately from autoregressive decode.
- RegNet covered four-stage grouped bottleneck CNNs, per-stage group count `max(1, out_channels // groups_width)`, RegNetY SE width based on block input channels, stage-stride effects, optional classifier absence on some repos, Conv+BN folding, grouped 3x3 provider requirements, and guarded NHWC residual stage regions.
- PVT covered hierarchical patch Conv2d embeddings, learned absolute positions with source-sensitive interpolation, final-stage CLS insertion, sequence-reduction Conv2d for K/V only, rectangular noncausal attention with large Q and small K/V lengths, and strict NCHW/sequence crossing guards.
- CvT covered convolutional patch embeddings, official `dw_bn` depthwise Conv+BN Q/K/V projections, final-stage CLS handling, rectangular attention with K/V stride 2, source-specific attention scale `embed_dim ** -0.5`, rejection of non-`dw_bn` and early-CLS variants, and ConvNeXT-style preprocessing reuse.
- FocalNet covered attention-free focal modulation with channel-last projections, depthwise NCHW focal levels 3/5/7, learned gates plus global context, right/bottom patch padding, optional post-LayerNorm and modulator normalization flags, MIM PixelShuffle head, and inaccessible SimMIM/LRF links recorded as 401/404 gaps.
- Perceiver covered modality-flexible preprocessing, learned latent input cross-attention, repeated latent self-attention, task query decoders, q/k/value width asymmetry, byte-token MLM, image Fourier/learned/conv preprocessors, optical-flow and multimodal `space_to_depth`, sorted modality packing, stochastic mask deferral, and no autoregressive KV cache.

### Batch 29 subagent reports

- MobileNetV1, MobileNetV2, BiT, Data2Vec, and UPerNet were produced by parallel subagents and coordinator-reviewed. No prompt change was needed; existing clauses covered dynamic padding, source-specific weight transforms, multi-source family routing, composite backbone heads, and no-layout-translation guards. MobileNetV1 lower-width raw configs returned 401 and were recorded with clickable links.
- MobileNetV1 covered TensorFlow SAME padding, 13 depthwise-separable Conv/BN/ReLU6 blocks, depth multipliers and 1001-label ImageNet heads, checkpoint-default dropout differences, hidden-state tuple shape/order, BN folding, ReLU6 epilogues, and guarded NHWC depthwise/pointwise regions.
- MobileNetV2 covered inverted residual schedules with `make_divisible`, `finegrained_output`, dynamic TF SAME padding with dilation for segmentation output stride, projection layers without activation, segmentation head consumption of the pre-final feature map, and no-layout guards around NCHW logits/postprocess axes.
- BiT covered native ResNetV2-style `model_type="bit"` scope, weight-standardized Conv2d, GroupNorm rather than BatchNorm, preactivation versus bottleneck layer types, static/dynamic padding differences from ViT-hybrid nested configs, backbone feature-map outputs, and inference pre-standardized-weight rewrites.
- Data2Vec covered three distinct native surfaces: raw-waveform audio with Conv1d frontend, downsampled attention masks, grouped convolutional positional embeddings, CTC and x-vector heads; RoBERTa-like text with optional decoder/cache branches; and BEiT-like vision with patch Conv2d, mixed-bias Q/K/V, relative bias modes, and optional segmentation.
- UPerNet covered nested `AutoBackbone` ownership, four-level NCHW feature-map ABI, ConvNeXT/Swin official variants, PSP pooling, FPN lateral/top-down resize/add/concat, `align_corners=False` everywhere, auxiliary-head inference pruning when labels are absent, Segformer preprocessing, and guarded NHWC decode-head fusion.

### Batch 30 subagent reports

- LayoutLM, LayoutLMv2, LayoutLMv3, Donut, and TrOCR were produced by parallel subagents and coordinator-reviewed. One prompt refinement was applied: document/OCR/layout models must explicitly document OCR ownership, coordinate normalization, special-token boxes, word-to-subword box expansion, overflow image duplication, bbox guards, and where boxes feed embedding or attention-bias math.
- LayoutLM covered text-plus-layout encoder inference with no image branch, nine-way embedding fan-in, BERT versus RoBERTa tokenizer variants under `model_type="layoutlm"`, bbox width/height embedding indices, ignored cache config fields, tied MLM projection, and gated FUNSD fine-tune metadata.
- LayoutLMv2 covered Detectron2 ResNet-FPN visual tokens, 7x7 visual bbox grids, text+49 visual token concatenation, absolute 2D embeddings, 1D and 2D relative attention bias, base packed QKV with q/v bias but zero k bias, OCR/Tesseract processor coupling, and gated FUNSD/DocVQA/CORD fine-tune metadata.
- LayoutLMv3 covered single-stream text+patch encoder construction, RoBERTa-style position IDs, NCHW patch Conv2d, text then visual-CLS/patch concat, shared 1D/2D relative bias across layers, CogView softmax semantics, token-head text slicing versus QA full-sequence logits, absent native MLM/MIM heads, and gated fine-tune configs.
- Donut covered composite VisionEncoderDecoder execution with native DonutSwin encoder plus delegated MBART decoder, huge document image grids, long-axis align/resize/thumbnail/center-pad preprocessing, shifted-window attention and masks, cross-attention cache over encoder tokens, task prompt/token2json postprocess, absent processor/generation JSON files, and legacy `donut-proto` routing to Swin.
- TrOCR covered VisionEncoderDecoder OCR generation with ViT/DeiT encoder composition, TrOCR causal decoder, learned versus sinusoidal decoder positions, `cross_attention_hidden_size` avoiding wrapper projections, optional self/cross `EncoderDecoderCache`, config-default cache caveats, tied versus untied LM projection, and greedy OCR generation staging.

### Batch 31 subagent reports

- Table Transformer, Deformable DETR, DAB-DETR, GLPN, and ZoeDepth were produced by parallel subagents and coordinator-reviewed. One prompt refinement was applied: deformable/sparse-sampling attention must be documented as its own attention family with reference points, level metadata, sampling offsets, interpolation, mask order, and custom-kernel/fallback details.
- Table Transformer covered a DETR-like table detector with delegated DETR image processor, ResNet backbone contracts, object-query decoder, sine or learned 2D feature positions, query counts 15 versus 125, table-specific class heads, center-box postprocess with no NMS, and legacy versus native backbone config handling.
- Deformable DETR covered generated/modular source boundaries, multi-scale flattened memory metadata (`spatial_shapes`, `level_start_index`, `valid_ratios`), encoder/decoder multi-scale deformable attention, `grid_sample` fallback, custom kernel hook, single-scale/DC5/two-stage/box-refine variants, DETIC label-width variants, and no-NMS sigmoid top-k postprocess.
- DAB-DETR covered learned 4D anchor/refpoint queries, strict `query_dim=4`, anchor sine embeddings, query-scale and anchor-size modulation, rectangular decoder cross-attention with Q/K width 512 and V width 256, iterative bbox refinement with tied bbox predictor alias, sigmoid multi-label top-k postprocess, and gated ResNet-101 links.
- GLPN covered native MixTransformer depth estimation with overlapping patch Conv2d, spatial-reduction attention, Mix-FFN depthwise Conv2d, forced hidden-state feature maps, selective feature fusion decoder, `sigmoid * max_depth` head, resize-down-to-divisor preprocessing, bicubic postprocess, and guarded channel-last decoder/head fusions.
- ZoeDepth covered nested BEiT backbone contracts, token reassembly, DPT-style neck fusion, relative depth plus metric bin/attractor heads, softplus versus normed bin centers, conditional log-binomial depth, multi-domain patch-transformer routing with host-visible branch selection, preprocessing/postprocess padding coupling, and ignored native attractor alpha/gamma config fields.

### Batch 32 subagent reports

- OneFormer, MaskFormer, SAM2, SAM2-Video, and DepthPro were produced by parallel subagents and coordinator-reviewed. One prompt refinement was applied: stateful video/tracking/interactive segmentation models must document session state separately from KV cache, including feature caches, per-object histories, memory tensor ABI, object pointers, update order, and propagation rules.
- OneFormer covered universal segmentation with task-token BPE inputs, missing `processor_config.json` fallback to source processor defaults, nested backbone contracts, multi-scale deformable pixel decoder, task-token-conditioned query transformer, masked decoder attention, semantic/instance/panoptic postprocess, and training-only text mapper deferral.
- MaskFormer covered Swin/ResNet backbone variants, simple FPN pixel decoder, DETR-style learned query decoder over final feature maps, mask/class heads, semantic/instance/panoptic postprocess, preprocessor JSON fallback, no source use of `pixel_mask` despite processor output, and differences from Mask2Former.
- SAM2 covered the shared image promptable segmentation subgraph inside official `sam2_video` checkpoints, Hiera encoder with mixed NCHW/NHWC layout, FPN high-res feature cache, point/box/mask prompt packing including `-1` and `-10` labels, two-way mask decoder, dynamic multimask stability fallback, object-score head, and image-feature cache rewrites.
- SAM2-Video covered the stateful tracking model: session dictionaries, processed frame/feature caches, per-object prompt histories, bf16 mask-memory tensors `[4096,B,64]`, object pointers split into memory-width tokens, memory attention with 2D RoPE and pointer exclusion, memory encoder/fuser, forward/reverse propagation, and gated official config links.
- DepthPro covered the single native `apple/DepthPro-hf` checkpoint, three separate DINOv2 encoders, 1536 image preprocessing, scaled patch extraction/unfold and merge ABI, DPT-like neck/fusion, inverse-depth head, optional FOV encoder/head and focal-length postprocess, and guarded conv/deconv NHWC regions.

### Batch 33 subagent reports

- DINOv3-ViT, DINOv3-ConvNeXt, TimmWrapper, SigLIP2, and I-JEPA were produced by parallel subagents and coordinator-reviewed. One prompt refinement was applied: wrapper or bridge families whose neural body is delegated to external libraries must document external owner/version, topology-selecting args, preprocessing source, output ABI variants, weight-key mapping, and reject/fallback/allowlist policy rather than claiming a fixed operator surface.
- DINOv3-ViT covered gated official config gaps, conversion-script-derived dimensions, CLS plus register-token ordering, patch-only dynamic 2D RoPE, optional gated SiLU MLP variants, very large 7B ViT dimensions, and guarded patch Conv2d/NHWC lowering.
- DINOv3-ConvNeXt covered gated official config gaps, ConvNeXt-style NCHW boundaries with channel-last islands inside blocks, DINO token output from pooled token plus flattened final map, absent attention/cache paths, and no GRN despite ConvNeXtV2 similarity.
- TimmWrapper covered the external-timm ownership boundary, config-driven `architecture` and `model_args` dispatch, pretrained processor metadata from `config.pretrained_cfg`, output ABI variation between classifier/backbone/feature modes, and reject-by-default admission unless exact timm bodies are separately audited.
- SigLIP2 covered native NaFlex packed-patch dual encoder ABI, `model_type: siglip` named-checkpoint rerouting, dynamic position resize/pad over flattened patches, learned single-query vision pooling attention, checkpoint tokenizer/config precedence, contrastive logits orientation, and local NHWC patchify fusion guards.
- I-JEPA covered ViT-like encoder-only image features without CLS tokens, NCHW non-overlap patch Conv2d plus learned absolute patch positions, optional mask-token path, noncausal MHA over patch tokens, mean-pool classification head, dynamic bicubic position interpolation, and no-layout guards outside the local patch-lowering region.

### Batch 34 subagent reports

- ALIGN, FLAVA, ViLT, VisualBERT, and LXMERT were produced by parallel subagents and coordinator-reviewed. One prompt refinement was applied: models that consume precomputed modality features must document the external extractor boundary, feature tensor ABI, ordering/coordinate metadata, masks, cacheability, and first-integration ownership policy instead of implying a raw-image graph.
- ALIGN covered a dual-encoder contrastive model with BERT text, EfficientNet-derived NCHW vision, text-only projection to 640, image pooled features without a separate image projection, learned temperature as a divisor rather than `exp(logit_scale)`, `[text,image]` and transposed output orientation, and an inaccessible COYO raw-config gap.
- FLAVA covered independent ViT image and BERT text encoders plus a third multimodal encoder, source-required hidden-state capture before final LayerNorm, bridge projections, optional contrastive/ITM/MLM/MIM/MMM heads, separate 112x112 image-codebook preprocessing, random pretraining mask generation, and gated Facebook image/text/full-weights links.
- ViLT covered single-stream image-text encoding with NCHW patch Conv2d, pixel-mask downsampling into patch masks, learned image-position interpolation, stochastic `max_image_length` patch selection, cross-encoder retrieval rather than cached CLIP-style embeddings, historical VQA architecture aliasing, and NLVR two-image looping.
- VisualBERT covered caller-supplied visual region features, checkpoint-specific visual widths 512/1024/2048, text+visual sequence concat, visual type/position embeddings including optional image-text alignment averaging, VQA last-text-token gather, NLVR/VCR heads, and absence of processor/tokenizer/detector assets in sampled repos.
- LXMERT covered required external ROI feature and box tensors, separate language/visual encoders, bidirectional cross-modality blocks with shared cross-attention module weights per layer, rectangular attention masks, checkpoint-specific QA label widths, optional object/attribute/feature heads, and gated Graphcore mirror config links.

### Batch 35 subagent reports

- mT5, ByT5, UMT5, M2M100, and mBART were produced by parallel subagents and coordinator-reviewed. One prompt refinement was applied: multilingual or tokenizer-controlled generation models must document language-code layouts, decoder start rules, forced BOS/EOS ids, generation metadata, and tokenizer/vocab variants separately from graph ops.
- mT5 covered T5-style seq2seq with multilingual SentencePiece routing through `T5Tokenizer`, huge 250112 vocab, forced tied embeddings despite serialized `tie_word_embeddings=false`, gated-GELU FFNs, block-0 shared relative bias per stack, MHA inner widths that can differ from `d_model`, and `EncoderDecoderCache` self/cross reuse.
- ByT5 covered the in-library T5 neural route with byte-level tokenizer coupling, byte id offset 3, 125 sentinel tokens inside a 384-row embedding/LM head, tokenizer `vocab_size` versus model `config.vocab_size` mismatch, asymmetric encoder/decoder depths, gated-GELU FFNs, and tiny-vocab last-token logits opportunities.
- UMT5 covered UMT5-specific source deltas from T5/mT5: per-layer learned relative self-attention bias, decoder output scaling before the tied LM projection, ignored historical `scalable_attention`/`output_past` fields, 256384 vocab with 300 sentinels, bias-free projections, and standard self/cross `EncoderDecoderCache`.
- M2M100 covered BART-like translation with shared embeddings, sinusoidal pad-aware positions that can grow, source/target language-code prefixes, forced target-language BOS via generation controller, M2M100 versus WMT21 language/vocab variants, ReLU FFNs, cross-attention K/V reuse, and last-token-only logits rewrites.
- mBART covered learned absolute positions with offset 2, shared embedding/LM head plus `final_logits_bias`, pre-norm encoder/decoder with final layer norms, mBART-25 suffix versus mBART-50 prefix language-token layouts, inconsistent decoder-start semantics across checkpoints, forced BOS/EOS generation metadata, activation/vocab variants, and a 404 tokenizer-config gap for `facebook/mbart-large-cc25`.

### Batch 36 subagent reports

- VideoMAE, TimeSformer, ViViT, X-CLIP, and TVP were produced by parallel subagents and coordinator-reviewed. One prompt refinement was applied: video reports must document frame sampling ownership, `[B,T,C,H,W]`/`NCTHW`/`NTHWC` layout, tubelet or per-frame patch order, temporal/spatial token order, frame-count positional dependencies, pooling, and layout guard boundaries.
- VideoMAE covered video classification with `[B,T,C,H,W]` input, immediate `NCTHW` Conv3d tubelet embedding, fixed sin/cos position table, no CLS token, mean-pool versus first-patch pooling modes, optional masked reconstruction decoder deferral, `video_preprocessor_config.json` 404 compatibility notes, and Conv3d-to-GEMM guarded lowering.
- TimeSformer covered `VideoMAEImageProcessor` delegation, per-frame Conv2d patch embedding, divided space-time attention with temporal `[B*patches,T,D]` and spatial `[B*T,1+patches,D]` attention islands, learned spatial/time embeddings with nearest interpolation, gated large-checkpoint links, and strict reshape/layout guards.
- ViViT covered generated/modular source boundaries, tubelet Conv3d embedding with learned positions, the absence of paper-style factorized encoder variants in current source, historical `video_size` config normalization, processor zero-centering differences, optional spatial-only position interpolation, and NTHWC tubelet optimization guards.
- X-CLIP covered CLIP text plus frame ViT plus cross-frame message tokens, MIT temporal encoder, visual prompt cross-attention into text labels, video-conditioned text embeddings, `[B_video,B_text]` similarity orientation with transposed text logits, branch cache points, and patch/projection fusion guards.
- TVP covered video-language temporal grounding with ResNet `AutoBackbone` composition, pixel-space visual prompt injection, temporal mean-pooled visual grids, row/column visual position embeddings, learned text prompt plus fused text/visual encoder, sigmoid normalized time head with duration postprocess, ignored `use_cache`, and unsafe `framedownpad` config mismatch.

### Batch 37 subagent reports

- GOT-OCR2, Nougat, MarkupLM, LayoutXLM, and UDOP were produced by parallel subagents and coordinator-reviewed. One prompt refinement was applied: markup/DOM-structured text models must document parser ownership, node extraction, XPath/tag/subscript dictionaries, pad/unknown IDs, clamping, node-to-subword expansion, overflow mapping, and whether structure IDs feed embeddings or attention bias.
- GOT-OCR2 covered native `got_ocr2` scope versus remote-code `GOT`/`qwen2_vl` variants, SAM-like vision tower plus projector, Qwen2 delegated decoder, placeholder `<imgpad>` stitch, crop-to-patches prompt growth, official `image_seq_length=576` versus actual 1024-preprocessor projector count 256 trap, Qwen2 cache admission, and guarded vision NHWC islands.
- Nougat covered the absence of a native modeling file and composition through `VisionEncoderDecoderModel` with DonutSwin encoder plus MBART decoder, fixed 896x672 document preprocessing, final 588-token image sequence, decoder self/cross cache reuse, generation metadata normalization, tokenizer Markdown postprocess, and guarded DonutSwin window rewrites.
- MarkupLM covered encoder-only RoBERTa-like text plus XPath structure embeddings, HTML parser and caller-supplied node modes, tag/subscript pad/unknown IDs, 50-depth XPath tensors expanded to subwords, source-ignored historical tree/relative-bias fields, absent native pretraining class despite config names, and gated Microsoft SQuAD/RICO fine-tune links.
- LayoutXLM covered config/processor/tokenizer ownership without native `modeling_layoutxlm.py`, official base routing through LayoutLMv2, multilingual tokenizer and OCR metadata, bbox expansion/special boxes, no-relative-bias common configs, rejection of unsupported `layoutxlm`/`layout_xlm`/no-visual bodies, and gated official large/fine-tune links.
- UDOP covered document multimodal seq2seq with LayoutLMv3 image processor composition, OCR boxes and patch embeddings, dynamic OCR-patch gather/removal and remaining-patch concat, bbox-derived 1D/horizontal/vertical relative biases, 224 versus 512 patch-count scaling, T5-like decoder cache, relu FFN official configs, and bbox scale normalization guards.

### Batch 38 subagent reports

- D-FINE, RT-DETRv2, LW-DETR, RF-DETR, and OmDet-Turbo were produced by parallel subagents and coordinator-reviewed. One prompt refinement was applied: query-based detector reports must document query/proposal construction, anchors, reference-point updates, distributional/DFL box decoding, top-k tie sensitivity, and whether decoder layers update references iteratively or only in the final head.
- D-FINE covered HGNetV2 AutoBackbone composition, D-FINE hybrid encoder/FPN/PAN, top-k encoder proposals, V2-style deformable attention, nonuniform distributional box refinement plus LQE score correction, no-NMS RT-DETR postprocess, nano/two-level and xlarge/384-channel variants, and a gated PekingU R50VD link.
- RT-DETRv2 covered generated/modular source boundaries, ResNet backbone variants, AIFI selected-level dense attention, FPN/PAN, V2 deformable cross-attention with `decoder_offset_scale` and `default` versus `discrete` methods, iterative box refinement, focal top-k postprocess without NMS, and accessible PekingU configs.
- LW-DETR covered a custom ViT patch/window backbone, same-resolution selected stages, scale-dependent projector paths, mixed top-k query selection plus learned references, Q/V-biased but K-bias-free backbone attention, one/two-level deformable attention, final-only reference refinement, no-NMS sigmoid top-k postprocess, and accessible AnnaZhang configs.
- RF-DETR covered DINOv2-like window/global backbone variants, processor/backbone image-size mismatch with position interpolation, C2F projector, proposal bridge top-k sensitivity, one-level deformable cross-attention, DETR softmax/no-object postprocess without NMS, optional segmentation head deferral, and accessible stevenbucaille configs.
- OmDet-Turbo covered open-vocabulary detection with CLIP class/task text embeddings, `classes_structure` dynamic class ABI, timm/Swin AutoBackbone delegation, OmDet neck/FPN/PAN, prompt/class fusion, top-k proposals, multi-scale deformable decoder, cosine class similarity, postprocess with threshold/top-k/batched NMS, and raw `.pth` original repo lacking HF config.

### Batch 39 subagent reports

- TimeSeriesTransformer, Informer, Autoformer, PatchTST, and PatchTSMixer were produced by parallel subagents and coordinator-reviewed. One prompt refinement was applied: time-series reports must document forecasting ABI, lag/history requirements, scaler and distribution-head contracts, sampling behavior, and patch/channel axes separately from text generation.
- TimeSeriesTransformer covered probabilistic encoder-decoder forecasting with `past_values`/`future_values`, observed masks, lagged subsequences, static/time features, mean/std/no-op scalers, StudentT/Normal/NegativeBinomial heads, `num_parallel_samples` batch expansion, and source generation that recomputes full decoder prefixes despite cache-capable layers.
- Informer covered the same forecasting ABI plus ProbSparse attention with random key sampling and top-u query selection, encoder distillation through Conv1d/BatchNorm/ELU/MaxPool, strict `attention_type="prob"` admission versus dense fallback, full-prefix generation without cache use, and a gated `hf-internal-testing/tiny-random-InformerModel` config gap.
- Autoformer covered moving-average decomposition with edge-repeat padding, FFT AutoCorrelation with top-k delay aggregation, trend/seasonality decoder accumulation, probabilistic heads/scalers, full-horizon decode without cache, and guarded FFT/roll/gather lowering candidates.
- PatchTST covered encoder-only patch forecasting with `[B,T,C] -> [B,C,N_patches,patch_length]`, shared versus per-channel embeddings/projections, optional CLS/channel attention, deterministic versus distribution heads, and a gated original `namctin/patchtst_etth1_forecast` config gap with Granite as the accessible representative fallback.
- PatchTSMixer covered attention-optional MLP-mixer time-series execution, temporal unfold patching, common versus mix-channel modes, gated Linear+Softmax axis blocks, forecast/pretrain/classification/regression/distribution heads, and the source/doc mismatch where `scaling=True` is treated as standard scaling.

### Batch 40 subagent reports

- GLM-OCR, Florence2, FastVLM, DeepSeek-VL, and ColPali were produced by parallel subagents and coordinator-reviewed. One prompt refinement was applied: multimodal reports must distinguish broad source scatter calls from processor-guaranteed bounded placeholder-stitch patterns that can lower to indexed or prefix copy with guards.
- GLM-OCR covered OCR generation with flattened GLM46V patch features, `image_grid_thw`, `mm_token_type_ids`, multimodal RoPE/`rope_deltas`, packed vision attention with `cu_seqlens`, explicit `head_dim`, placeholder count validation, and local-only NHWC vision rewrites.
- Florence2 covered task-token-driven vision-language generation with NCHW Conv2d/depthwise vision stages, alternating NCHW and NHWC/token window layouts, custom grouped channel attention, processor-generated prefix copy as a safer replacement for `masked_scatter`, 1000-bin box dequantization, OCR/polygon parsing, and no-NMS structured postprocess.
- FastVLM covered a delegated `timm_wrapper` FastViT vision tower plus Qwen2 decoder, public 0.5B/1.5B/7B configs, exact `fastvit_mci3` allowlist needs, NCHW feature flatten/permute layout guards, Qwen2 GQA/RoPE cache requirements, and strict image-placeholder indexed-copy guards.
- DeepSeek-VL covered plain `deepseek_vl` scope versus separate `deepseek_vl_hybrid`, SigLIP fixed-384 vision path, projector and image-token stitch, Qwen-style decoder cache, dynamic-image/SigLIP interpolation deferrals, and a checked placeholder-offset copy alternative to general `masked_scatter`.
- ColPali covered document retrieval rather than generation: composed PaliGemma/SigLIP/Gemma dependencies, separate query/page embedding passes, prefix-aware `token_type_ids`, MaxSim score orientation `[queries,passages]`, PEFT adapter and ColPali2 routing guards, and NCHW patch-region semantics with only guarded NHWC optimization.

### Batch 41 subagent reports

- LED, Pegasus, Pegasus-X, BigBird-Pegasus, and BlenderBot were produced by rolling subagents and coordinator-reviewed. No prompt change was needed; the existing sparse/long-context attention, generation-controller, cache, and layout-guard clauses covered the new findings.
- LED covered Longformer-style local/global encoder attention with runtime padding to attention-window multiples, dynamic global-token packing/scatter, decoder self and cross caches, and admission bounds for source length, padded length, window, and max global tokens.
- Pegasus covered dense seq2seq generation with frozen non-interleaved sinusoidal position tables, LayerNorm-heavy encoder/decoder blocks, self-cache append plus cross-cache reuse, and no channel-last layout opportunity for text-only tensors.
- Pegasus-X covered custom block-local-plus-global encoder attention, learned global tokens, block-size padding, optional half-block staggering, tokenizer truncation mismatch versus 16k model positions, disabled SDPA routing, and decoder self/cross cache requirements.
- BigBird-Pegasus covered encoder sparse/full dispatch at the default `S <= 704` threshold, deterministic all-zero eval random plans, global/local/random sparse regions, bias-free attention projections with biased FFNs, and decoder `EncoderDecoderCache` parity.
- BlenderBot covered full-size conversation generation with no-offset learned positions, pre-norm block ordering, fp16 encoder clamp, tied LM head plus `final_logits_bias`, generation-controller defaults, and explicit routing away from `model_type=blenderbot-small`.

### Batch 42 subagent reports

- BlenderBot Small, Marian, FSMT, PLBart, and MVP were produced by rolling subagents and coordinator-reviewed. No prompt change was needed; existing tokenizer/language-control, cache, ignored historical fields, and layout-guard clauses covered the reusable findings.
- BlenderBot Small covered post-norm blocks and embedding LayerNorms distinct from full BlenderBot, official `vocab_size=54944` and `scale_embedding=True` overrides, seq2seq self/cross cache ABI, beam reorder, tied LM head, and host generation controls.
- Marian covered translation generation with pad-token decoder start, forced EOS and pad suppression, tokenizer-driven target language prefixes, frozen sinusoidal positions, dense encoder/decoder/cross attention, LayerNorm, and cross-attention K/V reuse.
- FSMT covered dual source/target vocab and `langs` tokenizer ABI, internal `[T,B,C]` layout guarded behind a batch-major public ABI, decoder-owned `output_projection` logits, optional tied embeddings as aliasing contracts, and WMT19 config/default differences.
- PLBart covered code/text seq2seq with language-code suffix metadata, `plbart_shift_tokens_right`, fairseq/SentencePiece ID alignment, post-norm blocks, 404 tokenizer/generation sidecars on reachable official repos, and gated `uclanlp/plbart-multi_task`.
- MVP covered BART-like seq2seq plus optional prompts, source-version-specific decoder-prompt behavior, prompt K/V precompute and mask guards, optional sequence-classification/QA heads, and self/cross cache reuse.

### Batch 43 subagent reports

- BertGeneration, EncoderDecoder, VisionEncoderDecoder, SpeechEncoderDecoder, and VisionTextDualEncoder were produced by rolling subagents and coordinator-reviewed. No prompt change was needed; existing wrapper/delegated-library, non-generation, multimodal processor, audio preprocessor, cache, and admission-policy clauses covered the findings.
- BertGeneration covered encoder versus decoder config traps, the requirement that decoder generation needs `is_decoder=True`, cross-attention needing both `is_decoder` and `add_cross_attention`, `chunk_size_feed_forward` deferral, tensor-valued `logits_to_keep` deferral, and no layout translation for text sequences.
- EncoderDecoder covered generic wrapper admission: allowlist exact encoder/decoder/head/cache-width combinations, wrapper-owned `enc_to_dec_proj` only when widths differ and decoder does not own `cross_attention_hidden_size`, ignored historical `tie_encoder_decoder`, and clear generation-controller boundaries.
- VisionEncoderDecoder covered exact pair admission for ViT/DeiT/TrOCR/GPT-2-style combinations, rejecting encoders with LM heads, processor ABI composition, optional bridge projection, delegated decoder cache ownership, and NHWC rewrites only inside audited vision encoders.
- SpeechEncoderDecoder covered audio encoder plus text decoder composition, Wav2Vec2/XLS-R feature extractor as CPU/data-pipeline boundary, strict `encoder.hidden_size`/`output_hidden_size`/decoder width guards, mBART language-control metadata, and backend dispatch only when both delegated bodies support it.
- VisionTextDualEncoder covered dual-encoder contrastive wrapper ownership: stable rank-2 `pooler_output` from delegated encoders, wrapper projections, full-forward L2 normalization versus projected unnormalized `get_*_features`, `[B_text,B_image]` similarity orientation, and compositional image/tokenizer processor ABI.

### Batch 44 subagent reports

- DPR, BROS, LiLT, LUKE, and CANINE were produced by rolling subagents and coordinator-reviewed. No prompt change was needed; existing document/layout, non-generation, custom attention, and preprocessing-boundary clauses covered the reusable findings.
- DPR covered BERT-derived retrieval encoders, optional `projection_dim`, explicit external similarity orientation `Q @ C.T`, rejection of decoder/cache configs, cached context embeddings, and `DPRReader` staging due to tokenizer-owned span postprocess.
- BROS covered document text/layout encoding with query-dependent relative bbox attention scores, source-derived bbox sinusoid dimensions, bbox shape/range guards, OCR and word-to-subword boxes as data-pipeline work, and rejection of absent in-source document-classification configs.
- LiLT covered paired text/layout streams, six bbox embeddings, layout-width divisibility guards, coupled text/layout score softmax that can be shared in inference, token-classification first target, and ignored cache config fields.
- LUKE covered word/entity packed inputs, entity-aware four-block attention, `-1` sentinel masked entity-position averaging, entity classification/pair/span heads, tokenizer-owned entity packing, and ignored historical cache/classifier fields.
- CANINE covered tokenization-free codepoint inputs, multi-hash embedding tables, local shallow attention, Conv1d downsampling and same-padded projection, exact molecule upsample length math, position table guard against hash-bucket size, and MLM/generation deferrals.

### Batch 45 subagent reports

- MobileBERT, MPNet, Funnel, FNet, and I-BERT were produced by rolling subagents and coordinator-reviewed. No prompt change was needed; existing custom math, non-generation, quantized/packed weights, and layout-guard clauses covered the reusable findings.
- MobileBERT covered compact BERT with trigram embeddings, bottleneck/true-hidden-size attention, `NoNorm` affine normalization, multiple FFN sublayers, flag-dependent bottleneck variants, and MobileBERT-specific MLM decoder packing/tied weights.
- MPNet covered encoder-only relative-bucket attention with absolute positions, shared per-head relative-bias table, sentence-transformers mean pooling and optional L2 normalization, MLM tied aliases, and additive-bias attention requirements.
- Funnel covered sequence-length pooling with CLS/truncation, ceil-mode mean/max/min pooling, rectangular pooled attention, relative-position and token-type score bias, full-model upsampling, and `FunnelBaseModel` reduced-sequence ABI distinctions.
- FNet covered attention-free Fourier mixing with `real(fftn(x, dim=(1,2)))`, LayerNorm-heavy residual blocks, `gelu_new`, bounded FFT/DFT admission, no attention masks/cache, and strict `[B,S,H]` Fourier-axis layout guards.
- I-BERT covered dense public checkpoints versus source-supported `quant_mode=true`, scale-carrying quantized module ABI, integer GELU/softmax/LayerNorm custom math, fixed calibration buffers for inference, and ignored historical position fields.

### Batch 46 subagent reports

- RemBERT, SqueezeBERT, RoBERTa-PreLayerNorm, Megatron-BERT, and NomicBERT were produced by rolling subagents and coordinator-reviewed. No prompt change was needed; existing factorized-projection, external wrapper, ignored-field, and layout-guard clauses covered the reusable findings.
- RemBERT covered factorized input embeddings `256 -> 1152`, independent output projection `1152 -> 1664 -> 250300`, untied input/output embeddings, tokenizer special-ID ABI, encoder-only admission, and large-vocab logits planning.
- SqueezeBERT covered BERT-like public ABI with NCW encoder internals, grouped pointwise Conv1d projections/FFNs, NCW channel LayerNorm, BSC-to-NCW layout islands, grouped-GEMM rewrites, and tied MLM decoder alias handling.
- RoBERTa-PreLayerNorm covered source model-type spelling, padding-aware position IDs, embedding LayerNorm, pre-attention and pre-FFN LayerNorms, final model LayerNorm, optional decoder/cache deferral, and local-only attention layout fusion.
- Megatron-BERT covered pre-norm encoder blocks without embedding LayerNorm, final encoder LayerNorm, large hidden sizes, `gelu_new` checkpoint variants, tied MLM decoder/input embedding aliases, and decoder/cache rejection for first encoder target.
- NomicBERT covered native versus legacy/remote-code divergence, separated biasless Q/K/V projections versus remote packed `Wqkv`, native `rope_parameters` for long context, SentenceTransformers mean pooling and optional Matryoshka postprocess, remote MoE rejection, and legacy config mapping guards.

### Batch 47 subagent reports

- DeepSeek-VL-Hybrid, Qwen3-VL, Qwen3-VL-MoE, Qwen3-Omni-MoE, and Phi-4-Multimodal were produced by rolling subagents and coordinator-reviewed. No prompt change was needed; existing multimodal processor, placeholder row-copy, MoE, staged composite modality, audio ABI, quantized/packed weights, and layout-guard clauses covered the findings.
- DeepSeek-VL-Hybrid covered exact hybrid admission over plain DeepSeek-VL, paired low/high image tensor ABI, SigLIP plus SAM branches, 576 image-placeholder guards, SAM local/global attention with relative bias, SAM NCHW/NHWC boundaries, guarded indexed copy, and LLaMA cache/decode behavior.
- Qwen3-VL covered dense VL generation with packed image/video patch rows, `mm_token_type_ids`, 3D/4D M-RoPE and `rope_deltas`, DeepStack indexed additions, varlen vision attention with `cu_seqlens`, FP8 checkpoint gates, and MoE routing to separate `qwen3_vl_moe`.
- Qwen3-VL-MoE covered production top-k MoE decoder requirements, GQA with Q/K head RMSNorm and RoPE-before-cache, packed patch/grid ABI, `mm_token_type_ids`, guarded placeholder row copy, varlen vision attention, Conv3d patch-to-linear guards, NHWC fences, and blocked-FP8 provider deferral.
- Qwen3-Omni-MoE covered staged thinker text-output parity before talker/code2wav audio output, hidden-size versus attention-width mismatch, multimodal M-RoPE and `rope_deltas`, audio/video/image processor ABI, MoE top-k/grouped expert dispatch, batch-size-1 audio output, and local-only layout optimizations.
- Phi-4-Multimodal covered native-versus-legacy `phi4mm` schema mapping, image/audio branch composition, ordered row-copy replacement for `index_put`, image Conv2d/bucketized HD stitch guards, audio fbank/Conformer/Conv coverage, partial longrope, GQA cache, sliding-window admission, and cache reset at the longrope boundary.

### Batch 48 subagent reports

- DeepSeek-V4, Gemma4, GLM4V, InternVL, and MiniCPM-V 4.6 were produced by rolling subagents and coordinator-reviewed. No prompt change was needed; existing clauses for custom cache/state ABIs, multimodal packing, gated configs, placeholder row-copy, external wrappers, and NHWC guard boundaries covered the findings.
- DeepSeek-V4 covered HCA/CSA admission, sink-aware MQA, CSA compressor/indexer/top-k gather, MoE grouped experts, mHC Sinkhorn routing, explicit `head_dim=512` where attention width differs from hidden size, dynamic cache shape ABI, and FP8/FP4 metadata as provider-gated work.
- Gemma4 covered shared-KV attention ownership, `attention_k_eq_v` gating, 26B-A4B MoE constraints, image/video patch flatten order and position guards, audio feature extraction as CPU or precomputed ABI first, and external quantization mirrors as separate audits.
- GLM4V covered exact native scope rejection for `glm4v_moe`/GLM-4.5V and older `chatglm`, patch flattening plus `grid_sample`/spatial-merge guards, multimodal RoPE with `mm_token_type_ids` and `rope_deltas`, varlen vision `cu_seqlens`, and guarded indexed row-copy.
- InternVL covered native composition around Qwen2 decode/cache, GotOCR-style crop-to-patches preprocessing, video flattening, 38B vision RMSNorm plus Q/K norm variants, historical `image_token_index` normalization, and lowering source `masked_scatter` only as guarded row-copy.
- MiniCPM-V 4.6 covered the Qwen3.5 hybrid text core with 18 Gated Delta Net state layers and 6 full-attention KV layers, NaViT packed image/video `target_sizes`, vision varlen/window attention with `cu_seqlens`, custom conv/recurrent linear-attention states, and placeholder `masked_scatter` admission only through stricter processor-order guards.

### Batch 49 subagent reports

- PaddleOCR-VL, Qianfan-OCR, LightOn-OCR, PP-OCRv5-Mobile-Det, and PP-OCRv5-Mobile-Rec were produced by rolling subagents and coordinator-reviewed. No prompt change was needed; existing clauses for structured postprocess ownership, placeholder row-copy, config/tokenizer overrides, and guarded NHWC islands covered the findings.
- PaddleOCR-VL covered packed patch tensors plus `image_grid_thw`, varlen vision attention with `cu_seqlens`, mRoPE/`rope_deltas`, explicit Qwen-style attention widths, ordered image-placeholder row-copy, and external ownership of page-level JSON/Markdown parsing.
- Qianfan-OCR covered dynamic image tiling, `<IMG_CONTEXT>` expansion, checkpoint `image_token_id=151671` overriding a colliding source default, NCHW patch Conv2d plus NHWC-like pixel-shuffle guards, Qwen3 GQA with Q/K RMSNorm, and generated-token structured OCR parsing.
- LightOn-OCR covered public `model_type=mistral3` versus native `lighton_ocr` source drift, Pixtral vision plus Qwen3 decoder composition, explicit `head_dim` attention-width mismatch, first-iteration-only image forwarding with KV cache, and bbox-tuned outputs as text-generation behavior rather than native box postprocess.
- PP-OCRv5-Mobile-Det covered CNN text-region probability maps, BGR/HWC-to-NCHW processor traps, resize-to-32-multiple and `target_sizes` side-channel, DB/OpenCV contour/min-area-box/unclip postprocess, ConvTranspose2d head requirements, and NCHW-first layout admission.
- PP-OCRv5-Mobile-Rec covered PP-LCNetV3 recognition with height-48 dynamic-width inputs, height-collapse guard before sequence attention, SVTR noncausal blocks, CTC-style duplicate/blank postprocess, resolved-backbone admission away from server HGNetV2 defaults, and NHWC only as a guarded local optimization.

### Batch 50 subagent reports

- PP-OCRv5-Server-Det, PP-OCRv5-Server-Rec, PP-DocLayoutV2, PP-DocLayoutV3, and PP-FormulaNet were produced by rolling subagents and coordinator-reviewed. No prompt change was needed; existing clauses for postprocess ABI, config normalization, composed backbones, deformable attention, and layout boundaries covered the findings.
- PP-OCRv5-Server-Det covered HGNetV2 composition, detector neck/head probability maps, Paddle-era fields not read by native source, dynamic H/W multiple-of-32 guards, DB/OpenCV contour geometry, and NCHW-first admission with NHWC only as a fully guarded conv island.
- PP-OCRv5-Server-Rec covered HGNetV2 plus SVTR recognition, height-48 dynamic-width preprocessing, noncausal attention with `head_dim=15`, no-mask attention fallback needs, CTC-style greedy decode, and missing non-safetensors preprocessor metadata treated only as Paddle provenance.
- PP-DocLayoutV2 covered HGNetV2 plus hybrid encoder/decoder detection, top-k proposal selection, multiscale deformable cross-attention with eager `grid_sample` as semantic reference, reading-order transformer ABI, no-NMS postprocess, and exact box/order output parity.
- PP-DocLayoutV3 covered v3 mask prototypes and polygon postprocess, `feature_strides` versus `feat_strides` config-normalization gate, multiscale deformable attention, query mask logits, OpenCV polygon extraction, no-NMS semantics, and NCHW boundaries around flatten/grid-sample/postprocess.
- PP-FormulaNet covered image-to-LaTeX encoder-decoder generation with mixed NCHW/NHWC vision blocks, local/global relative-position attention, NCHW neck/projector, MBart-like decoder self/cross caches, tokenizer formula cleanup, PaddleOCR deployment configs routed out of native scope, and a generated-source `get_encoder()` helper caveat to verify before generic generation helpers rely on it.

### Batch 51 subagent reports

- SAM3, SAM3-Video, SAM3-Tracker, SAM3-Tracker-Video, SAM3-LiteText, EdgeTAM, and EdgeTAM-Video were produced by rolling subagents and coordinator-reviewed. No prompt change was needed; existing stateful video/tracking, postprocess, tokenizer ABI, cache/state, external wrapper, and guarded NHWC clauses covered the findings.
- SAM3 covered image promptable concept segmentation with CLIP text, optional box prompts, DETR-style prompt fusion, ROIAlign/mask heads, bounded prompt concat row-copy, mask-einsum-to-BMM, additive box RPB forcing non-Flash fallback, and official `facebook/sam3` raw config gating.
- SAM3-Video covered composite detector plus tracker orchestration, prompt embedding cache, frame vision feature cache, mask memory, object pointer tokens, streaming versus preloaded-video state differences, optional quality kernels for NMS/hole/sprinkle steps, and open-mirror config use only as non-authoritative metadata.
- SAM3-Tracker covered tracker-only image prompt segmentation, cached image embeddings, prompt encoder/mask decoder split, video-memory fields appearing in an ONNX mirror as a tracker-only admission trap, and the boundary to `sam3_tracker_video` for persistent state.
- SAM3-Tracker-Video covered the real session ABI: frame storage, object id maps, prompt histories, conditioning/non-conditioning output maps, vision feature cache, bf16 mask memory writeback, object pointer packing, memory attention with pointer-token RoPE exclusion, and explicit session operations instead of hidden graph mutation.
- SAM3-LiteText covered MobileCLIP-style text towers for S0/S1/L, S0 RepMixer blocks through `[B,C,1,T]` NCHW views, CLIP tokenizer EOT-pooling ID ordering, full-sequence text projection to detector hidden size, cached `text_embeds`/`vision_embeds`, and tracker keys ignored by this source.
- EdgeTAM covered image promptable segmentation as a wrapper around delegated RepViT/TimmWrapper features, cached image embeddings, NCHW/NHWC FPN boundaries, SAM-style prompt/mask decoder, accessible public configs that actually advertise `edgetam_video`, and a source-doc `danelcsb/edgetam.1_hiera_tiny` gated/auth gap.
- EdgeTAM-Video covered stateful video tracking with RepViT/TimmWrapper delegated vision, explicit session caches, memory attention with object-pointer key RoPE exclusion and repeat metadata, spatial perceiver memory compression, memory encoder writeback, no broad NHWC across persisted caches, and `facebook/EdgeTAM` as an open `.pt` artifact rather than native Transformers config.

### Batch 52 subagent reports

- Qwen3.5, Qwen3.5-MoE, RecurrentGemma, Falcon-H1, OLMo3, OLMo-Hybrid, OLMoE, GraniteMoE-Shared, LongCat-Flash, EXAONE4, and EXAONE-MoE were produced by rolling subagents and coordinator-reviewed where complete. No prompt change was needed; existing clauses for recurrent/state-space states, MoE dispatch, quantized metadata, multimodal processors, gated links, and explicit projection dims covered the findings.
- Qwen3.5 covered a packed multimodal VL processor path plus hybrid text core: Gated DeltaNet linear-attention state layers, full-attention GQA layers with Q output gates, M-RoPE/`rope_deltas`, packed image/video patch rows, processor-owned placeholder row-copy, and FP8 metadata as provider-only work.
- Qwen3.5-MoE covered the same hybrid cache plus sparse top-k MoE with dense shared expert, explicit attention/linear projection widths independent of hidden size, top-k 8/10 variants, FP8/GPTQ configs as loader/provider gates, and multimodal placeholder ordering left to a separate processor audit.
- RecurrentGemma covered Griffin/Hawk recurrent decoder layers, RG-LRU fp32 state, depthwise Conv1d rolling state, local/sliding attention layers, explicit recurrent session state outside `DynamicCache`, and gated official Google configs for 2B/9B variants.
- Falcon-H1 covered parallel Mamba2 SSM plus GQA attention in every layer, four-part hybrid cache (attention K/V plus conv/recurrent states), attention-width smaller than hidden size, Mamba scan/decode provider boundaries, large RoPE theta, and GPTQ metadata as a separate packed-weight admission path.
- OLMo3 covered norm-heavy dense decoder blocks with q/k RMSNorm, branch RMSNorm before residual, YaRN RoPE normalization from legacy fields, MHA 7B versus GQA 32B shapes, full/sliding per-layer cache, and gated AllenAI base/32B-instruct configs.
- OLMo-Hybrid covered three-linear-one-full layer scheduling, NoPE configs with null `rope_theta`, GatedDeltaNet q/k/v conv states and recurrent state, optional FLA provider treated as non-semantic acceleration, and a custom hybrid cache that default generation cache helpers cannot assume.
- OLMoE covered top-8-of-64 sparse MoE, q/k RMSNorm before RoPE, fp32 router softmax, packed expert gate/up weights, bounded dispatch/scatter-add rather than general scatter, and official configs differing from source defaults in expert intermediate size.
- GraniteMoE-Shared covered source-only shared-expert MoE semantics, optional shared dense SwiGLU branch controlled by `shared_intermediate_size`, softmax over selected top-k logits, public PowerMoE configs resolving to `granitemoe` rather than `granitemoeshared`, and unresolved/inaccessible exact deployed shared-expert dimensions.
- LongCat-Flash covered MLA attention with separate RoPE/non-RoPE Q/K widths, 56 cache sublayers from 28 logical layers, shortcut-connected MoE, identity zero experts, LongCat interleaved RoPE, unequal QK/V dimensions needing Flash padding or compatible kernels, Lite N-gram variants routed to remote-code audits, and FP8 as loader/provider metadata.
- EXAONE4 covered dense GQA decoder variants with Llama3 RoPE, Q/K RMSNorm, post-attention and post-MLP RMSNorm before residual add, 32B hybrid `LLLG` full/sliding attention where full layers use NoPE, AWQ/FP8 configs as provider contracts, and layer-pattern validation.
- EXAONE-MoE covered the 236B-A23B MoE shape, dense first layer plus sparse MoE layers, layer-specific RoPE/NoPE, GQA cache, top-k expert routing with shared expert, ignored historical fields such as `first_last_k_dense_replace` and `num_nextn_predict_layers`, and MLX quantized mirrors as non-native provider work.
- Granite Speech covered native 3.3 speech-to-text generation with log-mel frame stacking, Conformer encoder, BLIP-2 Q-Former window projector, guarded audio-token row-copy replacing `masked_scatter`, PEFT LoRA adapter requirements for intended audio behavior, delegated Granite GQA cache, and 3.2 remote-code `granite_speech_qformer` routed out of native scope.

### Batch 53 subagent reports

- Granite Speech Plus, Higgs Audio V2, Higgs Audio V2 Tokenizer, and VibeVoice ASR were produced by rolling subagents and coordinator-reviewed. No prompt change was needed; existing audio preprocessing, codec/tokenizer, composite modality staging, placeholder row-copy, and cache/state clauses covered the findings.
- Granite Speech Plus covered the Plus-only hidden-state concat before the BLIP-2 Q-Former projector, processor-derived audio placeholder counts, guarded audio row-copy, delegated Granite GQA cache, one public Plus checkpoint, missing `preprocessor_config.json`, and active LoRA deferral for this checkpoint.
- Higgs Audio V2 covered text/audio-code causal generation, Llama3 RoPE, GQA KV cache, audio embedding sums over eight codebooks, dual text/audio norm/MLP branches, delay-pattern generation and per-codebook logits processing, plus the generator-codebook `1026` versus codec-codebook `1024` coupling trap.
- Higgs Audio V2 Tokenizer covered waveform-to-code and code-to-waveform codec ABI: HuBERT semantic branch with hidden-state averaging, DAC Conv1d/ConvTranspose1d branch, RVQ nearest-codebook loops, bandwidth-selected quantizer prefixes, CPU resampling first, and a gated/docs-example `hf-audio/higgs_audio_v2_tokenizer-hubert-librispeech` config gap.
- VibeVoice ASR covered generation-style ASR rather than CTC, two causal Conv1d audio tokenizers with explicit padding-cache state, random VAE noise in acoustic latents as an admission decision, Qwen2 GQA decoder cache, guarded audio row-copy, and older/export/MLX mirrors routed out of native scope.

### Batch 54 subagent reports

- DAC, XCodec, Falcon-Mamba, EXAONE4.5, ERNIE4.5, and ERNIE4.5-MoE were produced by rolling subagents and coordinator-reviewed. No prompt change was needed; existing codec/tokenizer, recurrent state, quantized metadata, model-type drift, and MoE routing clauses covered the findings.
- DAC covered neural audio codec encode/decode with NCL Conv1d/ConvTranspose1d, Snake activations, RVQ nearest-codebook search, decode-only code ABI, and the important 16/24 kHz `hop_length` stale-config trap where source ratios/preprocessor imply 320 while config JSON says 512.
- XCodec covered composed HuBERT/WavLM semantic branches plus DAC acoustic branch, XCodec-specific RVQ buffers, adjusted DAC decoder `output_padding`/final activation, decode-only codebook gather-sum, RVQ GEMM+argmax encode, and HuBERT/WavLM treated as composed dependencies.
- Falcon-Mamba covered pure Mamba SSM decoder with no attention/RoPE/KV cache, fixed per-layer conv and recurrent states, selective scan/update provider boundaries, multiplicative attention-mask semantics, bitsandbytes FP4 as provider metadata, and the `expand=16` versus serialized `intermediate_size=8192` config/weight-shape admission trap.
- EXAONE4.5 covered wrapper/delegated `exaone4` text body plus vision processor metadata, historical `exaone4_5_text`/`rope_scaling` alias normalization, FP8 config `layer_types` length mismatch, hybrid full/sliding attention with RoPE/NoPE inherited from EXAONE4, and FP8/AWQ compressed-tensors metadata as loader/provider work.
- ERNIE4.5 covered dense GQA decoder with explicit `head_dim`, GLM-style even/odd RoPE, cached K after RoPE, q/k projection widths independent of hidden size, MoE/VL checkpoints as separate source families, and no source-coupled quantization path.
- ERNIE4.5-MoE covered 21B/300B top-k MoE variants, score-correction bias and top-k weight normalization clamp, optional shared experts, dense early layers before MoE start, grouped expert GEMM lowering, and ignored Thinking/MTP config fields (`moe_gate`, `moe_capacity`, `num_nextn_predict_layers`) as admission traps.

### Batch 55 subagent reports

- ERNIE4.5-VL-MoE, GLM4V-MoE, GLM46V, GLM-MoE-DSA, LFM2, and LFM2-MoE were produced by rolling subagents and coordinator-reviewed. No prompt change was needed; existing clauses for model-type drift, multimodal placeholder row-copy, recurrent/indexer state, hybrid layers, quantized metadata, and guarded NHWC islands covered the findings.
- ERNIE4.5-VL-MoE covered official legacy `model_type=ernie4_5_moe_vl` versus native `ernie4_5_vl_moe` conversion, missing processor config, flattened image/video patch rows, modality-isolated MoE, M-RoPE/`rope_deltas`, `moe_mm_token_type_ids`, and guarded placeholder row-copy.
- GLM4V-MoE covered GLM-4.5V as a VL MoE with 24-layer vision and 46-layer text, processor/preprocessor naming drift, partial M-RoPE, `rope_deltas`, image/video row-copy, dense first layer before MoE, vision varlen `cu_seqlens`, and FP8 compressed-tensors as provider work.
- GLM46V covered the local wrapper as non-default for current GLM-4.6V main configs: main/FP8 route to `glm4v_moe`, while Flash routes to `glm4v`. It also flagged token-id mismatch, admission checks before wrapper reuse, dense delegated `glm4v` paths, patch packing, and M-RoPE.
- GLM-MoE-DSA covered MLA-style attention, DeepSeek Sparse Attention top-k index selection, expanded K/V cache plus separate per-layer DSA `_cached_keys`, beam-reorder hazards, sparse-attention metadata opportunities, and dense-first/sparse-MoE feed-forward paths.
- LFM2 covered the hybrid dense decoder split between full-attention and short-conv layers, with GQA, q/k RMSNorm, RoPE, ordinary KV cache, and fixed conv-state cache. The report correctly avoids treating conv layers as sliding-window attention.
- LFM2-MoE covered LFM2 hybrid cache plus sparse sigmoid top-k MoE, expert bias before top-k, optional top-k renorm, grouped expert GEMM, and ONNX/MLX/GGUF metadata as provider/admission notes rather than core runtime requirements.

### Batch 56 subagent reports

- LFM2-VL, Voxtral, Voxtral-Realtime, Moonshine, Moonshine-Streaming, and Moshi were produced by rolling subagents and coordinator-reviewed. No prompt change was needed; existing clauses for streaming/session ABI, codec staging, bounded placeholder row-copy, head-dim padding, hybrid cache manifests, and local layout rewrites covered the findings.
- LFM2-VL covered patchified SigLIP2 input, resized positional embeddings, local projector layout constraints, bounded image-token row-copy replacing `masked_scatter`, mixed attention/conv LFM2 cache state, and no pixel forwarding after the first cached generation step.
- Voxtral covered offline audio-prefix generation with Whisper log-mel chunks, NCL Conv1d audio stem, projector grouping `[chunks,1500,1280] -> [chunks*375,5120]`, strict audio-placeholder row-copy, Llama GQA cache shapes, missing official `processor_config.json`, and separate routing for Realtime/TTS.
- Voxtral-Realtime covered additive audio/text embedding coupling rather than placeholder scatter, three-part realtime state (text KV, audio encoder KV, causal Conv1d padding cache), first-chunk versus later-chunk STFT centering, stream-exhaustion generation behavior, and an `audio_token_id` source/config gap.
- Moonshine covered raw waveform plus Wav2Vec2 feature extraction, Conv1d compressor, encoder-decoder self/cross attention, cross-cache reuse, backend head-dim padding (`36 -> 40`, `52 -> 56`) with original-dim scaling, and rejection of advertised GQA/MQA fields until source reshape behavior is validated.
- Moonshine-Streaming covered the gap between native whole-waveform `forward` and official ONNX streaming metadata, required frontend chunk/window state for true streaming, decoder self/cross cache ABI, width projection for small/medium, and the same GQA/MQA rejection guard for current source.
- Moshi covered native Transformers Moshi versus external Kyutai runtime repos, main decoder plus depth decoder plus Mimi codec staging, summed multi-codebook embeddings instead of scatter, `MoshiFlexibleLinear` grouped/batched GEMM, fixed short depth-generation loop, and Mimi encode/decode as a separate codec target.

### Batch 57 subagent reports

- AudioFlamingo3, MusicFlamingo, Kyutai Speech-to-Text, GLM-ASR, Parakeet, and FastSpeech2-Conformer were produced by rolling subagents and coordinator-reviewed. No prompt change was needed; existing clauses for bounded placeholder row-copy, external runtime/config rejection, streaming/session ABI, relative attention, dynamic postprocess, and local layout islands covered the findings.
- AudioFlamingo3 covered Whisper-style log-mel preprocessing, audio encoder/projector to Qwen2, bounded `<sound>` row-copy replacing `masked_scatter`, Qwen2 GQA cache with 4 KV heads, limited representative native configs, and rejection of legacy remote-code/LLaVA-style `think/config.json`.
- MusicFlamingo covered the same audio encoder family plus Music RoTE/timestamp conditioning, Qwen2 causal LM with public `use_cache=false`, private original NVIDIA source repo gap, absent `preprocessor_config.json`, no codec/token-generation path, and explicit optional-cache staging.
- Kyutai Speech-to-Text covered Mimi codec encode plus Moshi-like packed rank-3 text/audio-token causal decoder, feature-extractor delay/prefix padding, main decoder KV plus Mimi transformer KV plus Mimi Conv1d padding cache, tokenizer/model ID mismatch, and semantic VAD as external Rust-server functionality absent from Transformers.
- GLM-ASR covered Whisper log-mel preprocessing, noncausal no-mask audio encoder with valid-row selection after projection, partial audio RoPE, 4-frame projector, guarded audio placeholder row-copy, composed Llama GQA decode, and historical remote-code-style config rejection/normalization.
- Parakeet covered native CTC/encoder-only FastConformer ASR, Conv2d subsampling, custom relative-position attention bias that blocks drop-in FlashAttention, CTC 1x1 Conv head to Linear rewrite, CTC blank/pad postprocess, and rejection/routing of RNNT/TDT NeMo or `parakeet_tdt` configs.
- FastSpeech2-Conformer covered non-autoregressive TTS, duration/pitch/energy predictors, value-dependent duration rounding and `repeat_interleave` length regulation, relative-position Conformer attention, `[B,T,C]` with local `[B,C,T]` Conv1d/BatchNorm islands, optional HiFi-GAN vocoder boundary, and legacy standalone vocoder `model_type="hifigan"` admission handling.

### Batch 58 subagent reports

- EfficientLoFTR, LightGlue, SuperPoint, SAM-HQ, VitPose, and VitMatte were produced by rolling subagents and coordinator-reviewed. No prompt change was needed; the NHWC/no-translation guidance is landing correctly as source-faithful layouts first plus guarded local islands/fusion passes.
- EfficientLoFTR covered detector-free matching with RepVGG NCHW backbone, aggregated attention local layout islands, coarse mutual matching, fine unfold/window refinement, ragged match ABI, and direct matched-window gather as a post-coarse optimization.
- LightGlue covered native SuperPoint+LightGlue end-to-end matching, dynamic early-stop/pruning as the main graph-capture trap, mutual-nearest match extraction/postprocess as required ABI, and remote-code DISK detector rejection/routing.
- SuperPoint covered pure Conv2d detector/descriptor heads, source NCHW semantics, detector softmax/depth-to-space, value-dependent NMS/nonzero/top-k, descriptor `grid_sample(align_corners=True)`, ragged-to-padded output, and no broad NHWC translation before conv/head parity.
- SAM-HQ covered promptable segmentation with cacheable final image embeddings plus intermediate global-layer embeddings, HQ token path, NCHW/NHWC ViT boundaries, local/global relative attention, mask selection behavior, and processor-owned upsample/crop/threshold postprocess.
- VitPose covered padded patch `Conv2d(..., padding=2)` as an unsafe vanilla patchify rewrite, simple versus classic deconv/BatchNorm heads, VitPose+ dataset-index MoE expert suffix, flip helper, DARK heatmap refinement, and axis-sensitive target-size coordinate mapping.
- VitMatte covered strict RGB+trimap 4-channel packing, ViTDet allowlist/backbone contract, NHWC transformer islands interrupted by NCHW residual/decoder regions, relative/window attention, no native alpha crop/compositing helper, and guarded patch-conv/BN/upsample rewrites.

### Batch 59 subagent reports

- VitDet, VitPose-Backbone, Timm-Backbone, HGNet-V2, SwinV2, and PVTv2 were produced by rolling subagents and coordinator-reviewed. No prompt change was needed; the reports strengthen reusable backbone guidance around wrapper admission, NCHW feature-map ABIs, and internal NHWC/layout islands.
- VitDet covered backbone-only ownership with no image processor or detector/postprocess, NCHW patch/feature maps with NHWC transformer internals, optional window/global relative attention, residual NCHW conv blocks, and gated links for standalone Google VitDet plus one VitMatte nested config.
- VitPose-Backbone covered padded patch Conv2d, sequence feature maps consumed by parent NCHW reshape, source-default materialization for omitted public config fields, dataset-indexed MoE suffix, and stage selection through `BackboneConfigMixin`.
- Timm-Backbone covered wrapper/dispatch-only ownership, strict delegated-body allowlisting, feature-info ABI manifests, current-source `backbone` requirement versus Hub `architecture`/`timm_model_name` configs, no generic timm operator lowering, and NCHW preservation unless an exact body plus parent are jointly audited.
- HGNet-V2 covered pure NCHW CNN backbone behavior, Conv/BN/ReLU and depthwise-conv fusions, explicit pad/pool/concat details, optional learnable affine blocks, composed D-FINE/DEIM/PP-OCR/document-head boundaries, and external Intellindust config translation as separate work.
- SwinV2 covered shifted-window noncausal attention, continuous log-spaced relative position bias, key-bias loading policy, patch merging gather order, source mask-add parity, optional absolute embeddings, and NCHW public feature maps with guarded NHWC internals.
- PVTv2 covered pyramid overlap patch embeddings, spatial-reduction attention via Conv2d, `linear_attention=True` as AdaptiveAvgPool2d(7)+1x1 Conv2d path, ignored `reshape_last_stage`, depthwise Conv2d in the MLP, and NCHW feature-map outputs with token-layout internals.

### Batch 60 subagent reports

- GPT-Neo, CTRL, XGLM, XLM, XLM-RoBERTa-XL, and X-MOD were produced by rolling subagents and coordinator-reviewed. No prompt change was needed; existing clauses for old config traps, multilingual tokenizer/language ABI, cache gating, selected logits, and adapter routing covered the findings.
- GPT-Neo covered alternating global/local causal attention, GPT-Neo-specific `window_size`, full-growth versus pruned local-cache policy, FlashAttention/local-window parity gate, `gelu_new`, tied LM head, and `logits_to_keep`.
- CTRL covered fixed non-interleaved sinusoidal absolute positions with hard `n_positions`, tokenizer-level first-control-code convention, dense causal MHA with DynamicCache but no native SDPA/Flash dispatch, tied LM head aliasing, and ignored historical fields such as `attn_pdrop`, `n_ctx`, `summary_*`, and `is_decoder`.
- XGLM covered multilingual decoder generation with dynamic sinusoidal absolute positions, MHA across 564M..7.5B sizes, `facebook/xglm-4.5B` ReLU/FFN-width trap, left-padding/cache mask parity, `DynamicCache`, `logits_to_keep`, and tokenizer artifacts as required pipeline state.
- XLM covered native encoder-only construction despite source cache/cross-attention plumbing, optional causal triangular mask in encoder mode, language embeddings gated by real `langs`, `token_type_ids` reusing the word embedding table, and adaptive softmax (`asm=true`) as gated until real checkpoint/cutoffs are selected.
- XLM-RoBERTa-XL covered XL/XXL pre-LN encoder masked-LM, absolute position behavior despite ignored `position_embedding_type`, decoder/cross-attention/cache rejection for first target, 250880-vocab LM head staging, and XXL memory/perf probes.
- X-MOD covered XLM-R-tokenized multilingual encoder with language-specific bottleneck adapters, `lang_ids` as graph-significant input, base postnorm vs large prenorm adapter/LN placement, fixed-language and grouped adapter routing rewrites, and several 401 variant IDs tracked as unavailable.

### Batch 61 subagent reports

- GPT-NeoX Japanese, Flaubert, RoFormer, RoCBert, Splinter, and ModernBERT Decoder were produced by rolling subagents and coordinator-reviewed. No prompt change was needed; source/config divergence, tokenizer-coupled side inputs, question-position ABI, local attention cache policy, and legacy field rejection are already covered by the prompt.
- GPT-NeoX Japanese covered its bias-free packed per-head QKV layout, full-head RoPE, DynamicCache, separate last-layer attention `dense_bias`, tied LM projection, and one gated PPO config gap.
- Flaubert covered XLM-style French encoder masked-LM inference, noncausal masks, optional but deferred causal/cache and adaptive-softmax paths, monolingual `langs` behavior, tied LM head, and missing `special_tokens_map.json` files as tokenizer metadata gaps.
- RoFormer covered BERT-like encoder attention with RoPE, optional `rotary_value`, cached K/V after rotation for decoder mode, and the important RoFormer-v2 config trap where `norm_type=rms_norm` and `use_bias=false` are ignored by the inspected source.
- RoCBert covered BERT-style encoder plus shape/pronunciation side embeddings, `concat_input=True` mapping `[word,shape,pron] -> hidden`, tokenizer JSON dictionaries as required pipeline state, side-ID cache slicing for optional decoder mode, and multiple-choice flatten/reshape heads.
- Splinter covered encoder masked-LM/QA plus QASS question-aware span selection, `question_token_id=104` defaulting, explicit `question_positions` as preferred bounded ABI, dynamic pretraining `where/bincount/scatter` deferral, and QASS logits orientation.
- ModernBERT Decoder covered Ettin-style decoder-only ModernBERT with alternating full/sliding causal attention, source-computed `sliding_window = local_attention // 2`, full and bounded sliding cache layers, RoPE theta variants, GEGLU split order, tied LM head, and ignored downstream config fields.

### Batch 62 subagent reports

- SEW, SEW-D, UniSpeech, UniSpeech-SAT, Wav2Vec2-BERT, and Wav2Vec2-Conformer were produced by rolling subagents and coordinator-reviewed. No prompt change was needed; the current audio/layout guidance handled raw-waveform versus precomputed-feature ABIs, local BTC/BCT layout islands, relative-position math, and CTC/head postprocessing boundaries.
- SEW covered raw-waveform Conv1d feature extraction, squeeze/upsample encoder topology, grouped weight-normalized positional Conv1d, CTC logits, and the strict `sew` versus `sew-d` family boundary.
- SEW-D covered the same squeeze/upsample audio shape family plus DeBERTa-style disentangled relative attention, separate SEW-D configs, and CTC/classifier head staging.
- UniSpeech covered Wav2Vec2-style raw-waveform encoder inference, conv output-length/mask math, positional weight-norm Conv1d, stable/non-stable encoder ordering, CTC fine-tunes with phoneme tokenizers, and pretraining quantizer/adapters as deferred.
- UniSpeech-SAT covered base/large/CTC/speaker/frame-classification variants, XVector TDNN/statistic pooling, weighted layer-sum heads, unfinished source pretraining objective path, conv bias/norm variation, and CTC/postprocess separation.
- Wav2Vec2-BERT covered the rank-3 fbank feature-input ABI `[B,T,160]`, SeamlessM4T feature extraction as data-pipeline work, Conformer encoder blocks with default `relative_key` attention, optional stride-2 adapter, CTC/classification/XVector heads, and local Conv1d layout islands.
- Wav2Vec2-Conformer covered raw-waveform Conformer ASR, relative and rotary checkpoint variants, Transformer-XL-style shifted relative scores, source-specific RoPE before Q/K projection, Conformer Conv1d/BatchNorm module, and the warning that plain FlashAttention/QK-RoPE fusion is not semantically sufficient.

### Batch 63 subagent reports

- Speech2Text, SeamlessM4T v2, MusicGen Melody, Pop2Piano, UnivNet, and VITS were produced by rolling subagents and coordinator-reviewed. No prompt change was needed; the existing prompt already drives the right split between neural graph, processor, generation controller, stochastic/RNG contract, and codec/vocoder staging.
- Speech2Text covered fbank/CMVN input features, Conv1d+GLU subsampling, encoder-decoder MHA, `EncoderDecoderCache` self/cross cache semantics, generation config precedence over legacy config fields, and local feature-layout guards.
- SeamlessM4T v2 covered composite UnitY staging across text encoder-decoder, speech Conformer encoder, non-autoregressive T2U with duration expansion, language-map-driven generation ABI, and HiFi-GAN vocoder; it also rejected a remote-code speech-encoder derivative as out of native scope.
- MusicGen Melody covered text/chroma conditioning as a causal prefix rather than cross-attention, mono/stereo codebook ABI, delay-pattern masking, prefix KV cache ownership, grouped codebook heads, CFG controller behavior, and EnCodec decode composition.
- Pop2Piano covered T5-like audio-to-MIDI encoder-decoder generation, STFT/mel/beatstep preprocessing, decoder self/cross caches, relative bias, separator-row batching metadata, generated-token to MIDI/PrettyMIDI postprocess, and effective gated-FFN activation traps.
- UnivNet covered neural vocoder inference from mel features plus explicit noise, ConvTranspose1d upsampling, mel-conditioned location-variable convolution, kernel predictor caching opportunities, in-graph RNG deferral, and scarcity of public native configs rather than gated access.
- VITS covered stochastic TTS with tokenizer/phonemizer frontend, stochastic duration predictor reverse path, dynamic duration expansion, reverse residual coupling flows, HiFi-GAN vocoder, multi-speaker conditioning, explicit RNG input contracts, and a gated/inaccessible `facebook/mms-tts-cmn` config gap.

### Batch 64 subagent reports

- AIMv2, DiNAT, GroupViT, Hiera, ImageGPT, and ViT-MAE were produced by rolling subagents and coordinator-reviewed. No prompt change was needed; NHWC/NCHW local-layout guidance, custom attention/operator admission, tokenized-image preprocessing, and MAE/random-mask contracts were all captured well.
- AIMv2 covered fixed and native-resolution vision encoders, the `lit` image-text contrastive variant, RMSNorm/SwiGLU blocks, attention pooling, generated 2D sinusoidal native positions, EOS pooling, logit-scale similarity, and guarded patch Conv2d-to-GEMM/NHWC regions.
- DiNAT covered hierarchical NHWC vision stages with external NATTEN 2-D dilated neighborhood attention, relative positional bias, patch/downsample Conv2d layout folds, backbone NCHW feature maps, and square-input/NATTEN admission requirements.
- GroupViT covered CLIP-like dual encoding plus hierarchical group-token assignment, hard assignment attention with soft grouping outputs, optional zero-shot segmentation reconstruction, projection BatchNorm folding, and careful NCHW patch/segmentation layout boundaries.
- Hiera covered patch Conv2d, source-specific token unroll, mask-unit versus global attention, query-pooling reduce-max, classifier/backbone ABIs, MAE pretraining deferral, and feature-map reroll/NHWC LayerNorm/NCHW output packing.
- ImageGPT covered RGB cluster quantization into 512 image tokens plus SOS id 512, GPT-style causal image-token generation with KV cache, cluster-to-RGB postprocess, processor-owned nearest-cluster work, and gated/inaccessible OpenAI variant links.
- ViT-MAE covered MAE patch embedding, deterministic-noise random masking, kept-token encoder lengths, decoder mask-token restoration via `ids_restore`, fixed sin/cos position tables, patchify/loss path, and local NHWC patch-region rewrites with no broad layout translation.

### Batch 65 subagent reports

- DEIMv2, EoMT, EoMT-DINOv3, MM Grounding DINO, SwiftFormer, and Swin2SR were produced by rolling subagents and coordinator-reviewed. No prompt change was needed; the batch reinforced existing guidance around converted/original config admission, composed backbone boundaries, detection/segmentation postprocess ABI, and local layout islands.
- DEIMv2 covered query-based object detection with HGNetV2/DINOv3 composed backbones, NCHW FPN/PAN features, top-k query selection, multiscale deformable decoder cross-attention, DFL/integral boxes, source postprocess without NMS, and explicit rejection of raw original configs unless converted.
- EoMT covered DINOv2-style owned ViT segmentation, query insertion before final blocks, mask-conditioned final attention, mask-head upscaling, semantic/instance/panoptic postprocess including split-image stitching, and DINOv3 routing as a separate family.
- EoMT-DINOv3 covered public converted `tue-mps/eomt-dinov3-*` configs, DINOv3-style dynamic 2D RoPE, query mask conditioning, mask-logit and semantic GEMMs, gated standalone DINOv3 backbone gaps, and older underscore ID 404s.
- MM Grounding DINO covered Swin+BERT composition, text-image fusion, multiscale deformable attention with `grid_sample` fallback, label-token mask generation, text-conditioned class logits, grounded box/text-span postprocess, and no-NMS source behavior.
- SwiftFormer covered NCHW conv-heavy classification, Conv/BatchNorm folding, depthwise/pointwise conv blocks, source-specific efficient additive attention with singleton softmax behavior, final spatial pooling, and careful NCHW/NHWC axis rewrite notes.
- Swin2SR covered super-resolution/restoration with NCHW convs, SwinV2 window attention, continuous relative position bias, model-vs-processor padding distinctions, multiple PixelShuffle/nearest-conv heads, output crop ABI, and source-faithful double mask-add behavior.

### Batch 66 subagent reports

- SwitchTransformers, NLLB-MoE, Nystromformer, MRA, ProphetNet, and TAPAS were produced by rolling subagents and coordinator-reviewed. No prompt change was needed; existing guidance covered sparse MoE routing, approximation-attention admission, custom-kernel boundaries, table preprocessing ABI, and nonstandard seq2seq decoder state.
- SwitchTransformers covered T5-style encoder-decoder sparse top-1 MoE, router logits/probabilities, capacity/drop behavior, T5 relative bias placement, tied LM scaling, ignored historical config flags, and Google public config coverage without gated gaps.
- NLLB-MoE covered multilingual seq2seq with dense and sparse layers, Top-2 routing/dispatch, shared embedding aliases, cross-attention cache precompute, tokenizer/language-code ABI, and a gated/community `ArthurZ/nllb-moe-128` config gap.
- Nystromformer covered the official dense-branch behavior where UW configs set `num_landmarks == segment_means_seq_len`, plus guarded future work for true Nyström landmark attention, iterative pseudoinverse, and depthwise value convolution.
- MRA covered encoder-only long-context inference, the dynamically loaded `kernels-community/mra` 32x32 block attention dependency, no useful no-kernel inference fallback, tokenizer-file gaps in public repos, and custom provider admission requirements.
- ProphetNet covered encoder-decoder generation with ngram decoder streams, relative bucket helpers, predict attention masks, stream-0 logits for generation, cache staging constraints, and one inaccessible Microsoft config gap.
- TAPAS covered BERT-style table QA/classification with seven token-type channels, relative position segmented-min helpers, cell/column aggregation reductions, numeric aggregation heads, table postprocess ABI, and no gated config gaps.

### Batch 67 subagent reports

- AFMoE, Apertus, Arcee, BLT, CHMv2, and ColModernVBERT were produced by rolling subagents and coordinator-reviewed. No prompt change was needed; the batch reinforced existing guidance for official-vs-mirror provenance, PEFT adapter admission, byte/patch preprocessing, DINOv3 head composition, and guarded local NHWC rewrites.
- AFMoE covered Arcee Trinity text-only decoder/MoE variants, attention gating, dense and expert SwiGLU paths, grouped expert GEMM, DynamicCache decode, and public Trinity config snapshots without gated gaps.
- Apertus covered decoder-only GQA with Llama-3-style RoPE, xIELU MLP epilogue, tokenizer/generation snapshots, grouped GEMM handling where Q/K/V are not simply packable, and no reported gated official access gaps.
- Arcee covered Llama-like native decoder with ungated `relu(x) ** 2` FFN, YaRN long-context RoPE, GQA/cache behavior, rejection of `arcee_kda` custom-code variants, and no image/audio layout concerns.
- BLT covered byte-level local/global/local architecture, hash embeddings, patch cross-attention masks, patch reduction operations, byte tokenizer ABI, and gated official Meta BLT links (`facebook/blt-1b`, `facebook/blt-7b`, `facebook/blt-entropy`).
- CHMv2 covered gated official `facebook/dinov3-vitl16-chmv2-dpt-head`, ONNX mirror as non-authoritative snapshot, delegated DINOv3 ViT feature contract, NCHW DPT-like depth head, bin-weighted depth reduction, and guarded NHWC conv/fusion regions only after faithful parity.
- ColModernVBERT covered retrieval-oriented ModernVBERT/ColPali composition, accessible dense configs routing as `modernvbert`, official PEFT LoRA adapter handling, SigLIP NCHW patch embedding, ModernBERT gated MLP, MaxSim scoring, and adapter folding as load-time work.

### Batch 68 subagent reports

- ColQwen2, CPMAnt, CWM, Decision Transformer, DiffLlama, and Doge were produced by rolling subagents and coordinator-reviewed. No prompt change was needed; the batch reinforced existing guidance for legacy/native route guards, generated/modular source, trajectory-packing ABIs, mirror-vs-native config admission, and custom decoder math.
- ColQwen2 covered visual document retrieval over a delegated Qwen2-VL body, packed-patch Conv3d ABI, image masked-scatter to indexed-copy rewrite, late-interaction MaxSim scoring, rejection/routing of legacy `model_type=qwen2_vl` ColQwen repos, and no gated gaps.
- CPMAnt covered OpenBMB CPM-Ant 10B as the only native checkpoint found, internal prompt-prefix materialization, segment-relative bias, ignored caller attention masks, gated GELU FFN, tied embedding/LM projection, and cache decode as validation-gated due to source prompt-prefix behavior.
- CWM covered gated official `facebook/cwm`, `facebook/cwm-sft`, and `facebook/cwm-pretrain` repos, open MLX/Llama mirrors as dimensional evidence only, native `rope_parameters`, Llama-format mirror rejection unless explicitly mapped, and PEFT adapter deferral.
- Decision Transformer covered trajectory `(return,state,action)` interleaving, GPT-2 Conv1D blocks with constant zero GPT positions plus per-timestep embeddings, action-only serving specialization, and 401 gaps for two Hub config fetches.
- DiffLlama covered generated runtime source from modular DiffLlama, native `model_type=diffllama` versus remote-code `diff_llama` rejection, differential attention epilogue, RoPE/cache behavior, and no gated native gaps.
- Doge covered generated Doge decoder source, dynamic-mask attention, dense SwiGLU and CDMoE staged support, historical ignored config fields, DynamicCache decode, and public config snapshots without gated gaps.

### Batch 69 subagent reports

- Dots1, Emu3, ERNIE, ESM, EuroBERT, and Evolla were produced by rolling subagents and coordinator-reviewed. No prompt change was needed; the reports used existing source-faithfulness, generated/modular source, multimodal/tokenized-image, protein encoder, and gated-link guidance well.
- Dots1 covered public base/instruct/FP8-dynamic configs, MoE routing, RoPE/cache ABI, dense and expert SwiGLU, last-token logits, and no NCHW/NHWC concerns.
- Emu3 covered native `model_type=emu3` versus remote-code `Emu3` rejection, VQ-VAE NCHW/NCTHW regions, image placeholder embedding stitch, VQ distance GEMM, generated-image decode staging, and the pinned-source caveat that `Emu3ForConditionalGeneration.forward` accepts but does not pass image tensors to `Emu3Model`.
- ERNIE covered BERT-like encoder inference with optional `task_type_ids`, generated/modular source plus BERT inheritance, open mirror configs, and a gated/unavailable PaddlePaddle ERNIE 3.0 config link.
- ESM covered protein masked-LM and contact/folding variants, token dropout, contact APC kernel, ESMFold trunk/structure staging, atom-table constants, and public representative config snapshots.
- EuroBERT covered encoder-only RoPE/GQA BERT-like models, masked-LM/sequence-classification heads, historical `clf_pooling`/`classifier_pooling` naming, selected-token logits, and public configs without gated gaps.
- Evolla covered protein-text generation with SaProt encoder, latent protein compressor, tanh-gated adapter cross-attention into a text decoder, protein-prefix cache opportunities, `model_type` metadata drift, and guessed 80B HF URLs returning 401.

### Batch 70 subagent reports

- FlexOlmo, Gemma4 Assistant, GLM, GLM Image, GLM4 MoE Lite, and Granite4 Vision were produced by rolling subagents and coordinator-reviewed. No prompt change was needed; the batch reinforced existing guidance for assistant-only shared-state models, split-to-packed weight conversion, remote-code routing, nested config/source mismatch, quantized loader contracts, and packed image-token graphs.
- FlexOlmo covered Olmo-like MoE decoding, source-expected packed expert tensors versus public split per-expert checkpoint keys, conversion mapping as a loader contract, q/k normalization after projection, grouped MoE GEMM staging, and public configs without gated gaps.
- Gemma4 Assistant covered inference-time assisted decoding rather than standalone LM serving, shared-KV assistant attention, ordered masked vocabulary heads for smaller assistants, dense head variants, and integration with a Gemma4 backbone that returns shared KV states.
- GLM covered native `model_type=glm` for THUDM GLM-4 HF repos, rejection/routing of `chatglm` remote-code configs, partial interleaved RoPE, separate Q/K/V packing preconditions, GQA cache, and one 128k config fetch gap.
- GLM Image covered native Transformers autoregressive image-token generator scope, packed source-image patch ABI, M-RoPE, VQ/source-image tokenization as source-owned while Diffusers decode remains out of scope, NCHW/VQ layout guards, and quantized mirror policy deferral.
- GLM4 MoE Lite covered GLM-4.7 Flash 30B-A3B style MoE, MLA projection/split order, routed and shared experts with packed weights, quantized FP8/NVFP4 metadata as loader/provider contracts, and public official configs.
- Granite4 Vision covered SigLIP vision plus Window-QFormer projector plus pure-attention text model despite nested `granitemoehybrid` metadata, multi-crop feature packing/newline embeddings, image placeholder injection, and NCHW patch Conv2d rewrite guards.

### Batch 71 subagent reports

- Helium, Hunyuan V1 Dense, Hunyuan V1 MoE, HY-V3, HyperCLOVAX, and InstructBLIPVideo were produced by rolling subagents and coordinator-reviewed. No prompt change was needed; existing guidance covered native-vs-remote-code route guards, packed MoE/expert layouts, model-specific RoPE, delegated LM composition, and video layout islands.
- Helium covered Kyutai native `model_type=helium` versus newer Llama-routed Helium repos, even/odd interleaved RoPE, FlexAttention/packed-position mask deferral, and text-only GQA decode staging.
- Hunyuan V1 Dense covered dense decoder variants across sizes and quantized metadata, dynamic-alpha Hunyuan RoPE, widened Q/O widths, Q/K RMSNorm before cache, tied LM head, and quantized variants as separate provider/load-path work.
- Hunyuan V1 MoE covered native A13B MoE routing versus remote-code `model_type=hunyuan` quantized variants, GQA/RoPE/QK-norm details, packed expert tensors, and grouped expert GEMM staging.
- HY-V3 covered Tencent Hy3 preview configs, sparse expert packed tensors with FP32 `e_score_correction_bias`, Q/K RMSNorm plus RoPE fusion opportunities, tensor-parallel metadata deferral, and no layout-sensitive image regions.
- HyperCLOVAX covered native text-only causal LM, 14B public repo metadata, restricted-access 1.5B config gap, remote-code Omni/Think wrapper deferral, MuP/residual scalars, and GQA/SwiGLU/RoPE staging.
- InstructBLIPVideo covered generated/modular video source with no public native checkpoint found, Salesforce InstructBLIP configs as dimensional anchors only, `[B,T,C,H,W]` video to CLIP/Q-Former/projector pipeline, placeholder stitch into delegated decoder-only or seq2seq LM, and guarded local video/patch rewrites.

### Batch 72 subagent reports

- JAIS2, JetMoe, Jina Embeddings v3, Kosmos2.5, Laguna, and LASR were produced by rolling subagents and coordinator-reviewed. No prompt change was needed; the batch reinforced gated-link reporting, remote-code/native routing, Sentence Transformers/adapter boundaries, OCR/document preprocessing, compressed-tensor loader deferral, and ASR frontend split.
- JAIS2 covered generated native source, gated official Inception 8B/70B dense and GGUF repos, open derived configs without inventing 70B dimensions, ungated `relu2` MLP, split-half RoPE, and gated/GGUF admission deferral.
- JetMoe covered MoE MLP plus MoA attention routing, dynamic top-k/sort/split/index-add router topology, remote-code `auto_map` compatibility checks, expert weight layout, and decode/cache staging.
- Jina Embeddings v3 covered native encoder-only embedding extraction, external Sentence Transformers mean pooling/L2 normalize, remote-code original repo rejection, fixed-task LoRA pre-merge as a loader option, dynamic `adapter_mask` deferral, and optional non-generation heads.
- Kosmos2.5 covered document/OCR-style multimodal generation, flattened image patches with row/column IDs, separate vision/text/projector components, NCHW patch extraction, image placeholder row-copy, split/4-bit checkpoint policy, and a private `kirp/kosmos2_5` 401 gap.
- Laguna covered Poolside text MoE with full/sliding layer behavior, attention-output gating, packed expert SwiGLU, compressed-tensors FP8/NVFP4/INT4 loader/provider deferral, and no gated configs.
- LASR covered medical ASR with feature-extractor-owned STFT/RFFT/log-mel frontend, Conformer-style encoder with Conv1d/GLU/depthwise convs, CTC head/postprocess split, gated `google/medasr` config/processor gap, and open mirrors/finetunes as labeled evidence.

### Batch 73 subagent reports

- MetaCLIP 2, MGP-STR, MiniMax, MiniMax M2, Ministral, and Ministral3 were produced by rolling subagents and coordinator-reviewed. No prompt change was needed; the batch reinforced naming-trap handling, original-to-HF conversion notes, recurrent/lightning attention state, compressed-tensor loaders, and guarded NCHW image regions.
- MetaCLIP 2 covered dual text/vision encoders, NCHW patch Conv2d, EOS pooling with missing-EOS guard, CLIP similarity orientation, original MetaCLIP packed-QKV conversion as loader-only work, and no gated configs.
- MGP-STR covered scene-text recognition with ViT-like encoder plus A3 character/BPE/WP heads, exact `32x128` NCHW input, grouped 1x1 Conv2d/einsum rewrites, limited public checkpoint sweep, and decode/postprocess parity.
- MiniMax covered text-only decoder with alternating full and lightning attention, custom MiniMaxCache with recurrent linear state, packed 3D expert tensors, ignored/remote `minimax_m1` traps, and `rotary_dim` as ignored by the inspected in-library source.
- MiniMax M2 covered official massive MoE configs, packed expert layout, route/reject trap for an AWQ mirror with `model_type=mixtral`, GGUF/FP8/AWQ materialization as loader/provider work, and GQA/cache staging.
- Ministral covered the in-library text-only `ministral` source versus current accessible `Ministral-8B-Instruct-2410` routing to `model_type=mistral`, one gated base config gap, sliding-window layer dispatch, and no multimodal ownership.
- Ministral3 covered nested text decoder ownership inside top-level `mistral3` multimodal configs, Q/K RoPE weight conversion via `permute_for_rope`, GQA/cache/sliding-window handling, and FP8 static quantized checkpoint deferral.

### Batch 74 subagent reports

- Mistral3, Mistral4, MLCD, ModernVBERT, NanoChat, and Nemotron were produced by rolling subagents and coordinator-reviewed. No prompt change was needed; the batch reinforced wrapper/subconfig ownership, processor fallback boundaries, CLIP/MLCD routing, generated-source authority, and hybrid Mamba/MoE staging.
- Mistral3 covered Mistral Small multimodal wrapper with Pixtral vision tower, patch merger, image placeholder stitch, text body routing to Mistral versus nested Ministral3, missing 3.2 processor raw files, and guarded NCHW patch rewrites.
- Mistral4 covered text-decoder ownership under an official Mistral3 multimodal wrapper, MLA-style low-rank Q/KV projection splits, packed expert gate/up tensors, FP8 loader metadata, and projector/image stitching deferral to wrapper audits.
- MLCD covered native bigG RoPE2D vision encoder versus public MLCD-branded CLIPVisionModel routing, NCHW patch Conv2d, image preprocessor contract, packed QKV, and local NHWC patch-region optimization only under guards.
- ModernVBERT covered ModernVBERT/SigLIP/ModernBERT composition, absent local processor files with Idefics3Processor as the usable ABI, unimplemented BiModernVBert configs, structured image-token stitch, full/sliding ModernBERT attention, and NCHW SigLIP patch embedding.
- NanoChat covered native text-only causal LM with parameter-free RMSNorm, `relu2` MLP, custom-code/gated config traps, q/k norm plus RoPE, and straightforward cache/logits staging.
- Nemotron covered dense Nemotron with LayerNorm1P and `relu2` MLP plus `nemotron_h` hybrid scheduling with Mamba2/attention/MLP/MoE layers, Mamba state/provider requirements, NeMo-only 340B conversion deferral, and no multimodal source ownership.

### Batch 75 subagent reports

- Nemotron-H, OpenAI GPT, OpenAI Privacy Filter, OWLv2, PE Audio, and PE Audio-Video were produced by rolling subagents and coordinator-reviewed. No prompt change was needed; the batch reinforced hybrid stateful decoders, legacy Conv1D formats, structured postprocess/calibration, detection postprocess, audio/video processor ABIs, and delegated branch ownership.
- Nemotron-H covered native hybrid Mamba/attention/MLP/MoE scheduling, no RoPE in native attention despite helper definitions, Mamba conv/recurrent cache ABI, non-gated experts, FP8 loader deferral, and no multimodal path.
- OpenAI covered legacy GPT-1, not GPT-2, with no KV-cache source path, Conv1D weight layout `[in_features, out_features]`, packed QKV split order, `gelu_new` activation mapping, and full-prefix generation staging.
- OpenAI Privacy Filter covered local bidirectional token classification with GQA, local symmetric attention plus sink vectors, interleaved RoPE, original packed-QKV conversion metadata, BIOES class logits, and optional constrained Viterbi/calibration postprocess as a host/controller ABI.
- OWLv2 covered CLIP-like text/vision encoders plus open-vocabulary detection heads, NCHW square-padded image preprocessing, query masking via padded text query detection, normalized similarity and box heads, and postprocess thresholding/box conversion.
- PE Audio covered DAC/PE audio encoder plus delegated ModernBERT text branch, `[B,1,T]` waveform ABI, Conv1d/patch ResNet layout regions, Q/K RMSNorm with RoPE, similarity heads, and 401 gaps for `facebook/pe-a-base` and `facebook/pe-a-large`.
- PE Audio-Video covered composite audio/video/text/audio-video retrieval with delegated timm PE-core vision body and ModernBERT text body, fixed-vs-variable frame processor behavior, `[B,T,C,H,W]` video ABI, audio-video fusion alignment, and branch-wise staging over precomputed embeddings.

### Batch 76 subagent reports

- PE Video, Perception LM, Persimmon, PhiMoE, PI0, and Pixio were produced by rolling subagents and coordinator-reviewed. No prompt change was needed; the batch reinforced delegated timm/vision bodies, official-gated versus mirror-derived configs, robotics/action generation ABIs, and guarded NCHW image lowering.
- PE Video covered video retrieval over a delegated timm PE-core frame tower plus PE temporal encoder, fixed 16-frame versus variable-frame processor behavior, `[B,T,C,H,W]` frame ABI, temporal Conv1d/attention fusions, and no gated links.
- Perception LM covered gated official `facebook/Perception-LM-{1B,3B,8B}` metadata versus mirror-derived config details, delegated timm vision and Llama text bodies, media placeholder row-copy, image/video preprocessing, and official config recheck needs.
- Persimmon covered open Adept causal decoder checkpoints, packed fused QKV row order trap, per-head Q/K LayerNorm before partial RoPE, non-gated `relu2` MLP, and GPT-NeoX derivative routing rejection.
- PhiMoE covered native PhiMoE with LongRoPE normalization, official remote-code `auto_map` compatibility note, packed expert 3D tensors, top-k MoE dispatch, and a 401 internal tiny-random config gap.
- PI0 covered robotics action generation rather than text decoding, PaliGemma/VLM prefix plus DiT flow denoising, bounded camera/image/state/action processor ABI, broad `masked_scatter` lowered to guarded row copy, gated `google/paligemma-3b-pt-224`, and LeRobot policy config mismatch.
- Pixio covered gated official `facebook/pixio-*` repos, public mirror/config fallback plus converter size map, NCHW vision encoder/backbone, Conv2d patch stem and position interpolation, and original QKV conversion metadata.

### Batch 77 subagent reports

- PP-Chart2Table, PP-LCNet, PP-LCNet v3, Prompt Depth Anything, RAG, and Seed-OSS were produced by rolling subagents and coordinator-reviewed. No prompt change was needed; the batch reinforced that NHWC preference should be documented as guarded layout/fusion islands unless a local producer/consumer region is fully owned.
- PP-Chart2Table covered native routing through `GotOcr2ForConditionalGeneration`, the older remote-code `model_type=GOT` rejection path, contiguous chart image placeholder row-copy, mixed NCHW convolution and NHWC vision-transformer internals, and no gated representative links.
- PP-LCNet covered OCR/classification CNN variants, BGR-normalized NCHW processor behavior, depthwise/grouped Conv-BN-HardSwish-SE coverage, fixed text-line stride `[2,1]`, and guarded channel-last conv regions behind NCHW public ABI.
- PP-LCNet v3 covered a native backbone source with no released native checkpoint/config found, an unofficial Paddle `.pdparams` Hub repo only, RepLayer branch fusion, SE linearization, and NCHW feature-map ABI boundaries.
- Prompt Depth Anything covered Dinov2-backed metric depth, optional prompt-depth normalization/denormalization, `align_corners` resize traps, missing `vitb-hf` lookup, and global NHWC translation rejection in favor of local guarded regions.
- RAG covered wrapper-level DPR+BART retrieval generation, external FAISS/index CPU boundary, document score/marginalization reductions, beam/context expansion rewrites, and no gated config links.
- Seed-OSS covered ByteDance Seed-OSS GQA decoder, separated Q/K/V projections, RoPE, SwiGLU, tied LM head behavior, and no gated links.

### Batch 78 subagent reports

- SegGPT, ShieldGemma2, SLANet, SLANeXt, SmolLM3, and Solar Open were produced by rolling subagents and coordinator-reviewed. No prompt change was needed; the batch confirms the current prompt catches nested delegated models, mixed NCHW/NHWC vision internals, recurrent OCR heads, and nonstandard decoder width splits.
- SegGPT covered the official `BAAI/seggpt-vit-large` target, NCHW public tensors with NHWC encoder blocks, decomposed H/W relative position bias, RGB-mask postprocessing, and Conv2d/patch lowering guards.
- ShieldGemma2 covered gated official `google/shieldgemma-2-4b-it` with labeled open-mirror evidence, Gemma3 plus SigLIP delegation, image placeholder row-copy, hybrid full/sliding GQA cache, text residual width versus Q/O width mismatch, and guarded NCHW image front-end rewrites.
- SLANet covered Paddle table recognition as PP-LCNet backbone plus CSP-PAN plus recurrent GRU attention head, fixed-point processor resize, NCHW conv ABI, and no gated native checkpoints.
- SLANeXt covered native safetensors versus older PaddleOCR artifact repos, NCHW input with NHWC vision blocks and NCHW neck, packed QKV/relative-position attention, recurrent table-token head, ignored `loc_reg_num`, and fixed-point resize parity.
- SmolLM3 covered text-only GQA decoder, RoPE/NoPE layer alternation, unexpanded `[B,4,T,128]` KV cache, sliding-window source support not used by inspected configs, and ONNX/GGUF exclusion from native admission.
- Solar Open covered 100B MoE decoder, `hidden_size != num_heads * head_dim`, post-RoPE GQA cache, packed expert `gate_up_proj` gate/up split, legacy-to-normalized RoPE config handling, and no gated config blocker.

### Batch 79 subagent reports

- StableLM, SuperGlue, T5Gemma, T5Gemma2, TextNet, and TimesFM were produced by rolling subagents and coordinator-reviewed. No prompt change was needed; the batch adds more evidence for explicit source-vs-conversion provenance and non-text generation model handling.
- StableLM covered decoder-only LayerNorm rather than RMSNorm, partial RoPE over 25 percent of head dim, StableLM 2 QKV bias/QK norm/GQA/parallel residual variants, GGUF mirror admission gaps, and sequence/head layout guards.
- SuperGlue covered SuperPoint detector delegation, dynamic padded keypoint `K`, NCHW detector conv/grid-sample/NMS parity, 100-iteration Sinkhorn logsumexp, and nested detector allowlisting.
- T5Gemma covered gated official Google configs, open mirror/module evidence, text-only encoder-decoder GQA with RoPE, softcaps, sliding masks, `EncoderDecoderCache`, hidden width versus attention projection width traps, and no NHWC/NCHW path.
- T5Gemma2 covered manual-gated official configs, image-text encoder-decoder over SigLIP plus Gemma-style modules, broad `masked_scatter` lowered to guarded row-copy, merged decoder self+cross attention, dual local/global RoPE, and guarded NCHW vision/projector rewrites.
- TextNet covered CNN-only text-detection backbone, RepConv branch reparameterization, public tiny/small/base variants, tuple stem-kernel source trap, NCHW feature-map ABI, and guarded NHWC conv islands.
- TimesFM covered native forecasting `TimesFmModelForPrediction`, patched context/horizon tensors, no KV cache, masked first-valid-patch normalization, additive causal/padding masks, per-dimension Q scaling, legacy/custom repo rejection, and separate `timesfm2_5` scope.

### Batch 80 subagent reports

- TimesFM 2.5, UVDoc, VaultGemma, VibeVoice Acoustic Tokenizer, VideoLLaMA3, and VidEoMT were produced by rolling subagents and coordinator-reviewed. No prompt change was needed; the coordinator normalized TimesFM 2.5 section capitalization to the standard headings.
- TimesFM 2.5 covered the single official native checkpoint versus mirrors/original TimesFM variants, bounded padded rank-2 time-series ABI, Welford patch stats, RevIN, shifted RoPE, Q/K RMSNorm, per-dim query scaling, and a generated-source `window_size` method-name hazard.
- UVDoc covered document rectification as NCHW CNN plus dilated bridge and mesh head, RGB-to-BGR/resize processor behavior, optional interpolate plus `grid_sample` postprocess, and guarded NHWC CNN islands with channel/mesh boundary guards.
- VaultGemma covered gated official `google/vaultgemma-1b` raw config, mirror-derived 1B shape evidence, hidden width versus attention projection width, source-default versus 1B attention/softcap differences, and SDPA softcap incompatibility.
- VibeVoice Acoustic Tokenizer covered Conv1d/depthwise/ConvTranspose audio tokenizer paths, NCL/BLC layout islands, deterministic `sample=False` staging, streaming conv cache deferral, and legacy-key config normalization.
- VideoLLaMA3 covered native HF configs versus older DAMO remote-code configs, Qwen2 decoder delegation, flattened channel-first visual patch rows, placeholder row-copy, vision varlen attention, pixel-unshuffle/interpolate parity, and video compression metadata.
- VidEoMT covered fixed-square video segmentation/translation model shapes, missing public preprocessor configs, `[B,T,C,H,W]` runtime ABI, recurrent fixed-query state rather than KV cache, dense attention scaling risks for VSPW, and deferred Hungarian/grid-sample/postprocess variable-record work.

### Batch 81 subagent reports

- ViT-MSN, VJEPA2, xLSTM, YOSO, Youtu, Zamba, and Zamba2 were produced by rolling subagents and coordinator-reviewed. No prompt change was needed; this completed the remaining source-directory audit set.
- ViT-MSN covered NCHW vision encoder patch embedding, Conv2d-to-GEMM lowering, LayerNorm and dense noncausal MHA, p=4/p=7 attention-size risks, and guarded patch-region NHWC only after NCHW parity.
- VJEPA2 covered video encoder/classification first with `skip_predictor=True`, `[B,T,C,H,W]` to Conv3d `[B,C,T,H,W]` layout, 3-axis video RoPE, large-token dense attention, and deferred predictor/mask-token path.
- xLSTM covered recurrent causal LM semantics rather than KV attention, mLSTM C/N/M state ABI, fallback-native versus external Triton admission, RMSNorm/per-head LayerNorm/soft-cap math, and cache mutation/reset staging.
- YOSO covered non-SDPA expectation attention with `acos` cumulation, external LSH kernel gate, source mask quirks, MLM decoder tying, and dense long-context memory pressure.
- Youtu covered native text-only Tencent Youtu configs versus legacy `youtu_llm` remote-code configs, MLA-style low-rank Q/KV paths, interleaved RoPE, QK dim 192 versus V dim 128, and text-only layout guards.
- Zamba covered Mamba selective-scan requirement, mixed Mamba state plus hybrid-layer KV cache, shared/tied transformer branch weights, concat-original-embedding hybrid attention width, and no-RoPE despite config fields.
- Zamba2 covered Mamba2 prefill/decode provider needs, hybrid cache manifest, nonstandard attention scale/input width, tied hybrid blocks/adapters, no MoE despite gated MLP, and gated base 7B configs with Instruct config fallback.

## Suggested assessment order

Start with families that maximize architectural coverage before grinding through near-duplicates:

1. Decoder-only RoPE/GQA: `llama`, then `mistral` or `qwen2`.
2. Encoder-decoder relative bias: `t5`.
3. BERT-style encoder absolute/relative embeddings: `bert`, `roberta`, `deberta_v2`.
4. Vision transformer: `vit`, `swin`, `dinov2`.
5. Multimodal projector + LLM: `llava`, `qwen2_vl`, `gemma3`.
6. MoE decoder: `mixtral`, `qwen3_moe`, or similar.
7. Audio encoder/decoder or CTC: `whisper`, `wav2vec2`.

This order should expose most prompt gaps early: cache, masks, position math, convolution/patch preprocess, multimodal packing, MoE routing, and non-text preprocessing.
