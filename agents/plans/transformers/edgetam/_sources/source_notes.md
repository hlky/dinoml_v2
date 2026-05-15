# EdgeTAM Source Notes

## Local source basis

- Transformers checkout: `transformers`
- Inspected commit: `b75feb2af64c3e29cbbc1bd859958c5432cc7ed4`
- Primary model-family directory: `transformers/src/transformers/models/edgetam`
- Generated-file note: `configuration_edgetam.py` is generated from `modular_edgetam.py`; the generated warning says future source edits should be applied to the modular file.

## Files read

- `src/transformers/models/edgetam/modular_edgetam.py`
- `src/transformers/models/edgetam/configuration_edgetam.py`
- `src/transformers/models/edgetam/modeling_edgetam.py`
- `src/transformers/models/edgetam/convert_edgetam_to_hf.py`
- `src/transformers/models/sam2/processing_sam2.py`
- `src/transformers/models/sam2/image_processing_sam2.py`
- Boundary-only read for public checkpoint architecture/state ABI:
  - `src/transformers/models/edgetam_video/configuration_edgetam_video.py`
  - `src/transformers/models/edgetam_video/modeling_edgetam_video.py`

## HF config / processor snapshots

Fetched with `Invoke-RestMethod`; no model imports or DinoML tests were run.

- `https://huggingface.co/facebook/EdgeTAM/raw/refs%2Fpr%2F1/config.json`
- `https://huggingface.co/facebook/EdgeTAM/raw/refs%2Fpr%2F1/preprocessor_config.json`
- `https://huggingface.co/facebook/EdgeTAM/raw/refs%2Fpr%2F1/video_preprocessor_config.json`
- `https://huggingface.co/yonigozlan/EdgeTAM-hf/raw/main/config.json`
- `https://huggingface.co/api/models/facebook/EdgeTAM/revision/refs%2Fpr%2F1`
- `https://huggingface.co/api/models/yonigozlan/EdgeTAM-hf`

Observed config facts:

- `facebook/EdgeTAM` at `refs/pr/1` and `yonigozlan/EdgeTAM-hf` expose the same operator-significant public config shape in the fetched snapshots.
- Public checkpoint `architectures` is `["EdgeTamVideoModel"]` and `model_type` is `edgetam_video`, not plain `edgetam`.
- Plain `edgetam` source still owns the reusable single-image SAM-style prompt encoder and mask decoder contracts. Full public checkpoint parity needs `edgetam_video` state/memory coverage.
- `danelcsb/edgetam.1_hiera_tiny` was referenced in source docstrings, but the HF API request returned `Invalid username or password`; treat it as inaccessible/gated or removed for this audit.

## Key source line anchors

- `modeling_edgetam.py:53-76`: `EdgeTamLayerNorm`, channels-last native path and channels-first permute wrapper.
- `modeling_edgetam.py:105-124`: eager dense attention math, additive mask before fp32 softmax.
- `modeling_edgetam.py:127-195`: `EdgeTamAttention`, downsampled q/k/v width, noncausal attention ABI, FA fallback when additive target-guided mask is present.
- `modeling_edgetam.py:198-272`: two-way attention block: sparse self-attn, sparse-to-image cross-attn, MLP, image-to-sparse cross-attn.
- `modeling_edgetam.py:327-370`: 2D sine/cosine FPN positional embedding.
- `modeling_edgetam.py:373-422`: FPN neck; NHWC backbone features are permuted to NCHW for 1x1 convs and top-down nearest upsample.
- `modeling_edgetam.py:430-473`: `EdgeTamVisionModel`; TimmWrapper backbone output is treated as NCHW and converted to NHWC before the neck.
- `modeling_edgetam.py:510-654`: prompt encoder; random Fourier coordinate embedding, point/box sparse embeddings, dense mask embedding.
- `modeling_edgetam.py:657-711`: two-way transformer flattening image embeddings to token sequence.
- `modeling_edgetam.py:714-865`: mask decoder tokens, deconv upscaling, high-resolution feature skips, hypernetwork mask projection, IoU/object heads.
- `modeling_edgetam.py:867-914`: dynamic multimask stability fallback.
- `modeling_edgetam.py:923-1238`: image model forward, cached image feature path, prompt/mask decoder handoff.
- `processing_sam2.py:38-173` and `424-503`: point/box packing, coordinate normalization, and postprocess delegation.
- `image_processing_sam2.py:371-456` and `598-685`: processor defaults, channels-first output, mask resize/postprocess/non-overlap constraints.
- `edgetam_video/modeling_edgetam_video.py:2772-3063`: video boundary only: memory-conditioned feature path, single-frame inference, new memory encoding.

## Gaps / limits in this source pass

- No DinoML code tests, model imports, or checkpoint downloads were run.
- The RepViT body is delegated through `TimmWrapperConfig` and was not audited here; DinoML should not infer its complete operator surface from EdgeTAM wrapper code.
- Full video tracker state ABI belongs to `edgetam_video`; this report records its boundary but does not claim complete video parity coverage.
- Only two accessible public config snapshots were found, and they were not operator-distinct. The prompt requested 3-5 representative configs when available; this family appears to have one public EdgeTAM shape in the inspected sources/configs.
