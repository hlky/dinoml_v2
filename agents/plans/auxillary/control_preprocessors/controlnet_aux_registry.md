# controlnet_aux Registry

## Coverage

- Diffusers: not covered as core model implementations. Diffusers consumes the produced control images through ControlNet, T2I-Adapter, or related pipelines.
- Transformers: mixed. Some depth families overlap Transformers model coverage, especially ZoeDepth, DPT, and GLPN-like depth support, but the `controlnet_aux` package owns the ControlNet-specific detector wrappers and image rendering contract.
- Third-party/package: local source at `H:/controlnet_aux/src/controlnet_aux`.

## Runtime Contract

`controlnet_aux` exposes detector classes directly and also provides a `Processor` registry for UI-style preprocessor IDs. Checkpoint-backed detectors default to `from_pretrained("lllyasviel/Annotators")` unless a mapping overrides the repository or filename.

The registry maps:

- HED: `scribble_hed`, `softedge_hed`, `scribble_hedsafe`, `softedge_hedsafe`.
- MiDaS: `depth_midas`.
- MLSD: `mlsd`.
- OpenPose: `openpose`, `openpose_face`, `openpose_faceonly`, `openpose_full`, `openpose_hand`.
- DWPose: `dwpose`.
- PiDiNet: `scribble_pidinet`, `softedge_pidinet`, `scribble_pidsafe`, `softedge_pidsafe`.
- NormalBae: `normal_bae`.
- Lineart: `lineart_realistic`, `lineart_coarse`, `lineart_anime`.
- Depth: `depth_zoe`, `depth_leres`, `depth_leres++`.
- Non-checkpoint transforms: `canny`, `shuffle`, `mediapipe_face`.

Direct package exports additionally include `AnylineDetector`, `LineartStandardDetector`, `SamDetector`, `TEEDdetector`, and `ContentShuffleDetector`.

## DinoML Notes

Treat the registry names as compatibility aliases, not model names. A DinoML preprocessor manifest should preserve detector class, checkpoint identity, resize policy, output type, and mode flags such as `safe`, `scribble`, `coarse`, `boost`, and OpenPose body/hand/face inclusion.

The package reinforces a split contract: neural inference is only part of parity. Condition rendering, thresholding, NMS, random shuffle fields, keypoint remapping, and output polarity are user-visible behavior.

## Sources

- `H:/controlnet_aux/src/controlnet_aux/__init__.py`
- `H:/controlnet_aux/src/controlnet_aux/processor.py`
- `H:/controlnet_aux/README.md`
