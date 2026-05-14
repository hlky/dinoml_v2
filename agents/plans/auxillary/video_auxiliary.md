# Video Auxiliary Models and Runtime Contracts

## Why This Matters

The video UIs expose much more than a single video diffusion model. They need
temporal VAEs, frame interpolation, optical flow, video super-resolution, masks,
camera/trajectory conditioning, reference images, audio-video conditioning, and
flow-matching schedulers. These are runtime contracts as much as model
architectures.

## Video Postprocessing and Preprocessing

- RIFE, FILM, and GIMM video frame interpolation.
- RAFT and Unimatch optical flow.
- FlashVSR video super-resolution.
- MatAnyone and SAM video mask propagation.
- Depth Anything v3 video depth with temporal chunking.

## Code Anchors

- `H:/uis/Comfy-Org/ComfyUI/comfy_extras/nodes_frame_interpolation.py:9`
  RIFE/FILM frame interpolation nodes.
- `H:/uis/Comfy-Org/ComfyUI/comfy_extras/nodes_void.py:15`
  torchvision RAFT optical-flow helper.
- `H:/uis/deepbeepmeep/Wan2GP/preprocessing/flow.py:19`
  `FlowAnnotator`, loads RAFT.
- `H:/uis/deepbeepmeep/Wan2GP/preprocessing/raft/raft.py:24`
  RAFT implementation.
- `H:/uis/deepbeepmeep/Wan2GP/postprocessing/rife/RIFE_V4.py:141`
  RIFE `IFNet`.
- `H:/uis/deepbeepmeep/Wan2GP/postprocessing/rife/inference.py:122`
  temporal interpolation.
- `H:/uis/deepbeepmeep/Wan2GP/postprocessing/flashvsr/wgp_bridge.py:14`
  `FlashVSRBridge`.
- `H:/uis/deepbeepmeep/Wan2GP/postprocessing/flashvsr/wan_video_dit.py:657`
  FlashVSR Wan DiT.
- `H:/uis/deepbeepmeep/Wan2GP/postprocessing/flashvsr/tcdecoder.py:170`
  `TAEHV` temporal/spatial decoder.
- `H:/uis/deepbeepmeep/Wan2GP/preprocessing/depth_anything_v3/depth.py:266`
  `DepthV3VideoAnnotator`.

## Video Generation and Conditioning

- Wan and LTX custom video stacks.
- Hunyuan Video, CogVideo, Cosmos, Mochi, LTX, Wan and related video VAEs in
  Comfy/Swarm compatibility maps.
- VACE context, MultiTalk audio conditioning, camera embedding, recammaster,
  masks/control video, trajectory controls, reference images.
- FlowMatch, FlowDPM, and FlowUniPC scheduler variants.

## Code Anchors

- `H:/uis/Comfy-Org/ComfyUI/comfy/ldm/wan/model.py`.
- `H:/uis/Comfy-Org/ComfyUI/comfy/ldm/wan/vae.py`.
- `H:/uis/Comfy-Org/ComfyUI/comfy/ldm/lightricks/model.py`.
- `H:/uis/Comfy-Org/ComfyUI/comfy/ldm/lightricks/vae/causal_video_autoencoder.py`.
- `H:/uis/deepbeepmeep/Wan2GP/models/wan/any2video.py:86`
  `WanAny2V`.
- `H:/uis/deepbeepmeep/Wan2GP/models/wan/modules/model.py:214`
  `WanSelfAttention`.
- `H:/uis/deepbeepmeep/Wan2GP/models/wan/modules/model.py:506`
  `WanAttentionBlock`.
- `H:/uis/deepbeepmeep/Wan2GP/models/wan/modules/model.py:883`
  `WanModel`.
- `H:/uis/deepbeepmeep/Wan2GP/models/wan/modules/vae.py:928`
  `WanVAE`.
- `H:/uis/deepbeepmeep/Wan2GP/models/wan/wan_handler.py:322`
  mask/control preprocessing config.
- `H:/uis/deepbeepmeep/Wan2GP/models/wan/wan_handler.py:421`
  `i2v_trajectory`.
- `H:/uis/deepbeepmeep/Wan2GP/models/wan/any2video.py:912`
  camera embedding/recammaster path.
- `H:/uis/deepbeepmeep/Wan2GP/models/wan/any2video.py:1048`
  VACE context encode.
- `H:/uis/deepbeepmeep/Wan2GP/models/wan/any2video.py:1085`
  MultiTalk audio conditioning.
- `H:/uis/deepbeepmeep/Wan2GP/shared/utils/basic_flowmatch.py:7`
  `FlowMatchScheduler`.
- `H:/uis/deepbeepmeep/Wan2GP/shared/utils/fm_solvers.py:69`
  `FlowDPMSolverMultistepScheduler`.
- `H:/uis/deepbeepmeep/Wan2GP/shared/utils/fm_solvers_unipc.py:20`
  `FlowUniPCMultistepScheduler`.

## 3D and Camera Helpers

- `H:/uis/deepbeepmeep/Wan2GP/models/wan/scail/scail_pose_nlf.py:69`
  NLF pose video processing.
- `H:/uis/deepbeepmeep/Wan2GP/models/wan/scail/nlf/multiperson_model.py:17`
  `MultipersonNLF`.
- `H:/uis/deepbeepmeep/Wan2GP/models/wan/vista4d/preprocess.py:147`
  depth/camera point rendering helpers.
- `H:/uis/deepbeepmeep/Wan2GP/models/wan/vista4d/runtime.py:6`
  `add_vista4d_modules`.
- Packages: `smplfitter`, `chumpy`, `insightface`, `facexlib`.

## DinoML Gap

Very high. Required contracts include temporal shapes, frame rates, clip
chunking, tiled video VAE encode/decode, 3D RoPE, video masks, camera paths,
reference images, audio-conditioned branches, flow schedulers, and generated
runtime plans for long-running generation.

