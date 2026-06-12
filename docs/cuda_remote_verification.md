# CUDA Remote Verification

Use Runpod for CUDA runtime verification in this repository.

## Preferred Image

The current preferred image is:

- `hlky/dinoml:ubuntu-nodeps`

It is the validated DinoML v2 CUDA verification baseline and already contains:

- a live git checkout at `/opt/src/dinoml_v2`
- `/opt/venvs/dinoml`
- CUDA 12.9 tooling
- `transformers` source at `/opt/src/transformers`
- `diffusers` source at `/opt/src/diffusers`

## Preferred Pod Creation Flow

Use the repo-local Runpod helper and reuse the prebaked repo path instead of
recloning:

```powershell
python .codex/skills/runpod-codex-remote/scripts/create_runpod_codex_remote.py `
  --gpu-id "NVIDIA RTX 2000 Ada Generation" `
  --repo https://github.com/hlky/dinoml_v2.git `
  --name dinoml-ubuntu-nodeps-rtx2000ada `
  --image hlky/dinoml:ubuntu-nodeps `
  --existing-project-path /opt/src/dinoml_v2 `
  --volume-gb 20 `
  --ports 22/tcp `
  --auto-connect
```

For disposable verification pods, delete the pod when done. `runpodctl pod stop` only stops compute; volume storage remains billable until `runpodctl pod delete`.

## Current Smoke Baseline

The current image was validated on a fresh Runpod pod with these smoke checks:

- `torch` import and CUDA matmul
- tiny `transformers` GPT-2 forward on CUDA
- tiny `diffusers` UNet/scheduler forward on CUDA
- DinoML v2 trace/compile/load/run smoke on CUDA

The DinoML smoke used a tiny add module, compiled it for the detected `sm_XX`
target, loaded the artifact, ran it with NumPy inputs, and checked parity
against `reference_numpy(...)`.

## Image Notes

The image includes a reduced CUDA static-archive set that is sufficient for the
current DinoML CUDA compile flow:

- `libcudadevrt.a`
- `libcudart_static.a`
- `libculibos.a`
- `libcublas_static.a`
- `libcublasLt_static.a`
- `libcurand_static.a`

That set is intentionally narrower than the full CUDA devel image to reduce
image size while preserving the current DinoML CUDA toolchain path.
