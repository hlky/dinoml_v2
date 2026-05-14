# SuperPoint Config Sweep Snapshot

Fetched from Hugging Face raw files/API on 2026-05-13.

| Repo | Scope | model_type | Architecture field | Encoder | Decoder | Detector controls | Processor |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `magic-leap-community/superpoint` | Native Transformers SuperPoint | `superpoint` | `SuperPointModel` | `[64,64,128,128]` | hidden `256`, detector `65`, descriptor `256` | threshold `0.005`, max `-1`, NMS radius `4`, border `4` | `SuperPointImageProcessor`, resize `480x640`, rescale `1/255`, grayscale omitted/effective default false |
| `stevenbucaille/superpoint` | Native Transformers SuperPoint mirror/early upload | `superpoint` | `SuperPointForKeypointDetection` | `[64,64,128,128]` | hidden `256`, detector `65`, descriptor `256` | threshold `0.005`, max `-1`, NMS radius `4`, border `4` | Same as Magic Leap |
| `ETH-CVG/lightglue_superpoint` | LightGlue wrapper with full nested detector config | `lightglue` | `LightGlueForKeypointMatching` | nested `[64,64,128,128]` | nested hidden `256`, detector `65`, descriptor `256` | nested threshold `0.005`, max `-1`, NMS radius `4`, border `4` | `LightGlueImageProcessor`, resize `480x640`, rescale `1/255`, `do_grayscale=true` |
| `stevenbucaille/lightglue_superpoint` | LightGlue wrapper with minimal nested detector config | `lightglue` | `LightGlueForKeypointMatching` | omitted; source defaults apply | omitted; source defaults apply | omitted; source defaults apply | `LightGlueImageProcessor`, resize `480x640`, rescale `1/255`, `do_grayscale=true` |
| `AXERA-TECH/superpoint` | Out of scope export | `ONNX` | none | not native | not native | not native | no preprocessor config found |

Raw links:

- https://huggingface.co/magic-leap-community/superpoint/raw/main/config.json
- https://huggingface.co/magic-leap-community/superpoint/raw/main/preprocessor_config.json
- https://huggingface.co/stevenbucaille/superpoint/raw/main/config.json
- https://huggingface.co/stevenbucaille/superpoint/raw/main/preprocessor_config.json
- https://huggingface.co/ETH-CVG/lightglue_superpoint/raw/main/config.json
- https://huggingface.co/ETH-CVG/lightglue_superpoint/raw/main/preprocessor_config.json
- https://huggingface.co/stevenbucaille/lightglue_superpoint/raw/main/config.json
- https://huggingface.co/stevenbucaille/lightglue_superpoint/raw/main/preprocessor_config.json
- https://huggingface.co/AXERA-TECH/superpoint/raw/main/config.json
