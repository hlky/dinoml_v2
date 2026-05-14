# glm_moe_dsa source notes

Audit date: 2026-05-13

Transformers checkout:

- Path: `X:/H/transformers`
- Commit inspected: `b75feb2af64c3e29cbbc1bd859958c5432cc7ed4`
- Model directory: `src/transformers/models/glm_moe_dsa`

Copied source snapshots:

- `modeling_glm_moe_dsa.py` from the pinned checkout. SHA256 `BFB722EB830ABC1FADBBB35005861C7902C159ECDF6330835F4BCF56E9BC955B`.
- `modular_glm_moe_dsa.py` from the pinned checkout. SHA256 `7F291A7D7E55096DDB93EEAA6B35B75C538AC8E2F7199B4DFFC461DFF5B14B80`.
- `configuration_glm_moe_dsa.py` from the pinned checkout. SHA256 `662465C008048EB59EA96C167223770F82FA3A551AED78D074067A86F68D492F`.

Config snapshots:

- `hf_config_zai-org_GLM-5_main.json`: `https://huggingface.co/zai-org/GLM-5/raw/main/config.json`, SHA256 `308D3CEEBE7BD9C2246D20550D5B793098DD03838D1E504F8236063AE152BCAE`.
- `hf_config_zai-org_GLM-5.1_main.json`: `https://huggingface.co/zai-org/GLM-5.1/raw/main/config.json`, SHA256 `726E45E28D1BE1636A5834048A4101A2CFE384819AD2EBE08D41F6FE04656526`.
- `hf_config_zai-org_GLM-5-FP8_main.json`: `https://huggingface.co/zai-org/GLM-5-FP8/raw/main/config.json`, SHA256 `5EA2D79D3A77FE90836C18893DE00996D096CD01516B52B88C6FC18993CB4DF8`.
- `hf_config_zai-org_GLM-5-FP8_refs-pr-4.json`: `https://huggingface.co/zai-org/GLM-5-FP8/raw/refs%2Fpr%2F4/config.json`, SHA256 `5EA2D79D3A77FE90836C18893DE00996D096CD01516B52B88C6FC18993CB4DF8`.
- `hf_config_QuantTrio_GLM-5-AWQ_e18e264.json`: `https://huggingface.co/QuantTrio/GLM-5-AWQ/raw/e18e264928f517f7d433a29fb9e64fc1740e4753/config.json`, SHA256 `878B7F967BF86B754F02BBC91BFC4D0D6B6067054B83A4A988B1185D2C00B664`.
- `hf_config_spicyneuron_GLM-5.1-MLX-2.9bit_head.json`: `https://huggingface.co/spicyneuron/GLM-5.1-MLX-2.9bit/raw/main/config.json`, SHA256 `EE04B30FF93AE71E8D6AB8EF7A7858121223D96221E91FEED3627CFC6B414514`.
- `hf_config_tiny-random_glm-moe-dsa.json`: `https://huggingface.co/tiny-random/glm-moe-dsa/raw/main/config.json`, SHA256 `D17FB8A4AA977F07EEDF75D9C2E35563BAB1EE8A01E7B9479DA356C8C58C2E61`.

Scope notes:

- `modeling_glm_moe_dsa.py` and `configuration_glm_moe_dsa.py` are generated from `modular_glm_moe_dsa.py`; future Transformers source edits should use the modular file, but DinoML import/parsing parity should inspect generated `modeling_glm_moe_dsa.py`.
- No DinoML tests, Python imports, or model execution were run. The audit is static source/config inspection only.
- Official GLM-5 and GLM-5.1 main configs were accessible. Quantized configs were included to expose loading/provider traps, not to claim DinoML should support those quantized formats in the first runtime target.
