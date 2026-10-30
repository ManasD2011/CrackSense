# Model checkpoint requirements

Weights are not distributed in this repository. Train your own or obtain the corresponding project checkpoint from the owner. There is no published download URL asserted here.

| Consumer | Expected checkpoint |
|---|---|
| Explained B0/B1/B2 notebooks | Dictionary with `model_state_dict`, matching backbone; optional epoch/config/threshold metadata |
| Training resume | Matching architecture/config plus optimizer, scheduler and scaler state |
| Live `ground-station/model.py` | Serialized `torch.nn.Module`, either directly or under `model`; valid `best_threshold` metadata or explicit `--threshold` |
| `quantize_segformer.py` | Full saved module or supported B2 state dictionary; its reconstruction config is B2-specific |

Full-model packages use Python deserialization; only load trusted files. FP16 is supported by the live GPU path. Dynamic INT8 packages are experimental and rejected by that path. Converting FP16 weights to FP32 does not restore discarded weight precision.

A useful local layout is `03-segformer/checkpoints/latest.pt` for the source and `03-segformer/Quantised/latest_fp16.pt` after packaging. Pass explicit paths in live commands. If your source is named `best.pt`, exports are named `best_fp16.pt`, etc. Do not guess a threshold if metadata is absent; validate it first.
