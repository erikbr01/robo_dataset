# Config architecture

## Data configs (this repo)

```
config/
├── datasource/   ← one file per dataset / task / modality variant
│   ├── lift_image_abs.yaml
│   ├── pusht_image.yaml
│   └── ...
├── preprocess/   ← obs transforms before predict_action()
│   ├── robomimic_image.yaml
│   ├── pusht_image.yaml
│   └── identity.yaml
└── postprocess/  ← action transforms after predict_action()
    ├── rotation6d_to_axisangle.yaml
    └── identity.yaml
```

These configs are self-contained: `preprocess/` and `postprocess/` reference only
`robo_dataset.transforms.*` classes; `datasource/` declares the dataset and
env-runner structure consumed by the host learning-pipeline repo.

They are composed into any learning-pipeline config via Hydra `defaults:`:

```yaml
# my_pipeline/config/pipeline/lift_diffusion_unet.yaml
defaults:
  - _self_
  - /path/to/robo_dataset/config/datasource: lift_image_abs
  - /path/to/robo_dataset/config/preprocess: robomimic_image
  - /path/to/robo_dataset/config/postprocess: rotation6d_to_axisangle
  - model: diffusion_unet_hybrid   # lives in the learning-pipeline repo

horizon: 16
n_obs_steps: 2
n_action_steps: 8
```

Model configs and top-level pipeline configs live in the learning-pipeline repo,
not here.

---

## Layer responsibilities

### `datasource/`

Everything about the data: what it contains and how to load it. One config per
task / modality / action-space variant. Migrated from diffusion_policy `config/task/`.

Contains:
- `shape_meta` — obs keys with shapes and types (`rgb` / `low_dim`); action shape
- `dataset_path`, `task_name`, `dataset_type`, `abs_action`, `rotation_rep`
- `dataset:` — `_target_` for the dataset class + constructor args
- `env_runner:` — rollout environment for evaluation

Cross-references to pipeline-level params (`${horizon}`, `${n_obs_steps}`, etc.)
are resolved after Hydra merges the defaults list.

### `preprocess/`

**Observation transforms applied identically at train time and inference time**,
before `policy.predict_action()`. Serialised into the checkpoint so inference
requires only the `.ckpt` file.

| Config | Transforms |
|---|---|
| `robomimic_image` | `NormalizeImages` — agentview + eye-in-hand, ImageNet stats |
| `pusht_image` | `ToFloat` — single camera, [0, 1] |
| `identity` | no-op — for low-dim tasks |

Only include transforms with **fixed** (non-data-dependent) parameters here.
The policy's internal `LinearNormalizer` (fit from training data) lives inside
the model and is saved with its weights.

### `postprocess/`

**Action transforms applied identically at train time and inference time**,
after `policy.predict_action()`. Also serialised into the checkpoint.

| Config | Transforms |
|---|---|
| `rotation6d_to_axisangle` | `ConvertRotation(start=3, end=9, rotation_6d→axis_angle)` — robomimic abs-action, 10-dim → 7-dim |
| `identity` | no-op — relative-action / lowdim / push-t tasks |

The policy's internal unnormalisation already runs inside `predict_action()`;
postprocess covers structural conversions only.

---

## Adding new configs

1. **New datasource**: add `config/datasource/<task>.yaml` with `shape_meta`, `dataset:`, `env_runner:`.
2. **New preprocess**: add `config/preprocess/<name>.yaml` referencing `robo_dataset.transforms.*`.
3. **New postprocess**: add `config/postprocess/<name>.yaml` referencing `robo_dataset.transforms.*`.
