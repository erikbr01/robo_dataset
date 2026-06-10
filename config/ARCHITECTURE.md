# Config architecture

## Directory layout

```
config/
├── data_source/     ← one file per task/modality/action-space variant
│   ├── lift_image_abs.yaml
│   ├── lift_image.yaml
│   ├── transport_image_abs.yaml
│   └── ...
├── pipeline/        ← one file per (sensor setup × action space) combination
│   │                   each file bundles preprocess + postprocess transforms
│   ├── robomimic_single_arm_image_abs.yaml
│   ├── robomimic_single_arm_image_rel.yaml
│   ├── robomimic_bimanual_image_abs.yaml
│   ├── robomimic_bimanual_image_rel.yaml
│   ├── robomimic_lowdim_abs.yaml
│   ├── robomimic_lowdim_rel.yaml
│   ├── pusht_image.yaml
│   └── lowdim_identity.yaml
├── store/           ← storage backend configs
│   ├── zarr_disk.yaml
│   ├── hdf5.yaml
│   └── rlds_frame.yaml
├── transforms/      ← three-tier per-sample transforms (disk/cpu/gpu)
│   ├── disk/
│   ├── cpu/
│   └── gpu/
└── train.yaml       ← default robo_dataset training config
```

---

## `data_source/`

One config per task / modality / action-space variant.  Contains everything
about the data: what it contains, how to load it, and how to interpret it.

### Obs + action shape

Image tasks carry a `shape_meta` block:

```yaml
shape_meta:
  obs:
    agentview_image: {shape: [3, 84, 84], type: rgb}
    robot0_eef_pos:  {shape: [3]}         # type default: low_dim
    ...
  action:
    shape: [10]
    fields:                    # single source of truth for action field indices
      pos:      [0, 1, 2]
      rotation: [3, 4, 5, 6, 7, 8]
      gripper:  [9]
    rotation_rep: rotation_6d          # representation stored in the dataset
    output_rotation_rep: axis_angle    # representation expected by the robot
```

Low-dim tasks carry a top-level `action` block (no `shape_meta`):

```yaml
obs_dim: 19
action_dim: 10
obs_keys: [...]

action:
  shape: [10]
  fields:
    pos:      [0, 1, 2]
    rotation: [3, 4, 5, 6, 7, 8]
    gripper:  [9]
  rotation_rep: rotation_6d
  output_rotation_rep: axis_angle
```

### Why `action.fields` lives here

Index-to-field mapping is a property of *the dataset*, not the model or the
transform chain.  It must be defined exactly once.  Pipeline configs reference
it via Hydra interpolation (`${data_source.shape_meta.action.fields}` for image
tasks, `${data_source.action.fields}` for low-dim), so transforms never
hard-code numeric indices.

---

## `pipeline/`

One config per combination of sensor setup and action space.  Each file defines
**both** `preprocess` and `postprocess` as `InferencePipeline` instances.

```yaml
preprocess:
  _target_: robo_dataset.transforms.InferencePipeline
  transforms:
    - _target_: robo_dataset.transforms.NormalizeImages
      keys: [agentview_image, robot0_eye_in_hand_image]
      mean: [0.485, 0.456, 0.406]
      std:  [0.229, 0.224, 0.225]
      from_uint8: true

postprocess:
  _target_: robo_dataset.transforms.InferencePipeline
  transforms:
    - _target_: robo_dataset.transforms.ConvertRotation
      rotation_fields: [rotation]
      action_fields: ${data_source.shape_meta.action.fields}
      from_rep: ${data_source.shape_meta.action.rotation_rep}
      to_rep:   ${data_source.shape_meta.action.output_rotation_rep}
```

`preprocess` is applied to observations **before** `policy.predict_action()`.
`postprocess` is applied to actions **after** `policy.predict_action()`.
Both are applied identically at train time and inference time.

### Available pipeline configs

| Config | Preprocess | Postprocess |
|---|---|---|
| `robomimic_single_arm_image_abs` | NormalizeImages (agentview + eye-in-hand) | rotation_6d → axis_angle |
| `robomimic_single_arm_image_rel` | NormalizeImages (agentview + eye-in-hand) | identity |
| `robomimic_bimanual_image_abs` | NormalizeImages (4 cameras) | rotation_6d → axis_angle (×2) |
| `robomimic_bimanual_image_rel` | NormalizeImages (4 cameras) | identity |
| `robomimic_lowdim_abs` | identity | rotation_6d → axis_angle |
| `robomimic_lowdim_rel` | identity | identity |
| `pusht_image` | ToFloat (single camera, [0,1]) | identity |
| `lowdim_identity` | identity | identity |

---

## Transform classes (`robo_dataset.transforms`)

```python
from robo_dataset.transforms import (
    InferencePipeline,     # chains InferenceTransform objects
    NormalizeImages,       # ImageNet normalisation on obs image keys
    ToFloat,               # uint8 → float32 in [0, 1]
    IdentityObs,           # pass-through obs
    ConvertRotation,       # rotation-rep conversion on named action fields
    IdentityAction,        # pass-through action
    RotationTransformer,   # direct API: .forward() / .inverse()
)
```

### `ConvertRotation` design

```python
ConvertRotation(
    rotation_fields=['rotation'],            # field names to convert
    action_fields=<dict from data_source>,   # field → contiguous index list
    from_rep='rotation_6d',
    to_rep='axis_angle',
)
```

`action_fields` is the single source of index truth, loaded from the
data_source config via `${data_source.shape_meta.action.fields}`.
`ConvertRotation.__init__` validates contiguous indices and derives
`(start, end)` slices internally — numeric indices never appear in YAML.

When `rotation_fields` contains multiple entries (bimanual tasks), fields
are processed from highest to lowest index so earlier-field slices remain
valid after higher-field slices change size.

---

## Using in a learning pipeline

A learning pipeline config (lives in the learning pipeline repo, not here)
composes data_source and pipeline via Hydra:

```yaml
defaults:
  - data_source: lift_image_abs               # from robo_dataset/config/
  - pipeline: robomimic_single_arm_image_abs  # from robo_dataset/config/
  - _self_

# Resolved values available at runtime:
#   ${data_source.shape_meta}
#   ${preprocess}   → InferencePipeline
#   ${postprocess}  → InferencePipeline
```

Usage at runtime:

```python
preprocess  = hydra.utils.instantiate(cfg.preprocess)   # InferencePipeline
postprocess = hydra.utils.instantiate(cfg.postprocess)  # InferencePipeline

# training step
obs  = preprocess(batch['obs'])
loss = model.compute_loss({'obs': obs, 'action': batch['action']})

# inference step (identical transforms)
obs    = preprocess(live_observation)
action = postprocess(policy.predict_action(obs)['action'])
```

---

## Adding a new use-case

1. **New task**: add `config/data_source/<task>.yaml` with `shape_meta`
   (or `obs_dim`/`action_dim` for low-dim), `dataset`, `env_runner`, and
   `action.fields`.
2. **New sensor/action-space combination**: add `config/pipeline/<name>.yaml`
   with `preprocess` and `postprocess` blocks.
3. No Python changes required unless the transform class itself is novel.
