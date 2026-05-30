# robo_dataset

Unified trajectory dataset for robot learning, with:

- **Multi-backend TrajectoryStore**: read from HDF5, zarr, RLDS/TFRecord, or video with one interface
- **Streaming conversion**: convert between backends one episode at a time (never loads full dataset into RAM)
- **Three-tier transform system**: disk-cached offline, CPU-batch online, GPU-batch online
- **Hydra configuration**: all backends and transforms configurable via YAML
- **Equivalence tests**: verify data integrity across backends (exact for lossless, PSNR for lossy)

## Quick start

```bash
# Install (requires Python >=3.10)
uv pip install -e ".[dev]"
```

## Backends

| Backend | Class | Read | Write | Notes |
|---|---|---|---|---|
| Robomimic HDF5 | `RobomimicHDF5Store` | ✓ | – | Source format |
| zarr DirectoryStore | `ZarrStore` / `ZarrStoreWriter` | ✓ | ✓ | zarr v2/v3 compatible |
| RLDS/TFRecord | `RLDSFrameStore` / `RLDSFrameStoreWriter` | ✓ | ✓ | One record per frame, random access via index |

## Convert HDF5 → zarr

```python
from robo_dataset.backends.hdf5 import RobomimicHDF5Store
from robo_dataset.backends.zarr_backend import ZarrStoreWriter
from robo_dataset.convert import convert

src = RobomimicHDF5Store("image.hdf5", obs_keys=["agentview_image", "robot0_eye_in_hand_image"])
dst = ZarrStoreWriter.create(
    "output.zarr",
    key_shapes={k: src.frame_shape(k) for k in src.keys},
    key_dtypes={k: src.frame_dtype(k) for k in src.keys},
    image_keys=["agentview_image", "robot0_eye_in_hand_image"],
)
convert(src, dst)  # streams one episode at a time
```

## Convert HDF5 → RLDS/TFRecord

```python
from robo_dataset.convert import convert_rlds
convert_rlds(src, "output.rlds", image_keys=["agentview_image", "robot0_eye_in_hand_image"])
```

## Training pipeline with all three transform tiers

```python
from robo_dataset.backends.zarr_backend import ZarrStore
from robo_dataset.dataset import TrajectoryDataset
from robo_dataset.transforms.disk_cached import ResizeImages, apply_disk_cached
from robo_dataset.transforms.cpu_batch import ActionChunk, make_collate_fn
from robo_dataset.transforms.gpu_batch import RandomCrop

IMAGE_KEYS = ["agentview_image", "robot0_eye_in_hand_image"]

# Tier 1: offline resize (cached; reuses on subsequent runs)
store = ZarrStore("my_dataset.zarr")
resize = ResizeImages(image_keys=IMAGE_KEYS, target_h=84, target_w=84)
store = apply_disk_cached(store, resize, cache_root="/tmp/rd_cache", image_keys=IMAGE_KEYS)

# Tier 2: online CPU transform in DataLoader
cpu_transform = ActionChunk(n_obs_steps=2, image_keys=IMAGE_KEYS)
dataset = TrajectoryDataset(store, horizon=16, pad_before=1, pad_after=7)
loader = dataset.make_dataloader(
    batch_size=32, num_workers=8, collate_fn=make_collate_fn(cpu_transform)
)

# Tier 3: online GPU transform after DataLoader
gpu_transform = RandomCrop(image_keys=IMAGE_KEYS, size=(76, 76), pad=4)
device = torch.device("cuda")

for batch in loader:
    batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
    batch = gpu_transform(batch)  # applied on GPU before model
    output = model(batch)
```

## Hydra config

```bash
python examples/mock_train.py store.path=/data/my_zarr num_workers=8
```

## Run tests

```bash
# Runs in robocasa env (has working torch + zarr3)
conda run -n robocasa python -m pytest tests/ -q
```

## Benchmarks

Storage format benchmarks are in `tools/`:

- `tools/benchmark_storage.py` — zarr, HDF5 formats (run in `rerun_env`)
- `tools/benchmark_rlds.py` — RLDS/TFRecord format (run in `rerun_env`)
- `tools/benchmark_rrd.py` — Rerun RRD format (run in `rerun_env`)
- `tools/benchmark_lerobot.py` — video formats via torchcodec (run in `robocasa`)
- `tools/make_pareto_plot.py` — generate throughput vs size scatter plot

### Key results (Robomimic Lift PH, 12 DataLoader workers)

| Format | Size | vs raw | fps |
|---|---|---|---|
| **rlds_frame_jpeg_q75** | 27.6 MB | 14.8× | **31.6k** |
| zarr_disk_jpeg_q75 | 26.6 MB | 15.4× | 19.9k |
| h264_kf1 + cached decoder | 10 MB | 41× | 16.4k |
| zarr_disk_lz4 | 223 MB | 1.8× | 24.9k |

RLDS/TFRecord is the **Pareto-optimal** training format: smallest size AND fastest throughput.
Why: zarr uses one file per frame (19k `open()` syscalls per epoch), TFRecord uses one file total.
