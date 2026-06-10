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
| zarr ShardingCodec | `ZarrShardedStore` / `ZarrShardedStoreWriter` | ✓ | ✓ | zarr v3 only; shard_frames files instead of one-per-frame |
| RLDS/TFRecord | `RLDSFrameStore` / `RLDSFrameStoreWriter` | ✓ | ✓ | One record per frame, random access via index |
| Shared-memory RAM | `ShmStore` | ✓ | – | Zero-copy after one-time load; fork-safe |
| Per-episode MP4 | `VideoStore` / `VideoStoreWriter` | ✓ | ✓ | H.264/AV1 via PyAV; torchcodec for decode; non-video keys in zarr lz4 |

### Which backend to use

```
OK with lossy images? (fine for visuomotor policies)
├── Yes — need smallest file size? (H.264 ~28× smaller than zarr; 2.5× smaller than RLDS)
│    ├── Yes  →  VideoStore (H.264 kfi=1; torchcodec cached decoder; ~11 MB vs 27 MB RLDS)
│    │           Half RLDS throughput at 12 workers (4k vs 8k fps) but smallest on disk.
│    └── No   →  RLDSFrameStore (4–7× faster than zarr; JPEG q75; single TFRecord file)
│                Note: ZarrShardedStore + JPEG is NOT competitive — JPEG forces chunk_frames=1,
│                losing the vectorized benefit; use RLDS instead.
└── No (need lossless)
     ├── Dataset fits in RAM? (rule of thumb: < ½ of system RAM)
     │    └── Yes  →  ShmStore (load once from zarr; zero-copy, no disk I/O after load)
     └── Disk-bound, lossless
          ├── Default  →  ZarrShardedStore (1.6× vs directory zarr; set chunk_frames = horizon)
          └── Already have directory zarr, no perf budget for reconversion?
               └── ZarrStore (baseline)
```

`RLDSFrameStore` is the fastest disk-lossy backend — JPEG q75 reduces I/O by ~15× vs raw, and RLDS is optimized for per-frame random access without shard-index overhead. `VideoStore` uses H.264 (41× compression at crf=23) with a cached torchcodec decoder; run `--video` in the benchmark to compare throughput directly against RLDS on your hardware. `ZarrShardedStore` is the right lossless default: set `chunk_frames = horizon` to decompress only the needed frames in one call. `ShmStore` is fastest overall but requires the full dataset to fit in RAM.

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

## Load zarr into shared memory (ShmStore)

```python
from robo_dataset.backends.zarr_backend import ZarrStore
from robo_dataset.backends.shm import ShmStore

zarr_store = ZarrStore("output.zarr")
store = ShmStore.from_store(zarr_store)  # one-time load, ~4s for 9k frames

# Use store in TrajectoryDataset — workers read zero-copy from shared memory
dataset = TrajectoryDataset(store, horizon=16, pad_before=1, pad_after=7)
loader = dataset.make_dataloader(batch_size=32, num_workers=8)
```

## Convert HDF5 → sharded zarr (ZarrShardedStore)

```python
from robo_dataset.backends.hdf5 import RobomimicHDF5Store
from robo_dataset.backends.zarr_sharded import ZarrShardedStoreWriter
from robo_dataset.convert import convert

src = RobomimicHDF5Store("image.hdf5", obs_keys=["agentview_image", "robot0_eye_in_hand_image"])
writer = ZarrShardedStoreWriter.create(
    "output_sharded.zarr",
    key_shapes={k: src.frame_shape(k) for k in src.keys},
    key_dtypes={k: src.frame_dtype(k) for k in src.keys},
    chunk_frames=8,    # inner decompression unit; set ≤ horizon for vectorized advantage
    shard_frames=1024, # frames per shard file; 9k frames → ~10 files vs 9k+ for directory zarr
)
convert(src, writer)
```

`chunk_frames` is the key parameter: vectorized slice `arr[start:end]` decompresses only the inner chunks that overlap the range. Set `chunk_frames ≤ horizon` so each sequence request hits at most one decompression boundary. `shard_frames` controls how many frames share one shard file — larger values reduce `open()` overhead but increase the minimum I/O unit.

## Convert HDF5 → RLDS/TFRecord

```python
from robo_dataset.convert import convert_rlds
convert_rlds(src, "output.rlds", image_keys=["agentview_image", "robot0_eye_in_hand_image"])
```

## Convert HDF5 → VideoStore (H.264)

```python
from robo_dataset.backends.hdf5 import RobomimicHDF5Store
from robo_dataset.backends.video import VideoStoreWriter
from robo_dataset.convert import convert

src = RobomimicHDF5Store("image.hdf5", obs_keys=["agentview_image", "robot0_eye_in_hand_image"])
writer = VideoStoreWriter.create(
    "output.video",
    key_shapes={k: src.frame_shape(k) for k in src.keys},
    key_dtypes={k: src.frame_dtype(k) for k in src.keys},
    image_keys=["agentview_image", "robot0_eye_in_hand_image"],
    vcodec="libx264",  # or 'libsvtav1' for AV1
    crf=23,            # lower = higher quality
    kfi=1,             # keyframe every frame — required for random access
    fps=10,
)
convert(src, writer)

# Read back
from robo_dataset.backends.video import VideoStore
store = VideoStore("output.video")
```

Non-video keys (actions, proprioceptive obs) are stored losslessly as zarr lz4 alongside the MP4 files.  Requires `av` (PyAV) to encode and `torchcodec` + `torch` to decode.  Set `kfi=1` for random-access training; larger `kfi` compresses better but requires seeking back to the previous keyframe on every decode.

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

Benchmarks are in `tools/`:

- `tools/benchmark_storage.py` — zarr, HDF5 formats (run in `rerun_env`)
- `tools/benchmark_rlds.py` — RLDS/TFRecord format (run in `rerun_env`)
- `tools/benchmark_rrd.py` — Rerun RRD format (run in `rerun_env`)
- `tools/benchmark_lerobot.py` — video formats via torchcodec (run in `robocasa`)
- `tools/benchmark_acceleration.py` — zarr_disk vs zarr_sharded vs RLDS vs VideoStore (H.264) + channels_last conv + chunk_frames sweep (run in `robocasa`; add `--video` to include VideoStore)
- `tools/make_pareto_plot.py` — generate throughput vs size scatter plot

### Storage format — size vs raw throughput (Robomimic Lift PH, 12 workers, horizon=1)

These numbers are from `benchmark_storage.py` (horizon=1 sequential access, no trajectory slicing).

| Format | Size | vs raw | fps |
|---|---|---|---|
| **rlds_frame_jpeg_q75** | 27.6 MB | 14.8× | **31.6k** |
| zarr_disk_lz4 | 223 MB | 1.8× | 24.9k |
| zarr_disk_jpeg_q75 | 26.6 MB | 15.4× | 19.9k |
| h264_kf1 + cached decoder | 10 MB | 41× | 16.4k |

`rlds_frame_jpeg_q75` is Pareto-optimal: smallest size AND fastest throughput.
Root cause of zarr's bottleneck: one file per chunk → 19k `open()` syscalls per epoch per worker.

### zarr_disk vs zarr_sharded vs RLDS (Robomimic Lift PH, RTX 4090, horizon=8)

`sharded_lz4`: `chunk_frames=8`, `shard_frames=1024`.  `sharded_jpeg`: `chunk_frames=1` (forced by JPEG), `shard_frames=1024`.
`batch=32`, 3 measured epochs + 1 warmup.

| workers | zarr_disk_lz4 | zarr_sharded_lz4 | zarr_sharded_jpeg | rlds_jpeg_q75 | h264_kf1 |
|--------:|--------------|------------------|-------------------|---------------|----------|
| 0 | 211 fps | 414 fps | 200 fps | 1,911 fps | 510 fps |
| 4 | 1,038 fps | 1,573 fps | 770 fps | 5,844 fps | 1,777 fps |
| 8 | 1,708 fps | 2,852 fps | 1,403 fps | 9,157 fps | 3,245 fps |
| 12 | 2,294 fps | 3,858 fps | 1,812 fps | **8,392 fps** | 4,224 fps |

Storage: zarr_disk 305 MB · sharded_lz4 170 MB · sharded_jpeg 27 MB · rlds 27 MB · **h264_kf1 11 MB**

**H.264 kf1 (VideoStore)**: 1.8× faster than directory zarr at 12 workers; half of RLDS throughput but 2.5× smaller on disk. Best choice when storage is the bottleneck. Requires torchcodec + PyAV.

**Sharded + lz4 (lossless)**: 1.6× over directory zarr — same codec, fewer `open()` calls, and vectorized decompression when `horizon ≥ chunk_frames`.

**Sharded + JPEG (lossy)**: actually *slower* than directory zarr lz4. Root cause: JPEG forces `chunk_frames=1` (one image per inner chunk), losing the vectorized benefit. The shard structure helps open() overhead but JPEG decompression per frame dominates. zarr's shard index lookup per inner chunk adds overhead not present in RLDS.

**RLDS + JPEG**: 4–7× faster than directory zarr. RLDS is specifically designed for per-frame random access — no shard index overhead, and JPEG compression reduces I/O by ~15×. This is the correct backend for JPEG; sharded zarr + JPEG is not competitive.

### chunk_frames sweep (zarr_sharded, 8 workers, horizon=8)

Set `chunk_frames ≤ horizon` so each sequence request spans at most one decompression unit.
Beyond that threshold, throughput plateaus then drops as wasted decompressed bytes increase.

| chunk_frames | fps | notes |
|---|---|---|
| 1 | 1,428 | decompression overhead per frame; no vectorized benefit |
| 2 | 2,112 | |
| 4 | 2,701 | |
| **8** | **3,045** | **matches horizon=8 — peak** |
| 16 | 3,039 | same (horizon still ≤ chunk_frames; no waste at horizon=8) |
| 32 | 2,757 | decompresses 4× more data than needed per slice |

**Rule of thumb**: set `chunk_frames = horizon`. Default of 8 is optimal for standard robotics training (horizon 4–16).

### channels_last conv2d (RTX 4090, batch=32, 84×84)

1.01× — negligible on Ada Lovelace at robotics scale. The 10–30% gains reported in the literature are Ampere-era measurements with larger batches and deeper networks.
