"""Acceleration benchmark: zarr_disk vs zarr_sharded vs RLDS, and channels_last conv2d.

Part A — DataLoader throughput (frames/s):
    ZarrStore (disk, lz4)  vs  ZarrShardedStore (lz4)  vs  RLDSFrameStore (JPEG q75)
    at several worker counts.  If --zarr points to a robo_dataset ZarrStore with
    meta.json it is used directly.  Otherwise --hdf5 creates a fresh zarr under --out.

Part A2 — chunk_frames sweep (ZarrShardedStore only):
    Sweeps chunk_frames over {1, 2, 4, 8, 16} at --sweep-workers to show how inner
    chunk size affects throughput.  Vectorized advantage materialises when horizon >= chunk_frames.

Part B — channels_last conv2d microbenchmark:
    Contiguous (NCHW) vs channels_last (NHWC) layout on a single forward pass
    through a small conv stack, on the shape that robotics models actually use:
    (B*T, 3, 84, 84) float32.  Reports the speedup from channels_last on the
    current GPU.

Usage (from an existing robo_dataset zarr store)::

    conda run -n robocasa python tools/benchmark_acceleration.py \\
        --zarr  /path/to/existing.zarr \\
        --batch 32 --epochs 2

Usage (from raw HDF5 — creates new zarr stores under --out)::

    conda run -n robocasa python tools/benchmark_acceleration.py \\
        --hdf5  /mnt/ssdB/erik/cupid_data/robomimic/datasets/lift/ph/image.hdf5 \\
        --obs-keys agentview_image robot0_eye_in_hand_image \\
        --out   /mnt/ssdB/erik/cupid_data/robo_dataset_bench \\
        --batch 32 --epochs 2

Results are printed to stdout and saved as benchmark_acceleration.json under
--out (or next to the --zarr store).
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path.home() / "diffusion_policy"))


# ── helpers ───────────────────────────────────────────────────────────────────

def _fps(n_frames: int, elapsed_per_epoch: float) -> float:
    return n_frames / elapsed_per_epoch


def _run_loader(loader, n_frames: int, n_epochs: int) -> float:
    """Warm up one epoch, then return average seconds per epoch over n_epochs."""
    for _ in loader:
        pass

    t0 = time.perf_counter()
    for _ in range(n_epochs):
        for _ in loader:
            pass
    return (time.perf_counter() - t0) / n_epochs


# ── zarr preparation ──────────────────────────────────────────────────────────

def prepare_zarr(hdf5_path: str, obs_keys: list[str], out_dir: str) -> str:
    from robo_dataset.backends.hdf5 import RobomimicHDF5Store
    from robo_dataset.backends.zarr_backend import ZarrStoreWriter
    from robo_dataset.convert import convert

    zarr_path = os.path.join(out_dir, "zarr_disk_lz4")
    if os.path.exists(os.path.join(zarr_path, "meta.json")):
        print(f"Reusing existing zarr store: {zarr_path}")
        return zarr_path

    print(f"Converting {hdf5_path} → {zarr_path} …")
    os.makedirs(out_dir, exist_ok=True)
    src = RobomimicHDF5Store(hdf5_path, obs_keys=obs_keys)
    writer = ZarrStoreWriter.create(
        path=zarr_path,
        key_shapes={k: src.frame_shape(k) for k in src.keys},
        key_dtypes={k: src.frame_dtype(k) for k in src.keys},
        image_keys=obs_keys,
    )
    convert(src, writer, verbose=True)
    print(f"Conversion done → {zarr_path}")
    return zarr_path


def prepare_rlds(
    zarr_path: str,
    out_dir: str,
    obs_keys: list[str],
    jpeg_quality: int = 75,
) -> str:
    """Convert an existing directory zarr to an RLDS/TFRecord store."""
    from robo_dataset.backends.zarr_backend import ZarrStore
    from robo_dataset.convert import convert_rlds

    rlds_path = os.path.join(out_dir, f"rlds_jpeg_q{jpeg_quality}")
    if os.path.exists(os.path.join(rlds_path, "meta.json")):
        print(f"Reusing existing RLDS store: {rlds_path}")
        return rlds_path

    print(f"Converting {zarr_path} → {rlds_path} (JPEG q{jpeg_quality}) …")
    os.makedirs(out_dir, exist_ok=True)
    src = ZarrStore(zarr_path)
    convert_rlds(src, rlds_path, image_keys=obs_keys, jpeg_quality=jpeg_quality, verbose=False)
    print(f"Conversion done → {rlds_path}")
    return rlds_path


def prepare_zarr_sharded_jpeg(
    zarr_path: str,
    out_dir: str,
    obs_keys: list[str],
    shard_frames: int = 1024,
    jpeg_level: int = 75,
) -> str:
    """Convert an existing directory zarr to a sharded zarr-v3 store with JPEG for image keys."""
    from robo_dataset.backends.zarr_backend import ZarrStore
    from robo_dataset.backends.zarr_sharded import ZarrShardedStoreWriter
    from robo_dataset.convert import convert

    label = f"zarr_sharded_jpeg_q{jpeg_level}"
    sharded_path = os.path.join(out_dir, label)
    if os.path.exists(os.path.join(sharded_path, "meta.json")):
        print(f"Reusing existing sharded-jpeg store: {sharded_path}")
        return sharded_path

    print(f"Converting {zarr_path} → {sharded_path} (JPEG q{jpeg_level}, shard_frames={shard_frames}) …")
    os.makedirs(out_dir, exist_ok=True)
    src = ZarrStore(zarr_path)
    writer = ZarrShardedStoreWriter.create(
        path=sharded_path,
        key_shapes={k: src.frame_shape(k) for k in src.keys},
        key_dtypes={k: src.frame_dtype(k) for k in src.keys},
        chunk_frames=1,           # JPEG encodes one frame at a time
        shard_frames=shard_frames,
        image_keys=obs_keys,
        jpeg_level=jpeg_level,
    )
    convert(src, writer, verbose=False)
    print(f"Conversion done → {sharded_path}")
    return sharded_path


def prepare_video(
    zarr_path: str,
    out_dir: str,
    obs_keys: list[str],
    vcodec: str = "libx264",
    crf: int = 23,
    kfi: int = 1,
    fps: float = 10.0,
) -> str:
    """Convert an existing directory zarr to a per-episode MP4 VideoStore."""
    from robo_dataset.backends.zarr_backend import ZarrStore
    from robo_dataset.backends.video import VideoStoreWriter
    from robo_dataset.convert import convert

    label = f"h264_kf{kfi}_crf{crf}"
    video_path = os.path.join(out_dir, label)
    if os.path.exists(os.path.join(video_path, "meta.json")):
        print(f"Reusing existing video store: {video_path}")
        return video_path

    print(f"Converting {zarr_path} → {video_path} ({vcodec} kfi={kfi} crf={crf}) …")
    os.makedirs(out_dir, exist_ok=True)
    src = ZarrStore(zarr_path)
    writer = VideoStoreWriter.create(
        path=video_path,
        key_shapes={k: src.frame_shape(k) for k in src.keys},
        key_dtypes={k: src.frame_dtype(k) for k in src.keys},
        image_keys=obs_keys,
        vcodec=vcodec,
        crf=crf,
        kfi=kfi,
        fps=fps,
    )
    convert(src, writer, verbose=False)
    print(f"Conversion done → {video_path}")
    return video_path


def prepare_zarr_sharded(
    zarr_path: str,
    out_dir: str,
    chunk_frames: int = 8,
    shard_frames: int = 1024,
) -> str:
    """Convert an existing directory zarr to a sharded zarr-v3 store."""
    from robo_dataset.backends.zarr_backend import ZarrStore
    from robo_dataset.backends.zarr_sharded import ZarrShardedStoreWriter
    from robo_dataset.convert import convert

    label = f"zarr_sharded_cf{chunk_frames}"
    sharded_path = os.path.join(out_dir, label)
    if os.path.exists(os.path.join(sharded_path, "meta.json")):
        print(f"Reusing existing sharded store: {sharded_path}")
        return sharded_path

    print(f"Converting {zarr_path} → {sharded_path} (chunk_frames={chunk_frames}) …")
    os.makedirs(out_dir, exist_ok=True)
    src = ZarrStore(zarr_path)
    writer = ZarrShardedStoreWriter.create(
        path=sharded_path,
        key_shapes={k: src.frame_shape(k) for k in src.keys},
        key_dtypes={k: src.frame_dtype(k) for k in src.keys},
        chunk_frames=chunk_frames,
        shard_frames=shard_frames,
    )
    convert(src, writer, verbose=False)
    print(f"Conversion done → {sharded_path}")
    return sharded_path


# ── Part A: zarr_disk vs zarr_sharded vs RLDS throughput ─────────────────────

def benchmark_dataloader(
    zarr_path: str,
    sharded_lz4_path: str,
    sharded_jpeg_path: str,
    rlds_path: str,
    batch_size: int,
    n_epochs: int,
    worker_counts,
    horizon: int = 8,
    video_path: str = None,
    use_shm: bool = False,
):
    from robo_dataset.backends.zarr_backend import ZarrStore
    from robo_dataset.backends.zarr_sharded import ZarrShardedStore
    from robo_dataset.backends.rlds import RLDSFrameStore
    from robo_dataset.dataset import TrajectoryDataset

    print("\n=== Part A: DataLoader throughput ===")
    print(f"zarr_disk:      {zarr_path}")
    print(f"sharded_lz4:    {sharded_lz4_path}")
    print(f"sharded_jpeg:   {sharded_jpeg_path}")
    print(f"rlds:           {rlds_path}")
    if video_path:
        print(f"video:          {video_path}")
    print(f"batch={batch_size}, epochs={n_epochs} measured (+ 1 warmup), horizon={horizon}")

    zarr_src = ZarrStore(zarr_path)
    sharded_lz4_src = ZarrShardedStore(sharded_lz4_path)
    sharded_jpeg_src = ZarrShardedStore(sharded_jpeg_path)
    rlds_src = RLDSFrameStore(rlds_path)
    n_frames = int(zarr_src.episode_ends[-1])

    print(f"Dataset: {n_frames} frames, {zarr_src.n_episodes} episodes")

    stores = [
        ("zarr_disk_lz4",      zarr_src),
        ("zarr_sharded_lz4",   sharded_lz4_src),
        ("zarr_sharded_jpeg",  sharded_jpeg_src),
        ("rlds_jpeg_q75",      rlds_src),
    ]

    if video_path:
        from robo_dataset.backends.video import VideoStore
        video_label = os.path.basename(video_path)
        stores.append((video_label, VideoStore(video_path)))

    if use_shm:
        from robo_dataset.backends.shm import ShmStore
        print("Loading zarr_disk into shared memory (one-time cost) …")
        shm_src = ShmStore.from_store(zarr_src, verbose=True)
        stores.append(("shm_lz4", shm_src))
        print("ShmStore ready.")

    results = {}

    for nw in worker_counts:
        row = {}
        for label, store in stores:
            dataset = TrajectoryDataset(store, horizon=horizon, pad_before=1, pad_after=1)
            loader = dataset.make_dataloader(
                batch_size=batch_size,
                num_workers=nw,
                shuffle=True,
                persistent_workers=(nw > 0),
                pin_memory=(nw > 0),
            )
            elapsed = _run_loader(loader, n_frames, n_epochs)
            fps = _fps(n_frames, elapsed)
            row[label] = fps
            print(f"  workers={nw:2d}  {label:28s}  {fps:8.0f} fps")

        row["speedup_sharded_lz4_vs_disk"]  = row["zarr_sharded_lz4"]  / row["zarr_disk_lz4"]
        row["speedup_sharded_jpeg_vs_disk"] = row["zarr_sharded_jpeg"] / row["zarr_disk_lz4"]
        row["speedup_rlds_vs_disk"]         = row["rlds_jpeg_q75"]     / row["zarr_disk_lz4"]
        row["speedup_sharded_jpeg_vs_rlds"] = row["zarr_sharded_jpeg"] / row["rlds_jpeg_q75"]
        if video_path:
            vl = os.path.basename(video_path)
            row[f"speedup_{vl}_vs_disk"] = row[vl] / row["zarr_disk_lz4"]
            row[f"speedup_{vl}_vs_rlds"] = row[vl] / row["rlds_jpeg_q75"]
        results[str(nw)] = row
        print(f"           {'sharded_lz4 vs disk':28s}  {row['speedup_sharded_lz4_vs_disk']:.2f}x")
        print(f"           {'sharded_jpeg vs disk':28s}  {row['speedup_sharded_jpeg_vs_disk']:.2f}x")
        print(f"           {'rlds vs disk':28s}  {row['speedup_rlds_vs_disk']:.2f}x")
        print(f"           {'sharded_jpeg vs rlds':28s}  {row['speedup_sharded_jpeg_vs_rlds']:.2f}x")
        if video_path:
            vl = os.path.basename(video_path)
            print(f"           {f'{vl} vs disk':28s}  {row[f'speedup_{vl}_vs_disk']:.2f}x")
            print(f"           {f'{vl} vs rlds':28s}  {row[f'speedup_{vl}_vs_rlds']:.2f}x")

    return results


# ── Part A2: chunk_frames sweep (ZarrShardedStore) ────────────────────────────

def benchmark_chunk_frames_sweep(
    zarr_path: str,
    out_dir: str,
    batch_size: int,
    n_epochs: int,
    n_workers: int,
    chunk_frames_list: list[int],
    shard_frames: int = 1024,
    horizon: int = 8,
):
    from robo_dataset.backends.zarr_backend import ZarrStore
    from robo_dataset.backends.zarr_sharded import ZarrShardedStore
    from robo_dataset.dataset import TrajectoryDataset

    print(f"\n=== Part A2: chunk_frames sweep (workers={n_workers}, horizon={horizon}) ===")
    print(f"chunk_frames to test: {chunk_frames_list}")

    zarr_src = ZarrStore(zarr_path)
    n_frames = int(zarr_src.episode_ends[-1])

    results = {}
    for cf in chunk_frames_list:
        sharded_path = prepare_zarr_sharded(zarr_path, out_dir, chunk_frames=cf, shard_frames=shard_frames)
        store = ZarrShardedStore(sharded_path)
        dataset = TrajectoryDataset(store, horizon=horizon, pad_before=1, pad_after=1)
        loader = dataset.make_dataloader(
            batch_size=batch_size,
            num_workers=n_workers,
            shuffle=True,
            persistent_workers=(n_workers > 0),
            pin_memory=(n_workers > 0),
        )
        elapsed = _run_loader(loader, n_frames, n_epochs)
        fps = _fps(n_frames, elapsed)
        results[str(cf)] = fps
        print(f"  chunk_frames={cf:3d}  {fps:8.0f} fps")

    return results


# ── Part B: channels_last conv2d microbenchmark ───────────────────────────────

def _make_conv_stack(n_layers: int = 4, channels: int = 64) -> torch.nn.Module:
    layers = [torch.nn.Conv2d(3, channels, 3, padding=1), torch.nn.ReLU(inplace=True)]
    for _ in range(n_layers - 1):
        layers += [torch.nn.Conv2d(channels, channels, 3, padding=1), torch.nn.ReLU(inplace=True)]
    return torch.nn.Sequential(*layers)


def benchmark_channels_last(batch_size: int, n_warmup: int = 50, n_iters: int = 500):
    if not torch.cuda.is_available():
        print("\n=== Part B: channels_last conv2d ===")
        print("  SKIPPED — no CUDA device available")
        return {}

    device = torch.device("cuda:0")
    H, W = 84, 84

    print("\n=== Part B: channels_last conv2d microbenchmark ===")
    print(f"  shape=({batch_size}, 3, {H}, {W}), warmup={n_warmup}, iters={n_iters}")
    print(f"  device={torch.cuda.get_device_name(device)}")

    model = _make_conv_stack().to(device)
    model.eval()

    results = {}
    for label, memory_format in [
        ("contiguous_nchw", torch.contiguous_format),
        ("channels_last_nhwc", torch.channels_last),
    ]:
        x = torch.randn(batch_size, 3, H, W, device=device).contiguous(
            memory_format=memory_format
        )
        with torch.no_grad():
            for _ in range(n_warmup):
                _ = model(x)
        torch.cuda.synchronize(device)

        t0 = time.perf_counter()
        with torch.no_grad():
            for _ in range(n_iters):
                _ = model(x)
        torch.cuda.synchronize(device)
        elapsed = time.perf_counter() - t0

        ms_per_iter = elapsed / n_iters * 1000
        results[label] = ms_per_iter
        print(f"  {label:28s}  {ms_per_iter:.3f} ms/iter")

    speedup = results["contiguous_nchw"] / results["channels_last_nhwc"]
    results["speedup"] = speedup
    print(f"  {'channels_last speedup':28s}  {speedup:.2f}x")
    return results


# ── entry point ───────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description=__doc__)

    src = ap.add_mutually_exclusive_group()
    src.add_argument("--zarr", help="Existing robo_dataset ZarrStore path (has meta.json)")
    src.add_argument("--hdf5", help="Source Robomimic HDF5; zarr stores will be created under --out")

    ap.add_argument(
        "--obs-keys",
        nargs="+",
        default=["agentview_image", "robot0_eye_in_hand_image"],
        help="Observation image keys (used only when --hdf5 is provided)",
    )
    ap.add_argument("--out", default="/tmp/robo_dataset_bench", help="Output dir for zarr + results")
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--epochs", type=int, default=2, help="Measured epochs per config (+ 1 warmup)")
    ap.add_argument(
        "--workers",
        type=int,
        nargs="+",
        default=[0, 4, 8, 12],
        help="DataLoader worker counts to test (Part A)",
    )
    ap.add_argument(
        "--sweep-workers",
        type=int,
        default=8,
        help="Worker count to use for the chunk_frames sweep (Part A2)",
    )
    ap.add_argument(
        "--chunk-frames",
        type=int,
        default=8,
        help="Default chunk_frames for the sharded store used in Part A",
    )
    ap.add_argument(
        "--shard-frames",
        type=int,
        default=1024,
        help="Shard file size in frames",
    )
    ap.add_argument("--horizon", type=int, default=8, help="Sequence horizon for DataLoader")
    ap.add_argument(
        "--sweep-chunk-frames",
        type=int,
        nargs="+",
        default=[1, 2, 4, 8, 16],
        help="chunk_frames values to sweep in Part A2",
    )
    ap.add_argument(
        "--jpeg-quality",
        type=int,
        default=75,
        help="JPEG quality for RLDS store",
    )
    ap.add_argument(
        "--skip-sweep",
        action="store_true",
        help="Skip the chunk_frames sweep (Part A2)",
    )
    ap.add_argument(
        "--video",
        action="store_true",
        help="Also benchmark VideoStore (H.264 kfi=1); requires torchcodec and PyAV",
    )
    ap.add_argument("--video-crf", type=int, default=23, help="CRF for video encoding")
    ap.add_argument("--video-kfi", type=int, default=1, help="Keyframe interval for video encoding")
    ap.add_argument(
        "--shm",
        action="store_true",
        help="Also benchmark ShmStore (load zarr_disk into shared RAM; zero-copy workers)",
    )
    args = ap.parse_args()

    if args.zarr:
        zarr_path = args.zarr
        out_dir = os.path.dirname(args.zarr)
    elif args.hdf5:
        zarr_path = prepare_zarr(args.hdf5, args.obs_keys, args.out)
        out_dir = args.out
    else:
        ap.error("Provide either --zarr or --hdf5")

    obs_keys = args.obs_keys if hasattr(args, "obs_keys") and args.obs_keys else [
        "agentview_image", "robot0_eye_in_hand_image"
    ]

    sharded_lz4_path = prepare_zarr_sharded(
        zarr_path, out_dir,
        chunk_frames=args.chunk_frames,
        shard_frames=args.shard_frames,
    )

    sharded_jpeg_path = prepare_zarr_sharded_jpeg(
        zarr_path, out_dir,
        obs_keys=obs_keys,
        shard_frames=args.shard_frames,
        jpeg_level=args.jpeg_quality,
    )

    rlds_path = prepare_rlds(
        zarr_path, out_dir,
        obs_keys=obs_keys,
        jpeg_quality=args.jpeg_quality,
    )

    video_path = None
    if args.video:
        video_path = prepare_video(
            zarr_path, out_dir,
            obs_keys=obs_keys,
            vcodec="libx264",
            crf=args.video_crf,
            kfi=args.video_kfi,
        )

    dl_results = benchmark_dataloader(
        zarr_path, sharded_lz4_path, sharded_jpeg_path, rlds_path,
        args.batch, args.epochs, args.workers, args.horizon,
        video_path=video_path,
        use_shm=args.shm,
    )

    sweep_results = {}
    if not args.skip_sweep:
        sweep_results = benchmark_chunk_frames_sweep(
            zarr_path, out_dir,
            batch_size=args.batch,
            n_epochs=args.epochs,
            n_workers=args.sweep_workers,
            chunk_frames_list=args.sweep_chunk_frames,
            shard_frames=args.shard_frames,
            horizon=args.horizon,
        )

    cl_results = benchmark_channels_last(args.batch)

    out = {
        "config": {
            "zarr": zarr_path,
            "sharded_lz4": sharded_lz4_path,
            "sharded_jpeg": sharded_jpeg_path,
            "rlds": rlds_path,
            "video": video_path,
            "shm": args.shm,
            "batch": args.batch,
            "epochs": args.epochs,
            "workers": args.workers,
            "horizon": args.horizon,
            "chunk_frames": args.chunk_frames,
            "shard_frames": args.shard_frames,
        },
        "dataloader_fps": dl_results,
        "chunk_frames_sweep": sweep_results,
        "channels_last": cl_results,
    }

    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "benchmark_acceleration.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nResults saved → {out_path}")


if __name__ == "__main__":
    main()
