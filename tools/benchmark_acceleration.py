"""Acceleration benchmark: ShmStore vs zarr_disk, and channels_last conv2d.

Part A — DataLoader throughput (frames/s):
    ShmStore (RAM)  vs  ZarrStore (disk, lz4) at several worker counts.
    If --zarr points to a robo_dataset ZarrStore with meta.json it is used
    directly.  Otherwise --hdf5 is used to create a fresh zarr store under
    --out, which is then used as the zarr source.

Part B — channels_last conv2d microbenchmark:
    Contiguous (NCHW) vs channels_last (NHWC) layout on a single forward pass
    through a small conv stack, on the shape that robotics models actually use:
    (B*T, 3, 84, 84) float32.  Reports the speedup from channels_last on the
    current GPU.

Usage (from an existing robo_dataset zarr store)::

    conda run -n robocasa python tools/benchmark_acceleration.py \\
        --zarr  /path/to/existing.zarr \\
        --batch 32 --epochs 2

Usage (from raw HDF5 — creates a new zarr under --out)::

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
    from robo_dataset.backends.zarr_backend import ZarrStore, ZarrStoreWriter
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


# ── Part A: ShmStore vs zarr throughput ───────────────────────────────────────

def benchmark_dataloader(zarr_path: str, batch_size: int, n_epochs: int, worker_counts):
    from robo_dataset.backends.zarr_backend import ZarrStore
    from robo_dataset.backends.shm import ShmStore
    from robo_dataset.dataset import TrajectoryDataset

    print("\n=== Part A: DataLoader throughput ===")
    print(f"zarr: {zarr_path}")
    print(f"batch={batch_size}, epochs={n_epochs} measured (+ 1 warmup)")

    zarr_src = ZarrStore(zarr_path)
    n_frames = int(zarr_src.episode_ends[-1])

    print(f"Dataset: {n_frames} frames, {zarr_src.n_episodes} episodes")
    print("Loading ShmStore … ", end="", flush=True)
    t_load = time.perf_counter()
    shm = ShmStore.from_store(zarr_src, verbose=False)
    t_load = time.perf_counter() - t_load
    print(f"done in {t_load:.1f}s")

    results = {}

    for nw in worker_counts:
        row = {}
        for label, store in [("zarr_disk_lz4", zarr_src), ("shm_ram", shm)]:
            dataset = TrajectoryDataset(store, horizon=2, pad_before=1, pad_after=1)
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
            print(f"  workers={nw:2d}  {label:20s}  {fps:8.0f} fps")

        speedup = row["shm_ram"] / row["zarr_disk_lz4"]
        row["speedup"] = speedup
        results[str(nw)] = row
        print(f"           {'shm speedup':20s}  {speedup:.2f}x")

    return results, t_load


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
    src.add_argument("--hdf5", help="Source Robomimic HDF5; a zarr store will be created under --out")

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
        help="DataLoader worker counts to test",
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

    dl_results, shm_load_s = benchmark_dataloader(
        zarr_path, args.batch, args.epochs, args.workers
    )
    cl_results = benchmark_channels_last(args.batch)

    out = {
        "config": {
            "zarr": zarr_path,
            "batch": args.batch,
            "epochs": args.epochs,
            "workers": args.workers,
        },
        "shm_load_seconds": shm_load_s,
        "dataloader_fps": dl_results,
        "channels_last": cl_results,
    }

    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "benchmark_acceleration.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nResults saved → {out_path}")


if __name__ == "__main__":
    main()
