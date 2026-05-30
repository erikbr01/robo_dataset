"""Mock training pipeline demonstrating all three transform tiers.

Usage:
    python examples/mock_train.py \
        store=zarr_disk \
        store.path=/mnt/ssdB/erik/cupid_data/storage_bench/zarr_disk_jpeg_q75 \
        cache.root=/tmp/rd_cache \
        num_workers=8

The pipeline:
  Tier 1 (disk): ResizeImages — applied offline, cached on first run.
  Tier 2 (cpu):  ActionChunk  — applied in DataLoader collate_fn.
  Tier 3 (gpu):  RandomCrop   — applied after DataLoader returns, on GPU.
"""

import hydra
from omegaconf import DictConfig, OmegaConf
import torch
from tqdm import tqdm

import robo_dataset
from robo_dataset.dataset import TrajectoryDataset
from robo_dataset.transforms.cpu_batch import make_collate_fn
from robo_dataset.transforms.disk_cached import apply_disk_cached


@hydra.main(config_path="../conf", config_name="train", version_base="1.3")
def main(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")

    # ── Instantiate store ──────────────────────────────────────────────────────
    store = hydra.utils.instantiate(cfg.store)
    print(f"Source store: {store.n_episodes} episodes, {store.n_frames} frames")

    # ── Tier 1: Disk-cached transform ─────────────────────────────────────────
    if cfg.cache.enabled:
        disk_transform = hydra.utils.instantiate(cfg.disk_transforms)
        store = apply_disk_cached(
            store,
            disk_transform,
            cache_root=cfg.cache.root,
            image_keys=list(cfg.cache.image_keys),
        )
        print(f"Using disk-cached store (transform applied offline)")

    # ── Build TrajectoryDataset ────────────────────────────────────────────────
    dataset = TrajectoryDataset(
        store=store,
        horizon=cfg.horizon,
        pad_before=cfg.pad_before,
        pad_after=cfg.pad_after,
        val_ratio=cfg.val_ratio,
        seed=cfg.seed,
    )
    val_dataset = dataset.get_validation_dataset()
    print(f"Train samples: {len(dataset)}  Val samples: {len(val_dataset)}")

    # ── Tier 2: CPU batch transform (collate_fn) ───────────────────────────────
    cpu_transform = hydra.utils.instantiate(cfg.cpu_transform)
    collate_fn = make_collate_fn(cpu_transform)

    loader = dataset.make_dataloader(
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        collate_fn=collate_fn,
    )

    # ── Tier 3: GPU batch transform ────────────────────────────────────────────
    gpu_transform = hydra.utils.instantiate(cfg.gpu_transform)

    # ── Mock model ────────────────────────────────────────────────────────────
    # A tiny linear "model" just to benchmark the full pipeline
    obs_key = list(cfg.cache.image_keys)[0]
    dummy_model = torch.nn.Linear(1, 1).to(device)

    # ── Training loop ──────────────────────────────────────────────────────────
    print(f"\nRunning {cfg.max_epochs} epochs on {device}")
    for epoch in range(cfg.max_epochs):
        total_batches = 0
        for batch in tqdm(loader, desc=f"Epoch {epoch + 1}", leave=False):
            # Move to GPU
            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}

            # Tier 3: GPU transform
            batch = gpu_transform(batch)

            # Mock forward (just inspect shapes)
            imgs = batch[obs_key]  # (B, n_obs, C, H, W) or (B, n_obs, C, crop_h, crop_w)
            actions = batch["action"]  # (B, T, action_dim)
            total_batches += 1

        print(f"Epoch {epoch + 1}: {total_batches} batches | "
              f"img {tuple(imgs.shape)} | action {tuple(actions.shape)}")


if __name__ == "__main__":
    main()
