"""Mock training pipeline demonstrating all three transform tiers.

Usage:
    python examples/mock_train.py \
        store=zarr_disk \
        store.path=/mnt/ssdB/erik/cupid_data/storage_bench/zarr_disk_jpeg_q75 \
        cache.root=/tmp/rd_cache \
        num_workers=8

The pipeline:
  Tier 1 (disk): ResizeImages   — applied offline, cached on first run.
  Tier 2 (cpu):  SplitObsAction + ImageToTensor — applied in DataLoader collate_fn.
  Tier 3 (gpu):  RandomCrop    — applied after DataLoader returns, on GPU.
"""

import hydra
from omegaconf import DictConfig, OmegaConf
import torch
from tqdm import tqdm

from robo_dataset.dataset import TrajectoryDataset
from robo_dataset.transforms.cpu_batch import (
    SplitObsAction, ImageToTensor, ComposeCPU, make_collate_fn
)
from robo_dataset.transforms.disk_cached import ResizeImages, apply_disk_cached


def _keys_from_shape_meta(shape_meta):
    """Return (obs_keys, image_keys) derived from shape_meta config."""
    obs_keys = list(shape_meta.obs.keys())
    image_keys = [k for k, v in shape_meta.obs.items() if v.type == "rgb"]
    return obs_keys, image_keys


@hydra.main(config_path="../conf", config_name="train", version_base="1.3")
def main(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")

    obs_keys, image_keys = _keys_from_shape_meta(cfg.shape_meta)
    print(f"obs_keys={obs_keys}  image_keys={image_keys}")

    # ── Instantiate store ──────────────────────────────────────────────────────
    store = hydra.utils.instantiate(cfg.store)
    print(f"Source store: {store.n_episodes} episodes, {store.n_frames} frames")

    # ── Tier 1: Disk-cached transform ─────────────────────────────────────────
    if cfg.cache.enabled:
        disk_transform = ResizeImages(image_keys=image_keys, target_h=84, target_w=84)
        store = apply_disk_cached(
            store,
            disk_transform,
            cache_root=cfg.cache.root,
            image_keys=image_keys,
        )
        print("Using disk-cached store (transform applied offline)")

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
    # Build from shape_meta so obs_keys / image_keys stay in sync automatically.
    cpu_transform = ComposeCPU([
        SplitObsAction(n_obs_steps=cfg.n_obs_steps, obs_keys=obs_keys, action_key="action"),
        ImageToTensor(image_keys=image_keys),
    ])
    collate_fn = make_collate_fn(cpu_transform)

    loader = dataset.make_dataloader(
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        collate_fn=collate_fn,
    )

    # ── Tier 3: GPU batch transform ────────────────────────────────────────────
    gpu_transform = hydra.utils.instantiate(cfg.gpu_transform)

    # ── Training loop ──────────────────────────────────────────────────────────
    print(f"\nRunning {cfg.max_epochs} epochs on {device}")
    for epoch in range(cfg.max_epochs):
        total_batches = 0
        for batch in tqdm(loader, desc=f"Epoch {epoch + 1}", leave=False):
            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
            batch = gpu_transform(batch)

            imgs = batch[image_keys[0]]      # (B, n_obs_steps, C, H, W)
            actions = batch["action"]         # (B, horizon, action_dim)

            assert imgs.ndim == 5 and imgs.shape[1] == cfg.n_obs_steps, (
                f"Expected img shape (B, {cfg.n_obs_steps}, C, H, W), got {tuple(imgs.shape)}"
            )
            assert actions.ndim == 3 and actions.shape[1] == cfg.horizon, (
                f"Expected action shape (B, {cfg.horizon}, D), got {tuple(actions.shape)}"
            )
            total_batches += 1

        print(
            f"Epoch {epoch + 1}: {total_batches} batches | "
            f"img {tuple(imgs.shape)} | action {tuple(actions.shape)}"
        )


if __name__ == "__main__":
    main()
