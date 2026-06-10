"""Smoke-check VideoStore write/read round-trip in the current env.

Writes a tiny synthetic dataset to --out, reads it back, prints the decode
backend used and the image PSNR. Used to validate the decoder compat layer
across conda envs (modern torchcodec / legacy 0.0.3 / PyAV fallback).
"""

import argparse
import shutil

import numpy as np

from robo_dataset.backends.video import VideoStore, VideoStoreWriter, _pick_decoder_backend
from robo_dataset.convert import convert


class _SyntheticSource:
    def __init__(self, n_episodes=3, ep_len=20, hw=32):
        rng = np.random.default_rng(0)
        self._eps = []
        yy, xx = np.mgrid[0:hw, 0:hw].astype(np.float64) / hw
        for e in range(n_episodes):
            # Smooth moving gradient — compressible content, unlike noise.
            frames = []
            for t in range(ep_len):
                phase = (t / ep_len + e * 0.3) * 2 * np.pi
                r = 127 + 120 * np.sin(2 * np.pi * xx + phase)
                g = 127 + 120 * np.cos(2 * np.pi * yy + phase)
                b = 127 + 120 * np.sin(2 * np.pi * (xx + yy) - phase)
                frames.append(np.stack([r, g, b], axis=-1))
            img = np.clip(np.stack(frames), 0, 255).astype(np.uint8)
            self._eps.append({
                "img": img,
                "action": rng.standard_normal((ep_len, 7)).astype(np.float32),
            })
        self.keys = ["img", "action"]
        self.episode_ends = np.cumsum([ep_len] * n_episodes).astype(np.int64)
        self.n_episodes = n_episodes

    def iter_episodes(self):
        yield from self._eps

    def frame_shape(self, key):
        return self._eps[0][key].shape[1:]

    def frame_dtype(self, key):
        return self._eps[0][key].dtype


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    shutil.rmtree(args.out, ignore_errors=True)

    src = _SyntheticSource()
    writer = VideoStoreWriter.create(
        args.out,
        key_shapes={k: src.frame_shape(k) for k in src.keys},
        key_dtypes={k: src.frame_dtype(k) for k in src.keys},
        image_keys=["img"],
    )
    convert(src, writer, verbose=False)

    store = VideoStore(args.out)
    print("decode backend:", _pick_decoder_backend())

    # action round-trip must be exact
    ref_actions = np.concatenate([ep["action"] for ep in src._eps])
    np.testing.assert_array_equal(store.get_sequence("action", 0, 60), ref_actions)

    # image round-trip: lossy — check PSNR, including a cross-episode read
    ref_imgs = np.concatenate([ep["img"] for ep in src._eps])
    got = store.get_sequence("img", 10, 30)  # spans episodes 0 and 1
    ref = ref_imgs[10:30]
    mse = np.mean((got.astype(np.float64) - ref.astype(np.float64)) ** 2)
    psnr = 10 * np.log10(255.0**2 / max(mse, 1e-12))
    print(f"img PSNR: {psnr:.1f} dB (shape {got.shape})")
    assert got.shape == ref.shape
    assert psnr > 25, f"PSNR too low: {psnr}"
    print("ROUNDTRIP OK")


if __name__ == "__main__":
    main()
