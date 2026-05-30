"""
Fair LeRobot video benchmark using official decoders.

Runs in robocasa env (has lerobot 0.3.3, torchcodec 0.5, torch 2.7).

Tests:
  A) zarr_disk_jpeg_q75         — baseline (zarr2 in rerun_env, zarr3 here; same on-disk format)
  B) h264_kf1_lerobot_official  — H.264 kf=1, official decode_video_frames_torchcodec (new decoder/frame)
  C) h264_kf1_cached_decoder    — H.264 kf=1, cached VideoDecoder per worker (optimal torchcodec path)
  D) h264_kf2_cached_decoder    — H.264 kf=2 (≈lerobot default), cached VideoDecoder
  E) av1_kf2_lerobot_official   — AV1 kf=2 crf=30 (official LeRobot defaults), decode_video_frames_torchcodec
  F) av1_kf2_cached_decoder     — same AV1 files, cached VideoDecoder

Usage:
  conda run -n robocasa python -u tools/benchmark_lerobot.py \
      --hdf5  data/robomimic/datasets/lift/ph/image.hdf5 \
      --out   /mnt/ssdB/erik/cupid_data/lerobot_bench \
      --epochs 2
"""
import argparse, os, shutil, time, json, io, pathlib, importlib, logging
import numpy as np
import h5py
import torch
from torch.utils.data import Dataset, DataLoader
import av as av_mod
import zarr

logging.basicConfig(level=logging.WARNING)

_WORKER_COUNTS = [0, 1, 2, 4, 6, 8, 10, 12]
_BENCH_EPOCHS  = 2
_BENCH_BATCH   = 32
_FPS           = 10.0


def _path_stats(p):
    p = pathlib.Path(p)
    if not p.exists(): return 0, 0
    if p.is_file(): return 1, p.stat().st_size
    n, total = 0, 0
    for f in p.rglob('*'):
        if f.is_file(): n += 1; total += f.stat().st_size
    return n, total


def _load_hdf5(hdf5_path, cam_keys):
    parts = {k: [] for k in cam_keys}
    with h5py.File(hdf5_path, 'r') as f:
        for i in range(len(f['data'])):
            for k in cam_keys:
                parts[k].append(f['data'][f'demo_{i}']['obs'][k][:])
    return {k: np.concatenate(parts[k]) for k in cam_keys}


def _load_episode_ends(hdf5_path):
    ends, step = [], 0
    with h5py.File(hdf5_path, 'r') as f:
        for i in range(len(f['data'])):
            step += f['data'][f'demo_{i}']['actions'].shape[0]
            ends.append(step)
    return np.array(ends, dtype=np.int64)


# ── Encoders ──────────────────────────────────────────────────────────────────

def _encode_video_episodes(out_dir, images, episode_ends, cam_keys,
                            vcodec='libx264', crf=23, kfi=1, fps=10):
    """Write one MP4 per episode per camera using PyAV."""
    if os.path.exists(out_dir): shutil.rmtree(out_dir)
    os.makedirs(out_dir)
    t0, prev = time.perf_counter(), 0
    codec_opts = {'crf': str(crf), 'g': str(kfi), 'preset': 'fast'}
    if 'svt' in vcodec or 'av1' in vcodec:
        codec_opts = {'crf': str(crf), 'g': str(kfi)}  # svtav1 doesn't support preset
    for ep_idx, end in enumerate(episode_ends):
        for cam_key in cam_keys:
            ep_frames = images[cam_key][prev:int(end)]
            mp4 = os.path.join(out_dir, cam_key, f'episode_{ep_idx:06d}.mp4')
            os.makedirs(os.path.dirname(mp4), exist_ok=True)
            c = av_mod.open(mp4, 'w')
            s = c.add_stream(vcodec, rate=fps)
            s.width, s.height, s.pix_fmt = ep_frames.shape[2], ep_frames.shape[1], 'yuv420p'
            s.options = codec_opts
            for arr in ep_frames:
                for p in s.encode(av_mod.VideoFrame.from_ndarray(arr, format='rgb24')):
                    c.mux(p)
            for p in s.encode(): c.mux(p)
            c.close()
        prev = int(end)
    return time.perf_counter() - t0, *_path_stats(out_dir)


def _build_zarr(out_dir, images, cam_keys):
    """zarr3 disk store with jpeg75 compression (same on-disk format as rerun_env zarr2)."""
    import concurrent.futures
    from PIL import Image as PIL_Image
    if os.path.exists(out_dir): shutil.rmtree(out_dir)
    store = zarr.storage.LocalStore(out_dir)
    root  = zarr.group(store=store, overwrite=True)
    t0 = time.perf_counter()
    for k, arr in images.items():
        N, H, W, C = arr.shape
        # zarr3 uses codec pipeline syntax
        from zarr.codecs import BytesCodec
        try:
            import imagecodecs
            from zarr.codecs import ImagecodecsBlosc2Codec  # not standard
        except Exception:
            pass
        # Simpler: write as raw uint8 zarr3 and we'll handle jpeg decoding manually
        # Actually, use numcodecs via config_entries
        import numcodecs
        # zarr3 + numcodecs via zarr2-compat store approach
        # Fallback: just use raw zarr3 with blosc
        from zarr.codecs import BloscCodec
        za = root.require_array(k, shape=(N, H, W, C),
                                chunks=(1, H, W, C), dtype=np.uint8)
        def _copy(i, _za=za, _arr=arr): _za[i] = _arr[i]
        with concurrent.futures.ThreadPoolExecutor(8) as ex:
            list(ex.map(_copy, range(N)))
    return time.perf_counter() - t0, *_path_stats(out_dir)


# ── Dataset classes ───────────────────────────────────────────────────────────

class _ZarrDataset(Dataset):
    """zarr3 disk dataset."""
    def __init__(self, zarr_path, cam_keys, n):
        self._path = zarr_path
        self.cam_keys = cam_keys
        self._n = n
        self._root = None
        self._pid  = None

    def _get_root(self):
        pid = os.getpid()
        if self._pid != pid:
            store = zarr.storage.LocalStore(self._path, read_only=True)
            self._root = zarr.open_group(store=store, mode='r')
            self._pid  = pid
        return self._root

    def __len__(self): return self._n
    def __getitem__(self, i):
        r = self._get_root()
        return [r[k][int(i)] for k in self.cam_keys]


class _LeRobotOfficialDataset(Dataset):
    """Official LeRobot: new VideoDecoder per frame (as lerobot does it)."""
    def __init__(self, out_dir, cam_keys, episode_ends, n, fps=_FPS):
        self.out_dir      = out_dir
        self.cam_keys     = cam_keys
        self.episode_ends = episode_ends
        self._n  = n
        self._fps = fps
        self._tol = 1.0 / fps / 2.0

    def __len__(self): return self._n

    def __getitem__(self, gi):
        from lerobot.datasets.video_utils import decode_video_frames_torchcodec
        gi      = int(gi)
        ep_idx  = int(np.searchsorted(self.episode_ends, gi, side='right'))
        loc_idx = gi - (int(self.episode_ends[ep_idx - 1]) if ep_idx > 0 else 0)
        ts      = loc_idx / self._fps
        frames  = []
        for cam_key in self.cam_keys:
            mp4 = os.path.join(self.out_dir, cam_key, f'episode_{ep_idx:06d}.mp4')
            decoded = decode_video_frames_torchcodec(mp4, [ts], self._tol, device='cpu')
            frames.append(decoded[0])   # shape [C, H, W], float32 0-1
        return frames


class _CachedDecoderDataset(Dataset):
    """Cached torchcodec VideoDecoder per (cam, episode) per worker process."""
    def __init__(self, out_dir, cam_keys, episode_ends, n, fps=_FPS):
        self.out_dir      = out_dir
        self.cam_keys     = cam_keys
        self.episode_ends = episode_ends
        self._n   = n
        self._fps = fps
        self._decoders = {}
        self._pid      = os.getpid()

    def _get_decoder(self, cam_key, ep_idx):
        from torchcodec.decoders import VideoDecoder
        pid = os.getpid()
        if pid != self._pid:
            self._decoders = {}
            self._pid = pid
        key = (cam_key, ep_idx)
        if key not in self._decoders:
            mp4 = os.path.join(self.out_dir, cam_key, f'episode_{ep_idx:06d}.mp4')
            self._decoders[key] = VideoDecoder(mp4, device='cpu', seek_mode='approximate')
        return self._decoders[key]

    def __len__(self): return self._n

    def __getitem__(self, gi):
        gi      = int(gi)
        ep_idx  = int(np.searchsorted(self.episode_ends, gi, side='right'))
        loc_idx = gi - (int(self.episode_ends[ep_idx - 1]) if ep_idx > 0 else 0)
        frames  = []
        for cam_key in self.cam_keys:
            dec    = self._get_decoder(cam_key, ep_idx)
            result = dec.get_frames_at(indices=[loc_idx])
            frames.append(result.data[0])  # [C, H, W] float32 0-1
        return frames


# ── Benchmark core ────────────────────────────────────────────────────────────

def _bench_dl(make_dataset, worker_counts=_WORKER_COUNTS,
              n_epochs=_BENCH_EPOCHS, batch=_BENCH_BATCH, mp_context=None):
    scaling = {}
    for nw in worker_counts:
        ds = make_dataset()
        dl = DataLoader(ds, batch_size=batch, num_workers=nw,
                        shuffle=True, pin_memory=False,
                        persistent_workers=(nw > 0),
                        prefetch_factor=2 if nw > 0 else None,
                        multiprocessing_context=mp_context if nw > 0 else None)
        n, t0 = 0, time.perf_counter()
        for _ in range(n_epochs):
            for b in dl:
                if isinstance(b, (list, tuple)) and len(b) > 0:
                    n += b[0].shape[0] if hasattr(b[0], 'shape') else len(b[0])
                elif hasattr(b, 'shape'):
                    n += b.shape[0]
        fps = round(n / max(time.perf_counter() - t0, 1e-6), 1)
        scaling[nw] = fps
        print(f'    {nw}w → {fps:.0f} fps', flush=True)
    return scaling


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--hdf5',     required=True)
    ap.add_argument('--out',      required=True)
    ap.add_argument('--epochs',   type=int, default=_BENCH_EPOCHS)
    ap.add_argument('--cam-keys', default='agentview_image,robot0_eye_in_hand_image')
    args = ap.parse_args()

    cam_keys = args.cam_keys.split(',')
    out_root = pathlib.Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)
    n_ep = args.epochs

    print('Loading images from HDF5 …')
    images       = _load_hdf5(args.hdf5, cam_keys)
    episode_ends = _load_episode_ends(args.hdf5)
    N = images[cam_keys[0]].shape[0]
    raw_bytes = N * len(cam_keys) * images[cam_keys[0]][0].size
    print(f'  {N} frames × {len(cam_keys)} cams  raw={raw_bytes/1e6:.0f} MB')
    print(f'  epochs={n_ep}  workers={_WORKER_COUNTS}\n')

    out_json = out_root / 'lerobot_bench_results.json'
    _done, results = set(), []
    if out_json.exists():
        try:
            ex = json.loads(out_json.read_text())
            results = ex.get('results', [])
            _done   = {r['label'] for r in results}
            print(f'  Resuming: {sorted(_done)}\n')
        except Exception: pass

    def _save():
        with open(out_json, 'w') as f:
            json.dump({'N': N, 'raw_mb': round(raw_bytes/1e6,1),
                       'worker_counts': _WORKER_COUNTS, 'results': results}, f, indent=2)

    def _row(label, fmt, size_mb, ratio, sc):
        results.append(dict(label=label, format=fmt, size_mb=round(size_mb,2),
                            compression_ratio=round(ratio,2),
                            scaling={str(k): round(v,1) for k,v in sc.items()}))
        _save()

    # ── A. zarr_disk_jpeg_q75 baseline ───────────────────────────────────────
    # Reuse files already built by main benchmark (zarr2 format = zarr3 readable)
    # but zarr3 API may differ → just test read speed using zarr3
    zarr_src = '/mnt/ssdB/erik/cupid_data/storage_bench/zarr_disk_jpeg_q75'
    label = 'zarr_disk_jpeg_q75_zarr3'
    if label not in _done:
        print(f'[{label}]')
        _, n_bytes = _path_stats(zarr_src)
        ratio = raw_bytes / n_bytes if n_bytes else 0
        print(f'  size={n_bytes/1e6:.1f} MB  ratio={ratio:.1f}x  → scaling:')
        try:
            sc = _bench_dl(lambda: _ZarrDataset(zarr_src, cam_keys, N), n_epochs=n_ep)
            _row(label, 'zarr_disk', n_bytes/1e6, ratio, sc)
        except Exception:
            import traceback; traceback.print_exc()
    else:
        print(f'[{label}] (skip)')

    # ── B. H.264 kf=1, official LeRobot decode_video_frames_torchcodec ───────
    h264kf1_dir = str(out_root / 'h264_kf1')
    label = 'h264_kf1_official'
    if label not in _done:
        print(f'\n[{label}]')
        if not os.path.exists(h264kf1_dir):
            print('  Encoding H.264 kf=1 …', flush=True)
            enc_s, _, n_bytes = _encode_video_episodes(
                h264kf1_dir, images, episode_ends, cam_keys, 'libx264', crf=23, kfi=1)
            print(f'  enc={enc_s:.0f}s', flush=True)
        _, n_bytes = _path_stats(h264kf1_dir)
        ratio = raw_bytes / n_bytes if n_bytes else 0
        print(f'  size={n_bytes/1e6:.1f} MB  ratio={ratio:.1f}x  → scaling:')
        try:
            sc = _bench_dl(
                lambda: _LeRobotOfficialDataset(h264kf1_dir, cam_keys, episode_ends, N),
                n_epochs=n_ep, mp_context='spawn')
            _row(label, 'lerobot_official', n_bytes/1e6, ratio, sc)
        except Exception:
            import traceback; traceback.print_exc()
    else:
        print(f'\n[{label}] (skip)')

    # ── C. H.264 kf=1, cached VideoDecoder per worker ─────────────────────────
    label = 'h264_kf1_cached'
    if label not in _done:
        print(f'\n[{label}]')
        _, n_bytes = _path_stats(h264kf1_dir)
        ratio = raw_bytes / n_bytes if n_bytes else 0
        print(f'  size={n_bytes/1e6:.1f} MB  ratio={ratio:.1f}x  → scaling:')
        try:
            sc = _bench_dl(
                lambda: _CachedDecoderDataset(h264kf1_dir, cam_keys, episode_ends, N),
                n_epochs=n_ep)
            _row(label, 'lerobot_cached', n_bytes/1e6, ratio, sc)
        except Exception:
            import traceback; traceback.print_exc()
    else:
        print(f'\n[{label}] (skip)')

    # ── D. H.264 kf=2, cached VideoDecoder (≈ lerobot default codec) ──────────
    h264kf2_dir = str(out_root / 'h264_kf2')
    label = 'h264_kf2_cached'
    if label not in _done:
        print(f'\n[{label}]')
        if not os.path.exists(h264kf2_dir):
            print('  Encoding H.264 kf=2 …', flush=True)
            enc_s, _, n_bytes = _encode_video_episodes(
                h264kf2_dir, images, episode_ends, cam_keys, 'libx264', crf=23, kfi=2)
        _, n_bytes = _path_stats(h264kf2_dir)
        ratio = raw_bytes / n_bytes if n_bytes else 0
        print(f'  size={n_bytes/1e6:.1f} MB  ratio={ratio:.1f}x  → scaling:')
        try:
            sc = _bench_dl(
                lambda: _CachedDecoderDataset(h264kf2_dir, cam_keys, episode_ends, N),
                n_epochs=n_ep)
            _row(label, 'lerobot_cached', n_bytes/1e6, ratio, sc)
        except Exception:
            import traceback; traceback.print_exc()
    else:
        print(f'\n[{label}] (skip)')

    # ── E. AV1 kf=2 crf=30, official decode_video_frames_torchcodec ──────────
    av1kf2_dir = str(out_root / 'av1_kf2')
    label = 'av1_kf2_official'
    if label not in _done:
        print(f'\n[{label}]  (encoding AV1 is slow, ~5-10 min)')
        if not os.path.exists(av1kf2_dir):
            print('  Encoding AV1 kf=2 crf=30 …', flush=True)
            enc_s, _, n_bytes = _encode_video_episodes(
                av1kf2_dir, images, episode_ends, cam_keys, 'libsvtav1', crf=30, kfi=2)
            print(f'  enc={enc_s:.0f}s', flush=True)
        _, n_bytes = _path_stats(av1kf2_dir)
        ratio = raw_bytes / n_bytes if n_bytes else 0
        print(f'  size={n_bytes/1e6:.1f} MB  ratio={ratio:.1f}x  → scaling:')
        try:
            sc = _bench_dl(
                lambda: _LeRobotOfficialDataset(av1kf2_dir, cam_keys, episode_ends, N),
                n_epochs=n_ep, mp_context='spawn')
            _row(label, 'lerobot_official', n_bytes/1e6, ratio, sc)
        except Exception:
            import traceback; traceback.print_exc()
    else:
        print(f'\n[{label}] (skip)')

    # ── F. AV1 kf=2, cached VideoDecoder ─────────────────────────────────────
    label = 'av1_kf2_cached'
    if label not in _done:
        print(f'\n[{label}]')
        _, n_bytes = _path_stats(av1kf2_dir)
        ratio = raw_bytes / n_bytes if n_bytes else 0
        print(f'  size={n_bytes/1e6:.1f} MB  ratio={ratio:.1f}x  → scaling:')
        try:
            sc = _bench_dl(
                lambda: _CachedDecoderDataset(av1kf2_dir, cam_keys, episode_ends, N),
                n_epochs=n_ep)
            _row(label, 'lerobot_cached', n_bytes/1e6, ratio, sc)
        except Exception:
            import traceback; traceback.print_exc()
    else:
        print(f'\n[{label}] (skip)')

    # ── Summary ───────────────────────────────────────────────────────────────
    wc = _WORKER_COUNTS
    W  = 40 + 8 + 7 + 6 * len(wc)
    def _f(v):
        if v >= 10000: return f'{v/1000:.0f}k'
        if v >= 1000:  return f'{v/1000:.1f}k'
        return f'{v:.0f}'
    print(f'\n{"─"*W}')
    print(f'LEROBOT BENCHMARK  N={N}  epochs={n_ep}')
    print(f'{"─"*W}')
    print(f'  {"Backend":<38} {"MB":>6} {"Ratio":>6}  ' +
          '  '.join(f'{w}w' for w in wc))
    print('  ' + '-' * (W - 2))
    for r in results:
        sc  = r['scaling']
        fps = [sc.get(str(w), sc.get(w, 0)) for w in wc]
        fps_str = '  '.join(f'{_f(v):>5}' for v in fps)
        print(f'  {r["label"]:<38} {r["size_mb"]:>5.1f} {r["compression_ratio"]:>5.1f}x  {fps_str}')
    print(f'{"─"*W}')
    print(f'Results → {out_json}')


if __name__ == '__main__':
    main()
