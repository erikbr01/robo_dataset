"""
Storage compression benchmark for image datasets.

Tests three orthogonal dimensions:
  A) Codec:      raw vs lz4 vs zstd vs j2k vs jpeg vs png vs webp  (zarr disk)
  B) Location:   disk vs RAM vs HDF5                                (codec=raw/j2k)
  C) Format:     zarr vs hdf5 vs rrd vs lerobot vs robodm

All backends tested with torch DataLoader forked workers scaling 0–12,
simulating real training conditions.

Usage:
  conda run -n rerun_env python tools/benchmark_storage.py \
      --hdf5  data/robomimic/datasets/lift/ph/image.hdf5 \
      --out   /mnt/ssdB/erik/cupid_data/storage_bench \
      --epochs 10
"""
import argparse, os, shutil, time, json, io, tempfile
import pathlib, concurrent.futures, threading
import numpy as np
import h5py
import zarr
import numcodecs
import sys
from PIL import Image as PILImage

import rerun as rr
import rerun_bindings.rerun_bindings as rb
import rerun.server as rr_srv
import rerun.catalog as rr_cat
import datafusion as df
from robodm import Trajectory
import av as av_mod

import torch
from torch.utils.data import Dataset, DataLoader

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))
from diffusion_policy.codecs.imagecodecs_numcodecs import (
    register_codecs, Jpeg2k, Jpeg, Png, Webp
)
register_codecs()

_WORKER_COUNTS = [0, 1, 2, 4, 6, 8, 10, 12]
_BENCH_EPOCHS  = 10
_BENCH_BATCH   = 32


def _identity_collate(x): return x


# ── Dataset classes (fork-safe for torch DataLoader workers) ──────────────────

class _ZarrDataset(Dataset):
    """Works for disk path and in-RAM MemoryStore root (copy-on-write after fork)."""
    def __init__(self, zarr_path_or_root, cam_keys, n):
        self._path = zarr_path_or_root if isinstance(zarr_path_or_root, str) else None
        self._root = None if isinstance(zarr_path_or_root, str) else zarr_path_or_root
        self.cam_keys = cam_keys
        self._n = n
        self._open_root = None
        self._open_pid  = None

    def _get_root(self):
        if self._root is not None:
            return self._root
        pid = os.getpid()
        if self._open_pid != pid:
            self._open_root = zarr.open(self._path, mode='r')
            self._open_pid  = pid
        return self._open_root

    def __len__(self): return self._n
    def __getitem__(self, i):
        r = self._get_root()
        return [r[k][int(i)] for k in self.cam_keys]


class _HDF5Dataset(Dataset):
    """PID-based HDF5 file reopen (HDF5 handles are not fork-safe)."""
    def __init__(self, hdf5_path, cam_keys, index):
        self.hdf5_path = hdf5_path
        self.cam_keys  = cam_keys
        self.index     = index   # list of (demo_idx, frame_idx)
        self._file = None
        self._pid  = None

    def _f(self):
        pid = os.getpid()
        if self._pid != pid:
            if self._file is not None:
                try: self._file.close()
                except: pass
            self._file = h5py.File(self.hdf5_path, 'r')
            self._pid  = pid
        return self._file

    def __len__(self): return len(self.index)
    def __getitem__(self, i):
        di, fi = self.index[i]
        f = self._f()
        return [f[f'data/demo_{di}/obs/{k}'][fi] for k in self.cam_keys]


class _LeRobotDataset(Dataset):
    """LeRobot per-episode MP4. PyAV containers are reset on fork via PID check."""
    def __init__(self, out_dir, cam_keys, episode_ends, n):
        self.out_dir      = out_dir
        self.cam_keys     = cam_keys
        self.episode_ends = episode_ends
        self._n    = n
        self._conts = {}
        self._pid   = os.getpid()

    def _get_cont(self, cam_key, ep_idx):
        pid = os.getpid()
        if pid != self._pid:
            self._conts = {}
            self._pid   = pid
        key = (cam_key, ep_idx)
        if key not in self._conts:
            mp4 = os.path.join(self.out_dir, cam_key, f"episode_{ep_idx:09d}.mp4")
            c   = av_mod.open(mp4, 'r')
            vs  = c.streams.video[0]
            pts_per_f = 1.0 / float(vs.average_rate) / float(vs.time_base)
            self._conts[key] = (c, vs, pts_per_f)
        return self._conts[key]

    def __len__(self): return self._n
    def __getitem__(self, gi):
        gi      = int(gi)
        ep_idx  = int(np.searchsorted(self.episode_ends, gi, side='right'))
        loc_idx = gi - (int(self.episode_ends[ep_idx - 1]) if ep_idx > 0 else 0)
        frames  = []
        for cam_key in self.cam_keys:
            c, vs, pts_per_f = self._get_cont(cam_key, ep_idx)
            c.seek(int(loc_idx * pts_per_f), stream=vs, backward=True, any_frame=False)
            for pkt in c.demux(vs):
                for frm in pkt.decode():
                    frames.append(frm.to_ndarray(format='rgb24'))
                    break
                break
        return frames


class _RoboDMDataset(Dataset):
    """RoboDM VLA. Episode-level RAM cache per worker process."""
    def __init__(self, out_dir, cam_keys, episode_ends, n):
        self.out_dir      = out_dir
        self.cam_keys     = cam_keys
        self.episode_ends = episode_ends
        self._n     = n
        self._cache = {}
        self._pid   = os.getpid()

    def _get_ep(self, ep_idx):
        pid = os.getpid()
        if pid != self._pid:
            self._cache = {}
            self._pid   = pid
        if ep_idx not in self._cache:
            t = Trajectory(os.path.join(self.out_dir, f'ep_{ep_idx:06d}.vla'), mode='r')
            self._cache[ep_idx] = t.load()
            t.close()
        return self._cache[ep_idx]

    def __len__(self): return self._n
    def __getitem__(self, gi):
        gi      = int(gi)
        ep_idx  = int(np.searchsorted(self.episode_ends, gi, side='right'))
        loc_idx = gi - (int(self.episode_ends[ep_idx - 1]) if ep_idx > 0 else 0)
        ep_data = self._get_ep(ep_idx)
        return [ep_data[k][loc_idx] for k in self.cam_keys]


class _RRDServerDataset(Dataset):
    """RRD random access via Rerun catalog server + DataFusion.

    Each DataLoader worker connects to the server independently.
    Queries are batched internally to amortize gRPC overhead (~0.55 s/query).
    The Dataset item is a *pre-batch* of server_batch_size frames so that
    DataLoader batch_size=1 with collate_fn=lambda x:x[0] gives the right count.
    """
    def __init__(self, rrd_path, cam_keys, n, server_addr, server_batch=128):
        self.rrd_path     = rrd_path
        self.cam_keys     = cam_keys
        self._n           = n
        self.server_addr  = server_addr
        self.server_batch = server_batch
        self._dataset     = None
        self._pid         = None
        # Number of "items" = number of server queries needed for n frames
        self._n_items = max(1, n // server_batch)

    def _get_ds(self):
        pid = os.getpid()
        if self._pid != pid:
            client = rr_cat.CatalogClient(self.server_addr)
            self._dataset = client.get_dataset("bench")
            self._pid     = pid
        return self._dataset

    def __len__(self): return self._n_items

    def __getitem__(self, batch_i):
        rng   = np.random.default_rng(batch_i + os.getpid())
        steps = rng.choice(self._n, size=self.server_batch, replace=False).astype(np.int64)
        ds    = self._get_ds()
        cam_cols = [f"/cam/{k}:EncodedImage:blob" for k in self.cam_keys]
        n = 0
        for batch in ds.reader(index="step", using_index_values=steps).collect():
            for col in cam_cols:
                try: arr = batch.column(col)
                except KeyError: continue
                for outer in arr.to_pylist():
                    if outer and outer[0]:
                        _ = np.array(PILImage.open(io.BytesIO(bytes(outer[0]))))
                        n += 1
        return [n]   # return frame count so DataLoader can sum it


# ── DataLoader scaling benchmark ──────────────────────────────────────────────

def _bench_dl_scaling(make_dataset, worker_counts=_WORKER_COUNTS,
                      n_epochs=_BENCH_EPOCHS, batch=_BENCH_BATCH, mp_context=None):
    """Run DataLoader for n_epochs at each worker count; return {n_workers: fps}."""
    scaling = {}
    for nw in worker_counts:
        ds = make_dataset()
        dl = DataLoader(
            ds, batch_size=batch, num_workers=nw,
            shuffle=True, pin_memory=False,
            persistent_workers=(nw > 0),
            prefetch_factor=2 if nw > 0 else None,
            multiprocessing_context=mp_context if nw > 0 else None,
        )
        n, t0 = 0, time.perf_counter()
        for _ in range(n_epochs):
            for b in dl:
                if isinstance(b, (list, tuple)) and len(b) > 0:
                    n += b[0].shape[0] if hasattr(b[0], 'shape') else len(b[0])
                elif hasattr(b, 'shape'):
                    n += b.shape[0]
        fps = round(n / (time.perf_counter() - t0), 1)
        scaling[nw] = fps
        print(f"      {nw}w → {fps:.0f} fps", flush=True)
    return scaling


def _bench_rrd_server_scaling(rrd_path, cam_keys, n, worker_counts=_WORKER_COUNTS,
                               n_epochs=_BENCH_EPOCHS, server_batch=128):
    """Start one Rerun server; benchmark DataLoader workers querying it."""
    with rr_srv.Server(datasets={"bench": [rrd_path]}) as server:
        addr = server.url()
        scaling = {}
        for nw in worker_counts:
            ds = _RRDServerDataset(rrd_path, cam_keys, n,
                                   server_addr=addr, server_batch=server_batch)
            # batch_size=1: each __getitem__ is already a pre-batch of server_batch frames
            dl = DataLoader(ds, batch_size=1, num_workers=nw,
                            shuffle=True, pin_memory=False,
                            persistent_workers=(nw > 0),
                            prefetch_factor=2 if nw > 0 else None,
                            multiprocessing_context='spawn' if nw > 0 else None,
                            collate_fn=_identity_collate)
            n_frames, t0 = 0, time.perf_counter()
            for _ in range(n_epochs):
                for b in dl:
                    # b is a list of 1 item; item is [frame_count]
                    n_frames += b[0][0].item() if isinstance(b[0][0], torch.Tensor) else b[0][0]
            fps = round(n_frames / (time.perf_counter() - t0), 1)
            scaling[nw] = fps
            print(f"      {nw}w → {fps:.0f} fps", flush=True)
    return scaling


# ── utilities ─────────────────────────────────────────────────────────────────

def _path_stats(p):
    p = pathlib.Path(p)
    if not p.exists():
        return 0, 0
    if p.is_file():
        return 1, p.stat().st_size
    n, total = 0, 0
    for f in p.rglob("*"):
        if f.is_file():
            n += 1; total += f.stat().st_size
    return n, total


def _load_hdf5(hdf5_path, cam_keys, max_frames=None):
    parts = {k: [] for k in cam_keys}
    with h5py.File(hdf5_path, 'r') as f:
        for i in range(len(f['data'])):
            for k in cam_keys:
                parts[k].append(f['data'][f'demo_{i}']['obs'][k][:])
    arrs = {k: np.concatenate(parts[k]) for k in cam_keys}
    if max_frames:
        arrs = {k: v[:max_frames] for k, v in arrs.items()}
    return arrs


def _load_episode_ends(hdf5_path):
    ends, step = [], 0
    with h5py.File(hdf5_path, 'r') as f:
        for i in range(len(f['data'])):
            step += f['data'][f'demo_{i}']['actions'].shape[0]
            ends.append(step)
    return np.array(ends, dtype=np.int64)


# ── encode helpers ────────────────────────────────────────────────────────────

def _build_zarr_dir(out_dir, images, compressor, n_workers=8):
    if os.path.exists(out_dir):
        shutil.rmtree(out_dir)
    store = zarr.DirectoryStore(out_dir)
    root  = zarr.group(store, overwrite=True)
    t0 = time.perf_counter()
    for k, arr in images.items():
        N, H, W, C = arr.shape
        za = root.require_dataset(k, shape=(N,H,W,C), chunks=(1,H,W,C),
                                  compressor=compressor, dtype=np.uint8)
        def _copy(i, _za=za, _arr=arr):
            _za[i] = _arr[i]
        with concurrent.futures.ThreadPoolExecutor(n_workers) as ex:
            list(ex.map(_copy, range(N)))
    return time.perf_counter() - t0, *_path_stats(out_dir)


def _build_zarr_mem(images, compressor):
    store = zarr.MemoryStore()
    root  = zarr.group(store)
    t0 = time.perf_counter()
    for k, arr in images.items():
        N, H, W, C = arr.shape
        za = root.require_dataset(k, shape=(N,H,W,C), chunks=(1,H,W,C),
                                  compressor=compressor, dtype=np.uint8)
        for i in range(N):
            za[i] = arr[i]
    enc_s = time.perf_counter() - t0
    n_bytes = sum(len(v) for v in store.store.values()) if hasattr(store, 'store') else 0
    return enc_s, n_bytes, root


def _write_rrd(rrd_path, images, mode):
    """mode: 'lz4'|'jpeg75'|'jpeg85'|'jpeg95'|'png'. Returns (encode_s, n_files, n_bytes)."""
    if os.path.exists(rrd_path):
        os.remove(rrd_path)
    quality = int(mode[4:]) if mode.startswith('jpeg') else None
    fmt = 'jpeg' if mode.startswith('jpeg') else ('png' if mode == 'png' else 'lz4')
    cam_keys = list(images.keys())
    N = images[cam_keys[0]].shape[0]
    t0 = time.perf_counter()
    rr.init('storage_bench', spawn=False)
    rr.save(rrd_path)
    for i in range(N):
        rr.set_time('step', sequence=i)
        for k in cam_keys:
            arr = images[k][i]
            if fmt == 'lz4':
                rr.log(f'cam/{k}', rr.Tensor(arr))
            elif fmt == 'jpeg':
                buf = io.BytesIO()
                PILImage.fromarray(arr).save(buf, 'JPEG', quality=quality)
                rr.log(f'cam/{k}', rr.EncodedImage(contents=buf.getvalue(), media_type='image/jpeg'))
            elif fmt == 'png':
                buf = io.BytesIO()
                PILImage.fromarray(arr).save(buf, 'PNG')
                rr.log(f'cam/{k}', rr.EncodedImage(contents=buf.getvalue(), media_type='image/png'))
    rr.disconnect()
    return time.perf_counter() - t0, *_path_stats(rrd_path)


def _decode_rrd_all(rrd_path, cam_keys):
    """Full sequential decode. Returns (decode_s, n_frames)."""
    archive = rb.load_archive(rrd_path)
    rec     = archive.all_recordings()[0]
    chunks  = list(rec.chunks())
    t0, n = time.perf_counter(), 0
    for chunk in chunks:
        ep = chunk.entity_path
        if not any(k in ep for k in cam_keys):
            continue
        batch       = chunk.to_record_batch()
        field_names = [f.name for f in batch.schema]
        if 'Tensor:data' in field_names:
            for item in batch.column('Tensor:data').to_pylist():
                if item:
                    b, sh = item[0]['buffer'], item[0]['shape']
                    _ = np.array(b, dtype=np.uint8).reshape(sh); n += 1
        elif 'EncodedImage:blob' in field_names:
            for item in batch.column('EncodedImage:blob').to_pylist():
                if item and item[0]:
                    _ = np.array(PILImage.open(io.BytesIO(bytes(item[0])))); n += 1
    return time.perf_counter() - t0, n


def _build_lerobot(out_dir, images, episode_ends, cam_keys, crf=23, kfi=1, codec='libx264'):
    if os.path.exists(out_dir):
        shutil.rmtree(out_dir)
    os.makedirs(out_dir)
    t0, prev = time.perf_counter(), 0
    for ep_idx, end in enumerate(episode_ends):
        for cam_key in cam_keys:
            ep_frames = images[cam_key][prev:int(end)]
            mp4_path  = os.path.join(out_dir, cam_key, f"episode_{ep_idx:09d}.mp4")
            os.makedirs(os.path.dirname(mp4_path), exist_ok=True)
            c = av_mod.open(mp4_path, 'w')
            s = c.add_stream(codec, rate=10)
            s.width, s.height, s.pix_fmt = ep_frames.shape[2], ep_frames.shape[1], 'yuv420p'
            s.options = {'crf': str(crf), 'g': str(kfi), 'preset': 'fast'}
            for arr in ep_frames:
                for p in s.encode(av_mod.VideoFrame.from_ndarray(arr, format='rgb24')):
                    c.mux(p)
            for p in s.encode():
                c.mux(p)
            c.close()
        prev = int(end)
    return time.perf_counter() - t0, *_path_stats(out_dir)


def _build_robodm(out_dir, images, episode_ends, cam_keys, codec='libx264', crf=23):
    import logging; logging.getLogger('robodm').setLevel(logging.WARNING)
    if os.path.exists(out_dir):
        shutil.rmtree(out_dir)
    os.makedirs(out_dir)
    t0, prev = time.perf_counter(), 0
    for ep_idx, end in enumerate(episode_ends):
        ep = {k: list(images[k][prev:int(end)]) for k in cam_keys}
        Trajectory.from_dict_of_lists(
            ep, path=os.path.join(out_dir, f'ep_{ep_idx:06d}.vla'),
            video_codec=codec, codec_options={'crf': crf}, fps=10)
        prev = int(end)
    return time.perf_counter() - t0, *_path_stats(out_dir)


# ── results helpers ───────────────────────────────────────────────────────────

def _row(results, label, fmt, enc_s, n_files, size_mb, ratio, scaling):
    results.append(dict(
        label=label, format=fmt,
        encode_s=round(enc_s, 2),
        n_files=n_files,
        size_mb=round(size_mb, 2),
        compression_ratio=round(ratio, 2),
        scaling={str(k): round(v, 1) for k, v in scaling.items()},
    ))


def _fmt_fps(fps):
    if fps <= 0:     return '  -'
    if fps >= 10000: return f'{fps/1000:.0f}k'
    if fps >= 1000:  return f'{fps/1000:.1f}k'
    return f'{fps:.0f}'


def _print_table(results, N, n_cams, raw_mb):
    wc  = _WORKER_COUNTS
    hdr_workers = ''.join(f'{w:>6}w' for w in wc)
    W   = 38 + 8 + 8 + 7 + 6 + 7 * len(wc) + 4
    sep = '─' * W
    print(f"\n{sep}")
    print(f"STORAGE BENCHMARK  |  {N}×{n_cams} frames  raw={raw_mb:.0f} MB")
    print(sep)
    print(f"  {'Backend':<38} {'SizeMB':>7} {'Files':>7} {'Ratio':>6} {'Enc s':>6}  {hdr_workers}")
    print(f"  {'-'*(W-2)}")
    for r in results:
        sc       = r.get('scaling', {})
        fps_cols = ''.join(f"{_fmt_fps(sc.get(w, sc.get(str(w), 0))):>7}" for w in wc)
        note     = ''
        if r['format'] == 'rrd_seq':    note = ' (seq)'
        elif r['format'] == 'rrd_server': note = ' (rnd)'
        elif r['format'] == 'robodm_vla': note = ' (ep$)'
        label_str = r['label'] + note
        print(f"  {label_str:<38} {r['size_mb']:>7.1f} {r['n_files']:>7,} "
              f"{r['compression_ratio']:>5.1f}x {r['encode_s']:>6.1f}  {fps_cols}")
    print(sep)
    print("  (ep$) = episode-level RAM cache per worker")


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--hdf5',      required=True)
    ap.add_argument('--out',       required=True)
    ap.add_argument('--n-frames',  type=int, default=None)
    ap.add_argument('--cam-keys',  default='agentview_image,robot0_eye_in_hand_image')
    ap.add_argument('--epochs',    type=int, default=_BENCH_EPOCHS,
                    help='DataLoader epochs per (backend, n_workers) measurement')
    args = ap.parse_args()

    cam_keys  = args.cam_keys.split(',')
    out_root  = pathlib.Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)
    n_epochs  = args.epochs

    print("Loading images from HDF5 …")
    images       = _load_hdf5(args.hdf5, cam_keys, args.n_frames)
    episode_ends = _load_episode_ends(args.hdf5)
    N, img_H, img_W, img_C = images[cam_keys[0]].shape
    raw_bytes = N * len(cam_keys) * img_H * img_W * img_C
    raw_mb    = raw_bytes / 1e6
    print(f"  {N} frames × {len(cam_keys)} cams  ({img_H}×{img_W}×{img_C})  raw={raw_mb:.0f} MB")
    print(f"  worker_counts={_WORKER_COUNTS}  epochs={n_epochs}\n")

    hdf5_bytes = _path_stats(args.hdf5)[1]

    hdf5_index = []
    with h5py.File(args.hdf5, 'r') as f:
        for i in range(len(f['data'])):
            T = f['data'][f'demo_{i}']['obs'][cam_keys[0]].shape[0]
            for t in range(T):
                hdf5_index.append((i, t))

    # Load existing results to allow resume (skip already-benchmarked backends)
    out_json = out_root / "storage_results.json"
    _done = set()
    if out_json.exists():
        try:
            existing = json.loads(out_json.read_text())
            results = existing.get('results', [])
            _done   = {r['label'] for r in results}
            print(f"  Resuming: {len(_done)} backends already done → {sorted(_done)}\n")
        except Exception:
            results = []
    else:
        results = []

    def _save():
        with open(out_json, 'w') as _f:
            json.dump({'n_frames': N, 'n_cams': len(cam_keys),
                       'raw_mb': round(raw_mb, 1),
                       'worker_counts': _WORKER_COUNTS,
                       'results': results}, _f, indent=2)

    # ─── A. Codec sweep (zarr disk) ───────────────────────────────────────────
    print("=== A. Codec sweep (zarr disk) ===")
    codecs_disk = [
        ('raw',           None),
        ('lz4',           numcodecs.LZ4(acceleration=1)),
        ('zstd_l1',       numcodecs.Zstd(level=1)),
        ('zstd_l3',       numcodecs.Zstd(level=3)),
        ('zstd_l9',       numcodecs.Zstd(level=9)),
        ('j2k_q50',       Jpeg2k(level=50)),
        ('j2k_q75',       Jpeg2k(level=75)),
        ('j2k_q90',       Jpeg2k(level=90)),
        ('jpeg_q75',      Jpeg(level=75)),
        ('jpeg_q85',      Jpeg(level=85)),
        ('jpeg_q95',      Jpeg(level=95)),
        ('png',           Png()),
        ('webp_q75',      Webp(level=75)),
        ('webp_lossless', Webp(level=0)),
    ]
    for suffix, comp in codecs_disk:
        label = f'zarr_disk_{suffix}'
        if label in _done:
            print(f"  [{label}] (skip — already in results)", flush=True)
            continue
        d     = str(out_root / label)
        print(f"  [{label}]", flush=True)
        if os.path.exists(d):
            enc_s = 0.0
            n_files, n_bytes = _path_stats(d)
        else:
            enc_s, n_files, n_bytes = _build_zarr_dir(d, images, comp, n_workers=8)
        ratio = raw_bytes / n_bytes if n_bytes else 0
        print(f"    size={n_bytes/1e6:.1f} MB  ratio={ratio:.1f}x  enc={enc_s:.0f}s  →  scaling:")
        scaling = _bench_dl_scaling(lambda d=d: _ZarrDataset(d, cam_keys, N),
                                    n_epochs=n_epochs)
        _row(results, label, 'zarr_disk', enc_s, n_files, n_bytes / 1e6, ratio, scaling)
        _save()

    # ─── B. Location / format sweep ───────────────────────────────────────────
    print("\n=== B. Location / format sweep ===")

    if 'hdf5_disk' not in _done:
        print("  [hdf5_disk]  →  scaling:", flush=True)
        scaling = _bench_dl_scaling(lambda: _HDF5Dataset(args.hdf5, cam_keys, hdf5_index),
                                    n_epochs=n_epochs)
        ratio_hdf5 = raw_bytes / hdf5_bytes if hdf5_bytes else 0
        _row(results, 'hdf5_disk', 'hdf5', 0, 1, hdf5_bytes / 1e6, ratio_hdf5, scaling)
        _save()
    else:
        print("  [hdf5_disk] (skip)", flush=True)

    if 'zarr_ram_raw' not in _done:
        print("  [zarr_ram_raw] encoding …", flush=True)
        enc_s_r, nb_r, root_r = _build_zarr_mem(images, None)
        print(f"    enc={enc_s_r:.0f}s  →  scaling:")
        scaling = _bench_dl_scaling(lambda r=root_r: _ZarrDataset(r, cam_keys, N),
                                    n_epochs=n_epochs)
        _row(results, 'zarr_ram_raw', 'zarr_ram', enc_s_r, 0, raw_mb, 1.0, scaling)
        _save()
    else:
        print("  [zarr_ram_raw] (skip)", flush=True)

    if 'zarr_ram_j2k' not in _done:
        print("  [zarr_ram_j2k] encoding …", flush=True)
        enc_s_rj, nb_rj, root_rj = _build_zarr_mem(images, Jpeg2k(level=50))
        print(f"    enc={enc_s_rj:.0f}s  →  scaling:")
        scaling = _bench_dl_scaling(lambda r=root_rj: _ZarrDataset(r, cam_keys, N),
                                    n_epochs=n_epochs)
        _row(results, 'zarr_ram_j2k', 'zarr_ram', enc_s_rj, 0, raw_mb, 1.0, scaling)
        _save()
    else:
        print("  [zarr_ram_j2k] (skip)", flush=True)

    # ─── C. Rerun RRD ─────────────────────────────────────────────────────────
    print("\n=== C. Rerun RRD variants ===")
    rrd_modes = [
        ('rrd_lz4',    'lz4'),
        ('rrd_jpeg75', 'jpeg75'),
        ('rrd_jpeg85', 'jpeg85'),
        ('rrd_jpeg95', 'jpeg95'),
        ('rrd_png',    'png'),
    ]
    rrd_jpeg75_path = None
    for label, mode in rrd_modes:
        rrd_path = str(out_root / f"{label}.rrd")
        if os.path.exists(rrd_path):
            if label == 'rrd_jpeg75':
                rrd_jpeg75_path = rrd_path
        if label in _done:
            print(f"  [{label}] (skip)", flush=True)
            if label == 'rrd_jpeg75':
                rrd_jpeg75_path = rrd_path
            continue
        print(f"  [{label}]", flush=True)
        try:
            if not os.path.exists(rrd_path):
                enc_s, n_files, n_bytes = _write_rrd(rrd_path, images, mode)
            else:
                enc_s = 0.0
                n_files, n_bytes = _path_stats(rrd_path)
            ratio = raw_bytes / n_bytes if n_bytes else 0
            dec_s, dec_n = _decode_rrd_all(rrd_path, cam_keys)
            fps = dec_n / dec_s if dec_s else 0
            print(f"    size={n_bytes/1e6:.1f} MB  ratio={ratio:.2f}x  seq={fps:.0f} fps")
            scaling = {nw: fps for nw in _WORKER_COUNTS}
            _row(results, label, 'rrd_seq', enc_s, 1, n_bytes / 1e6, ratio, scaling)
            _save()
            if label == 'rrd_jpeg75':
                rrd_jpeg75_path = rrd_path
        except Exception:
            import traceback; traceback.print_exc()

    # C2: RRD random access via catalog server, DataLoader workers
    if rrd_jpeg75_path and os.path.exists(rrd_jpeg75_path) and 'rrd_jpeg75_server' not in _done:
        print(f"  [rrd_jpeg75_server] server_batch=128  →  scaling:", flush=True)
        try:
            scaling = _bench_rrd_server_scaling(
                rrd_jpeg75_path, cam_keys, N,
                worker_counts=_WORKER_COUNTS, n_epochs=n_epochs, server_batch=128)
            rrd_j75_bytes = _path_stats(rrd_jpeg75_path)[1]
            ratio_j75 = raw_bytes / rrd_j75_bytes if rrd_j75_bytes else 0
            _row(results, 'rrd_jpeg75_server', 'rrd_server', 0,
                 1, rrd_j75_bytes / 1e6, ratio_j75, scaling)
            _save()
        except Exception:
            import traceback; traceback.print_exc()
    elif 'rrd_jpeg75_server' in _done:
        print(f"  [rrd_jpeg75_server] (skip)", flush=True)

    # ─── D. LeRobot MP4 ───────────────────────────────────────────────────────
    print("\n=== D. LeRobot MP4 (H.264) ===")
    lerobot_variants = [
        ('lerobot_h264_crf18_kf1',  18, 1),
        ('lerobot_h264_crf23_kf1',  23, 1),
        ('lerobot_h264_crf28_kf1',  28, 1),
        ('lerobot_h264_crf23_kf10', 23, 10),
    ]
    for label, crf, kfi in lerobot_variants:
        if label in _done:
            print(f"  [{label}] (skip)", flush=True)
            continue
        d = str(out_root / label)
        print(f"  [{label}]", flush=True)
        try:
            if os.path.exists(d):
                enc_s = 0.0
                n_files, n_bytes = _path_stats(d)
            else:
                enc_s, n_files, n_bytes = _build_lerobot(
                    d, images, episode_ends, cam_keys, crf, kfi)
            ratio = raw_bytes / n_bytes if n_bytes else 0
            print(f"    size={n_bytes/1e6:.1f} MB  ratio={ratio:.2f}x  enc={enc_s:.0f}s  →  scaling:")
            scaling = _bench_dl_scaling(
                lambda d=d: _LeRobotDataset(d, cam_keys, episode_ends, N),
                n_epochs=n_epochs, mp_context='spawn')
            _row(results, label, 'lerobot_mp4', enc_s, n_files, n_bytes / 1e6, ratio, scaling)
            _save()
        except Exception:
            import traceback; traceback.print_exc()

    # ─── E. RoboDM VLA ────────────────────────────────────────────────────────
    print("\n=== E. RoboDM VLA ===")
    robodm_variants = [
        ('robodm_h264_crf23', 'libx264', 23),
        ('robodm_h264_crf28', 'libx264', 28),
        ('robodm_h265_crf28', 'libx265', 28),
    ]
    for label, codec, crf in robodm_variants:
        if label in _done:
            print(f"  [{label}] (skip)", flush=True)
            continue
        d = str(out_root / label)
        print(f"  [{label}]", flush=True)
        try:
            if os.path.exists(d):
                enc_s = 0.0
                n_files, n_bytes = _path_stats(d)
            else:
                enc_s, n_files, n_bytes = _build_robodm(
                    d, images, episode_ends, cam_keys, codec, crf)
            ratio = raw_bytes / n_bytes if n_bytes else 0
            print(f"    size={n_bytes/1e6:.1f} MB  ratio={ratio:.2f}x  enc={enc_s:.0f}s  →  scaling:")
            scaling = _bench_dl_scaling(
                lambda d=d: _RoboDMDataset(d, cam_keys, episode_ends, N),
                n_epochs=n_epochs, mp_context='spawn')
            _row(results, label, 'robodm_vla', enc_s, n_files, n_bytes / 1e6, ratio, scaling)
            _save()
        except Exception:
            import traceback; traceback.print_exc()

    # ─── Summary ──────────────────────────────────────────────────────────────
    _print_table(results, N, len(cam_keys), raw_mb)
    _save()
    print(f"Results → {out_json}")


if __name__ == '__main__':
    main()
