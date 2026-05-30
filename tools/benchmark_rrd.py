"""
Focused RRD storage benchmark — no artificial bottlenecks.

Tests three RRD access patterns against zarr_disk_jpeg_q75 as baseline:
  1. rrd_seq        — sequential full-file scan (current baseline, single-threaded)
  2. rrd_server_pyl — catalog server, to_pylist() (what main benchmark used)
  3. rrd_server_np  — catalog server, to_numpy() (doc-recommended)
  4. rrd_ep_files   — per-episode .rrd files, direct rb.load_archive per worker, no server

All tested at 0–12 DataLoader workers, 2 epochs.

Usage:
  conda run -n rerun_env python -u tools/benchmark_rrd.py \
      --hdf5  data/robomimic/datasets/lift/ph/image.hdf5 \
      --out   /mnt/ssdB/erik/cupid_data/rrd_bench \
      --epochs 2
"""
import argparse, os, shutil, time, json, io, pathlib
import numpy as np
import h5py
from PIL import Image as PILImage

import rerun as rr
import rerun_bindings.rerun_bindings as rb
import rerun.server as rr_srv
import rerun.catalog as rr_cat

import torch
from torch.utils.data import Dataset, DataLoader

import sys
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))
from diffusion_policy.codecs.imagecodecs_numcodecs import register_codecs, Jpeg
register_codecs()
import zarr, numcodecs

_WORKER_COUNTS = [0, 1, 2, 4, 6, 8, 10, 12]
_BENCH_EPOCHS  = 2
_BENCH_BATCH   = 32


def _identity_collate(x): return x


# ── loaders ───────────────────────────────────────────────────────────────────

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


def _path_stats(p):
    p = pathlib.Path(p)
    if not p.exists(): return 0, 0
    if p.is_file(): return 1, p.stat().st_size
    n, total = 0, 0
    for f in p.rglob('*'):
        if f.is_file(): n += 1; total += f.stat().st_size
    return n, total


# ── encoders ──────────────────────────────────────────────────────────────────

def _write_rrd_single(rrd_path, images, cam_keys, mode='jpeg75'):
    """Write a single monolithic RRD file."""
    if os.path.exists(rrd_path): os.remove(rrd_path)
    quality = int(mode[4:]) if mode.startswith('jpeg') else None
    N = images[cam_keys[0]].shape[0]
    t0 = time.perf_counter()
    rr.init('rrd_bench', spawn=False)
    rr.save(rrd_path)
    for i in range(N):
        rr.set_time('step', sequence=i)
        for k in cam_keys:
            arr = images[k][i]
            if mode.startswith('jpeg'):
                buf = io.BytesIO()
                PILImage.fromarray(arr).save(buf, 'JPEG', quality=quality)
                rr.log(f'cam/{k}', rr.EncodedImage(contents=buf.getvalue(), media_type='image/jpeg'))
            else:
                rr.log(f'cam/{k}', rr.Tensor(arr))
    rr.disconnect()
    return time.perf_counter() - t0, *_path_stats(rrd_path)


def _write_rrd_episodes(out_dir, images, episode_ends, cam_keys, mode='jpeg75'):
    """Write one RRD file per episode."""
    if os.path.exists(out_dir): shutil.rmtree(out_dir)
    os.makedirs(out_dir)
    quality = int(mode[4:]) if mode.startswith('jpeg') else None
    t0, prev = time.perf_counter(), 0
    for ep_idx, end in enumerate(episode_ends):
        rrd_path = os.path.join(out_dir, f'ep_{ep_idx:06d}.rrd')
        rr.init(f'ep_{ep_idx}', spawn=False)
        rr.save(rrd_path)
        ep_len = int(end) - prev
        for i in range(ep_len):
            rr.set_time('step', sequence=i)
            for k in cam_keys:
                arr = images[k][prev + i]
                if mode.startswith('jpeg'):
                    buf = io.BytesIO()
                    PILImage.fromarray(arr).save(buf, 'JPEG', quality=quality)
                    rr.log(f'cam/{k}', rr.EncodedImage(contents=buf.getvalue(), media_type='image/jpeg'))
                else:
                    rr.log(f'cam/{k}', rr.Tensor(arr))
        rr.disconnect()
        prev = int(end)
    return time.perf_counter() - t0, *_path_stats(out_dir)


def _build_zarr_jpeg75(out_dir, images, cam_keys):
    if os.path.exists(out_dir): shutil.rmtree(out_dir)
    store = zarr.DirectoryStore(out_dir)
    root  = zarr.group(store, overwrite=True)
    import concurrent.futures
    t0 = time.perf_counter()
    for k, arr in images.items():
        N, H, W, C = arr.shape
        comp = Jpeg(level=75)
        za = root.require_dataset(k, shape=(N, H, W, C), chunks=(1, H, W, C),
                                  compressor=comp, dtype=np.uint8)
        def _copy(i, _za=za, _arr=arr): _za[i] = _arr[i]
        with concurrent.futures.ThreadPoolExecutor(8) as ex:
            list(ex.map(_copy, range(N)))
    return time.perf_counter() - t0, *_path_stats(out_dir)


# ── Dataset classes ───────────────────────────────────────────────────────────

class _ZarrDataset(Dataset):
    def __init__(self, zarr_path, cam_keys, n):
        self._path = zarr_path
        self.cam_keys = cam_keys
        self._n = n
        self._root = None
        self._pid  = None

    def _get_root(self):
        pid = os.getpid()
        if self._pid != pid:
            self._root = zarr.open(self._path, mode='r')
            self._pid  = pid
        return self._root

    def __len__(self): return self._n
    def __getitem__(self, i):
        r = self._get_root()
        return [r[k][int(i)] for k in self.cam_keys]


class _RRDSeqDataset(Dataset):
    """Load full RRD file into RAM on first access per worker (no server)."""
    def __init__(self, rrd_path, cam_keys, n):
        self.rrd_path = rrd_path
        self.cam_keys = cam_keys
        self._n       = n
        self._frames  = None   # {cam_key: list[np.ndarray]}
        self._pid     = None

    def _load(self):
        pid = os.getpid()
        if self._pid == pid: return
        frames = {k: [] for k in self.cam_keys}
        archive = rb.load_archive(self.rrd_path)
        rec     = archive.all_recordings()[0]
        for chunk in rec.chunks():
            ep = chunk.entity_path
            cam_key = next((k for k in self.cam_keys if f'cam/{k}' in ep), None)
            if cam_key is None: continue
            batch = chunk.to_record_batch()
            cols  = [f.name for f in batch.schema]
            if 'EncodedImage:blob' in cols:
                col = batch.column('EncodedImage:blob')
                # to_numpy() on a list-of-binary column → numpy object array
                arr = col.to_numpy(zero_copy_only=False)
                for cell in arr:
                    if cell is not None and len(cell) > 0:
                        blob = bytes(cell[0]) if hasattr(cell[0], '__len__') else bytes(cell)
                        frames[cam_key].append(
                            np.array(PILImage.open(io.BytesIO(blob))))
            elif 'Tensor:data' in cols:
                col = batch.column('Tensor:data')
                for item in col.to_pylist():
                    if item and item[0]:
                        b, sh = item[0]['buffer'], item[0]['shape']
                        frames[cam_key].append(
                            np.array(b, dtype=np.uint8).reshape(sh))
        self._frames = frames
        self._pid    = pid

    def __len__(self): return self._n
    def __getitem__(self, i):
        self._load()
        return [self._frames[k][int(i)] for k in self.cam_keys]


class _RRDServerPylDataset(Dataset):
    """Catalog server + to_pylist() — current main-benchmark approach."""
    def __init__(self, rrd_path, cam_keys, n, server_addr, server_batch=128):
        self.rrd_path    = rrd_path
        self.cam_keys    = cam_keys
        self._n          = n
        self.server_addr = server_addr
        self.server_batch = server_batch
        self._n_items    = max(1, n // server_batch)
        self._dataset    = None
        self._pid        = None

    def _get_ds(self):
        pid = os.getpid()
        if self._pid != pid:
            self._dataset = rr_cat.CatalogClient(self.server_addr).get_dataset('bench')
            self._pid     = pid
        return self._dataset

    def __len__(self): return self._n_items
    def __getitem__(self, batch_i):
        rng   = np.random.default_rng(batch_i + os.getpid())
        steps = rng.choice(self._n, size=self.server_batch, replace=False).astype(np.int64)
        ds    = self._get_ds()
        cols  = [f'/cam/{k}:EncodedImage:blob' for k in self.cam_keys]
        n = 0
        for batch in ds.reader(index='step', using_index_values=steps).collect():
            for col in cols:
                try: arr = batch.column(col)
                except KeyError: continue
                for outer in arr.to_pylist():   # <-- to_pylist path
                    if outer and outer[0]:
                        _ = np.array(PILImage.open(io.BytesIO(bytes(outer[0]))))
                        n += 1
        return [n]


class _RRDServerNpDataset(Dataset):
    """Catalog server + to_numpy() — doc-recommended path."""
    def __init__(self, rrd_path, cam_keys, n, server_addr, server_batch=128):
        self.rrd_path     = rrd_path
        self.cam_keys     = cam_keys
        self._n           = n
        self.server_addr  = server_addr
        self.server_batch = server_batch
        self._n_items     = max(1, n // server_batch)
        self._dataset     = None
        self._pid         = None

    def _get_ds(self):
        pid = os.getpid()
        if self._pid != pid:
            self._dataset = rr_cat.CatalogClient(self.server_addr).get_dataset('bench')
            self._pid     = pid
        return self._dataset

    def __len__(self): return self._n_items
    def __getitem__(self, batch_i):
        rng   = np.random.default_rng(batch_i + os.getpid())
        steps = rng.choice(self._n, size=self.server_batch, replace=False).astype(np.int64)
        ds    = self._get_ds()
        cols  = [f'/cam/{k}:EncodedImage:blob' for k in self.cam_keys]
        n = 0
        for batch in ds.reader(index='step', using_index_values=steps).collect():
            for col in cols:
                try: arr_col = batch.column(col)
                except KeyError: continue
                # to_numpy on a ListArray of binary → numpy object array of lists/bytes
                raw = arr_col.to_numpy(zero_copy_only=False)
                for cell in raw:
                    if cell is not None:
                        blob = bytes(cell[0]) if hasattr(cell, '__len__') and len(cell) > 0 else None
                        if blob:
                            _ = np.array(PILImage.open(io.BytesIO(blob)))
                            n += 1
        return [n]


class _RRDEpisodeDataset(Dataset):
    """Per-episode RRD files, direct rb.load_archive per worker. No server."""
    def __init__(self, ep_dir, cam_keys, episode_ends, n):
        self.ep_dir       = ep_dir
        self.cam_keys     = cam_keys
        self.episode_ends = episode_ends
        self._n           = n
        self._cache       = {}   # {ep_idx: {cam_key: np.ndarray}}
        self._pid         = os.getpid()

    def _get_ep(self, ep_idx):
        pid = os.getpid()
        if pid != self._pid:
            self._cache = {}
            self._pid   = pid
        if ep_idx not in self._cache:
            rrd_path = os.path.join(self.ep_dir, f'ep_{ep_idx:06d}.rrd')
            ep_data  = {k: [] for k in self.cam_keys}
            archive  = rb.load_archive(rrd_path)
            rec      = archive.all_recordings()[0]
            for chunk in rec.chunks():
                ep = chunk.entity_path
                cam_key = next((k for k in self.cam_keys if f'cam/{k}' in ep), None)
                if cam_key is None: continue
                batch = chunk.to_record_batch()
                cols  = [f.name for f in batch.schema]
                if 'EncodedImage:blob' in cols:
                    col = batch.column('EncodedImage:blob')
                    raw = col.to_numpy(zero_copy_only=False)
                    for cell in raw:
                        if cell is not None:
                            blob = bytes(cell[0]) if hasattr(cell, '__len__') and len(cell) > 0 else None
                            if blob:
                                ep_data[cam_key].append(
                                    np.array(PILImage.open(io.BytesIO(blob))))
                elif 'Tensor:data' in cols:
                    col = batch.column('Tensor:data')
                    for item in col.to_pylist():
                        if item and item[0]:
                            b, sh = item[0]['buffer'], item[0]['shape']
                            ep_data[cam_key].append(
                                np.array(b, dtype=np.uint8).reshape(sh))
            self._cache[ep_idx] = {k: np.stack(v) if v else np.array([]) for k, v in ep_data.items()}
        return self._cache[ep_idx]

    def __len__(self): return self._n
    def __getitem__(self, gi):
        gi      = int(gi)
        ep_idx  = int(np.searchsorted(self.episode_ends, gi, side='right'))
        loc_idx = gi - (int(self.episode_ends[ep_idx - 1]) if ep_idx > 0 else 0)
        ep_data = self._get_ep(ep_idx)
        return [ep_data[k][loc_idx] for k in self.cam_keys]


# ── benchmark core ────────────────────────────────────────────────────────────

def _bench_dl(make_dataset, worker_counts=_WORKER_COUNTS,
              n_epochs=_BENCH_EPOCHS, batch=_BENCH_BATCH,
              mp_context=None, is_server_ds=False):
    """Run DataLoader for n_epochs at each worker count; return {nw: fps}."""
    scaling = {}
    for nw in worker_counts:
        ds = make_dataset()
        collate = _identity_collate if is_server_ds else None
        bs      = 1 if is_server_ds else batch
        dl = DataLoader(ds, batch_size=bs, num_workers=nw,
                        shuffle=True, pin_memory=False,
                        persistent_workers=(nw > 0),
                        prefetch_factor=2 if nw > 0 else None,
                        multiprocessing_context=mp_context if nw > 0 else None,
                        collate_fn=collate)
        n, t0 = 0, time.perf_counter()
        for _ in range(n_epochs):
            for b in dl:
                if is_server_ds:
                    # b is list of 1 item; item is [frame_count]
                    val = b[0][0]
                    n += val.item() if hasattr(val, 'item') else int(val)
                elif isinstance(b, (list, tuple)) and len(b) > 0:
                    n += b[0].shape[0] if hasattr(b[0], 'shape') else len(b[0])
                elif hasattr(b, 'shape'):
                    n += b.shape[0]
        fps = round(n / max(time.perf_counter() - t0, 1e-6), 1)
        scaling[nw] = fps
        print(f'    {nw}w → {fps:.0f} fps', flush=True)
    return scaling


def _bench_server(rrd_path, cam_keys, n, ds_class, worker_counts, n_epochs, server_batch=128):
    with rr_srv.Server(datasets={'bench': [rrd_path]}) as server:
        addr = server.url()
        return _bench_dl(
            lambda: ds_class(rrd_path, cam_keys, n, addr, server_batch),
            worker_counts=worker_counts, n_epochs=n_epochs,
            mp_context='spawn', is_server_ds=True)


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--hdf5',     required=True)
    ap.add_argument('--out',      required=True)
    ap.add_argument('--epochs',   type=int, default=_BENCH_EPOCHS)
    ap.add_argument('--cam-keys', default='agentview_image,robot0_eye_in_hand_image')
    ap.add_argument('--server-batch', type=int, default=128)
    args = ap.parse_args()

    cam_keys  = args.cam_keys.split(',')
    out_root  = pathlib.Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)
    n_ep      = args.epochs
    sb        = args.server_batch

    print('Loading images from HDF5 …')
    images       = _load_hdf5(args.hdf5, cam_keys)
    episode_ends = _load_episode_ends(args.hdf5)
    N = images[cam_keys[0]].shape[0]
    raw_bytes = N * len(cam_keys) * images[cam_keys[0]][0].size
    print(f'  {N} frames × {len(cam_keys)} cams  raw={raw_bytes/1e6:.0f} MB')
    print(f'  worker_counts={_WORKER_COUNTS}  epochs={n_ep}  server_batch={sb}\n')

    out_json = out_root / 'rrd_bench_results.json'
    _done = set()
    results = []
    if out_json.exists():
        try:
            ex = json.loads(out_json.read_text())
            results = ex.get('results', [])
            _done   = {r['label'] for r in results}
            print(f'  Resuming: {sorted(_done)} already done\n')
        except Exception: pass

    def _save():
        with open(out_json, 'w') as f:
            json.dump({'N': N, 'n_cams': len(cam_keys),
                       'raw_mb': round(raw_bytes/1e6, 1),
                       'worker_counts': _WORKER_COUNTS,
                       'results': results}, f, indent=2)

    def _row(label, fmt, size_mb, ratio, scaling):
        results.append(dict(label=label, format=fmt,
                            size_mb=round(size_mb, 2),
                            compression_ratio=round(ratio, 2),
                            scaling={str(k): round(v, 1) for k, v in scaling.items()}))
        _save()

    # ── 0. zarr_disk_jpeg_q75 baseline ────────────────────────────────────────
    zarr_dir = str(out_root / 'zarr_disk_jpeg_q75')
    label = 'zarr_disk_jpeg_q75'
    if label not in _done:
        print(f'[{label}]')
        if not os.path.exists(zarr_dir):
            enc_s, n_files, n_bytes = _build_zarr_jpeg75(zarr_dir, images, cam_keys)
        else:
            _, n_bytes = _path_stats(zarr_dir)
        ratio = raw_bytes / n_bytes if n_bytes else 0
        print(f'  size={n_bytes/1e6:.1f} MB  ratio={ratio:.1f}x  → scaling:')
        sc = _bench_dl(lambda: _ZarrDataset(zarr_dir, cam_keys, N), n_epochs=n_ep)
        _row(label, 'zarr_disk', n_bytes/1e6, ratio, sc)
    else:
        print(f'[{label}] (skip)')

    # ── 1. RRD single file, RAM-preload (no server) ────────────────────────────
    rrd_j75 = str(out_root / 'rrd_jpeg75.rrd')
    label = 'rrd_jpeg75_ram_preload'
    if label not in _done:
        print(f'\n[{label}]')
        if not os.path.exists(rrd_j75):
            enc_s, _, n_bytes = _write_rrd_single(rrd_j75, images, cam_keys, 'jpeg75')
        else:
            _, n_bytes = _path_stats(rrd_j75)
        ratio = raw_bytes / n_bytes if n_bytes else 0
        print(f'  size={n_bytes/1e6:.1f} MB  ratio={ratio:.1f}x  → scaling:')
        sc = _bench_dl(lambda: _RRDSeqDataset(rrd_j75, cam_keys, N), n_epochs=n_ep)
        _row(label, 'rrd_ram', n_bytes/1e6, ratio, sc)
    else:
        print(f'\n[{label}] (skip)')

    # ── 2. RRD server + to_pylist() ───────────────────────────────────────────
    label = 'rrd_server_to_pylist'
    if label not in _done:
        print(f'\n[{label}]  server_batch={sb}')
        _, n_bytes = _path_stats(rrd_j75)
        ratio = raw_bytes / n_bytes if n_bytes else 0
        sc = _bench_server(rrd_j75, cam_keys, N, _RRDServerPylDataset,
                           _WORKER_COUNTS, n_ep, sb)
        _row(label, 'rrd_server', n_bytes/1e6, ratio, sc)
    else:
        print(f'\n[{label}] (skip)')

    # ── 3. RRD server + to_numpy() ────────────────────────────────────────────
    label = 'rrd_server_to_numpy'
    if label not in _done:
        print(f'\n[{label}]  server_batch={sb}')
        _, n_bytes = _path_stats(rrd_j75)
        ratio = raw_bytes / n_bytes if n_bytes else 0
        sc = _bench_server(rrd_j75, cam_keys, N, _RRDServerNpDataset,
                           _WORKER_COUNTS, n_ep, sb)
        _row(label, 'rrd_server', n_bytes/1e6, ratio, sc)
    else:
        print(f'\n[{label}] (skip)')

    # ── 4. Per-episode RRD files, direct rb.load_archive ─────────────────────
    ep_dir = str(out_root / 'rrd_ep_jpeg75')
    label  = 'rrd_ep_jpeg75'
    if label not in _done:
        print(f'\n[{label}]')
        if not os.path.exists(ep_dir):
            enc_s, _, n_bytes = _write_rrd_episodes(
                ep_dir, images, episode_ends, cam_keys, 'jpeg75')
        else:
            _, n_bytes = _path_stats(ep_dir)
        ratio = raw_bytes / n_bytes if n_bytes else 0
        print(f'  size={n_bytes/1e6:.1f} MB  ratio={ratio:.1f}x  → scaling:')
        sc = _bench_dl(
            lambda: _RRDEpisodeDataset(ep_dir, cam_keys, episode_ends, N),
            n_epochs=n_ep)
        _row(label, 'rrd_ep', n_bytes/1e6, ratio, sc)
    else:
        print(f'\n[{label}] (skip)')

    # ── summary ───────────────────────────────────────────────────────────────
    wc = _WORKER_COUNTS
    W  = 40 + 8 + 7 + 7 * len(wc)
    print(f'\n{"─"*W}')
    print(f'RRD BENCHMARK  N={N}  epochs={n_ep}')
    print(f'{"─"*W}')
    hdr = '  ' + f'{"Backend":<38}' + f'{"MB":>7}' + f'{"Ratio":>6}' + \
          ''.join(f'{w:>5}w' for w in wc)
    print(hdr)
    print('  ' + '-' * (W - 2))
    for r in results:
        sc  = r['scaling']
        fps = [sc.get(str(w), sc.get(w, 0)) for w in wc]
        def _f(v):
            if v >= 10000: return f'{v/1000:.0f}k'
            if v >= 1000:  return f'{v/1000:.1f}k'
            return f'{v:.0f}'
        fps_str = ''.join(f'{_f(v):>6}' for v in fps)
        print(f'  {r["label"]:<38} {r["size_mb"]:>6.1f} {r["compression_ratio"]:>5.1f}x  {fps_str}')
    print(f'{"─"*W}')
    print(f'Results → {out_json}')


if __name__ == '__main__':
    main()
