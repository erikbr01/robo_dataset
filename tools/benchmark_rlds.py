#!/usr/bin/env python3
"""Benchmark RLDS/TFRecord storage format for image DataLoaders.

Runs in: rerun_env  (tfrecord, zarr, numcodecs, torch, h5py)

Two variants:
  rlds_frame_jpeg_q75    — one tf.train.Example per frame, index-based random access
  rlds_episode_jpeg_q75  — one tf.train.SequenceExample per episode, episode-cache per worker

zarr_disk_jpeg_q75 is included as a baseline cross-check.
"""
import os, sys, json, struct, time
import numpy as np
import cv2
import h5py
import io

# ── Config ──────────────────────────────────────────────────────────────────
HDF5_PATH   = '/mnt/ssdB/erik/cupid_data/robomimic/datasets/lift/ph/image.hdf5'
OUT_DIR     = '/mnt/ssdB/erik/cupid_data/rlds_bench'
RESULTS_JSON = os.path.join(OUT_DIR, 'rlds_bench_results.json')
WORKER_COUNTS = [0, 1, 2, 4, 6, 8, 10, 12]
N_EPOCHS    = 2
BATCH_SIZE  = 32
JPEG_QUALITY = 75

os.makedirs(OUT_DIR, exist_ok=True)

# ── TFRecord utils ──────────────────────────────────────────────────────────
def _load_index(idx_path):
    """Return array of byte offsets, one per record."""
    return np.loadtxt(idx_path, dtype=np.int64)[:, 0]

def _read_record(f, offset):
    """Read one TFRecord at byte offset; return raw proto bytes."""
    f.seek(offset)
    length = struct.unpack('<Q', f.read(8))[0]
    f.read(4)           # masked crc32 of length
    data = f.read(length)
    f.read(4)           # masked crc32 of data
    return data

# ── Dataset classes ─────────────────────────────────────────────────────────
class _TFRFrameDataset:
    """Map-style Dataset: one tf.train.Example per frame, random access via index."""
    def __init__(self, tfr_path, idx_path, cam_keys, n_total):
        from tfrecord import example_pb2 as _pb2
        self.tfr_path = tfr_path
        self.cam_keys = cam_keys
        self.offsets = _load_index(idx_path)
        self.n_total = n_total
        self._pb2 = _pb2
        self._file = None
        self._pid = None

    def __len__(self): return self.n_total

    def _file_handle(self):
        pid = os.getpid()
        if pid != self._pid:
            if self._file is not None:
                self._file.close()
            self._file = open(self.tfr_path, 'rb')
            self._pid = pid
        return self._file

    def __getitem__(self, idx):
        raw = _read_record(self._file_handle(), self.offsets[idx])
        ex = self._pb2.Example()
        ex.ParseFromString(raw)
        frames = []
        for cam in self.cam_keys:
            blob = bytes(ex.features.feature[cam].bytes_list.value[0])
            img = cv2.imdecode(np.frombuffer(blob, np.uint8), cv2.IMREAD_COLOR)
            frames.append(img)
        return np.stack(frames)


class _TFREpisodeDataset:
    """Map-style Dataset: one tf.train.SequenceExample per episode, episode-cache per worker."""
    def __init__(self, tfr_path, idx_path, cam_keys, episode_ends, n_total):
        from tfrecord import example_pb2 as _pb2
        self.tfr_path = tfr_path
        self.cam_keys = cam_keys
        self.ep_offsets = _load_index(idx_path)  # one offset per episode
        self.episode_ends = np.asarray(episode_ends)
        self.ep_starts = np.concatenate([[0], self.episode_ends[:-1]])
        self.n_total = n_total
        self._pb2 = _pb2
        self._file = None
        self._pid = None
        self._cache = {}

    def __len__(self): return self.n_total

    def _file_handle(self):
        pid = os.getpid()
        if pid != self._pid:
            if self._file is not None:
                self._file.close()
            self._file = open(self.tfr_path, 'rb')
            self._pid = pid
            self._cache = {}
        return self._file

    def _get_ep(self, ep_idx):
        self._file_handle()  # ensure pid reset
        pid = os.getpid()
        if ep_idx not in self._cache:
            raw = _read_record(self._file, self.ep_offsets[ep_idx])
            seq = self._pb2.SequenceExample()
            seq.ParseFromString(raw)
            ep_data = {}
            for cam in self.cam_keys:
                fl = seq.feature_lists.feature_list[cam]
                ep_data[cam] = [bytes(f.bytes_list.value[0]) for f in fl.feature]
            self._cache[ep_idx] = ep_data
        return self._cache[ep_idx]

    def __getitem__(self, global_idx):
        ep_idx = int(np.searchsorted(self.episode_ends, global_idx, side='right'))
        local_idx = global_idx - self.ep_starts[ep_idx]
        ep = self._get_ep(ep_idx)
        frames = []
        for cam in self.cam_keys:
            blob = ep[cam][local_idx]
            img = cv2.imdecode(np.frombuffer(blob, np.uint8), cv2.IMREAD_COLOR)
            frames.append(img)
        return np.stack(frames)


class _ZarrJpegDataset:
    """Zarr disk JPEG q75 baseline (zarr 2.x API)."""
    def __init__(self, zarr_dir, cam_keys, n_total):
        import zarr
        self.arrays = {cam: zarr.open(os.path.join(zarr_dir, cam), 'r') for cam in cam_keys}
        self.cam_keys = cam_keys
        self.n_total = n_total

    def __len__(self): return self.n_total

    def __getitem__(self, idx):
        frames = []
        for cam in self.cam_keys:
            blob = bytes(self.arrays[cam][idx])
            img = cv2.imdecode(np.frombuffer(blob, np.uint8), cv2.IMREAD_COLOR)
            frames.append(img)
        return np.stack(frames)


# ── Data building ────────────────────────────────────────────────────────────
def _load_hdf5(hdf5_path):
    """Return (images_by_cam, episode_ends, cam_keys). images shape: (N, H, W, C) uint8."""
    with h5py.File(hdf5_path, 'r') as f:
        demos = sorted(f['data'].keys(), key=lambda k: int(k.split('_')[1]))
        cam_keys = [k for k in f[f'data/{demos[0]}/obs'].keys() if 'image' in k]
        print(f'  demos={len(demos)}  cam_keys={cam_keys}')
        images = {cam: [] for cam in cam_keys}
        ep_ends = []
        total = 0
        for demo in demos:
            obs = f[f'data/{demo}/obs']
            n = obs[cam_keys[0]].shape[0]
            for cam in cam_keys:
                images[cam].append(obs[cam][:])
            total += n
            ep_ends.append(total)
    for cam in cam_keys:
        images[cam] = np.concatenate(images[cam], axis=0)
    print(f'  {total} frames × {len(cam_keys)} cams  raw={images[cam_keys[0]].nbytes * len(cam_keys) / 1e6:.0f} MB')
    return images, ep_ends, cam_keys


def _encode_jpeg(img_hwc, quality=75):
    """uint8 HWC → JPEG bytes."""
    ok, buf = cv2.imencode('.jpg', img_hwc, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return bytes(buf.tobytes())


def _build_tfr_frame(images, cam_keys, quality, out_dir):
    """Write one tf.train.Example per frame."""
    from tfrecord.writer import TFRecordWriter
    from tfrecord.tools.tfrecord2idx import create_index
    name = f'frame_jpeg_q{quality}'
    tfr_path = os.path.join(out_dir, f'{name}.tfrecord')
    idx_path = os.path.join(out_dir, f'{name}.index')
    if os.path.exists(tfr_path) and os.path.exists(idx_path):
        print(f'  [skip] {name} already built')
        size_mb = os.path.getsize(tfr_path) / 1e6
        offsets = _load_index(idx_path)
        return tfr_path, idx_path, size_mb, len(offsets)
    print(f'  Building {name} …', flush=True)
    n = images[cam_keys[0]].shape[0]
    t0 = time.time()
    writer = TFRecordWriter(tfr_path)
    for i in range(n):
        datum = {cam: (_encode_jpeg(images[cam][i], quality), 'byte') for cam in cam_keys}
        writer.write(datum)
    writer.close()
    create_index(tfr_path, idx_path)
    elapsed = time.time() - t0
    size_mb = os.path.getsize(tfr_path) / 1e6
    offsets = _load_index(idx_path)
    print(f'  → {size_mb:.1f} MB  {n} records  {elapsed:.1f}s encode')
    return tfr_path, idx_path, size_mb, n


def _build_tfr_episode(images, cam_keys, episode_ends, quality, out_dir):
    """Write one tf.train.SequenceExample per episode."""
    from tfrecord.writer import TFRecordWriter
    from tfrecord.tools.tfrecord2idx import create_index
    name = f'episode_jpeg_q{quality}'
    tfr_path = os.path.join(out_dir, f'{name}.tfrecord')
    idx_path = os.path.join(out_dir, f'{name}.index')
    if os.path.exists(tfr_path) and os.path.exists(idx_path):
        print(f'  [skip] {name} already built')
        size_mb = os.path.getsize(tfr_path) / 1e6
        offsets = _load_index(idx_path)
        return tfr_path, idx_path, size_mb, len(offsets)
    print(f'  Building {name} …', flush=True)
    ep_starts = [0] + list(episode_ends[:-1])
    t0 = time.time()
    writer = TFRecordWriter(tfr_path)
    for ep_idx, (ep_start, ep_end) in enumerate(zip(ep_starts, episode_ends)):
        seq_datum = {}
        for cam in cam_keys:
            frames_bytes = [_encode_jpeg(images[cam][fi], quality)
                            for fi in range(ep_start, ep_end)]
            seq_datum[cam] = (frames_bytes, 'byte')
        writer.write({}, sequence_datum=seq_datum)
    writer.close()
    create_index(tfr_path, idx_path)
    elapsed = time.time() - t0
    size_mb = os.path.getsize(tfr_path) / 1e6
    n_eps = len(episode_ends)
    print(f'  → {size_mb:.1f} MB  {n_eps} episode records  {elapsed:.1f}s encode')
    return tfr_path, idx_path, size_mb, n_eps


def _build_zarr_jpeg(images, cam_keys, quality, out_dir):
    """Write zarr disk with JPEG numcodecs codec (baseline)."""
    import zarr
    import numcodecs
    name = f'zarr_jpeg_q{quality}'
    zarr_dir = os.path.join(out_dir, name)
    if os.path.exists(zarr_dir):
        print(f'  [skip] {name} already built')
        total_mb = sum(os.path.getsize(os.path.join(dp, f)) / 1e6
                       for dp, _, fns in os.walk(zarr_dir) for f in fns)
        return zarr_dir, total_mb
    print(f'  Building {name} …', flush=True)
    n = images[cam_keys[0]].shape[0]
    codec = numcodecs.JPEG(quality=quality)
    t0 = time.time()
    for cam in cam_keys:
        arr = zarr.open(os.path.join(zarr_dir, cam), mode='w',
                        shape=(n,), dtype=object,
                        object_codec=codec,
                        chunks=(1,))
        for i in range(n):
            arr[i] = images[cam][i]
    elapsed = time.time() - t0
    total_mb = sum(os.path.getsize(os.path.join(dp, f)) / 1e6
                   for dp, _, fns in os.walk(zarr_dir) for f in fns)
    print(f'  → {total_mb:.1f} MB  {elapsed:.1f}s')
    return zarr_dir, total_mb


# ── DataLoader benchmark ─────────────────────────────────────────────────────
def _bench_dl(dataset_factory, n_total, worker_counts, n_epochs, batch_size):
    from torch.utils.data import DataLoader
    results = {}
    for nw in worker_counts:
        ds = dataset_factory()
        dl = DataLoader(ds, batch_size=batch_size, num_workers=nw,
                        shuffle=True, pin_memory=False,
                        persistent_workers=(nw > 0),
                        prefetch_factor=(2 if nw > 0 else None))
        frames_loaded = 0
        t0 = time.perf_counter()
        for _ in range(n_epochs):
            for batch in dl:
                frames_loaded += batch.shape[0] if hasattr(batch, 'shape') else len(batch)
        elapsed = time.perf_counter() - t0
        fps = frames_loaded / elapsed
        results[str(nw)] = round(fps, 1)
        print(f'    {nw:2d}w → {fps/1000:.1f}k fps', flush=True)
        del dl, ds
    return results


# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    # Load existing results for resume
    if os.path.exists(RESULTS_JSON):
        with open(RESULTS_JSON) as f:
            all_results = json.load(f)
        print(f'Resuming — {len(all_results.get("results", []))} backends already done')
    else:
        all_results = {
            'N': None, 'n_cams': None, 'raw_mb': None,
            'worker_counts': WORKER_COUNTS, 'results': []
        }
    done_labels = {r['label'] for r in all_results['results']}

    def save():
        with open(RESULTS_JSON, 'w') as f:
            json.dump(all_results, f, indent=2)

    # Load dataset
    print('Loading images from HDF5 …')
    images, episode_ends, cam_keys = _load_hdf5(HDF5_PATH)
    n_total = images[cam_keys[0]].shape[0]
    raw_mb = sum(a.nbytes for a in images.values()) / 1e6
    all_results.update({'N': n_total, 'n_cams': len(cam_keys), 'raw_mb': round(raw_mb, 1)})
    ep_starts = [0] + list(episode_ends[:-1])
    save()

    # ── A. rlds_frame_jpeg_q75 ────────────────────────────────────────────
    label = 'rlds_frame_jpeg_q75'
    if label not in done_labels:
        print(f'\n[{label}]')
        tfr_path, idx_path, size_mb, n_records = _build_tfr_frame(
            images, cam_keys, JPEG_QUALITY, OUT_DIR)
        ratio = round(raw_mb / size_mb, 2)
        print(f'  size={size_mb:.1f} MB  ratio={ratio}x  → scaling:')
        scaling = _bench_dl(
            lambda: _TFRFrameDataset(tfr_path, idx_path, cam_keys, n_total),
            n_total, WORKER_COUNTS, N_EPOCHS, BATCH_SIZE)
        all_results['results'].append({
            'label': label, 'format': 'rlds_tfrecord_frame',
            'size_mb': round(size_mb, 2), 'compression_ratio': ratio,
            'n_records': n_records, 'scaling': scaling,
        })
        save()
    else:
        print(f'[skip] {label}')

    # ── B. rlds_episode_jpeg_q75 ──────────────────────────────────────────
    label = 'rlds_episode_jpeg_q75'
    if label not in done_labels:
        print(f'\n[{label}]')
        tfr_path, idx_path, size_mb, n_eps = _build_tfr_episode(
            images, cam_keys, episode_ends, JPEG_QUALITY, OUT_DIR)
        ratio = round(raw_mb / size_mb, 2)
        print(f'  size={size_mb:.1f} MB  ratio={ratio}x  n_eps={n_eps}  → scaling:')
        scaling = _bench_dl(
            lambda: _TFREpisodeDataset(tfr_path, idx_path, cam_keys, episode_ends, n_total),
            n_total, WORKER_COUNTS, N_EPOCHS, BATCH_SIZE)
        all_results['results'].append({
            'label': label, 'format': 'rlds_tfrecord_episode',
            'size_mb': round(size_mb, 2), 'compression_ratio': ratio,
            'n_episodes': n_eps, 'scaling': scaling,
        })
        save()
    else:
        print(f'[skip] {label}')

    # zarr baseline pulled from main storage_bench results (imagecodecs_jpeg not in this env)
    MAIN_STORAGE = '/mnt/ssdB/erik/cupid_data/storage_bench/storage_results.json'
    label = 'zarr_disk_jpeg_q75'
    if label not in done_labels and os.path.exists(MAIN_STORAGE):
        with open(MAIN_STORAGE) as f:
            main_data = json.load(f)
        for r in main_data['results']:
            if r['label'] == 'zarr_disk_jpeg_q75':
                all_results['results'].append({
                    'label': label, 'format': 'zarr_disk',
                    'size_mb': r['size_mb'], 'compression_ratio': r['compression_ratio'],
                    'scaling': r['scaling'], 'note': 'from storage_bench',
                })
                save()
                print(f'[zarr baseline] imported from storage_bench: '
                      f'{r["size_mb"]:.1f} MB  12w={r["scaling"]["12"]/1000:.1f}k fps')
                break

    # ── Summary table ─────────────────────────────────────────────────────
    print()
    print('─' * 90)
    print(f'RLDS BENCHMARK  N={n_total}  epochs={N_EPOCHS}')
    print('─' * 90)
    wc_hdr = ''.join(f'{w:>6}w' for w in WORKER_COUNTS)
    print(f'  {"Backend":<35} {"MB":>6} {"Ratio":>7}  {wc_hdr}')
    print('  ' + '-' * 88)
    for r in all_results['results']:
        fps_str = ''.join(f'{r["scaling"].get(str(w), 0)/1000:>6.1f}k' for w in WORKER_COUNTS)
        print(f'  {r["label"]:<35} {r["size_mb"]:>6.1f} {r["compression_ratio"]:>6.1f}x  {fps_str}')
    print('─' * 90)
    print(f'Results → {RESULTS_JSON}')


if __name__ == '__main__':
    main()
