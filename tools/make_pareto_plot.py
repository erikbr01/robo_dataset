"""Generate storage-format Pareto scatter plots.

Two panels:
  Left  — horizon=1 raw-frame benchmark (existing storage/lerobot/rlds/rrd data)
  Right — horizon=8 training-scenario benchmark (acceleration benchmark data)
"""
import json
import pathlib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import matplotlib.ticker
import numpy as np

# ── Load data ──────────────────────────────────────────────────────────────
MAIN  = '/mnt/ssdB/erik/cupid_data/storage_bench/storage_results.json'
RRD   = '/mnt/ssdB/erik/cupid_data/rrd_bench/rrd_bench_results.json'
LERO  = '/mnt/ssdB/erik/cupid_data/lerobot_bench/lerobot_bench_results.json'
RLDS  = '/mnt/ssdB/erik/cupid_data/rlds_bench/rlds_bench_results.json'
ACCEL = '/mnt/ssdB/erik/cupid_data/robo_dataset_bench/benchmark_acceleration.json'

def fps12(r): return r['scaling']['12']

with open(MAIN)  as f: main  = json.load(f)
with open(RRD)   as f: rrd   = json.load(f)
with open(LERO)  as f: lero  = json.load(f)
with open(RLDS)  as f: rlds  = json.load(f)
with open(ACCEL) as f: accel = json.load(f)

main_by_label = {r['label']: r for r in main['results']}
rrd_by_label  = {r['label']: r for r in rrd['results']}
lero_by_label = {r['label']: r for r in lero['results']}
rlds_by_label = {r['label']: r for r in rlds['results'] if r['label'] != 'zarr_disk_jpeg_q75'}
accel_fps12   = accel['dataloader_fps']['12']


def _dir_size_mb(path):
    return sum(f.stat().st_size for f in pathlib.Path(path).rglob('*') if f.is_file()) / 1e6


# ── Styling ────────────────────────────────────────────────────────────────
BG = '#0F1117'
FG = '#E0E0E0'

CATS = {
    'rlds':           dict(color='#FF6B6B', marker='*',  label='RLDS/TFRecord'),
    'zarr_lossless':  dict(color='#4FC3F7', marker='o',  label='zarr disk (lossless)'),
    'zarr_lossy':     dict(color='#81C784', marker='s',  label='zarr disk (lossy)'),
    'zarr_sharded':   dict(color='#29B6F6', marker='D',  label='zarr sharded'),
    'hdf5':           dict(color='#CE93D8', marker='d',  label='HDF5'),
    'rrd':            dict(color='#FFB74D', marker='^',  label='RRD (various)'),
    'video_cached':   dict(color='#F06292', marker='P',  label='video + cached decoder'),
    'video_official': dict(color='#EF9A9A', marker='X',  label='video + official decoder'),
    'video_store':    dict(color='#FF8A65', marker='h',  label='VideoStore (robo_dataset)'),
    'robodm':         dict(color='#A5D6A7', marker='v',  label='RoboDM episode cache'),
}


# ── Pareto frontier ─────────────────────────────────────────────────────────
def pareto_front(sizes, fps):
    order = np.argsort(sizes)
    front, max_fps = [], -1
    for i in order:
        if fps[i] > max_fps:
            front.append(i)
            max_fps = fps[i]
    return front


# ── Panel helper ────────────────────────────────────────────────────────────
def _draw_panel(ax, entries, title, raw_mb=None):
    sizes = np.array([e['mb']  for e in entries])
    fps   = np.array([e['fps'] for e in entries])

    pareto_idx = set(pareto_front(sizes, fps))

    plotted_cats = set()
    for i, e in enumerate(entries):
        if fps[i] == 0:
            continue
        cfg = CATS[e['cat']]
        kw = dict(
            color=cfg['color'], marker=cfg['marker'],
            s=80, zorder=3, linewidths=0,
            label=cfg['label'] if e['cat'] not in plotted_cats else '_',
        )
        if i in pareto_idx:
            kw.update(s=160, edgecolors='#FFD700', linewidths=1.8, zorder=5)
        ax.scatter(sizes[i], fps[i] / 1000, **kw)
        plotted_cats.add(e['cat'])

    # Pareto staircase
    pf = sorted(pareto_idx, key=lambda i: sizes[i])
    pf_x = [sizes[i] for i in pf if fps[i] > 0]
    pf_y = [fps[i] / 1000 for i in pf if fps[i] > 0]
    ax.step(pf_x, pf_y, where='post', color='#FFD700', linewidth=1.5,
            linestyle='--', alpha=0.7, zorder=4, label='Pareto frontier')

    # Labels on Pareto points
    for i in pf:
        if fps[i] == 0 or 'label' not in entries[i]:
            continue
        lbl, (dx, dy) = entries[i]['label'], entries[i].get('lbl_offset', (5, 8))
        ax.annotate(
            lbl,
            xy=(sizes[i], fps[i] / 1000),
            xytext=(dx, dy),
            textcoords='offset points',
            fontsize=7.5, color='#FFD700', ha='left',
            path_effects=[pe.withStroke(linewidth=2, foreground=BG)],
        )

    ax.set_xscale('log')
    ax.set_xlabel('Storage size (MB, log scale)', fontsize=10)
    ax.set_ylabel('Throughput at 12 workers (k fps)', fontsize=10)
    ax.set_title(title, fontsize=9.5, color=FG, pad=10)
    ax.tick_params(colors=FG, labelsize=8)
    ax.xaxis.label.set_color(FG)
    ax.yaxis.label.set_color(FG)
    ax.grid(True, which='both', color='#2A2A2A', linestyle='-', linewidth=0.5, zorder=0)
    for spine in ax.spines.values():
        spine.set_edgecolor('#444')

    xticks = [2, 5, 10, 30, 100, 300, 1000]
    ax.set_xticks(xticks)
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.set_xticklabels([str(x) for x in xticks], fontsize=7.5)

    if raw_mb:
        ax.axvline(raw_mb, color='#555', linewidth=0.8, linestyle=':')
        ax.text(raw_mb * 1.06, ax.get_ylim()[1] * 0.05,
                f'raw ({raw_mb:.0f} MB)', color='#888', fontsize=7,
                rotation=90, va='bottom')

    ax.legend(loc='lower right', fontsize=7.5, framealpha=0.3,
              facecolor='#1A1A2E', edgecolor='#444', labelcolor=FG, markerscale=1.1)


# ── Panel A: horizon=1 raw-frame benchmark ─────────────────────────────────
entries_h1 = [
    # RLDS
    dict(mb=rlds_by_label['rlds_frame_jpeg_q75']['size_mb'],
         fps=fps12(rlds_by_label['rlds_frame_jpeg_q75']),
         cat='rlds', label='rlds\nframe', lbl_offset=(-80, 8)),
    dict(mb=rlds_by_label['rlds_episode_jpeg_q75']['size_mb'],
         fps=fps12(rlds_by_label['rlds_episode_jpeg_q75']),
         cat='rlds', label='rlds\nepisode', lbl_offset=(-80, -18)),
    # zarr lossless
    dict(mb=main_by_label['zarr_disk_raw']['size_mb'],      fps=fps12(main_by_label['zarr_disk_raw']),      cat='zarr_lossless'),
    dict(mb=main_by_label['zarr_disk_lz4']['size_mb'],      fps=fps12(main_by_label['zarr_disk_lz4']),      cat='zarr_lossless'),
    dict(mb=main_by_label['zarr_disk_zstd_l1']['size_mb'],  fps=fps12(main_by_label['zarr_disk_zstd_l1']),  cat='zarr_lossless'),
    dict(mb=main_by_label['zarr_disk_zstd_l3']['size_mb'],  fps=fps12(main_by_label['zarr_disk_zstd_l3']),  cat='zarr_lossless'),
    dict(mb=main_by_label['zarr_disk_zstd_l9']['size_mb'],  fps=fps12(main_by_label['zarr_disk_zstd_l9']),  cat='zarr_lossless'),
    # zarr lossy
    dict(mb=main_by_label['zarr_disk_jpeg_q75']['size_mb'], fps=fps12(main_by_label['zarr_disk_jpeg_q75']), cat='zarr_lossy',
         label='zarr\njpeg q75', lbl_offset=(5, -16)),
    dict(mb=main_by_label['zarr_disk_jpeg_q85']['size_mb'], fps=fps12(main_by_label['zarr_disk_jpeg_q85']), cat='zarr_lossy'),
    dict(mb=main_by_label['zarr_disk_jpeg_q95']['size_mb'], fps=fps12(main_by_label['zarr_disk_jpeg_q95']), cat='zarr_lossy'),
    dict(mb=main_by_label['zarr_disk_j2k_q50']['size_mb'],  fps=fps12(main_by_label['zarr_disk_j2k_q50']),  cat='zarr_lossy'),
    dict(mb=main_by_label['zarr_disk_png']['size_mb'],      fps=fps12(main_by_label['zarr_disk_png']),      cat='zarr_lossy'),
    dict(mb=main_by_label['zarr_disk_webp_q75']['size_mb'], fps=fps12(main_by_label['zarr_disk_webp_q75']), cat='zarr_lossy'),
    # HDF5
    dict(mb=main_by_label['hdf5_disk']['size_mb'], fps=fps12(main_by_label['hdf5_disk']), cat='hdf5'),
    # RRD
    dict(mb=rrd_by_label['rrd_ep_jpeg75']['size_mb'],         fps=fps12(rrd_by_label['rrd_ep_jpeg75']),          cat='rrd'),
    dict(mb=rrd_by_label['rrd_jpeg75_ram_preload']['size_mb'],fps=fps12(rrd_by_label['rrd_jpeg75_ram_preload']),  cat='rrd'),
    dict(mb=rrd_by_label['rrd_server_to_numpy']['size_mb'],   fps=fps12(rrd_by_label['rrd_server_to_numpy']),    cat='rrd'),
    # video cached
    dict(mb=lero_by_label['h264_kf1_cached']['size_mb'], fps=fps12(lero_by_label['h264_kf1_cached']),
         cat='video_cached', label='H.264 kf1\ncached', lbl_offset=(5, 8)),
    dict(mb=lero_by_label['h264_kf2_cached']['size_mb'], fps=fps12(lero_by_label['h264_kf2_cached']), cat='video_cached'),
    dict(mb=lero_by_label['av1_kf2_cached']['size_mb'],  fps=fps12(lero_by_label['av1_kf2_cached']),
         cat='video_cached', label='AV1 kf2\ncached', lbl_offset=(5, -16)),
    # video official
    dict(mb=lero_by_label['h264_kf1_official']['size_mb'], fps=fps12(lero_by_label['h264_kf1_official']), cat='video_official'),
    dict(mb=lero_by_label['av1_kf2_official']['size_mb'],  fps=fps12(lero_by_label['av1_kf2_official']),  cat='video_official'),
    # RoboDM
    dict(mb=main_by_label['robodm_h264_crf28']['size_mb'], fps=fps12(main_by_label['robodm_h264_crf28']),
         cat='robodm', label='RoboDM\nh264', lbl_offset=(5, 8)),
]

# ── Panel B: horizon=8 training-scenario benchmark ─────────────────────────
BENCH_ROOT = '/mnt/ssdB/erik/cupid_data/robo_dataset_bench'

entries_h8 = [
    dict(mb=_dir_size_mb(f'{BENCH_ROOT}/zarr_disk_lz4'),
         fps=accel_fps12['zarr_disk_lz4'],
         cat='zarr_lossless', label='zarr disk\nlz4', lbl_offset=(5, 8)),
    dict(mb=_dir_size_mb(f'{BENCH_ROOT}/zarr_sharded_cf8'),
         fps=accel_fps12['zarr_sharded_lz4'],
         cat='zarr_sharded', label='zarr sharded\nlz4 cf=8', lbl_offset=(5, 8)),
    dict(mb=_dir_size_mb(f'{BENCH_ROOT}/zarr_sharded_jpeg_q75'),
         fps=accel_fps12['zarr_sharded_jpeg'],
         cat='zarr_sharded'),
    dict(mb=_dir_size_mb(f'{BENCH_ROOT}/rlds_jpeg_q75'),
         fps=accel_fps12['rlds_jpeg_q75'],
         cat='rlds', label='rlds\njpeg q75', lbl_offset=(-75, 8)),
    dict(mb=_dir_size_mb(f'{BENCH_ROOT}/h264_kf1_crf23'),
         fps=accel_fps12['h264_kf1_crf23'],
         cat='video_store', label='VideoStore\nH.264 kf1', lbl_offset=(5, 8)),
]

# ── Build figure ────────────────────────────────────────────────────────────
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7))
fig.patch.set_facecolor(BG)
ax1.set_facecolor(BG)
ax2.set_facecolor(BG)

_draw_panel(
    ax1, entries_h1,
    'Raw frame throughput  (horizon=1)\n'
    'Robomimic Lift PH · 9666 frames × 2 cams · 84×84×3 · 12 workers\n'
    'Gold outline = Pareto-optimal',
    raw_mb=409,
)

_draw_panel(
    ax2, entries_h8,
    'Training throughput  (horizon=8)\n'
    'Robomimic Lift PH · 9666 frames × 2 cams · 84×84×3 · 12 workers\n'
    'Gold outline = Pareto-optimal',
    raw_mb=409,
)

fig.suptitle(
    'robo_dataset — Storage Format Benchmark: Throughput vs Size',
    fontsize=13, color=FG, y=1.01,
)

plt.tight_layout()

out = '/home/erbauer/diffusion_policy/media/storage_benchmark_pareto.png'
plt.savefig(out, dpi=150, bbox_inches='tight', facecolor=BG)
print(f'Saved → {out}')
