"""Generate storage-format Pareto scatter plot."""
import json
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import numpy as np

# ── Load data ──────────────────────────────────────────────────────────────
MAIN = '/mnt/ssdB/erik/cupid_data/storage_bench/storage_results.json'
RRD  = '/mnt/ssdB/erik/cupid_data/rrd_bench/rrd_bench_results.json'
LERO = '/mnt/ssdB/erik/cupid_data/lerobot_bench/lerobot_bench_results.json'
RLDS = '/mnt/ssdB/erik/cupid_data/rlds_bench/rlds_bench_results.json'

def fps12(r): return r['scaling']['12']

with open(MAIN) as f: main = json.load(f)
with open(RRD)  as f: rrd  = json.load(f)
with open(LERO) as f: lero = json.load(f)
with open(RLDS) as f: rlds = json.load(f)

main_by_label = {r['label']: r for r in main['results']}
rrd_by_label  = {r['label']: r for r in rrd['results']}
lero_by_label = {r['label']: r for r in lero['results']}
rlds_by_label = {r['label']: r for r in rlds['results'] if r['label'] != 'zarr_disk_jpeg_q75'}

# ── Curated entry list ──────────────────────────────────────────────────────
# (label, display_name, MB, fps_at_12w, category)
BG  = '#0F1117'
FG  = '#E0E0E0'

CATS = {
    'rlds':          dict(color='#FF6B6B', marker='*',  label='RLDS/TFRecord'),
    'zarr_lossless': dict(color='#4FC3F7', marker='o',  label='zarr disk (lossless)'),
    'zarr_lossy':    dict(color='#81C784', marker='s',  label='zarr disk (lossy)'),
    'hdf5':          dict(color='#CE93D8', marker='D',  label='HDF5'),
    'rrd':           dict(color='#FFB74D', marker='^',  label='RRD (various)'),
    'video_cached':  dict(color='#F06292', marker='P',  label='video + cached decoder'),
    'video_official':dict(color='#EF9A9A', marker='X',  label='video + official decoder'),
    'robodm':        dict(color='#A5D6A7', marker='v',  label='RoboDM episode cache'),
}

entries = [
    # RLDS/TFRecord
    ('rlds_frame_jpeg_q75',   'rlds frame',     rlds_by_label['rlds_frame_jpeg_q75'],   'rlds'),
    ('rlds_episode_jpeg_q75', 'rlds episode',   rlds_by_label['rlds_episode_jpeg_q75'], 'rlds'),
    # zarr lossless
    ('zarr_disk_raw',       'zarr raw',       main_by_label['zarr_disk_raw'],       'zarr_lossless'),
    ('zarr_disk_lz4',       'zarr lz4',       main_by_label['zarr_disk_lz4'],       'zarr_lossless'),
    ('zarr_disk_zstd_l1',   'zarr zstd-l1',   main_by_label['zarr_disk_zstd_l1'],   'zarr_lossless'),
    ('zarr_disk_zstd_l3',   'zarr zstd-l3',   main_by_label['zarr_disk_zstd_l3'],   'zarr_lossless'),
    ('zarr_disk_zstd_l9',   'zarr zstd-l9',   main_by_label['zarr_disk_zstd_l9'],   'zarr_lossless'),
    # zarr lossy
    ('zarr_disk_jpeg_q75',  'zarr jpeg q75',  main_by_label['zarr_disk_jpeg_q75'],  'zarr_lossy'),
    ('zarr_disk_jpeg_q85',  'zarr jpeg q85',  main_by_label['zarr_disk_jpeg_q85'],  'zarr_lossy'),
    ('zarr_disk_jpeg_q95',  'zarr jpeg q95',  main_by_label['zarr_disk_jpeg_q95'],  'zarr_lossy'),
    ('zarr_disk_j2k_q50',   'zarr j2k',       main_by_label['zarr_disk_j2k_q50'],   'zarr_lossy'),
    ('zarr_disk_png',       'zarr png',       main_by_label['zarr_disk_png'],       'zarr_lossy'),
    ('zarr_disk_webp_q75',  'zarr webp',      main_by_label['zarr_disk_webp_q75'],  'zarr_lossy'),
    # HDF5
    ('hdf5_disk',           'HDF5',           main_by_label['hdf5_disk'],           'hdf5'),
    # RRD
    ('rrd_ep_jpeg75',       'rrd ep/file',    rrd_by_label['rrd_ep_jpeg75'],        'rrd'),
    ('rrd_jpeg75_ram_preload','rrd RAM preload', rrd_by_label['rrd_jpeg75_ram_preload'], 'rrd'),
    ('rrd_server_to_numpy', 'rrd gRPC server',rrd_by_label['rrd_server_to_numpy'],  'rrd'),
    # Video cached
    ('h264_kf1_cached',     'H.264 kf1 cached', lero_by_label['h264_kf1_cached'],  'video_cached'),
    ('h264_kf2_cached',     'H.264 kf2 cached', lero_by_label['h264_kf2_cached'],  'video_cached'),
    ('av1_kf2_cached',      'AV1 kf2 cached',   lero_by_label['av1_kf2_cached'],   'video_cached'),
    # Video official
    ('h264_kf1_official',   'H.264 official',   lero_by_label['h264_kf1_official'], 'video_official'),
    ('av1_kf2_official',    'AV1 official',     lero_by_label['av1_kf2_official'],  'video_official'),
    # RoboDM
    ('robodm_h264_crf28',   'RoboDM h264',    main_by_label['robodm_h264_crf28'],   'robodm'),
]

sizes = np.array([r['size_mb'] for _, _, r, _ in entries])
fps   = np.array([fps12(r)     for _, _, r, _ in entries])
cats  = [c for _, _, _, c in entries]
names = [n for _, n, _, _ in entries]

# ── Pareto frontier ─────────────────────────────────────────────────────────
def pareto_front(sizes, fps):
    order = np.argsort(sizes)
    front = []
    max_fps = -1
    for i in order:
        if fps[i] > max_fps:
            front.append(i)
            max_fps = fps[i]
    return front

pareto_idx = set(pareto_front(sizes, fps))

PARETO_LABELS = {
    'rlds_frame_jpeg_q75':   ('rlds\nframe',      (-80, 8)),
    'rlds_episode_jpeg_q75': ('rlds\nepisode',    (-80, -18)),
    'zarr_disk_jpeg_q75':    ('zarr\njpeg q75',   ( 5, -16)),
    'h264_kf1_cached':       ('H.264 kf1\ncached',( 5,  8)),
    'av1_kf2_cached':        ('AV1 kf2\ncached',  ( 5, -16)),
    'robodm_h264_crf28':     ('RoboDM\nh264',     ( 5,  8)),
}

# ── Plot ────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(11, 7))
fig.patch.set_facecolor(BG)
ax.set_facecolor(BG)

for spine in ax.spines.values():
    spine.set_edgecolor('#444')

ax.tick_params(colors=FG, labelsize=9)
ax.xaxis.label.set_color(FG)
ax.yaxis.label.set_color(FG)

# Grid
ax.grid(True, which='both', color='#2A2A2A', linestyle='-', linewidth=0.5, zorder=0)

# Plot all points
plotted_cats = set()
for i, (key, name, rec, cat) in enumerate(entries):
    if fps[i] == 0:
        continue
    cfg = CATS[cat]
    kw = dict(
        color=cfg['color'], marker=cfg['marker'],
        s=80, zorder=3, linewidths=0,
        label=cfg['label'] if cat not in plotted_cats else '_',
    )
    if i in pareto_idx:
        kw['s'] = 160
        kw['edgecolors'] = '#FFD700'
        kw['linewidths'] = 1.8
        kw['zorder'] = 5
    ax.scatter(sizes[i], fps[i] / 1000, **kw)
    plotted_cats.add(cat)

# Pareto staircase
pf = sorted(pareto_idx, key=lambda i: sizes[i])
pf_x = [sizes[i] for i in pf if fps[i] > 0]
pf_y = [fps[i] / 1000 for i in pf if fps[i] > 0]
ax.step(pf_x, pf_y, where='post', color='#FFD700', linewidth=1.5,
        linestyle='--', alpha=0.7, zorder=4, label='Pareto frontier')

# Labels on Pareto points
for i in pf:
    if fps[i] == 0:
        continue
    key = list(PARETO_LABELS.keys())
    # find this entry's key
    entry_key = entries[i][0]
    if entry_key in PARETO_LABELS:
        lbl, (dx, dy) = PARETO_LABELS[entry_key]
        ax.annotate(
            lbl,
            xy=(sizes[i], fps[i] / 1000),
            xytext=(dx, dy),
            textcoords='offset points',
            fontsize=7.5,
            color='#FFD700',
            ha='left',
            path_effects=[pe.withStroke(linewidth=2, foreground=BG)],
        )

# Axes
ax.set_xscale('log')
ax.set_xlabel('Storage size (MB, log scale)', fontsize=11)
ax.set_ylabel('Throughput at 12 workers (k fps)', fontsize=11)
ax.set_title(
    'Storage Format Benchmark — Throughput vs Size\n'
    '(Robomimic Lift PH, 9666 frames × 2 cameras, 84×84×3, 12 DataLoader workers)\n'
    'Gold outline = Pareto-optimal (higher throughput AND smaller size)',
    fontsize=10, color=FG, pad=12,
)

# X-axis ticks
ax.set_xticks([2, 5, 10, 30, 100, 300, 1000])
ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
ax.set_xticklabels(['2', '5', '10', '30', '100', '300', '1000'], fontsize=8)

# Legend
leg = ax.legend(
    loc='lower right',
    fontsize=8.5,
    framealpha=0.3,
    facecolor='#1A1A2E',
    edgecolor='#444',
    labelcolor=FG,
    markerscale=1.2,
)

# Size annotations on x-axis (raw=409 MB)
ax.axvline(409, color='#555', linewidth=0.8, linestyle=':')
ax.text(409 * 1.06, 1.0, 'raw (409 MB)', color='#888', fontsize=7.5,
        rotation=90, va='bottom')

out = '/home/erbauer/diffusion_policy/media/storage_benchmark_pareto.png'
plt.tight_layout()
plt.savefig(out, dpi=150, bbox_inches='tight', facecolor=BG)
print(f'Saved → {out}')
