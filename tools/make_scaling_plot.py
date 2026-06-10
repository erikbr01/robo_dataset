"""Generate throughput-vs-workers scaling line plot from benchmark_acceleration.json.

Shows how each backend's DataLoader throughput (fps) scales with num_workers,
produced from the output of benchmark_acceleration.py.
"""
import json
import pathlib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import numpy as np

ACCEL = '/mnt/ssdB/erik/cupid_data/robo_dataset_bench/benchmark_acceleration.json'

with open(ACCEL) as f:
    accel = json.load(f)

dl_fps = accel['dataloader_fps']

# ── Styling ──────────────────────────────────────────────────────────────────
BG = '#0F1117'
FG = '#E0E0E0'

BACKEND_STYLE = {
    'zarr_disk_lz4':       dict(color='#4FC3F7', linestyle='-',  marker='o', label='zarr disk  lz4'),
    'zarr_sharded_lz4':    dict(color='#29B6F6', linestyle='-',  marker='D', label='zarr sharded  lz4  cf=8'),
    'zarr_sharded_jpeg':   dict(color='#81C784', linestyle='--', marker='s', label='zarr sharded  jpeg'),
    'rlds_jpeg_q75':       dict(color='#FF6B6B', linestyle='-',  marker='*', label='RLDS  jpeg q75',    markersize=10),
    'h264_kf1_crf23':      dict(color='#FF8A65', linestyle='-',  marker='h', label='VideoStore  H.264 kf1'),
    'shm_lz4':             dict(color='#FFD700', linestyle='-',  marker='^', label='ShmStore  (RAM, zero-copy)'),
}

workers = sorted(int(k) for k in dl_fps.keys())
# collect which backend keys exist
all_labels = set()
for w in workers:
    for k in dl_fps[str(w)]:
        if not k.startswith('speedup'):
            all_labels.add(k)
all_labels = [l for l in BACKEND_STYLE if l in all_labels]  # keep defined order

# ── Figure ───────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 6))
fig.patch.set_facecolor(BG)
ax.set_facecolor(BG)

for label in all_labels:
    style = BACKEND_STYLE[label]
    xs, ys = [], []
    for w in workers:
        v = dl_fps[str(w)].get(label)
        if v is not None:
            xs.append(w)
            ys.append(v / 1000)
    kw = dict(linewidth=2, markersize=style.get('markersize', 7), zorder=3)
    ax.plot(xs, ys, color=style['color'], linestyle=style['linestyle'],
            marker=style['marker'], label=style['label'], **kw)
    # label at end of line
    if xs:
        ax.annotate(
            style['label'].split('  ')[0],
            xy=(xs[-1], ys[-1]),
            xytext=(6, 0), textcoords='offset points',
            fontsize=8, color=style['color'], va='center',
            path_effects=[pe.withStroke(linewidth=2, foreground=BG)],
        )

ax.set_xlabel('DataLoader num_workers', fontsize=11, color=FG)
ax.set_ylabel('Throughput (k fps)', fontsize=11, color=FG)

horizon = accel['config'].get('horizon', 8)
batch   = accel['config'].get('batch', 32)
ax.set_title(
    f'DataLoader throughput scaling  (horizon={horizon}, batch={batch})\n'
    'Robomimic Lift PH · 9 666 frames × 2 cams · 84×84×3',
    fontsize=10.5, color=FG, pad=10,
)

ax.tick_params(colors=FG, labelsize=9)
ax.xaxis.label.set_color(FG)
ax.yaxis.label.set_color(FG)
ax.set_xticks(workers)
ax.grid(True, which='both', color='#2A2A2A', linestyle='-', linewidth=0.5, zorder=0)
for spine in ax.spines.values():
    spine.set_edgecolor('#444')

ax.legend(loc='upper left', fontsize=8.5, framealpha=0.35,
          facecolor='#1A1A2E', edgecolor='#444', labelcolor=FG)

ax.set_xlim(left=-0.5)
ax.set_ylim(bottom=0)

plt.tight_layout()

out_repo   = '/home/erbauer/robo_dataset/media/scaling_plot.png'
out_shared = '/home/erbauer/diffusion_policy/media/scaling_plot.png'
for out in (out_repo, out_shared):
    pathlib.Path(out).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=150, bbox_inches='tight', facecolor=BG)
    print(f'Saved → {out}')
