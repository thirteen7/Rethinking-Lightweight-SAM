"""Render the paper's Dense / FSD-SAM timing comparison for the README."""
from pathlib import Path
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / 'site/assets/results'
VIOLET = '#7860de'
BASELINE = '#c5cad6'
INK = '#202333'
MUTED = '#737887'


def plot_timing():
    data = json.loads((ROOT / 'site/paper-everything.json').read_text(encoding='utf-8'))
    rows = {(r['backbone'], r['policy']): r for r in data['rows']}
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'text.color': INK,
                         'svg.fonttype': 'none', 'svg.hashsalt': 'fsd-paper-timing'})
    fig = plt.figure(figsize=(14, 6.5), facecolor='white')
    ax = fig.add_axes([.17, .27, .56, .5])
    fig.text(.035, .94, 'Segment Everything inference time', fontsize=24, weight='bold')
    fig.text(.035, .885, 'Dense SAM vs FSD-SAM · same frozen backbone · lower is faster', fontsize=13, color=MUTED)
    fig.legend([Patch(facecolor=BASELINE), Patch(facecolor=VIOLET)], ['Dense SAM', 'FSD-SAM'],
               loc='upper left', bbox_to_anchor=(.163, .82), frameon=False, ncol=2,
               fontsize=12, handlelength=1.15, columnspacing=2, borderaxespad=0)
    ax.set_xlim(0, 7000)
    ax.set_ylim(2.6, -.6)
    ax.set_xticks([0, 2000, 4000, 6000])
    ax.set_xlabel('Mean end-to-end time (ms/image)', fontsize=12, color=MUTED, labelpad=10)
    ax.tick_params(axis='x', colors=MUTED, labelsize=11, length=0, pad=8)
    ax.set_yticks([])
    ax.grid(axis='x', color='#e9ebf2', linewidth=.8)
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.axhspan(-.48, .48, color='#f7f3ff', zorder=0)
    fig.text(.79, .79, 'Speedup', fontsize=12, weight='bold')
    fig.text(.915, .79, 'Time saved', fontsize=12, weight='bold', ha='center')
    for y, backbone in enumerate(('SAM ViT-H', 'TinySAM', 'EdgeSAM')):
        dense, fsd = rows[backbone, 'Dense'], rows[backbone, 'FSD-SAM']
        ax.text(-140, y, backbone, fontsize=13, ha='right', va='center',
                weight='bold' if y == 0 else 'normal', clip_on=False)
        for row, offset, color in ((dense, -.19, BASELINE), (fsd, .19, VIOLET)):
            value = row['milliseconds']
            ax.barh(y + offset, value, height=.27, color=color)
            ax.text(value + 70, y + offset, f'{value:,} ms', fontsize=12, va='center',
                    color=VIOLET if offset > 0 else MUTED,
                    weight='bold' if offset > 0 else 'normal')
        # Speedups retain the paper's precision; time savings use its displayed ms.
        relative_y = .27 + .5 * (2.6 - y) / 3.2
        fig.text(.815, relative_y, f"{fsd['speedup']:.2f}×", fontsize=23, color=VIOLET,
                 weight='bold', ha='center', va='center')
        saved = 100 * (1 - fsd['milliseconds'] / dense['milliseconds'])
        fig.text(.915, relative_y, f'{saved:.1f}%', fontsize=17, color=INK, ha='center', va='center')
    fig.text(.035, .153,
             'Paper Table 7 · COCO100 development subset (100 images / 709 targets) · NVIDIA RTX 5060 Ti · FP32 · 32 × 32 grid · batch 64',
             fontsize=10.5, color=MUTED)
    fig.text(.035, .103,
             'CUDA-synchronized full generation, including image encoding and mask processing · 3 warm-ups · 3 timed repetitions per image',
             fontsize=10.5, color=MUTED)
    fig.text(.035, .053,
             'AR@300 change vs Dense: ViT-H −0.085 pp · TinySAM −0.465 pp · EdgeSAM −0.240 pp. Full policy comparisons: docs/results.md.',
             fontsize=10.5, color=MUTED)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    title = 'Dense SAM versus FSD-SAM end-to-end inference time'
    description = 'Paper Table 7; data: site/paper-everything.json; COCO100 development subset.'
    fig.savefig(OUTPUT / 'fsd-time-comparison.png', dpi=150, metadata={'Title': title, 'Description': description})
    fig.savefig(OUTPUT / 'fsd-time-comparison.svg', metadata={'Title': title, 'Description': description, 'Date': None})
    plt.close(fig)
    print('Rendered ViT-H, TinySAM and EdgeSAM Dense / FSD-SAM timing')


if __name__ == '__main__':
    plot_timing()
