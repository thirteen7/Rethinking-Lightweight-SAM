"""Render README result figures from the paper values in site/benchmarks.json.

Run from any directory with Python and matplotlib installed:
    python docs/figures/plot_prompt_results.py
"""

from pathlib import Path
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.patches import Patch, Rectangle


ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "site" / "assets" / "results"
MODELS = (("tinysam", "TinySAM"), ("mobilesam", "MobileSAM"), ("vith", "ViT-H"))
DATASETS = ("coco", "lvis", "sa1b")
INK = "#202333"
MUTED = "#737887"
VIOLET = "#7860de"
BASELINE = "#c5cad6"


def plot_comparison(datasets, prompt):
    """Show initial absolute IoU and every stage's reported gain together."""
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 12,
        "text.color": INK,
        "axes.labelcolor": MUTED,
        "xtick.color": MUTED,
        "svg.fonttype": "none",
        "svg.hashsalt": "rethinking-lightweight-sam-prompt-results",
    })
    fig = plt.figure(figsize=(14, 8.7), facecolor="white")
    bars = fig.add_axes([0.195, 0.19, 0.445, 0.59])
    gains = fig.add_axes([0.705, 0.19, 0.26, 0.59], sharey=bars)
    title = "Point prompts: the first click" if prompt == "point" else "Box prompts: the initial box"
    subtitle = "One foreground click, before any correction" if prompt == "point" else "One annotation box, before any correction"
    fig.text(0.035, 0.94, title, fontsize=24, weight="bold")
    fig.text(0.035, 0.895, subtitle, fontsize=13, color=MUTED)
    fig.text(0.195, 0.835, "Initial IoU (%)", fontsize=14, weight="bold")
    fig.text(0.705, 0.835, "Gain at each stage (pp)", fontsize=14, weight="bold")
    fig.legend(
        [Patch(facecolor=BASELINE), Patch(facecolor=VIOLET)],
        ["Original", "Refined"],
        loc="upper left", bbox_to_anchor=(0.186, 0.819),
        frameon=False, ncol=2, fontsize=11, handlelength=1.15,
        borderaxespad=0, columnspacing=1.8,
    )
    bars.set_xlim(0, 100)
    bars.set_ylim(11.9, -0.7)
    bars.set_xticks([0, 25, 50, 75, 100])
    bars.set_xlabel("IoU (%) · 0–100 scale", fontsize=11, labelpad=10)
    bars.tick_params(axis="x", length=0, pad=8, labelsize=11)
    bars.set_yticks([])
    bars.set_axisbelow(True)
    bars.grid(axis="x", color="#e9ebf2", linewidth=0.8)
    for spine in bars.spines.values():
        spine.set_visible(False)
    gains.set_xlim(-0.58, 2.58)
    gains.set_xticks([])
    gains.set_yticks([])
    for spine in gains.spines.values():
        spine.set_visible(False)

    # The same 0–12 pp color scale is used for both Point and Box figures.
    colors = LinearSegmentedColormap.from_list("gain", ["#f7f4ff", "#d2c6f4", "#7860de"])
    norm = Normalize(vmin=0, vmax=12)
    for stage, label in enumerate(["Initial", "+1 click", "+2 clicks"]):
        gains.text(stage, -0.47, label, ha="center", va="center", fontsize=11,
                   color=VIOLET if stage == 0 else MUTED,
                   weight="bold" if stage == 0 else "normal")
    gains.add_patch(Rectangle(
        (-0.49, 0.51), 0.98, 11.03, facecolor="#fbf9ff",
        edgecolor=VIOLET, linewidth=1.6, zorder=0,
    ))
    for group, dataset_key in enumerate(DATASETS):
        dataset = datasets[dataset_key]
        header_y = group * 4 + 0.25
        bars.text(-34, header_y, dataset["label"], fontsize=12, weight="bold", color=INK,
                  va="center", clip_on=False)
        for index, (model_key, model_label) in enumerate(MODELS):
            y = group * 4 + index + 1
            historical = dataset["baseline_kind"][model_key] == "historical_reference"
            label = model_label + (" †" if historical else "")
            original = dataset["original"][model_key][prompt][0]
            refined = dataset[model_key][prompt][0]
            bars.text(-3, y, label, fontsize=12, ha="right", va="center", clip_on=False)
            for value, offset, color, weight in [
                (original, -0.22, BASELINE, "normal"),
                (refined, 0.22, VIOLET, "bold"),
            ]:
                bars.barh(y + offset, value, height=0.32, color=color, zorder=2)
                bars.text(value + 1.2, y + offset, f"{value:.2f}",
                          color=MUTED if offset < 0 else VIOLET,
                          va="center", fontsize=10.5, weight=weight)
            for stage, delta in enumerate(dataset["gains"][model_key][prompt]):
                gains.add_patch(Rectangle(
                    (stage - 0.44, y - 0.39), 0.88, 0.78,
                    facecolor=colors(norm(delta)), edgecolor="white", linewidth=1.1,
                ))
                gains.text(stage, y, f"+{delta:.2f}", ha="center", va="center",
                           color="white" if delta >= 8 else INK,
                           fontsize=13, weight="bold" if stage == 0 else "normal")
    fig.text(0.035, 0.084,
             "Paper Tables 1, 3, 5, 6 · Legacy IoU (%) · † Historical original-model references; see Appendix E.",
             fontsize=10, color=MUTED)
    fig.text(0.035, 0.046,
             "Gains use the paper’s pre-rounding differences. +1 / +2 are cumulative corrective clicks; each method follows its own trajectory.",
             fontsize=10, color=MUTED)
    metadata = {"Title": title, "Description": subtitle + "; data: site/benchmarks.json; manuscript Tables 1, 3, 5, 6."}
    fig.savefig(OUTPUT / f"{prompt}-comparison.png", dpi=150, metadata={"Title": title, "Description": metadata["Description"]})
    fig.savefig(OUTPUT / f"{prompt}-comparison.svg", metadata={**metadata, "Date": None})
    plt.close(fig)


if __name__ == "__main__":
    datasets = json.loads((ROOT / "site" / "benchmarks.json").read_text(encoding="utf-8"))["datasets"]
    OUTPUT.mkdir(parents=True, exist_ok=True)
    for prompt in ("point", "box"):
        plot_comparison(datasets, prompt)
        print(f"Rendered {prompt}: 9 initial comparisons, 27 stage gains")
