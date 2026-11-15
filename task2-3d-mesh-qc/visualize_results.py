"""Generate defect detection frequency chart for Task 2 (3D Mesh Quality Control)."""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


DEFECT_COLS = [
    "abstract",
    "artifacts",
    "intersection",
    "lowpoly",
    "noisy",
    "open",
    "partial",
    "scale",
    "set",
    "simple",
]

PALETTE_CYBER = [
    "#FF1E6C",
    "#FA26A0",
    "#D946EF",
    "#A855F7",
    "#818CF8",
    "#6366F1",
    "#3B82F6",
    "#00B4D8",
    "#00F0FF",
    "#10B981",
]


def plot_defect_distribution(csv_path: Path, out_path: Path) -> None:
    df = pd.read_csv(csv_path)
    defect_counts = df[DEFECT_COLS].sum().sort_values(ascending=False)

    plt.rcParams.update({
        "text.color": "#F8FAFC",
        "axes.labelcolor": "#E2E8F0",
        "axes.edgecolor": "#475569",
        "xtick.color": "#E2E8F0",
        "ytick.color": "#E2E8F0",
        "figure.facecolor": "#0F172A",
        "axes.facecolor": "#0F172A",
    })

    fig, ax = plt.subplots(figsize=(10, 5.6))
    sns.barplot(
        x=defect_counts.values,
        y=defect_counts.index,
        hue=defect_counts.index,
        palette=PALETTE_CYBER,
        legend=False,
        ax=ax,
    )

    ax.set_title("Test Defect Detection Frequency — 3D Mesh Quality Control (n=669)", fontsize=13, fontweight="bold", pad=14)
    ax.set_xlabel("Detection Count", fontsize=11)
    ax.set_ylabel("Defect Class", fontsize=11)
    ax.set_xlim(0, max(defect_counts.values) * 1.12)

    for patch in ax.patches:
        width = int(patch.get_width())
        ax.annotate(
            f"{width}",
            (width, patch.get_y() + patch.get_height() / 2.0),
            ha="left",
            va="center",
            fontsize=10.5,
            fontweight="bold",
            color="#F8FAFC",
            xytext=(6, 0),
            textcoords="offset points",
        )

    sns.despine(ax=ax)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=220, facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"Saved chart to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=Path, default=Path("../submissions/3d_mesh_qc_submission.csv"))
    parser.add_argument("--out", type=Path, default=Path("../docs/images/mesh_defect_distribution.png"))
    args = parser.parse_args()
    plot_defect_distribution(args.csv, args.out)
