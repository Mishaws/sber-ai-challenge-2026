"""Generate class distribution chart for Task 1 (Noisy Weather Outside) predictions."""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


# Official competition class mapping (0=Rain, 1=Fog, 2=Snow)
LABEL_MAP = {0: "Rain", 1: "Fog", 2: "Snow"}


def plot_weather_distribution(csv_path: Path, out_path: Path) -> None:
    df = pd.read_csv(csv_path)
    df["weather_class"] = df["label"].map(LABEL_MAP)

    plt.rcParams.update({
        "text.color": "#F8FAFC",
        "axes.labelcolor": "#E2E8F0",
        "axes.edgecolor": "#475569",
        "xtick.color": "#E2E8F0",
        "ytick.color": "#E2E8F0",
        "figure.facecolor": "#0F172A",
        "axes.facecolor": "#0F172A",
    })

    fig, ax = plt.subplots(figsize=(8, 4.8))
    order = ["Rain", "Fog", "Snow"]
    palette = ["#38BDF8", "#818CF8", "#34D399"]
    sns.countplot(data=df, x="weather_class", hue="weather_class", order=order, palette=palette, legend=False, ax=ax)

    ax.set_title("Test Prediction Distribution — Noisy Weather Outside (n=1,431)", fontsize=13, fontweight="bold", pad=14)
    ax.set_xlabel("Weather Condition", fontsize=11)
    ax.set_ylabel("Image Count", fontsize=11)
    ax.set_ylim(0, max(df["weather_class"].value_counts()) * 1.15)

    for patch in ax.patches:
        height = int(patch.get_height())
        ax.annotate(
            f"{height}",
            (patch.get_x() + patch.get_width() / 2.0, height),
            ha="center",
            va="bottom",
            fontsize=11,
            fontweight="bold",
            color="#F8FAFC",
            xytext=(0, 5),
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
    parser.add_argument("--csv", type=Path, default=Path("../submissions/noisy_weather_submission.csv"))
    parser.add_argument("--out", type=Path, default=Path("../docs/images/weather_distribution.png"))
    args = parser.parse_args()
    plot_weather_distribution(args.csv, args.out)
