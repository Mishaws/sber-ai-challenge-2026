"""Exploratory Data Analysis for 3D Mesh Quality Control dataset."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image


DEFECTS = [
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


def inspect_sample_files(train_dir: Path) -> None:
    png_files = sorted(train_dir.glob("*.png"))
    if not png_files:
        print(f"No .png samples found in {train_dir}")
        return

    sample_png = png_files[0]
    sample_npz = sample_png.with_suffix(".npz")

    with Image.open(sample_png) as img:
        print(f"Sample render: {sample_png.name} | size={img.size} | mode={img.mode}")

    if sample_npz.exists():
        data = np.load(sample_npz)
        shapes = {k: (data[k].shape, str(data[k].dtype)) for k in data.files}
        print(f"Sample mesh NPZ: {sample_npz.name} | arrays={shapes}")


def analyze_labels(train_csv: Path) -> None:
    df = pd.read_csv(train_csv)
    print(f"Total training samples: {len(df)}")

    print("\nDefect frequencies:")
    freq = df[DEFECTS].sum().sort_values(ascending=False)
    for name, count in freq.items():
        print(f"  {name:<14s}: {int(count):>4d} ({count / len(df) * 100:5.2f}%)")

    print("\nQuality distribution:")
    print(df["quality"].value_counts().to_dict())

    has_defect = df[DEFECTS].sum(axis=1) > 0
    inconsistent_good = df[has_defect & (df["quality"] == 1)]
    inconsistent_bad = df[~has_defect & (df["quality"] == 0)]
    print(f"\nDeterministic quality rule mismatches: {len(inconsistent_good) + len(inconsistent_bad)}")


def main() -> None:
    parser = argparse.ArgumentParser(description="EDA for 3D Mesh Quality Control")
    parser.add_argument("--train-csv", type=Path, default=Path("data/train.csv"))
    parser.add_argument("--train-dir", type=Path, default=Path("data/train"))
    args = parser.parse_args()

    if args.train_dir.exists():
        inspect_sample_files(args.train_dir)
    if args.train_csv.exists():
        analyze_labels(args.train_csv)


if __name__ == "__main__":
    main()
