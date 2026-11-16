"""Generate reproducible EDA outputs from downloaded Nutrition5k manifest."""
import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=Path("data/nutrition5k/manifest.csv"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/nutrition_eda"))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    with args.manifest.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError("Manifest is empty. Run nutrition.prepare first.")
    ids = [r["dish_id"] for r in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate dish_id would leak across splits")
    calories = np.array([float(r["calories"]) for r in rows])
    splits = {s: [float(r["calories"]) for r in rows if r["split"] == s] for s in ("train", "val", "test")}
    report = {
        "count": len(rows),
        "split_counts": {k: len(v) for k, v in splits.items()},
        "calories": {k: float(v) for k, v in zip(
            ("min", "p25", "median", "mean", "p75", "p95", "max"),
            (calories.min(), *np.percentile(calories, [25, 50]), calories.mean(), *np.percentile(calories, [75, 95]), calories.max())
        )},
        "missing_images": [r["dish_id"] for r in rows if not (args.manifest.parent / r["image"]).is_file()],
    }
    (args.output / "summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(calories, bins=35)
    ax.set(xlabel="Calories (kcal)", ylabel="Dish count", title="Nutrition5k calorie distribution")
    fig.tight_layout()
    fig.savefig(args.output / "calorie_histogram.png", dpi=160)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.boxplot([v for v in splits.values() if v], tick_labels=[k for k, v in splits.items() if v])
    ax.set(ylabel="Calories (kcal)", title="Calorie labels by split")
    fig.tight_layout()
    fig.savefig(args.output / "split_boxplot.png", dpi=160)
    plt.close(fig)
    fig, axes = plt.subplots(3, 3, figsize=(9, 9))
    for ax, row in zip(axes.flat, rows[:9]):
        with Image.open(args.manifest.parent / row["image"]) as image:
            ax.imshow(image.convert("RGB"))
        ax.set_title(f'{row["calories"]} kcal', fontsize=9)
        ax.axis("off")
    for ax in list(axes.flat)[min(9, len(rows)):]:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(args.output / "sample_meals.png", dpi=140)
    plt.close(fig)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
