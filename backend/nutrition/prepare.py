"""Download a bounded Nutrition5k RGB subset and create dish-level splits.

Uses the community-hosted overhead RGB mirror (not the official train/test splits).
The original Nutrition5k official dataset is the authoritative source.
No model training is performed.
"""
import argparse
import csv
import hashlib
import math
from pathlib import Path

from huggingface_hub import hf_hub_download

DATASET = "ai5labsOfficial/nutrition5k"


def split_for(dish_id: str, seed: int) -> str:
    value = int(hashlib.sha256(f"{seed}:{dish_id}".encode()).hexdigest()[:12], 16)
    bucket = value % 10
    return "train" if bucket < 8 else "val" if bucket == 8 else "test"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("data/nutrition5k"))
    parser.add_argument("--max-images", type=int, default=500, help="Bounded download; 0 downloads all mirrored overhead RGB")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.max_images < 0:
        parser.error("--max-images must be >= 0")
    args.output.mkdir(parents=True, exist_ok=True)
    metadata = hf_hub_download(repo_id=DATASET, repo_type="dataset", filename="metadata.csv")
    with open(metadata, newline="", encoding="utf-8") as handle:
        source_rows = list(csv.DictReader(handle))
    samples = []
    seen = set()
    for row in source_rows:
        dish_id = row.get("dish_id", "").strip()
        try:
            calories = float(row.get("total_calories", ""))
        except (ValueError, TypeError):
            continue
        if not dish_id or dish_id in seen or not math.isfinite(calories) or calories < 0:
            continue
        seen.add(dish_id)
        samples.append((dish_id, calories))
    # Reproducible subset, chosen independently of order of metadata file.
    samples.sort(key=lambda r: hashlib.sha256(f"{args.seed}:{r[0]}".encode()).hexdigest())
    if args.max_images:
        samples = samples[:args.max_images]
    if not samples:
        raise RuntimeError("No valid dish records found")
    rows = []
    for idx, (dish_id, calories) in enumerate(samples, 1):
        filename = f"images/{dish_id}.png"
        try:
            downloaded = hf_hub_download(repo_id=DATASET, repo_type="dataset", filename=filename, local_dir=args.output)
        except Exception as exc:
            print(f"Skipping {dish_id}: {exc}")
            continue
        if not Path(downloaded).is_file():
            continue
        rows.append({"dish_id": dish_id, "image": filename, "calories": calories, "split": split_for(dish_id, args.seed)})
        if idx % 50 == 0:
            print(f"Processed {idx}/{len(samples)} images")
    manifest = args.output / "manifest.csv"
    with manifest.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["dish_id", "image", "calories", "split"])
        writer.writeheader()
        writer.writerows(rows)
    counts = {split: sum(r["split"] == split for r in rows) for split in ("train", "val", "test")}
    print(f"Manifest: {manifest}; downloaded {len(rows)} dishes; splits: {counts}")
    if min(counts.values()) == 0:
        print("WARNING: a split is empty. Increase --max-images before running training.")


if __name__ == "__main__":
    main()
