"""RGB-only Nutrition5k dataset from a prebuilt manifest."""
import csv
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms as T

MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)


def transform_for(split: str, size: int):
    if split == "train":
        return T.Compose([
            T.Resize((size + 32, size + 32)),
            T.RandomResizedCrop(size, scale=(0.8, 1.0), ratio=(0.9, 1.1)),
            T.RandomHorizontalFlip(),
            T.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
            T.ToTensor(),
            T.Normalize(MEAN, STD),
        ])
    return T.Compose([
        T.Resize((size, size)),
        T.ToTensor(),
        T.Normalize(MEAN, STD),
    ])


class NutritionDataset(Dataset):
    def __init__(self, manifest: str | Path, split: str, image_size: int = 224):
        self.manifest = Path(manifest).resolve()
        self.root = self.manifest.parent
        with self.manifest.open(newline="", encoding="utf-8") as handle:
            self.rows = [r for r in csv.DictReader(handle) if r["split"] == split]
        if not self.rows:
            raise ValueError(f"No rows for split {split!r} in {manifest}")
        self.transform = transform_for(split, image_size)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index: int):
        row = self.rows[index]
        image_path = self.root / row["image"]
        with Image.open(image_path) as image:
            tensor = self.transform(image.convert("RGB"))
        return tensor, torch.tensor(float(row["calories"]), dtype=torch.float32)
