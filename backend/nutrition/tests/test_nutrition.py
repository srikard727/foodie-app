"""Non-training smoke tests. Run: python -m unittest discover -s nutrition/tests"""
import csv
import tempfile
import unittest
from pathlib import Path

import torch
from PIL import Image

from nutrition.data import NutritionDataset
from nutrition.model import CalorieCNN, EfficientNetCalorie
from nutrition.prepare import split_for


class NutritionTests(unittest.TestCase):
    def test_split_reproducible(self):
        self.assertEqual(split_for("dish_0000000001", 42), split_for("dish_0000000001", 42))

    def test_custom_model_output(self):
        model = CalorieCNN().eval()
        with torch.no_grad():
            out = model(torch.randn(2, 3, 224, 224))
        self.assertEqual(tuple(out.shape), (2,))
        self.assertTrue(bool((out >= 0).all()))

    def test_dataset(self):
        with tempfile.TemporaryDirectory() as temp:
            base = Path(temp)
            (base / "images").mkdir()
            Image.new("RGB", (64, 64), "red").save(base / "images" / "dish_1.png")
            with (base / "manifest.csv").open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=["dish_id", "image", "calories", "split"])
                writer.writeheader()
                writer.writerow({"dish_id": "dish_1", "image": "images/dish_1.png", "calories": "321", "split": "train"})
            dataset = NutritionDataset(base / "manifest.csv", "train", 64)
            tensor, label = dataset[0]
            self.assertEqual(tuple(tensor.shape), (3, 64, 64))
            self.assertEqual(float(label), 321.0)


if __name__ == "__main__":
    unittest.main()
