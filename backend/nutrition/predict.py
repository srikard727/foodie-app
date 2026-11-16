"""Predict calories from an image using a locally trained checkpoint."""
import argparse
from pathlib import Path

import torch
from PIL import Image

from nutrition.data import transform_for
from nutrition.model import make_model


def predict(image_path: str | Path, checkpoint_path: str | Path, device: str = "cpu") -> float:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model = make_model(checkpoint["architecture"], pretrained=False).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    transform = transform_for("test", checkpoint["image_size"])
    with Image.open(image_path) as image:
        tensor = transform(image.convert("RGB")).unsqueeze(0).to(device)
    with torch.inference_mode():
        estimate = model(tensor)[0].item() * checkpoint["target_scale"]
    return max(0.0, float(estimate))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("image", type=Path)
    parser.add_argument("--checkpoint", type=Path, default=Path("artifacts/nutrition_training/best.pt"))
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    print(f"Estimated calories: {predict(args.image, args.checkpoint, args.device):.1f} kcal")
