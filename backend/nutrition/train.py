"""Train/evaluate an RGB calorie estimator. NEVER runs on import.

Example: python -m nutrition.train --model custom --epochs 50
"""
import argparse
import csv
import json
import math
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from nutrition.data import NutritionDataset
from nutrition.model import make_model

TARGET_SCALE = 500.0  # Makes regression gradients well-conditioned (~1 at init)


def seed_all(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def compute_metrics(pred: torch.Tensor, target: torch.Tensor) -> dict:
    error = pred.float() - target.float()
    return {"mae": float(error.abs().mean()), "rmse": float(error.square().mean().sqrt())}


@torch.no_grad()
def evaluate(model, loader, device, use_amp, amp_dtype):
    model.eval()
    predictions, targets = [], []
    for images, calories in loader:
        images = images.to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
            output = model(images).float() * TARGET_SCALE
        predictions.append(output.cpu())
        targets.append(calories)
    return compute_metrics(torch.cat(predictions), torch.cat(targets))


def train_epoch(model, loader, optimizer, device, scaler, use_amp, amp_dtype):
    model.train()
    total = 0.0
    samples = 0
    for images, calories in loader:
        images = images.to(device, non_blocking=True)
        calories = calories.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
            prediction = model(images)
            loss = nn.functional.smooth_l1_loss(prediction.float(), calories / TARGET_SCALE, beta=0.2)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()
        batch_size = images.shape[0]
        total += loss.item() * batch_size
        samples += batch_size
    return total / samples


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", type=Path, default=Path("data/nutrition5k/manifest.csv"))
    p.add_argument("--output", type=Path, default=Path("artifacts/nutrition_training"))
    p.add_argument("--model", choices=("custom", "efficientnet_v2_s"), default="custom")
    p.add_argument("--pretrained", action="store_true", help="Download ImageNet pretrained weights (EfficientNet only)")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--image-size", type=int, default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--patience", type=int, default=10)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    if args.epochs < 1 or args.batch_size < 1:
        p.error("--epochs and --batch-size must be positive")
    seed_all(args.seed)
    args.output.mkdir(parents=True, exist_ok=True)
    size = args.image_size or (384 if args.model == "efficientnet_v2_s" else 224)
    lr = args.lr or (1e-4 if args.pretrained else 3e-4)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = device.type == "cuda"
    train_set = NutritionDataset(args.manifest, "train", size)
    val_set = NutritionDataset(args.manifest, "val", size)
    test_set = NutritionDataset(args.manifest, "test", size)
    loader_kwargs = {"batch_size": args.batch_size, "num_workers": args.workers,
                     "pin_memory": device.type == "cuda", "persistent_workers": args.workers > 0}
    train_loader = DataLoader(train_set, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_set, shuffle=False, **loader_kwargs)
    test_loader = DataLoader(test_set, shuffle=False, **loader_kwargs)
    mean_calories = float(np.mean([float(r["calories"]) for r in train_set.rows]))
    baseline_val = compute_metrics(
        torch.full((len(val_set),), mean_calories),
        torch.tensor([float(r["calories"]) for r in val_set.rows]),
    )
    baseline_test = compute_metrics(
        torch.full((len(test_set),), mean_calories),
        torch.tensor([float(r["calories"]) for r in test_set.rows]),
    )
    print(f"Device={device}, train/val/test={len(train_set)}/{len(val_set)}/{len(test_set)}")
    print(f"Mean predictor validation={baseline_val}, test={baseline_test}")
    model = make_model(args.model, pretrained=args.pretrained).to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=args.weight_decay)
    warmup = min(3, max(0, args.epochs // 10))
    def lr_factor(epoch):
        if epoch < warmup:
            return 0.1 + 0.9 * (epoch + 1) / warmup
        progress = (epoch - warmup) / max(1, args.epochs - warmup - 1)
        return 0.05 + 0.95 * (1 + math.cos(math.pi * progress)) / 2
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_factor)
    amp_dtype = torch.bfloat16 if device.type == "cuda" and torch.cuda.is_bf16_supported() else torch.float16
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp and amp_dtype == torch.float16)
    best, stale = float("inf"), 0
    history = []
    checkpoint = args.output / "best.pt"
    for epoch in range(args.epochs):
        loss = train_epoch(model, train_loader, optimizer, device, scaler, use_amp, amp_dtype)
        val = evaluate(model, val_loader, device, use_amp, amp_dtype)
        scheduler.step()
        history.append({"epoch": epoch + 1, "train_smooth_l1": loss, **{f"val_{k}": v for k, v in val.items()},
                        "lr": optimizer.param_groups[0]["lr"]})
        print(f"Epoch {epoch+1:03d}: train loss={loss:.4f} val MAE={val['mae']:.1f} kcal RMSE={val['rmse']:.1f}")
        if val["mae"] < best:
            best, stale = val["mae"], 0
            torch.save({"model_state": model.state_dict(), "architecture": args.model,
                        "image_size": size, "target_scale": TARGET_SCALE,
                        "pretrained_at_training": args.pretrained, "epoch": epoch + 1,
                        "val_mae": best}, checkpoint)
        else:
            stale += 1
        if stale >= args.patience:
            print("Early stopping on validation MAE")
            break
    (args.output / "history.json").write_text(json.dumps(history, indent=2))
    state = torch.load(checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(state["model_state"])
    test = evaluate(model, test_loader, device, use_amp, amp_dtype)
    report = {"model": args.model, "best_validation_mae": best, "test": test,
              "mean_baseline_validation": baseline_val, "mean_baseline_test": baseline_test,
              "checkpoint": str(checkpoint), "sample_counts": {"train": len(train_set),
              "val": len(val_set), "test": len(test_set)}}
    (args.output / "results.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
