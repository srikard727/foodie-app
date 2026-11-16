# Foodie Nutrition5k: CNN calorie-regression research

**Status: code only; no data downloaded and no model trained for this PR.** This module does not change the existing FastAPI recommendations endpoint.

## Research basis and tradeoffs

- [Nutrition5k (Google Research)](https://github.com/google-research-datasets/Nutrition5k) contains 5,006 dishes with measured calories, macronutrients, ingredient masses and RGB-D data. The full release is ~181 GB. Original paper: [Thames et al., CVPR 2021](https://openaccess.thecvf.com/content/CVPR2021/html/Thames_Nutrition5k_Towards_Automatic_Nutritional_Understanding_of_Generic_Food_CVPR_2021_paper.html).
- [Image-only regression](https://openaccess.thecvf.com/content/CVPR2021/papers/Thames_Nutrition5k_Towards_Automatic_Nutritional_Understanding_of_Generic_Food_CVPR_2021_paper.pdf) is a useful baseline; depth and portion/volume measurements can reduce calorie error substantially. Exact portion mass cannot be inferred reliably from one photo.
- [EfficientNetV2-S (Torchvision)](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.efficientnet_v2_s.html) provides an ImageNet-pretrained CNN comparison, not a nutrition-pretrained model.
- [Overhead RGB community mirror](https://huggingface.co/datasets/ai5labsOfficial/nutrition5k) provides a smaller ~1.45 GB subset of overhead photos and calorie labels. It is **not** the complete official dataset. Validate its provenance before research publication.

## Training setup

From `backend/` on a GPU-capable computer (e.g. RTX 5080):

1. Create a virtual environment, then install the current PyTorch + torchvision CUDA build compatible with your GPU from [pytorch.org/get-started/locally](https://pytorch.org/get-started/locally/). Do **not** assume the default pip build supports your GPU.
2. Install the small research dependencies:

```bash
pip install -r nutrition/requirements.txt
```

3. Download a bounded sample (500 images by default; this is a convenience/dev subset, not a representative benchmark):

```bash
python -m nutrition.prepare --max-images 500
python -m nutrition.eda
```

4. Inspect `artifacts/nutrition_eda/summary.json`, `calorie_histogram.png`, `split_boxplot.png`, and `sample_meals.png`. The script checks for duplicate dish IDs, missing photos, and reports label ranges.

5. **When you are ready to train**, explicitly run:

```bash
python -m nutrition.prepare --max-images 0
python -m nutrition.train --model custom --epochs 50 --batch-size 32
python -m nutrition.train --model efficientnet_v2_s --pretrained --epochs 40 --batch-size 16 --output artifacts/nutrition_efficientnet
python -m nutrition.predict data/nutrition5k/images/REPLACE_WITH_DISH_ID.png --checkpoint artifacts/nutrition_training/best.pt
```

Note: `--max-images 0` attempts all images in the mirror and can take considerable bandwidth/storage. The dataset preparation script currently creates reproducible *hash-based dish-level 80/10/10 splits*, **not Google's official splits**. Images for the same dish never cross splits; however, Nutrition5k includes incremental related dishes and this mirror does not establish grouping across those. Consequently, these numbers are **development-only**, and must not be compared directly to the original paper's official benchmark. For publishable comparisons, source the official train/test ID lists and group related incremental plates.

## Baseline architecture and optimization

- `CalorieCNN`: custom train-from-scratch CNN, 4 convolutional stages, BatchNorm, SiLU, adaptive pooling, dropout, and a Softplus scalar output. Input size 224 × 224.
- `EfficientNetCalorie`: Torchvision EfficientNetV2-S with a new calorie regression head, optional ImageNet weights, 384 × 384 input. Included to measure the benefit of transfer learning.
- Regression in units of 500 kcal to improve numeric conditioning; Smooth L1 (Huber) loss, AdamW (default 3e-4 from scratch / 1e-4 pretrained), cosine LR decay with warmup, gradient clipping, CUDA AMP (BF16 if supported), early stopping by validation calorie MAE. These are **starting hyperparameters, not experimentally optimized values**.
- Best checkpoint selected by **validation MAE**; untouched test data used only after training; also reports RMSE and mean-calorie constant baseline. `history.json` and `results.json` preserve experiment outputs.
- Learned weights/biases **cannot be optimized before training**. Initialization is random for the custom model and pretrained ImageNet weights are optionally loaded for EfficientNet. Tune learning rates and hyperparameters on the validation set, not the test set.

## Limitations and next steps

- Single-view RGB images cannot reveal hidden butter/oil or exact food mass. Show results as estimates with appropriate uncertainty, never exact nutritional facts.
- Nutrition5k is from a limited cafeteria population and may not generalize to home cooking, other cuisines or typical cellphone angles.
- This milestone predicts **calories only**. Protein, fat, carbs, dish recognition, segmentation, and portion-size corrections are separate follow-ups.
- Do not expose an inference endpoint until a checkpoint has been trained and tested. The untrained network returns meaningless numbers.
- Avoid committing dataset images, downloaded model weights, or generated artifacts to Git.
