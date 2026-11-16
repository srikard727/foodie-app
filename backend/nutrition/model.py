"""CNN calorie regression models. All outputs are nonnegative kcal."""
import torch
from torch import nn
from torchvision.models import efficientnet_v2_s, EfficientNet_V2_S_Weights


class ConvBlock(nn.Sequential):
    def __init__(self, in_ch: int, out_ch: int, stride: int = 1):
        super().__init__(
            nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.SiLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.SiLU(inplace=True),
        )


class CalorieCNN(nn.Module):
    """Train-from-scratch baseline with residual stages and adaptive pooling."""

    def __init__(self, dropout: float = 0.25):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.SiLU(inplace=True),
        )
        self.stages = nn.Sequential(
            ConvBlock(32, 64, stride=2),
            ConvBlock(64, 96, stride=2),
            ConvBlock(96, 160, stride=2),
            ConvBlock(160, 256, stride=2),
        )
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(256, 128),
            nn.SiLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, 1),
            nn.Softplus(),  # Nonnegative prediction; stable gradient at initialization
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.stages(self.stem(x))).squeeze(1)


class EfficientNetCalorie(nn.Module):
    """Transfer-learning comparison; ImageNet weights optionally downloaded."""

    def __init__(self, pretrained: bool = True, dropout: float = 0.25):
        super().__init__()
        weights = EfficientNet_V2_S_Weights.DEFAULT if pretrained else None
        self.net = efficientnet_v2_s(weights=weights)
        in_features = self.net.classifier[1].in_features
        self.net.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_features, 1),
            nn.Softplus(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(1)


def make_model(name: str, pretrained: bool = False) -> nn.Module:
    if name == "custom":
        return CalorieCNN()
    if name == "efficientnet_v2_s":
        return EfficientNetCalorie(pretrained=pretrained)
    raise ValueError(f"Unknown model {name!r}")
