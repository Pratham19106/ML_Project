"""Baseline ResNet-18 Architecture for GTSRB (Master v1)

Implements:
1. ResNet-18 backbone with ImageNet pretrained weights (or scratch toggle).
2. Adapted 43-class classification head with Kaiming normal initialization.
3. Bag-of-Tricks zero-gamma initialization on residual blocks.
4. Parameter separation implementing the No-Bias-Decay rule.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models import ResNet18_Weights


class BaselineResNet18(nn.Module):
    """Standard black-box ResNet-18 baseline for traffic sign recognition."""

    def __init__(self, num_classes: int = 43, pretrained: bool = True, zero_init_residual: bool = True):
        super().__init__()
        weights = ResNet18_Weights.DEFAULT if pretrained else None
        self.backbone = models.resnet18(weights=weights)

        # Replace classification head
        in_features = self.backbone.fc.in_features  # 512
        self.backbone.fc = nn.Linear(in_features, num_classes)

        # Initialize the new FC layer
        nn.init.kaiming_normal_(self.backbone.fc.weight, mode="fan_out", nonlinearity="relu")
        nn.init.constant_(self.backbone.fc.bias, 0.0)

        # Zero-gamma initialization for the last BN in each residual block (Bag of Tricks)
        if zero_init_residual:
            for m in self.backbone.modules():
                if isinstance(m, models.resnet.BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass returning logits for the 43 sign classes."""
        return self.backbone(x)

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extracts 512-dim pooled feature embeddings prior to classification."""
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)

        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)
        x = self.backbone.layer3(x)
        x = self.backbone.layer4(x)

        x = self.backbone.avgpool(x)
        return torch.flatten(x, 1)


def get_parameter_groups(model: nn.Module, weight_decay: float = 1e-4) -> list[dict[str, Any]]:
    """Separates parameters to apply weight decay ONLY to conv/linear weights,
    omitting biases and 1D Batch Normalization parameters (No-Bias-Decay rule).
    """
    decay_params: list[torch.nn.Parameter] = []
    no_decay_params: list[torch.nn.Parameter] = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        # Biases and 1D normalization weights/biases receive 0 weight decay
        if param.ndim <= 1 or name.endswith(".bias") or "bn" in name:
            no_decay_params.append(param)
        else:
            decay_params.append(param)

    return [
        {"params": decay_params, "weight_decay": weight_decay},
        {"params": no_decay_params, "weight_decay": 0.0},
    ]

