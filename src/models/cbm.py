"""
Concept Bottleneck Model (CBM) Architecture for GTSRB (Master v1)

Implements:
1. ResNet-18 feature extractor backbone (ImageNet pretrained weights, zero-gamma residual BN).
2. Multi-head concept predictors corresponding to the 4-axis Vienna/StVO ontology:
   - Shape: 6 classes (circle, triangle_up, triangle_down, octagon, diamond, rectangle)
   - Colour Profile: 5 classes (white_red, blue_white, yellow_white, white_black, red_white)
   - Border Type: 4 classes (red_border, white_border, black_border, none)
   - Symbol Category: 7 classes (speed_numeral, warning_hazard, mandatory_direction,
                                prohibition_restriction, priority_geometry,
                                deregulation_cancel, none)
   Total bottleneck dimension = 6 + 5 + 4 + 7 = 22 dimensions.
3. Auxiliary Symbol / Visual Head:
   - Maps 512-dim visual features to 43-way symbol identity (mitigating the 25.58%
     vocabulary degeneracy ceiling discovered during Oracle CBM analysis).
4. Concept-to-Class Head:
   - Maps concept probabilities (and optional visual residual) to 43 traffic sign classes.
5. Concept intervention support:
   - Supports test-time concept intervention: evaluate g(c*) with intervened concept values.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torchvision.models import ResNet18_Weights

from .concept_encoder import (
    BORDER_OFFSET,
    BOTTLENECK_DIM,
    COLOUR_OFFSET,
    N_BORDER,
    N_COLOUR,
    N_SHAPE,
    N_SYMCAT,
    SHAPE_OFFSET,
    SYMCAT_OFFSET,
    encode_all_classes,
)


class ConceptBottleneckModel(nn.Module):
    """
    Concept Bottleneck Model for Traffic Sign Recognition.

    Parameters
    ----------
    num_classes : int
        Number of final sign classes (43 for GTSRB).
    pretrained : bool
        Whether to load ImageNet pretrained weights for the ResNet-18 backbone.
    zero_init_residual : bool
        Whether to apply zero-gamma initialization to the final BN in each residual block.
    bottleneck_mode : str
        "residual" (default): y = class_head(c_hat) + aux_head(features).
                              Solves within-concept collisions while preserving concept supervision.
        "pure":               y = class_head(c_hat). Strict CBM using strictly the 22-dim concept layer.
        "concat":             y = class_head(torch.cat([c_hat, features])).
    """

    def __init__(
        self,
        num_classes: int = 43,
        pretrained: bool = True,
        zero_init_residual: bool = True,
        bottleneck_mode: str = "residual",
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.bottleneck_mode = bottleneck_mode
        self.bottleneck_dim = BOTTLENECK_DIM  # 22

        # 1. Feature Extractor Backbone (ResNet-18)
        weights = ResNet18_Weights.DEFAULT if pretrained else None
        base_resnet = models.resnet18(weights=weights)

        self.conv1 = base_resnet.conv1
        self.bn1 = base_resnet.bn1
        self.relu = base_resnet.relu
        self.maxpool = base_resnet.maxpool
        self.layer1 = base_resnet.layer1
        self.layer2 = base_resnet.layer2
        self.layer3 = base_resnet.layer3
        self.layer4 = base_resnet.layer4
        self.avgpool = base_resnet.avgpool

        # Zero-gamma initialization for the last BN in each residual block
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, models.resnet.BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0.0)

        in_features = 512

        # 2. Concept Prediction Heads
        self.head_shape = nn.Linear(in_features, N_SHAPE)
        self.head_colour = nn.Linear(in_features, N_COLOUR)
        self.head_border = nn.Linear(in_features, N_BORDER)
        self.head_symcat = nn.Linear(in_features, N_SYMCAT)

        # 3. Auxiliary Visual / Symbol Head (512 -> 43)
        self.aux_head = nn.Linear(in_features, num_classes)

        # 4. Concept-to-Class Head
        if bottleneck_mode == "concat":
            self.class_head = nn.Linear(BOTTLENECK_DIM + in_features, num_classes)
        else:
            self.class_head = nn.Linear(BOTTLENECK_DIM, num_classes)

        # Initialize linear layers with Kaiming Normal
        for head in [
            self.head_shape,
            self.head_colour,
            self.head_border,
            self.head_symcat,
            self.aux_head,
            self.class_head,
        ]:
            nn.init.kaiming_normal_(head.weight, mode="fan_out", nonlinearity="relu")
            nn.init.constant_(head.bias, 0.0)

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extracts 512-dim pooled feature embeddings from the ResNet-18 backbone."""
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        return torch.flatten(x, 1)

    def predict_concepts_from_features(
        self, features: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Computes concept logits and the concatenated probability bottleneck vector.

        Returns
        -------
        logits_shape  : [B, 6]
        logits_colour : [B, 5]
        logits_border : [B, 4]
        logits_symcat : [B, 7]
        c_hat         : [B, 22] concatenated softmax probability vector
        """
        logits_shape = self.head_shape(features)
        logits_colour = self.head_colour(features)
        logits_border = self.head_border(features)
        logits_symcat = self.head_symcat(features)

        p_shape = F.softmax(logits_shape, dim=-1)
        p_colour = F.softmax(logits_colour, dim=-1)
        p_border = F.softmax(logits_border, dim=-1)
        p_symcat = F.softmax(logits_symcat, dim=-1)

        c_hat = torch.cat([p_shape, p_colour, p_border, p_symcat], dim=-1)
        return logits_shape, logits_colour, logits_border, logits_symcat, c_hat

    def forward(
        self,
        x: torch.Tensor,
        c_oracle: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass of the Concept Bottleneck Model.

        Parameters
        ----------
        x : torch.Tensor
            Input images [B, 3, 64, 64].
        c_oracle : Optional[torch.Tensor]
            Optional ground-truth / intervened concept vectors [B, 22].
            If provided, bypasses predicted concepts and tests concept intervention.

        Returns
        -------
        dict containing:
            "logits":        Final class logits [B, 43]
            "c_hat":         Predicted concept probabilities [B, 22]
            "logits_shape":  Shape logits [B, 6]
            "logits_colour": Colour logits [B, 5]
            "logits_border": Border logits [B, 4]
            "logits_symcat": Symbol Category logits [B, 7]
            "logits_aux":    Auxiliary symbol/visual logits [B, 43]
            "features":      512-dim visual features [B, 512]
        """
        features = self.extract_features(x)
        logits_shape, logits_colour, logits_border, logits_symcat, c_hat = (
            self.predict_concepts_from_features(features)
        )

        # Concept vector to use for class prediction (supports oracle intervention)
        c_effective = c_oracle if c_oracle is not None else c_hat

        # Auxiliary visual logits
        logits_aux = self.aux_head(features)

        # Class prediction based on bottleneck mode
        if self.bottleneck_mode == "pure":
            logits_class = self.class_head(c_effective)
        elif self.bottleneck_mode == "concat":
            logits_class = self.class_head(torch.cat([c_effective, features], dim=-1))
        else:  # "residual"
            logits_class = self.class_head(c_effective) + logits_aux

        return {
            "logits": logits_class,
            "c_hat": c_hat,
            "logits_shape": logits_shape,
            "logits_colour": logits_colour,
            "logits_border": logits_border,
            "logits_symcat": logits_symcat,
            "logits_aux": logits_aux,
            "features": features,
        }

    def intervene(
        self,
        x: torch.Tensor,
        intervened_concepts: torch.Tensor,
    ) -> torch.Tensor:
        """
        Applies concept intervention: replaces the predicted concept vector with
        user-specified ground-truth or counterfactual concept vectors.
        """
        output = self.forward(x, c_oracle=intervened_concepts)
        return output["logits"]


class CBMLoss(nn.Module):
    """
    Multi-Task Loss for Concept Bottleneck Model.

    Loss = L_task(y, y_hat) + lambda_c * sum(L_concept_k) + lambda_aux * L_aux

    Parameters
    ----------
    lambda_c : float
        Weight for the concept supervision losses (default: 1.0).
    lambda_aux : float
        Weight for the auxiliary visual/symbol classification loss (default: 0.5).
    """

    def __init__(self, lambda_c: float = 1.0, lambda_aux: float = 0.5) -> None:
        super().__init__()
        self.lambda_c = lambda_c
        self.lambda_aux = lambda_aux
        self.ce = nn.CrossEntropyLoss()

    def forward(
        self,
        outputs: Dict[str, torch.Tensor],
        target_class: torch.Tensor,
        target_shape: torch.Tensor,
        target_colour: torch.Tensor,
        target_border: torch.Tensor,
        target_symcat: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        # Task loss
        loss_task = self.ce(outputs["logits"], target_class)

        # Concept losses
        loss_shape = self.ce(outputs["logits_shape"], target_shape)
        loss_colour = self.ce(outputs["logits_colour"], target_colour)
        loss_border = self.ce(outputs["logits_border"], target_border)
        loss_symcat = self.ce(outputs["logits_symcat"], target_symcat)
        loss_concepts = loss_shape + loss_colour + loss_border + loss_symcat

        # Auxiliary symbol loss (using target_class since symbol_content uniquely maps to class_id)
        loss_aux = self.ce(outputs["logits_aux"], target_class)

        # Total combined objective
        total_loss = loss_task + self.lambda_c * loss_concepts + self.lambda_aux * loss_aux

        metrics = {
            "loss_task": float(loss_task.item()),
            "loss_concepts": float(loss_concepts.item()),
            "loss_shape": float(loss_shape.item()),
            "loss_colour": float(loss_colour.item()),
            "loss_border": float(loss_border.item()),
            "loss_symcat": float(loss_symcat.item()),
            "loss_aux": float(loss_aux.item()),
            "total_loss": float(total_loss.item()),
        }
        return total_loss, metrics


def get_concept_lookup_tensors(
    device: torch.device,
    csv_path: Optional[str] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Creates pre-computed GPU/CPU lookup tensors for mapping class_id (0..42)
    directly to ground-truth concept indices without CPU dictionary lookups.

    Returns
    -------
    shape_lut  : LongTensor of shape (43,)
    colour_lut : LongTensor of shape (43,)
    border_lut : LongTensor of shape (43,)
    symcat_lut : LongTensor of shape (43,)
    """
    enc = encode_all_classes(csv_path)
    shape_arr = np.array([enc[i]["shape_idx"] for i in range(43)], dtype=np.int64)
    colour_arr = np.array([enc[i]["colour_idx"] for i in range(43)], dtype=np.int64)
    border_arr = np.array([enc[i]["border_idx"] for i in range(43)], dtype=np.int64)
    symcat_arr = np.array([enc[i]["symcat_idx"] for i in range(43)], dtype=np.int64)

    shape_lut = torch.tensor(shape_arr, dtype=torch.long, device=device)
    colour_lut = torch.tensor(colour_arr, dtype=torch.long, device=device)
    border_lut = torch.tensor(border_arr, dtype=torch.long, device=device)
    symcat_lut = torch.tensor(symcat_arr, dtype=torch.long, device=device)

    return shape_lut, colour_lut, border_lut, symcat_lut
