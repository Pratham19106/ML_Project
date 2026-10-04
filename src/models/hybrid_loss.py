"""
Hybrid CBM Loss Module (Experiment 5: AugMix + Concept Bottleneck Model)

Methodological Grounding:
1. Koh et al. (ICML 2020) — Concept Bottleneck Models (CBM).
2. Hendrycks et al. (ICLR 2020) — AugMix: Consistency Regularization via JSD.

Formulation:
Given an input triplet (x_clean, x_aug1, x_aug2), the model produces:
- Class logits: y_clean, y_aug1, y_aug2
- Concept logits for Shape, Colour, Border, and Symbol Category
- Auxiliary symbol logits: aux_clean

The loss function enforces dual-level supervision:
1. Supervised multi-task CBM loss on clean images:
   L_clean = L_CE(y_clean, y) + lambda_c * sum(L_CE(c_clean_k, c_k)) + lambda_aux * L_CE(aux_clean, y)
2. Task-level consistency regularization:
   L_jsd_task = JSD(y_clean, y_aug1, y_aug2)
3. Concept-level semantic invariance:
   L_jsd_cpt = (1/4) * [ JSD(shape) + JSD(colour) + JSD(border) + JSD(symcat) ]

Total Objective:
L_total = L_clean + lambda_jsd_task * L_jsd_task + lambda_jsd_cpt * L_jsd_cpt
"""

from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def compute_jsd(
    logits_clean: torch.Tensor,
    logits_aug1: torch.Tensor,
    logits_aug2: torch.Tensor,
    eps: float = 1e-7,
) -> torch.Tensor:
    """
    Computes numerically stable symmetric Jensen-Shannon Divergence across 3 distributions.

    Parameters
    ----------
    logits_clean : Tensor of shape [B, C]
    logits_aug1  : Tensor of shape [B, C]
    logits_aug2  : Tensor of shape [B, C]
    eps          : Clamping threshold for log-space numerical stability

    Returns
    -------
    Scalar float32 tensor representing JSD(P_clean, P_aug1, P_aug2).
    """
    # Cast to float32 for AMP stability
    p_clean = F.softmax(logits_clean.float(), dim=-1)
    p_aug1 = F.softmax(logits_aug1.float(), dim=-1)
    p_aug2 = F.softmax(logits_aug2.float(), dim=-1)

    # Mixture distribution: M = (P1 + P2 + P3) / 3
    p_mean = (p_clean + p_aug1 + p_aug2) / 3.0
    log_p_mean = torch.log(p_mean.clamp(min=eps))

    # Symmetrical KL divergences: KL(P || M)
    kl_clean = F.kl_div(log_p_mean, p_clean, reduction="batchmean")
    kl_aug1 = F.kl_div(log_p_mean, p_aug1, reduction="batchmean")
    kl_aug2 = F.kl_div(log_p_mean, p_aug2, reduction="batchmean")

    return (kl_clean + kl_aug1 + kl_aug2) / 3.0


class HybridCBMLoss(nn.Module):
    """
    Unified Multi-Scale Consistency and Concept Supervision Objective for Experiment 5.

    Parameters
    ----------
    lambda_c : float
        Supervision weight for concept classification on clean view (default: 1.0).
    lambda_aux : float
        Supervision weight for auxiliary symbol classification on clean view (default: 0.5).
    lambda_jsd_task : float
        Consistency regularization weight for class prediction (default: 12.0).
    lambda_jsd_cpt : float
        Consistency regularization weight for semantic concepts (default: 4.0).
    """

    def __init__(
        self,
        lambda_c: float = 1.0,
        lambda_aux: float = 0.5,
        lambda_jsd_task: float = 12.0,
        lambda_jsd_cpt: float = 4.0,
    ) -> None:
        super().__init__()
        self.lambda_c = lambda_c
        self.lambda_aux = lambda_aux
        self.lambda_jsd_task = lambda_jsd_task
        self.lambda_jsd_cpt = lambda_jsd_cpt
        self.ce = nn.CrossEntropyLoss()

    def forward(
        self,
        outputs_clean: Dict[str, torch.Tensor],
        outputs_aug1: Dict[str, torch.Tensor],
        outputs_aug2: Dict[str, torch.Tensor],
        target_class: torch.Tensor,
        target_shape: torch.Tensor,
        target_colour: torch.Tensor,
        target_border: torch.Tensor,
        target_symcat: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        # 1. Supervised task and auxiliary losses on clean images
        loss_task = self.ce(outputs_clean["logits"], target_class)
        loss_aux = self.ce(outputs_clean["logits_aux"], target_class)

        # 2. Supervised concept losses on clean images
        loss_shape = self.ce(outputs_clean["logits_shape"], target_shape)
        loss_colour = self.ce(outputs_clean["logits_colour"], target_colour)
        loss_border = self.ce(outputs_clean["logits_border"], target_border)
        loss_symcat = self.ce(outputs_clean["logits_symcat"], target_symcat)
        loss_concepts = loss_shape + loss_colour + loss_border + loss_symcat

        loss_clean_total = loss_task + self.lambda_c * loss_concepts + self.lambda_aux * loss_aux

        # 3. Task-level consistency regularization (Class JSD)
        loss_jsd_task = compute_jsd(
            outputs_clean["logits"],
            outputs_aug1["logits"],
            outputs_aug2["logits"],
        )

        # 4. Concept-level consistency regularization (Per-axis Concept JSDs)
        jsd_shape = compute_jsd(
            outputs_clean["logits_shape"],
            outputs_aug1["logits_shape"],
            outputs_aug2["logits_shape"],
        )
        jsd_colour = compute_jsd(
            outputs_clean["logits_colour"],
            outputs_aug1["logits_colour"],
            outputs_aug2["logits_colour"],
        )
        jsd_border = compute_jsd(
            outputs_clean["logits_border"],
            outputs_aug1["logits_border"],
            outputs_aug2["logits_border"],
        )
        jsd_symcat = compute_jsd(
            outputs_clean["logits_symcat"],
            outputs_aug1["logits_symcat"],
            outputs_aug2["logits_symcat"],
        )
        loss_jsd_cpt = (jsd_shape + jsd_colour + jsd_border + jsd_symcat) / 4.0

        # 5. Total combined objective
        total_loss = (
            loss_clean_total
            + self.lambda_jsd_task * loss_jsd_task
            + self.lambda_jsd_cpt * loss_jsd_cpt
        )

        metrics = {
            "total_loss": float(total_loss.item()),
            "loss_task": float(loss_task.item()),
            "loss_concepts": float(loss_concepts.item()),
            "loss_aux": float(loss_aux.item()),
            "loss_jsd_task": float(loss_jsd_task.item()),
            "loss_jsd_cpt": float(loss_jsd_cpt.item()),
            "loss_shape": float(loss_shape.item()),
            "loss_colour": float(loss_colour.item()),
            "loss_border": float(loss_border.item()),
            "loss_symcat": float(loss_symcat.item()),
        }
        return total_loss, metrics
