"""
DL-AG Layer 2 losses: ARD + difficulty-aware omega + hidden alignment (PKD-style).
Matches paper Eq. ARD / omega / L_align.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .utils import entropy_from_logits


class FeatureProjector(nn.Module):
    """Maps student hidden dim -> teacher hidden dim for selected layers."""

    def __init__(self, layer_map: Dict[int, int], student_dim: int, teacher_dim: int):
        super().__init__()
        self.layer_map = layer_map  # student_idx -> teacher_idx (1-based transformer layers)
        self.projs = nn.ModuleDict(
            {str(s): nn.Linear(student_dim, teacher_dim) for s in layer_map.keys()}
        )

    def forward(
        self,
        student_hidden: Tuple[torch.Tensor, ...],
        teacher_hidden: Tuple[torch.Tensor, ...],
        omega: torch.Tensor,
    ) -> torch.Tensor:
        # hidden_states[0]=embeddings, [i]=layer i
        loss = torch.tensor(0.0, device=omega.device)
        n = 0
        for s_idx, t_idx in self.layer_map.items():
            hs = student_hidden[s_idx]
            ht = teacher_hidden[t_idx]
            # mean-pool over sequence for stability
            hs_p = self.projs[str(s_idx)](hs.mean(dim=1))
            ht_p = ht.mean(dim=1)
            # omega: [B]
            per = F.mse_loss(hs_p, ht_p, reduction="none").mean(dim=-1)
            loss = loss + (omega * per).mean()
            n += 1
        return loss / max(n, 1)


def ard_total_loss(
    student_logits_adv: torch.Tensor,
    teacher_logits_clean: torch.Tensor,
    teacher_logits_adv: torch.Tensor,
    labels: torch.Tensor,
    student_hidden: Optional[Tuple[torch.Tensor, ...]],
    teacher_hidden: Optional[Tuple[torch.Tensor, ...]],
    projector: Optional[FeatureProjector],
    T: float = 2.0,
    alpha: float = 0.5,
    beta: float = 1.0,
    align_weight: float = 1.0,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    L_total = (1-a) L_hard + a T^2 omega L_soft + L_align

    Teacher soft targets come from CLEAN inputs.
    omega from teacher entropy on ADV inputs.
    """
    hard = F.cross_entropy(student_logits_adv, labels)

    soft = F.kl_div(
        F.log_softmax(student_logits_adv / T, dim=-1),
        F.softmax(teacher_logits_clean / T, dim=-1),
        reduction="batchmean",
    )

    with torch.no_grad():
        H = entropy_from_logits(teacher_logits_adv)
        omega = torch.exp(-beta * H)  # [B]

    # broadcast omega into soft: use mean omega as scalar weight (paper applies per-example)
    soft_w = (omega.mean()) * soft

    align = torch.tensor(0.0, device=student_logits_adv.device)
    if projector is not None and student_hidden is not None and teacher_hidden is not None:
        align = projector(student_hidden, teacher_hidden, omega)

    total = (1.0 - alpha) * hard + alpha * (T * T) * soft_w + align_weight * align
    stats = {
        "hard": float(hard.detach().cpu()),
        "soft": float(soft.detach().cpu()),
        "align": float(align.detach().cpu()),
        "omega_mean": float(omega.mean().detach().cpu()),
        "total": float(total.detach().cpu()),
    }
    return total, stats


def default_layer_map(teacher_n_layers: int, student_n_layers: int = 6) -> Dict[int, int]:
    """
    PKD-Skip style uniform mapping from DistilBERT layers -> teacher layers.
    Indices are hidden_states indices (1..N), skipping embeddings at 0.
    """
    mapping: Dict[int, int] = {}
    for s in range(1, student_n_layers + 1):
        t = max(1, int(round(s * teacher_n_layers / student_n_layers)))
        t = min(t, teacher_n_layers)
        mapping[s] = t
    return mapping
