from __future__ import annotations

from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F

from .priors import DarkRegionPriorConfig, HolographicDarkRegionPriorLoss


@dataclass
class ReconstructionLossConfig:
    weight_l1: float = 1.0
    weight_mse: float = 0.0
    weight_hdrp: float = 0.1
    hdrp: DarkRegionPriorConfig | None = None


class ReconstructionWithHDRPLoss(nn.Module):
    """Drop-in image reconstruction loss combining ordinary fidelity and HDRP."""

    def __init__(self, config: ReconstructionLossConfig | None = None) -> None:
        super().__init__()
        self.config = config or ReconstructionLossConfig()
        self.hdrp = HolographicDarkRegionPriorLoss(self.config.hdrp)

    def forward(self, reconstruction: torch.Tensor, target: torch.Tensor):
        l1 = F.l1_loss(reconstruction, target)
        mse = F.mse_loss(reconstruction, target)
        hdrp, hdrp_terms = self.hdrp(reconstruction, target)
        total = (
            self.config.weight_l1 * l1
            + self.config.weight_mse * mse
            + self.config.weight_hdrp * hdrp
        )
        terms = {"total": total, "l1": l1, "mse": mse, **hdrp_terms}
        return total, terms
