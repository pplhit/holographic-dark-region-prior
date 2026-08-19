from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


def _ensure_bchw(x: torch.Tensor) -> torch.Tensor:
    """Convert HW / CHW / BCHW tensors to BCHW."""
    if x.ndim == 2:
        x = x[None, None]
    elif x.ndim == 3:
        x = x[None]
    elif x.ndim != 4:
        raise ValueError(f"Expected HW, CHW, or BCHW tensor, got shape {tuple(x.shape)}")
    return x


def _validate_kernel(kernel_size: int) -> None:
    if kernel_size < 1 or kernel_size % 2 == 0:
        raise ValueError("kernel_size must be a positive odd integer")


def _channel_min(x: torch.Tensor) -> torch.Tensor:
    """Reduce RGB/multi-channel intensity to a single local-darkness channel."""
    return x.amin(dim=1, keepdim=True)


def dark_channel(intensity: torch.Tensor, kernel_size: int = 15) -> torch.Tensor:
    """Hard dark channel using exact channel-min + local spatial min pooling."""
    _validate_kernel(kernel_size)
    x = _ensure_bchw(intensity)
    x = _channel_min(x)
    pad = kernel_size // 2
    return -F.max_pool2d(-x, kernel_size=kernel_size, stride=1, padding=pad)


def soft_dark_channel(
    intensity: torch.Tensor,
    kernel_size: int = 15,
    temperature: float = 0.02,
) -> torch.Tensor:
    """Differentiable soft-min approximation of the dark channel."""
    _validate_kernel(kernel_size)
    if temperature <= 0:
        raise ValueError("temperature must be > 0")

    x = _ensure_bchw(intensity)
    b, c, h, w = x.shape
    pad = kernel_size // 2
    x = F.pad(x, (pad, pad, pad, pad), mode="replicate")
    patches = F.unfold(x, kernel_size=kernel_size)
    softmin = -temperature * torch.logsumexp(-patches / temperature, dim=1)
    softmin = softmin + temperature * torch.log(
        torch.tensor(float(c * kernel_size * kernel_size), device=x.device, dtype=x.dtype)
    )
    return softmin.view(b, 1, h, w)


def multiscale_dark_channel(
    intensity: torch.Tensor,
    scales: Sequence[int] = (3, 7, 15),
    *,
    soft: bool = True,
    temperature: float = 0.02,
) -> list[torch.Tensor]:
    outputs: list[torch.Tensor] = []
    for k in scales:
        outputs.append(
            soft_dark_channel(intensity, k, temperature)
            if soft
            else dark_channel(intensity, k)
        )
    return outputs


def dark_region_mask(
    target_intensity: torch.Tensor,
    threshold: float = 0.05,
    *,
    relative: bool = True,
    softness: float = 0.01,
) -> torch.Tensor:
    """Return a soft mask selecting dark target regions."""
    if threshold < 0:
        raise ValueError("threshold must be >= 0")
    if softness <= 0:
        raise ValueError("softness must be > 0")

    x = _ensure_bchw(target_intensity)
    gray = x.mean(dim=1, keepdim=True) if x.shape[1] > 1 else x
    if relative:
        peak = gray.amax(dim=(-2, -1), keepdim=True).clamp_min(1e-8)
        gray = gray / peak
    return torch.sigmoid((threshold - gray) / softness)


def _normalize_per_image(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    x = _ensure_bchw(x)
    peak = x.amax(dim=(-2, -1), keepdim=True).clamp_min(eps)
    return x / peak


@dataclass
class DarkRegionPriorConfig:
    """Configuration for Holographic Dark-Region Prior (HDRP)."""

    scales: tuple[int, ...] = (3, 7, 15)
    scale_weights: tuple[float, ...] = (1.0, 1.0, 1.0)
    temperature: float = 0.02
    use_softmin: bool = True
    normalize_intensity: bool = True
    weight_dark_consistency: float = 1.0
    weight_background_leakage: float = 0.5
    dark_threshold: float = 0.05
    mask_softness: float = 0.01
    weight_excess_floor: float = 0.25
    excess_margin: float = 0.0
    weight_depth_crosstalk: float = 0.0

    def __post_init__(self) -> None:
        if len(self.scales) == 0:
            raise ValueError("At least one scale is required")
        if len(self.scales) != len(self.scale_weights):
            raise ValueError("scales and scale_weights must have the same length")
        for k in self.scales:
            _validate_kernel(k)
        if self.temperature <= 0:
            raise ValueError("temperature must be > 0")


class HolographicDarkRegionPriorLoss(nn.Module):
    """Physics-oriented prior for neural or iterative computer-generated holography."""

    def __init__(self, config: DarkRegionPriorConfig | None = None) -> None:
        super().__init__()
        self.config = config or DarkRegionPriorConfig()

    def _dc(self, x: torch.Tensor, kernel_size: int) -> torch.Tensor:
        if self.config.use_softmin:
            return soft_dark_channel(x, kernel_size, self.config.temperature)
        return dark_channel(x, kernel_size)

    def forward(
        self,
        reconstruction: torch.Tensor,
        target: torch.Tensor,
        *,
        depth_reconstructions: Iterable[torch.Tensor] | None = None,
        depth_targets: Iterable[torch.Tensor] | None = None,
        depth_weights: Iterable[float] | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        cfg = self.config
        rec = _ensure_bchw(reconstruction)
        tgt = _ensure_bchw(target)

        if rec.shape != tgt.shape:
            raise ValueError(f"reconstruction {rec.shape} and target {tgt.shape} must match")

        if cfg.normalize_intensity:
            rec = _normalize_per_image(rec)
            tgt = _normalize_per_image(tgt)

        scale_weight_sum = max(sum(cfg.scale_weights), 1e-12)
        consistency = rec.new_zeros(())
        excess_floor = rec.new_zeros(())

        for k, w in zip(cfg.scales, cfg.scale_weights):
            rec_dc = self._dc(rec, k)
            tgt_dc = self._dc(tgt, k)
            consistency = consistency + w * F.l1_loss(rec_dc, tgt_dc)
            excess_floor = excess_floor + w * F.relu(
                rec_dc - tgt_dc - cfg.excess_margin
            ).mean()

        consistency = consistency / scale_weight_sum
        excess_floor = excess_floor / scale_weight_sum

        mask = dark_region_mask(
            tgt,
            threshold=cfg.dark_threshold,
            relative=not cfg.normalize_intensity,
            softness=cfg.mask_softness,
        )
        rec_gray = rec.mean(dim=1, keepdim=True) if rec.shape[1] > 1 else rec
        background_leakage = (mask * rec_gray).sum() / mask.sum().clamp_min(1.0)

        depth_crosstalk = rec.new_zeros(())
        if depth_reconstructions is not None or depth_targets is not None:
            if depth_reconstructions is None or depth_targets is None:
                raise ValueError("Provide both depth_reconstructions and depth_targets")
            depth_recs = list(depth_reconstructions)
            depth_tgts = list(depth_targets)
            if len(depth_recs) != len(depth_tgts):
                raise ValueError("depth_reconstructions and depth_targets must have equal length")
            if not depth_recs:
                raise ValueError("depth lists must not be empty")
            weights = list(depth_weights) if depth_weights is not None else [1.0] * len(depth_recs)
            if len(weights) != len(depth_recs):
                raise ValueError("depth_weights must match the number of depth planes")
            denom = max(sum(weights), 1e-12)
            for ri, ti, wi in zip(depth_recs, depth_tgts, weights):
                ri = _ensure_bchw(ri)
                ti = _ensure_bchw(ti)
                if cfg.normalize_intensity:
                    ri = _normalize_per_image(ri)
                    ti = _normalize_per_image(ti)
                plane_mask = dark_region_mask(
                    ti,
                    threshold=cfg.dark_threshold,
                    relative=not cfg.normalize_intensity,
                    softness=cfg.mask_softness,
                )
                ri_gray = ri.mean(dim=1, keepdim=True) if ri.shape[1] > 1 else ri
                depth_crosstalk = depth_crosstalk + wi * (
                    (plane_mask * ri_gray).sum() / plane_mask.sum().clamp_min(1.0)
                )
            depth_crosstalk = depth_crosstalk / denom

        total = (
            cfg.weight_dark_consistency * consistency
            + cfg.weight_background_leakage * background_leakage
            + cfg.weight_excess_floor * excess_floor
            + cfg.weight_depth_crosstalk * depth_crosstalk
        )

        terms = {
            "hdrp_total": total,
            "dark_consistency": consistency,
            "background_leakage": background_leakage,
            "excess_floor": excess_floor,
            "depth_crosstalk": depth_crosstalk,
        }
        return total, terms
