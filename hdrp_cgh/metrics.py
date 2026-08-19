from __future__ import annotations

import torch
import torch.nn.functional as F

from .priors import _ensure_bchw, dark_channel, dark_region_mask


def background_floor(
    reconstruction: torch.Tensor,
    target: torch.Tensor,
    *,
    threshold: float = 0.05,
) -> torch.Tensor:
    """Mean reconstructed intensity over regions that are dark in the target."""
    rec = _ensure_bchw(reconstruction)
    tgt = _ensure_bchw(target)
    rec = rec / rec.amax(dim=(-2, -1), keepdim=True).clamp_min(1e-8)
    tgt = tgt / tgt.amax(dim=(-2, -1), keepdim=True).clamp_min(1e-8)
    mask = dark_region_mask(tgt, threshold=threshold, relative=False, softness=0.005)
    rec_gray = rec.mean(dim=1, keepdim=True) if rec.shape[1] > 1 else rec
    return (mask * rec_gray).sum() / mask.sum().clamp_min(1.0)


def dark_channel_l1(
    reconstruction: torch.Tensor,
    target: torch.Tensor,
    *,
    kernel_size: int = 15,
) -> torch.Tensor:
    """L1 distance between normalized hard dark-channel maps."""
    rec = _ensure_bchw(reconstruction)
    tgt = _ensure_bchw(target)
    rec = rec / rec.amax(dim=(-2, -1), keepdim=True).clamp_min(1e-8)
    tgt = tgt / tgt.amax(dim=(-2, -1), keepdim=True).clamp_min(1e-8)
    return F.l1_loss(dark_channel(rec, kernel_size), dark_channel(tgt, kernel_size))


def speckle_contrast(intensity: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
    """Standard speckle contrast sigma / mean, optionally within a supplied mask."""
    x = _ensure_bchw(intensity)
    if x.shape[1] > 1:
        x = x.mean(dim=1, keepdim=True)
    if mask is None:
        mean = x.mean()
        std = x.std(unbiased=False)
    else:
        m = _ensure_bchw(mask).to(dtype=x.dtype, device=x.device)
        if m.shape[1] != 1:
            m = m.mean(dim=1, keepdim=True)
        denom = m.sum().clamp_min(1.0)
        mean = (m * x).sum() / denom
        var = (m * (x - mean).square()).sum() / denom
        std = torch.sqrt(var + 1e-12)
    return std / mean.clamp_min(1e-8)
