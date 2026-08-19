import torch

from hdrp_cgh.priors import (
    DarkRegionPriorConfig,
    HolographicDarkRegionPriorLoss,
    dark_channel,
    soft_dark_channel,
)


def test_hard_dark_channel_shape_and_minimum():
    x = torch.ones(1, 1, 9, 9)
    x[..., 4, 4] = 0.0
    y = dark_channel(x, kernel_size=3)
    assert y.shape == (1, 1, 9, 9)
    assert y[..., 4, 4].item() == 0.0


def test_soft_dark_channel_has_dense_gradients():
    x = torch.rand(1, 1, 9, 9, requires_grad=True)
    y = soft_dark_channel(x, kernel_size=3, temperature=0.05).mean()
    y.backward()
    assert x.grad is not None
    assert torch.isfinite(x.grad).all()
    assert (x.grad.abs() > 0).sum() > 1


def test_hdrp_prefers_lower_background_leakage():
    target = torch.zeros(1, 1, 32, 32)
    target[..., 8:24, 8:24] = 1.0
    clean = target.clone()
    lifted = (target + 0.15).clamp(max=1.0)

    criterion = HolographicDarkRegionPriorLoss(
        DarkRegionPriorConfig(scales=(3, 7), scale_weights=(1.0, 1.0))
    )
    clean_loss, _ = criterion(clean, target)
    lifted_loss, _ = criterion(lifted, target)
    assert clean_loss < lifted_loss


def test_hdrp_backward():
    target = torch.rand(2, 1, 24, 24)
    rec = torch.rand(2, 1, 24, 24, requires_grad=True)
    criterion = HolographicDarkRegionPriorLoss()
    loss, terms = criterion(rec, target)
    loss.backward()
    assert rec.grad is not None
    assert torch.isfinite(rec.grad).all()
    assert set(terms) == {
        "hdrp_total",
        "dark_consistency",
        "background_leakage",
        "excess_floor",
        "depth_crosstalk",
    }


def test_multidepth_crosstalk_backward():
    target = torch.rand(1, 1, 20, 20)
    rec = torch.rand(1, 1, 20, 20, requires_grad=True)
    depth_rec = [torch.rand(1, 1, 20, 20, requires_grad=True) for _ in range(2)]
    depth_tgt = [torch.rand(1, 1, 20, 20) for _ in range(2)]
    criterion = HolographicDarkRegionPriorLoss(
        DarkRegionPriorConfig(weight_depth_crosstalk=0.5)
    )
    loss, _ = criterion(
        rec,
        target,
        depth_reconstructions=depth_rec,
        depth_targets=depth_tgt,
    )
    loss.backward()
    assert rec.grad is not None
    assert all(x.grad is not None for x in depth_rec)
