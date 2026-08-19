"""Template showing how to add HDRP to an existing neural holography loop."""

import torch.nn.functional as F

from hdrp_cgh import DarkRegionPriorConfig, HolographicDarkRegionPriorLoss

prior = HolographicDarkRegionPriorLoss(
    DarkRegionPriorConfig(
        scales=(3, 7, 15),
        temperature=0.02,
        weight_dark_consistency=1.0,
        weight_background_leakage=0.5,
        weight_excess_floor=0.25,
    )
)


def holography_loss(reconstructed_intensity, target_intensity):
    image_loss = F.l1_loss(reconstructed_intensity, target_intensity)
    hdrp_loss, hdrp_terms = prior(reconstructed_intensity, target_intensity)
    total = image_loss + 0.1 * hdrp_loss
    return total, {"image": image_loss, **hdrp_terms}
