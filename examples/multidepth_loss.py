"""Minimal example of the 3D HDRP crosstalk term."""

import torch

from hdrp_cgh import DarkRegionPriorConfig, HolographicDarkRegionPriorLoss

batch, height, width = 2, 128, 128
reconstruction = torch.rand(batch, 1, height, width, requires_grad=True)
target = torch.rand(batch, 1, height, width)

depth_reconstructions = [
    torch.rand(batch, 1, height, width, requires_grad=True) for _ in range(3)
]
depth_targets = [torch.rand(batch, 1, height, width) for _ in range(3)]

criterion = HolographicDarkRegionPriorLoss(
    DarkRegionPriorConfig(weight_depth_crosstalk=0.5)
)

loss, terms = criterion(
    reconstruction,
    target,
    depth_reconstructions=depth_reconstructions,
    depth_targets=depth_targets,
    depth_weights=[1.0, 1.0, 1.0],
)
loss.backward()

print({name: float(value.detach()) for name, value in terms.items()})
