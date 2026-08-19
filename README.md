# Holographic Dark-Region Prior (HDRP)

A compact PyTorch research package exploring a **Dark-Channel-inspired prior for neural and iterative computer-generated holography (CGH)**.

The central idea is not to reuse atmospheric dehazing physics literally. Instead, HDRP transfers the inverse-problem strategy behind the Dark Channel Prior: identify a simple image statistic, give it an optical interpretation, and use it to constrain an otherwise highly non-unique hologram optimization problem.

## Why HDRP?

Phase-only CGH can achieve low pixel loss while still exhibiting:

- raised background intensity,
- coherent speckle pedestal,
- zero-order / stray-light leakage,
- model-mismatch artifacts,
- cross-depth ghost energy in 3D holography.

A naive `dark_channel(I_rec) -> 0` loss is dangerous because coherent destructive interference can create isolated black nulls without improving image quality. HDRP therefore uses **target-conditioned local-minimum consistency** plus explicit **dark-region leakage suppression**.

## Method

For reconstruction `I` and target `T`:

```math
L_{dc}=\sum_s w_s\|D_s(I)-D_s(T)\|_1,
```

where `D_s` is a local dark-channel operator at scale `s`.

Dark target regions are selected with a soft mask `M_T`, then background leakage is penalized:

```math
L_{bg}=\frac{\sum_x M_T(x)I(x)}{\sum_x M_T(x)}.
```

A one-sided term penalizes only excess local background:

```math
L_{excess}=\sum_s w_s[D_s(I)-D_s(T)-m]_+.
```

The full prior is

```math
L_{HDRP}=\lambda_{dc}L_{dc}+\lambda_{bg}L_{bg}
+\lambda_{excess}L_{excess}+\lambda_zL_{depth}.
```

See [`docs/method.md`](docs/method.md) for the full rationale and 3D extension.

## Features

- Exact hard dark-channel operator.
- Differentiable soft-min dark channel with temperature control.
- Multi-scale dark-channel consistency.
- Target-conditioned background leakage loss.
- One-sided excess-background-floor penalty.
- Multi-depth crosstalk regularization for 3D CGH.
- Differentiable angular-spectrum propagation example.
- Optics-oriented metrics: background floor, dark-channel error, speckle contrast.
- Unit tests and GitHub Actions CI.

## Installation

```bash
git clone https://github.com/pplhit/holographic-dark-region-prior.git
cd holographic-dark-region-prior
pip install -e .[dev]
```

## Minimal usage

```python
import torch.nn.functional as F
from hdrp_cgh import DarkRegionPriorConfig, HolographicDarkRegionPriorLoss

criterion = HolographicDarkRegionPriorLoss(
    DarkRegionPriorConfig(
        scales=(3, 7, 15),
        temperature=0.02,
        weight_dark_consistency=1.0,
        weight_background_leakage=0.5,
        weight_excess_floor=0.25,
    )
)

image_loss = F.l1_loss(reconstructed_intensity, target_intensity)
hdrp_loss, terms = criterion(reconstructed_intensity, target_intensity)
loss = image_loss + 0.1 * hdrp_loss
loss.backward()
```

## Phase-only CGH demo

```bash
python examples/optimize_phase.py --steps 400
```

With your own target image:

```bash
python examples/optimize_phase.py --target path/to/target.png --steps 800
```

The demo optimizes a phase-only hologram with differentiable angular-spectrum propagation and saves the target, reconstruction, and optimized phase under `outputs/demo/`.

## Suggested ablation table

| Variant | Multi-scale | Soft-min | Dark-region leakage | Excess floor | Multi-depth |
|---|---:|---:|---:|---:|---:|
| Baseline | ✗ | ✗ | ✗ | ✗ | ✗ |
| + DCP | ✗ | ✗ | ✗ | ✗ | ✗ |
| + MS-DCP | ✓ | ✗ | ✗ | ✗ | ✗ |
| + Soft MS-DCP | ✓ | ✓ | ✗ | ✗ | ✗ |
| + HDRP | ✓ | ✓ | ✓ | ✓ | ✗ |
| + 3D HDRP | ✓ | ✓ | ✓ | ✓ | ✓ |

## Research hypothesis

HDRP should be judged by whether it improves **optical failure modes that standard image loss does not explicitly control**. The strongest experiments are therefore not only PSNR/SSIM comparisons, but also measurements of background floor, zero-order leakage, speckle contrast, signal efficiency, and cross-depth ghost energy.

## Status

Research prototype. The code is suitable for simulation studies and integration into neural holography training loops, but no experimental performance claims are made in this repository.

## License

MIT
