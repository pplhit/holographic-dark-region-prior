# Method: Holographic Dark-Region Prior (HDRP)

## Motivation

Dark Channel Prior (DCP) is useful here as a **design pattern for inverse problems**, not as a literal reuse of atmospheric transmission. In coherent computer-generated holography, the analogous failure is a raised background floor caused by speckle pedestal, zero-order leakage, stray light, imperfect calibration, or model mismatch.

A naive objective such as `mean(dark_channel(I_rec)) -> 0` is unsafe for coherent optics because destructive interference can create isolated zero-intensity nulls without improving reconstruction quality. HDRP is therefore **target-conditioned**.

## 1. Multi-scale dark-channel consistency

For reconstructed intensity `I` and target `T`, define a local minimum operator at scale `s`:

```math
D_s(I)(x)=\min_{y\in\Omega_s(x)}\min_c I_c(y).
```

HDRP matches the reconstructed and target local-minimum statistics:

```math
L_{dc}=\sum_s w_s\|D_s(I)-D_s(T)\|_1.
```

Multiple scales separate fine speckle/background structure from broader low-frequency veiling.

## 2. Soft dark channel

Exact `min` pooling sends gradients only through arg-min locations. During training we use a temperature-controlled soft minimum:

```math
\operatorname{softmin}_\tau(x_1,\ldots,x_N)
=-\tau\log\left(\frac{1}{N}\sum_i e^{-x_i/\tau}\right).
```

As `tau -> 0`, the approximation approaches the hard minimum while still providing denser gradients for finite temperature.

## 3. Target-conditioned background leakage

A soft dark-region mask is derived from the target:

```math
M_T(x)=\sigma\left(\frac{\eta-T(x)}{\epsilon}\right).
```

The background leakage term is

```math
L_{bg}=\frac{\sum_x M_T(x)I(x)}{\sum_x M_T(x)}.
```

This explicitly penalizes optical energy where the desired image is dark.

## 4. One-sided excess-floor term

To avoid rewarding artificial black interference nulls, HDRP additionally penalizes only reconstructed local minima that are brighter than the target:

```math
L_{excess}=\sum_s w_s\,[D_s(I)-D_s(T)-m]_+.
```

## 5. Multi-depth extension

For 3D holography, the same target-conditioned dark-region leakage can be evaluated at each propagation plane:

```math
L_{depth}=\sum_z \alpha_z\,
\frac{\sum_x M_{T_z}(x) I_z(x)}{\sum_x M_{T_z}(x)}.
```

This is intended to suppress ghost energy and cross-depth crosstalk, not to identify DCP transmission with propagation distance.

## Recommended starting objective

```math
L=L_{image}+\lambda_{hdrp}
(\lambda_{dc}L_{dc}+\lambda_{bg}L_{bg}+\lambda_{excess}L_{excess}+\lambda_zL_{depth}).
```

Suggested first sweep:

- `lambda_hdrp`: 0.02, 0.05, 0.1, 0.2
- scales: `(3, 7, 15)` or `(3, 9, 21)`
- soft-min temperature: 0.01, 0.02, 0.05
- dark threshold: 0.02, 0.05, 0.10 of normalized target peak

## Recommended ablations

1. Baseline image loss only.
2. Baseline + single-scale hard dark channel.
3. Baseline + multi-scale dark-channel consistency.
4. Baseline + soft-min multi-scale consistency.
5. + target-conditioned background leakage.
6. + one-sided excess-floor penalty.
7. 3D task: + multi-depth crosstalk term.

## Metrics

Report standard reconstruction metrics together with optics-specific diagnostics:

- PSNR / SSIM / LPIPS
- background floor in target-dark regions
- dark-channel L1 error
- speckle contrast
- diffraction / signal efficiency
- zero-order leakage if the experimental setup permits measurement
- cross-depth leakage for 3D holography
