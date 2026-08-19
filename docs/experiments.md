# Experimental plan

This project is strongest if HDRP is evaluated as an **optical prior**, rather than only as another image-space loss.

## 1. Core hypothesis

A conventional reconstruction loss can match bright structures while leaving an elevated optical background. HDRP should selectively reduce this background without simply manufacturing isolated interference zeros.

The key claim to test is:

> Target-conditioned local-minimum statistics provide information about holographic background quality that is complementary to ordinary pixel/perceptual fidelity.

## 2. Datasets

Start with three regimes:

1. **Binary / sparse targets** — strongest test of dark-region leakage.
2. **Natural grayscale images** — tests generality and whether the prior over-suppresses low-level texture.
3. **RGB natural images** — enables the original cross-channel dark-channel structure.

For 3D CGH, use RGB-D or layered focal stacks with known target planes.

## 3. Training comparisons

Keep network, propagation model, sampling, optimizer, schedule, dataset split, and random seeds fixed. Vary only the prior.

Recommended variants are provided in `configs/ablation.yaml`.

## 4. Metrics

### Fidelity

- PSNR
- SSIM
- LPIPS (optional external dependency)

### Optical-background metrics

- background floor over target-dark regions
- mean / percentile dark-channel error
- zero-order energy ratio
- signal-to-background ratio
- diffraction efficiency

### Coherence artifacts

- speckle contrast
- high-frequency residual energy in target-dark regions

### 3D

- cross-depth leakage
- focal-stack PSNR / SSIM by plane
- energy deposited at incorrect depth planes

## 5. Important controls

### Control A: `D(I) -> 0`

Include the naive zero-dark-channel objective as a negative control. It may reduce the numerical dark-channel value by creating destructive-interference nulls without lowering the broader background.

### Control B: ordinary dark-region mask only

Compare HDRP against simply weighting pixel loss more heavily in dark target regions. This determines whether local-minimum statistics contribute beyond a foreground/background weighting trick.

### Control C: high-pass / TV background regularization

Compare against a generic smoothness or high-frequency penalty to test whether HDRP is specifically useful for coherent background structure.

## 6. Simulation-to-experiment path

1. Train with ideal differentiable ASM/Fresnel propagation.
2. Add SLM quantization, fill factor, pixel crosstalk, finite aperture, and zero-order leakage.
3. Add measured or learned camera-in-the-loop calibration.
4. Evaluate whether the HDRP gain grows under model mismatch.

A particularly interesting result would be that HDRP offers only modest gains in ideal simulation but larger gains in the physical system, supporting the interpretation that it regularizes unmodeled background leakage.

## 7. Suggested paper figures

- **Fig. 1**: DCP-to-HDRP conceptual transition and CGH pipeline.
- **Fig. 2**: Histograms of dark-region intensities for target, baseline reconstruction, and HDRP reconstruction.
- **Fig. 3**: Qualitative reconstructions + magnified dark/background regions.
- **Fig. 4**: Ablation over scale, soft-min temperature, and prior weight.
- **Fig. 5**: Experimental optical reconstructions and zero-order/background measurements.
- **Fig. 6**: 3D focal stack / cross-depth crosstalk if included.
