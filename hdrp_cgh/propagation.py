from __future__ import annotations

import math
import torch


def angular_spectrum_propagate(
    field: torch.Tensor,
    distance: float,
    wavelength: float,
    pixel_pitch: float,
    *,
    bandlimit: bool = True,
) -> torch.Tensor:
    """Differentiable scalar angular-spectrum propagation."""
    if not torch.is_complex(field):
        raise TypeError("field must be a complex PyTorch tensor")
    if wavelength <= 0 or pixel_pitch <= 0:
        raise ValueError("wavelength and pixel_pitch must be positive")

    h, w = field.shape[-2:]
    device = field.device
    real_dtype = field.real.dtype

    fy = torch.fft.fftfreq(h, d=pixel_pitch, device=device, dtype=real_dtype)
    fx = torch.fft.fftfreq(w, d=pixel_pitch, device=device, dtype=real_dtype)
    fy, fx = torch.meshgrid(fy, fx, indexing="ij")

    inv_lambda2 = 1.0 / (wavelength * wavelength)
    radial2 = fx.square() + fy.square()
    propagating = radial2 <= inv_lambda2
    kz = 2.0 * math.pi * torch.sqrt(torch.clamp(inv_lambda2 - radial2, min=0.0))

    transfer = torch.exp(1j * distance * kz)
    if bandlimit:
        transfer = transfer * propagating

    spectrum = torch.fft.fft2(field)
    return torch.fft.ifft2(spectrum * transfer)
