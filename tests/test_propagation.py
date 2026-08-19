import torch

from hdrp_cgh.propagation import angular_spectrum_propagate


def test_asm_identity_at_zero_distance():
    phase = torch.rand(1, 1, 32, 32)
    field = torch.exp(1j * phase)
    out = angular_spectrum_propagate(field, 0.0, 532e-9, 8e-6)
    assert torch.allclose(out, field, atol=1e-5, rtol=1e-5)
