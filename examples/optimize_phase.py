"""Minimal phase-only CGH optimization with the HDRP regularizer."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision.transforms.functional import pil_to_tensor

from hdrp_cgh import DarkRegionPriorConfig, HolographicDarkRegionPriorLoss
from hdrp_cgh.propagation import angular_spectrum_propagate


def load_target(path: str | None, size: int, device: torch.device) -> torch.Tensor:
    if path is None:
        y, x = torch.meshgrid(
            torch.linspace(-1, 1, size, device=device),
            torch.linspace(-1, 1, size, device=device),
            indexing="ij",
        )
        target = (
            ((x.abs() < 0.34) & (y.abs() < 0.28))
            | ((x.square() + (y + 0.34).square()) < 0.08)
        ).float()
        target[(x.abs() < 0.10) & (y > 0.02) & (y < 0.28)] = 0.0
        return target[None, None]

    image = Image.open(path).convert("L").resize((size, size))
    return pil_to_tensor(image).float().div(255.0)[None].to(device)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", type=str, default=None)
    parser.add_argument("--size", type=int, default=256)
    parser.add_argument("--steps", type=int, default=400)
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument("--distance", type=float, default=0.20, help="meters")
    parser.add_argument("--wavelength", type=float, default=532e-9, help="meters")
    parser.add_argument("--pixel-pitch", type=float, default=8e-6, help="meters")
    parser.add_argument("--hdrp-weight", type=float, default=0.10)
    parser.add_argument("--out", type=str, default="outputs/demo")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    target = load_target(args.target, args.size, device)
    phase = torch.nn.Parameter(2 * torch.pi * torch.rand_like(target))
    optimizer = torch.optim.Adam([phase], lr=args.lr)

    prior = HolographicDarkRegionPriorLoss(
        DarkRegionPriorConfig(
            scales=(3, 7, 15),
            scale_weights=(1.0, 1.0, 1.0),
            temperature=0.02,
            weight_dark_consistency=1.0,
            weight_background_leakage=0.5,
            weight_excess_floor=0.25,
        )
    )

    for step in range(args.steps):
        field = torch.exp(1j * phase)
        propagated = angular_spectrum_propagate(
            field,
            distance=args.distance,
            wavelength=args.wavelength,
            pixel_pitch=args.pixel_pitch,
        )
        intensity = propagated.abs().square()
        intensity = intensity / intensity.amax(dim=(-2, -1), keepdim=True).clamp_min(1e-8)

        image_loss = F.l1_loss(intensity, target)
        prior_loss, terms = prior(intensity, target)
        loss = image_loss + args.hdrp_weight * prior_loss

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        if step % 50 == 0 or step == args.steps - 1:
            print(
                f"step={step:04d} total={loss.item():.5f} image={image_loss.item():.5f} "
                f"hdrp={prior_loss.item():.5f} bg={terms['background_leakage'].item():.5f}"
            )

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    torch.save(phase.detach().cpu(), out / "phase.pt")

    with torch.no_grad():
        field = torch.exp(1j * phase)
        propagated = angular_spectrum_propagate(
            field,
            distance=args.distance,
            wavelength=args.wavelength,
            pixel_pitch=args.pixel_pitch,
        )
        intensity = propagated.abs().square()
        intensity = intensity / intensity.amax().clamp_min(1e-8)

    for name, tensor in {"target": target, "reconstruction": intensity, "phase": phase}.items():
        plt.figure(figsize=(4, 4))
        plt.imshow(tensor.detach().cpu().squeeze(), cmap="gray")
        plt.axis("off")
        plt.tight_layout()
        plt.savefig(out / f"{name}.png", dpi=180, bbox_inches="tight")
        plt.close()


if __name__ == "__main__":
    main()
