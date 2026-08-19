from __future__ import annotations

import argparse
import csv
import json
import math
import random
import time
import zipfile
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F

from hdrp_cgh.priors import dark_channel
from hdrp_cgh.propagation import angular_spectrum_propagate


@dataclass
class ExperimentConfig:
    crop_size: int = 64
    epochs: int = 10
    batch_size: int = 32
    lr: float = 1e-3
    seed: int = 1234
    wavelength: float = 532e-9
    pixel_pitch: float = 8e-6
    distance: float = 0.02
    residual_phase_scale: float = math.pi / 2
    prior_weight: float = 0.05
    dark_threshold: float = 0.15
    soft_temperature_start: float = 0.05
    soft_temperature_end: float = 0.015
    scales: tuple[int, ...] = (3, 7, 15)
    val_batch_size: int = 25
    num_visual_examples: int = 4


METHODS = [
    "baseline",
    "naive_dcp_zero",
    "dcp_consistency",
    "ms_dcp",
    "soft_ms_dcp",
    "hdrp_no_excess",
    "hdrp_full",
]

DISPLAY_NAMES = {
    "physics_only": "Physics-only",
    "baseline": "Neural baseline",
    "naive_dcp_zero": "Naive DCP->0",
    "dcp_consistency": "DCP consistency",
    "ms_dcp": "MS-DCP",
    "soft_ms_dcp": "Soft MS-DCP",
    "hdrp_no_excess": "HDRP w/o excess",
    "hdrp_full": "HDRP (full)",
}


class ResidualBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.gelu(x + self.conv2(F.gelu(self.conv1(x))))


class TinyResidualPhaseNet(nn.Module):
    def __init__(self, base: int = 12) -> None:
        super().__init__()
        self.in_conv = nn.Conv2d(3, base, 3, padding=1)
        self.enc1 = ResidualBlock(base)
        self.down1 = nn.Conv2d(base, base * 2, 4, stride=2, padding=1)
        self.enc2 = ResidualBlock(base * 2)
        self.down2 = nn.Conv2d(base * 2, base * 3, 4, stride=2, padding=1)
        self.mid = nn.Sequential(ResidualBlock(base * 3), ResidualBlock(base * 3))
        self.up2 = nn.Conv2d(base * 3 + base * 2, base * 2, 3, padding=1)
        self.dec2 = ResidualBlock(base * 2)
        self.up1 = nn.Conv2d(base * 2 + base, base, 3, padding=1)
        self.dec1 = ResidualBlock(base)
        self.out = nn.Conv2d(base, 1, 3, padding=1)

    def forward(self, target_display: torch.Tensor, phase0: torch.Tensor) -> torch.Tensor:
        x = torch.cat([target_display, torch.sin(phase0), torch.cos(phase0)], dim=1)
        e1 = self.enc1(F.gelu(self.in_conv(x)))
        e2 = self.enc2(F.gelu(self.down1(e1)))
        z = self.mid(F.gelu(self.down2(e2)))
        z = F.interpolate(z, size=e2.shape[-2:], mode="bilinear", align_corners=False)
        z = self.dec2(F.gelu(self.up2(torch.cat([z, e2], dim=1))))
        z = F.interpolate(z, size=e1.shape[-2:], mode="bilinear", align_corners=False)
        z = self.dec1(F.gelu(self.up1(torch.cat([z, e1], dim=1))))
        return torch.tanh(self.out(z))


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def srgb_to_linear(x: np.ndarray) -> np.ndarray:
    return np.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4)


def load_linear_luminance(path: Path) -> np.ndarray:
    rgb = np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0
    rgb = srgb_to_linear(rgb)
    return (0.2126 * rgb[..., 0] + 0.7152 * rgb[..., 1] + 0.0722 * rgb[..., 2]).astype(np.float32)


def deterministic_crop(img: np.ndarray, size: int, seed: int) -> np.ndarray:
    h, w = img.shape
    rng = np.random.default_rng(seed)
    y = int(rng.integers(0, h - size + 1))
    x = int(rng.integers(0, w - size + 1))
    crop = img[y:y + size, x:x + size]
    if rng.random() < 0.5:
        crop = crop[:, ::-1]
    if rng.random() < 0.5:
        crop = crop[::-1, :]
    return np.ascontiguousarray(crop)


def center_crop(img: np.ndarray, size: int) -> np.ndarray:
    h, w = img.shape
    y = (h - size) // 2
    x = (w - size) // 2
    return np.ascontiguousarray(img[y:y + size, x:x + size])


def extract_if_needed(zip_path: Path, out_dir: Path) -> None:
    if out_dir.exists() and any(out_dir.glob("*.png")):
        return
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(out_dir.parent)


def prepare_patch_cache(data_root: Path, cache_path: Path, cfg: ExperimentConfig):
    if cache_path.exists():
        return torch.load(cache_path, map_location="cpu", weights_only=False)
    train_files = sorted((data_root / "DIV2K_train_HR").glob("*.png"))
    val_files = sorted((data_root / "DIV2K_valid_HR").glob("*.png"))
    if len(train_files) != 800 or len(val_files) != 100:
        raise RuntimeError(f"Expected DIV2K 800/100 images, found {len(train_files)}/{len(val_files)}")
    train = torch.empty(cfg.epochs, 800, 1, cfg.crop_size, cfg.crop_size, dtype=torch.float16)
    for i, path in enumerate(train_files):
        img = load_linear_luminance(path)
        for epoch in range(cfg.epochs):
            crop = deterministic_crop(img, cfg.crop_size, cfg.seed + epoch * 10000 + i)
            train[epoch, i, 0] = torch.from_numpy(crop).to(torch.float16)
        if (i + 1) % 100 == 0:
            print(f"prepared train images: {i + 1}/800", flush=True)
    val = torch.empty(100, 1, cfg.crop_size, cfg.crop_size, dtype=torch.float16)
    for i, path in enumerate(val_files):
        val[i, 0] = torch.from_numpy(center_crop(load_linear_luminance(path), cfg.crop_size)).to(torch.float16)
    cache = {"train": train, "val": val}
    torch.save(cache, cache_path)
    return cache


def normalize_target_energy(x: torch.Tensor) -> torch.Tensor:
    return x / x.mean(dim=(-2, -1), keepdim=True).clamp_min(0.03)


def analytic_phase0(target: torch.Tensor, cfg: ExperimentConfig) -> torch.Tensor:
    amp = torch.sqrt(target.clamp_min(0.0))
    field = torch.complex(amp, torch.zeros_like(amp))
    return torch.angle(angular_spectrum_propagate(field, -cfg.distance, cfg.wavelength, cfg.pixel_pitch))


def reconstruct_from_phase(phase: torch.Tensor, cfg: ExperimentConfig) -> torch.Tensor:
    field = torch.polar(torch.ones_like(phase), phase)
    out = angular_spectrum_propagate(field, cfg.distance, cfg.wavelength, cfg.pixel_pitch)
    return out.abs().square()


def forward_model(model, display: torch.Tensor, cfg: ExperimentConfig):
    target = normalize_target_energy(display)
    phase0 = analytic_phase0(target, cfg)
    if model is None:
        phase = phase0
    else:
        residual = model(display, phase0) * cfg.residual_phase_scale
        phase = torch.remainder(phase0 + residual + math.pi, 2 * math.pi) - math.pi
    return reconstruct_from_phase(phase, cfg), target, phase


def peak_norm(x: torch.Tensor) -> torch.Tensor:
    return x / x.amax(dim=(-2, -1), keepdim=True).clamp_min(1e-8)


def soft_dc_fast(x: torch.Tensor, k: int, tau: float) -> torch.Tensor:
    pad = k // 2
    x = F.pad(x, (pad, pad, pad, pad), mode="replicate")
    mean_exp = torch.exp(-x / tau).mean(dim=1, keepdim=True)
    mean_exp = F.avg_pool2d(mean_exp, k, stride=1)
    return -tau * torch.log(mean_exp.clamp_min(torch.finfo(x.dtype).tiny))


def prior_loss(method: str, rec: torch.Tensor, tgt: torch.Tensor, epoch: int, cfg: ExperimentConfig):
    rec_n, tgt_n = peak_norm(rec), peak_norm(tgt)
    zero = rec.new_zeros(())
    if method == "baseline":
        return zero
    if method == "naive_dcp_zero":
        return dark_channel(rec_n, 7).mean()
    if method == "dcp_consistency":
        return F.l1_loss(dark_channel(rec_n, 7), dark_channel(tgt_n, 7))
    use_soft = method in {"soft_ms_dcp", "hdrp_no_excess", "hdrp_full"}
    if use_soft:
        a = epoch / max(cfg.epochs - 1, 1)
        tau = cfg.soft_temperature_start * (1 - a) + cfg.soft_temperature_end * a
        dc_fn = lambda z, k: soft_dc_fast(z, k, tau)
    else:
        dc_fn = lambda z, k: dark_channel(z, k)
    dc_losses, excess_losses = [], []
    for k in cfg.scales:
        rdc, tdc = dc_fn(rec_n, k), dc_fn(tgt_n, k)
        dc_losses.append(F.l1_loss(rdc, tdc))
        excess_losses.append(F.relu(rdc - tdc).mean())
    dc = torch.stack(dc_losses).mean()
    if method in {"ms_dcp", "soft_ms_dcp"}:
        return dc
    mask = torch.sigmoid((cfg.dark_threshold - tgt_n) / 0.02)
    bg = (mask * rec_n).sum() / mask.sum().clamp_min(1.0)
    if method == "hdrp_no_excess":
        return dc + 0.5 * bg
    return dc + 0.5 * bg + 0.25 * torch.stack(excess_losses).mean()


def ssim_simple(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    k = 7
    mx = F.avg_pool2d(x, k, 1, k // 2)
    my = F.avg_pool2d(y, k, 1, k // 2)
    vx = F.avg_pool2d(x * x, k, 1, k // 2) - mx.square()
    vy = F.avg_pool2d(y * y, k, 1, k // 2) - my.square()
    vxy = F.avg_pool2d(x * y, k, 1, k // 2) - mx * my
    c1, c2 = 0.01 ** 2, 0.03 ** 2
    s = ((2 * mx * my + c1) * (2 * vxy + c2)) / ((mx.square() + my.square() + c1) * (vx + vy + c2) + 1e-12)
    return s.mean(dim=(1, 2, 3))


def batch_metrics(rec: torch.Tensor, tgt: torch.Tensor, cfg: ExperimentConfig):
    rn, tn = peak_norm(rec).clamp(0, 1), peak_norm(tgt).clamp(0, 1)
    mse = (rn - tn).square().mean(dim=(1, 2, 3)).clamp_min(1e-12)
    dark = (tn < cfg.dark_threshold).to(rn.dtype)
    bright = (tn > 0.5).to(rn.dtype)
    smooth = F.avg_pool2d(rn, 7, 1, 3).clamp_min(1e-4)
    residual = (rn - smooth) / smooth
    mid = (tn > 0.35).to(rn.dtype)
    rmean = (residual * mid).sum(dim=(1, 2, 3)) / mid.sum(dim=(1, 2, 3)).clamp_min(1.0)
    rvar = (((residual - rmean[:, None, None, None]) * mid).square().sum(dim=(1, 2, 3)) / mid.sum(dim=(1, 2, 3)).clamp_min(1.0))
    return {
        "psnr": -10 * torch.log10(mse),
        "ssim": ssim_simple(rn, tn),
        "background_floor": (rn * dark).sum(dim=(1, 2, 3)) / dark.sum(dim=(1, 2, 3)).clamp_min(1.0),
        "dark_channel_mae": (dark_channel(rn, 7) - dark_channel(tn, 7)).abs().mean(dim=(1, 2, 3)),
        "bright_efficiency": (rn * bright).sum(dim=(1, 2, 3)) / rn.sum(dim=(1, 2, 3)).clamp_min(1e-8),
        "local_speckle": torch.sqrt(rvar.clamp_min(0.0)),
    }


@torch.no_grad()
def evaluate(model, val: torch.Tensor, cfg: ExperimentConfig):
    if model is not None:
        model.eval()
    accum = {}
    for start in range(0, len(val), cfg.val_batch_size):
        display = val[start:start + cfg.val_batch_size].float()
        rec, tgt, _ = forward_model(model, display, cfg)
        for k, v in batch_metrics(rec, tgt, cfg).items():
            accum.setdefault(k, []).append(v.cpu())
    return {k: float(torch.cat(v).mean()) for k, v in accum.items()}


def train_method(method, init_state, train, val, cfg, out_dir):
    model = TinyResidualPhaseNet()
    model.load_state_dict(init_state)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    history = []
    for epoch in range(cfg.epochs):
        model.train()
        order = torch.randperm(train.shape[1], generator=torch.Generator().manual_seed(cfg.seed + epoch))
        loss_sum = rec_sum = prior_sum = 0.0
        nb = 0
        for start in range(0, len(order), cfg.batch_size):
            display = train[epoch, order[start:start + cfg.batch_size]].float()
            rec, tgt, _ = forward_model(model, display, cfg)
            rec_loss = F.l1_loss(rec, tgt)
            p_loss = prior_loss(method, rec, tgt, epoch, cfg)
            loss = rec_loss + cfg.prior_weight * p_loss
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            optimizer.step()
            loss_sum += float(loss.detach()); rec_sum += float(rec_loss.detach()); prior_sum += float(p_loss.detach()); nb += 1
        row = {"epoch": epoch + 1, "train_loss": loss_sum / nb, "train_reconstruction": rec_sum / nb, "train_prior": prior_sum / nb, **evaluate(model, val, cfg)}
        history.append(row)
        print(method, json.dumps(row), flush=True)
    ckpt = out_dir / "checkpoints"
    ckpt.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), ckpt / f"{method}.pt")
    return model, history


def plot_curves(histories, out_dir):
    for metric, ylabel in [("psnr", "Validation PSNR (dB)"), ("ssim", "Validation SSIM"), ("background_floor", "Dark-region background floor"), ("dark_channel_mae", "Dark-channel MAE"), ("local_speckle", "Local residual contrast")]:
        fig, ax = plt.subplots(figsize=(8, 5))
        for method, hist in histories.items():
            ax.plot([r["epoch"] for r in hist], [r[metric] for r in hist], marker="o", ms=3, label=DISPLAY_NAMES[method])
        ax.set_xlabel("Epoch"); ax.set_ylabel(ylabel); ax.grid(alpha=0.25); ax.legend(fontsize=8, ncol=2)
        fig.tight_layout(); fig.savefig(out_dir / f"curve_{metric}.png", dpi=180); plt.close(fig)


def plot_metric_summary(final_metrics, out_dir):
    metrics = ["psnr", "ssim", "background_floor", "dark_channel_mae", "bright_efficiency", "local_speckle"]
    labels = ["PSNR higher is better", "SSIM higher is better", "Background floor lower is better", "Dark-channel MAE lower is better", "Bright efficiency higher is better", "Local residual contrast lower is better"]
    methods = list(final_metrics)
    fig, axes = plt.subplots(2, 3, figsize=(13, 7))
    for ax, metric, label in zip(axes.flat, metrics, labels):
        ax.bar(range(len(methods)), [final_metrics[m][metric] for m in methods]); ax.set_title(label, fontsize=9)
        ax.set_xticks(range(len(methods))); ax.set_xticklabels([DISPLAY_NAMES[m] for m in methods], rotation=55, ha="right", fontsize=7); ax.grid(axis="y", alpha=0.2)
    fig.tight_layout(); fig.savefig(out_dir / "metric_summary.png", dpi=180); plt.close(fig)


def tensor_image(x):
    x = x.detach().cpu().float(); x = x / x.max().clamp_min(1e-8); return x.squeeze().numpy()


@torch.no_grad()
def save_qualitative(models, val, cfg, out_dir):
    ids = np.linspace(0, len(val) - 1, cfg.num_visual_examples, dtype=int)
    methods = list(models)
    for idx in ids:
        display = val[idx:idx + 1].float(); target = normalize_target_energy(display)
        fig, axes = plt.subplots(2, len(methods) + 1, figsize=(2.2 * (len(methods) + 1), 4.7))
        axes[0, 0].imshow(tensor_image(target), cmap="gray", vmin=0, vmax=1); axes[0, 0].set_title("Target"); axes[1, 0].axis("off")
        for col, method in enumerate(methods, 1):
            rec, _, phase = forward_model(models[method], display, cfg)
            axes[0, col].imshow(tensor_image(rec), cmap="gray", vmin=0, vmax=1); axes[0, col].set_title(DISPLAY_NAMES[method], fontsize=8)
            axes[1, col].imshow(phase.squeeze().cpu(), cmap="twilight", vmin=-math.pi, vmax=math.pi); axes[1, col].set_title("Phase", fontsize=8)
        for ax in axes.flat: ax.set_xticks([]); ax.set_yticks([])
        fig.suptitle(f"DIV2K validation example {idx + 801}", fontsize=11); fig.tight_layout(); fig.savefig(out_dir / f"qualitative_{idx + 801:04d}.png", dpi=170); plt.close(fig)
        fig, axes = plt.subplots(1, len(methods) + 1, figsize=(2.2 * (len(methods) + 1), 2.5))
        axes[0].imshow(tensor_image(dark_channel(peak_norm(target), 7)), cmap="magma", vmin=0, vmax=1); axes[0].set_title("Target DC", fontsize=8)
        for col, method in enumerate(methods, 1):
            rec, _, _ = forward_model(models[method], display, cfg)
            axes[col].imshow(tensor_image(dark_channel(peak_norm(rec), 7)), cmap="magma", vmin=0, vmax=1); axes[col].set_title(DISPLAY_NAMES[method], fontsize=8)
        for ax in axes: ax.set_xticks([]); ax.set_yticks([])
        fig.tight_layout(); fig.savefig(out_dir / f"dark_channel_{idx + 801:04d}.png", dpi=170); plt.close(fig)


def write_report(final_metrics, cfg, out_dir):
    methods = list(final_metrics)
    best_psnr = max(methods, key=lambda m: final_metrics[m]["psnr"])
    best_bg = min(methods, key=lambda m: final_metrics[m]["background_floor"])
    best_dc = min(methods, key=lambda m: final_metrics[m]["dark_channel_mae"])
    lines = ["# DIV2K 10-epoch HDRP benchmark", "", "All 800 DIV2K training images are used. Each epoch draws one deterministic 64x64 crop from every image. Validation uses fixed center crops from all 100 validation images.", "", "## Protocol", "", f"- Epochs: {cfg.epochs}", f"- Crop size: {cfg.crop_size}x{cfg.crop_size}", f"- Batch size: {cfg.batch_size}", f"- Wavelength: {cfg.wavelength*1e9:.0f} nm", f"- Pixel pitch: {cfg.pixel_pitch*1e6:.1f} um", f"- Propagation distance: {cfg.distance*1e3:.1f} mm", "- Linear-luminance targets normalized to unit mean optical energy", "- Analytic backward-propagation phase initialization; neural methods learn residual phase", "", "## Final validation metrics", "", "| Method | PSNR up | SSIM up | Background floor down | DC MAE down | Bright efficiency up | Local residual contrast down |", "|---|---:|---:|---:|---:|---:|---:|"]
    for m in methods:
        v = final_metrics[m]
        lines.append(f"| {DISPLAY_NAMES[m]} | {v['psnr']:.3f} | {v['ssim']:.4f} | {v['background_floor']:.4f} | {v['dark_channel_mae']:.4f} | {v['bright_efficiency']:.4f} | {v['local_speckle']:.4f} |")
    lines += ["", "## Automatic observations", "", f"- Highest PSNR: **{DISPLAY_NAMES[best_psnr]}** ({final_metrics[best_psnr]['psnr']:.3f} dB).", f"- Lowest dark-region background floor: **{DISPLAY_NAMES[best_bg]}** ({final_metrics[best_bg]['background_floor']:.4f}).", f"- Lowest dark-channel MAE: **{DISPLAY_NAMES[best_dc]}** ({final_metrics[best_dc]['dark_channel_mae']:.4f}).", "", "Simulation-only result under the stated 64x64 single-plane scalar ASM protocol."]
    (out_dir / "REPORT.md").write_text("\n".join(lines))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", type=Path, required=True)
    p.add_argument("--output", type=Path, default=Path("results/div2k_10epoch"))
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--crop-size", type=int, default=64)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--methods", nargs="*", default=METHODS)
    args = p.parse_args()
    cfg = ExperimentConfig(epochs=args.epochs, crop_size=args.crop_size, batch_size=args.batch_size)
    set_seed(cfg.seed); args.output.mkdir(parents=True, exist_ok=True); (args.output / "config.json").write_text(json.dumps(asdict(cfg), indent=2))
    extract_if_needed(args.data_root / "DIV2K_train_HR.zip", args.data_root / "DIV2K_train_HR")
    extract_if_needed(args.data_root / "DIV2K_valid_HR.zip", args.data_root / "DIV2K_valid_HR")
    cache = prepare_patch_cache(args.data_root, args.data_root / f"patch_cache_{cfg.crop_size}_{cfg.epochs}ep.pt", cfg)
    train, val = cache["train"], cache["val"]
    set_seed(cfg.seed); init = TinyResidualPhaseNet(); init_state = {k: v.detach().clone() for k, v in init.state_dict().items()}
    physics = evaluate(None, val, cfg); print("physics_only", json.dumps(physics), flush=True)
    histories = {}; final_metrics = {"physics_only": physics}; models = {"physics_only": None}; t0 = time.time()
    for method in args.methods:
        model, hist = train_method(method, init_state, train, val, cfg, args.output); histories[method] = hist; final_metrics[method] = {k: hist[-1][k] for k in physics}; models[method] = model
    rows = [{"method": m, **r} for m, h in histories.items() for r in h]
    with (args.output / "history.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    (args.output / "metrics.json").write_text(json.dumps(final_metrics, indent=2))
    plot_curves(histories, args.output); plot_metric_summary(final_metrics, args.output); save_qualitative(models, val, cfg, args.output); write_report(final_metrics, cfg, args.output)
    (args.output / "runtime.txt").write_text(f"{time.time()-t0:.1f} seconds\n"); print(f"completed in {time.time()-t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
