import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.models import GaussianDiffusion, LatentNorm, VAE, build_denoiser
from src.scripts.utils import (AverageMeter, get_device,
                   load_checkpoint, load_config, save_checkpoint, set_seed)
from src.scripts.utils.data import (
    build_dataloaders,
    load_label_classes,
)
from src.scripts.utils.paths import default_config_path, prepare_run_dir, resolve_run_dir

LEAD_NAMES = ["I", "V1", "V2", "V3", "V4", "V5", "V6", "aVF"]


def load_frozen_vae(cfg, vae_run, device):
    run_dir = resolve_run_dir(cfg, "vae", vae_run)
    ckpt_path = os.path.join(run_dir, "checkpoints", "vae_last.pt")
    ckpt = load_checkpoint(ckpt_path, map_location=device)
    vae = VAE(cfg).to(device)
    vae.load_state_dict(ckpt["model"])
    vae.eval()
    for p in vae.parameters():
        p.requires_grad_(False)
    latent_mean, latent_std = ckpt["latent_mean"], ckpt["latent_std"]
    return vae, LatentNorm(latent_mean, latent_std, device=device)


def _label_names(label_vec, classes, max_names=6):
    indices = np.flatnonzero(np.asarray(label_vec) > 0.5)
    if len(indices) == 0:
        return ["unlabeled"]
    names = [str(classes[i]) for i in indices[:max_names]]
    if len(indices) > max_names:
        names.append(f"+{len(indices) - max_names} more")
    return names


def random_label_batch(train_loader, num_samples, device):
    dataset = train_loader.dataset
    total = len(dataset.labels)
    idx = np.random.choice(total, size=num_samples, replace=total < num_samples)
    labels = np.asarray(dataset.labels[idx], dtype=np.float32)
    return torch.from_numpy(labels).to(device), labels


class StateDictEMA:

    def __init__(self, model, cfg, device, beta=0.995, update_after_step=1000, update_every=1):
        self.model = model
        self.beta = beta
        self.update_after_step = update_after_step
        self.update_every = update_every
        self.step = 0
        self.ema_model = build_denoiser(cfg).to(device)
        self.ema_model.load_state_dict(model.state_dict())
        self.ema_model.eval()
        for p in self.ema_model.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self):
        self.step += 1
        if self.step < self.update_after_step:
            self.ema_model.load_state_dict(self.model.state_dict())
            return
        if self.step % self.update_every != 0:
            return
        src = self.model.state_dict()
        dst = self.ema_model.state_dict()
        for name, ema_value in dst.items():
            value = src[name].detach()
            if torch.is_floating_point(ema_value):
                ema_value.lerp_(value, 1.0 - self.beta)
            else:
                ema_value.copy_(value)


@torch.no_grad()
def save_ddpm200_visual(epoch, cfg, vae, ema_model, latent_norm, train_loader,
                        norm_stats, classes, device, vis_dir, num_samples=4,
                        lead_names=LEAD_NAMES):
    os.makedirs(vis_dir, exist_ok=True)
    labels, label_np = random_label_batch(train_loader, num_samples, device)
    down = 2 ** (len(cfg.vae.ch_mult) - 1)
    latent_len = cfg.data.signal_length // down
    shape = (num_samples, cfg.vae.z_channels, latent_len)

    ema_model.eval()
    sampler = GaussianDiffusion(ema_model, cfg).to(device)
    z = sampler.ddpm_sample(shape, labels, device=device)
    x = vae.decode(latent_norm.decode(z))
    mean, std = norm_stats
    mean_t = torch.from_numpy(mean).to(device=device, dtype=x.dtype)
    std_t = torch.from_numpy(std).to(device=device, dtype=x.dtype)
    x_mv = (x * std_t + mean_t).detach().cpu().numpy()

    sample_names = [_label_names(label_np[i], classes) for i in range(num_samples)]
    sampling_rate_hz = float(cfg.data.sampling_rate_hz)
    time = np.arange(cfg.data.signal_length) / sampling_rate_hz
    fig, axes = plt.subplots(len(lead_names), num_samples, figsize=(18, 12), sharex=True)
    if num_samples == 1:
        axes = axes[:, None]
    for col in range(num_samples):
        title = ", ".join(sample_names[col])
        if len(title) > 70:
            title = title[:67] + "..."
        for row, lead in enumerate(lead_names):
            ax = axes[row, col]
            ax.plot(time, x_mv[col, row], color="#2563eb", linewidth=0.7)
            if col == 0:
                ax.set_ylabel(lead, rotation=0, labelpad=18, fontsize=8)
            if row == 0:
                ax.set_title(title, fontsize=8)
            ax.grid(True, linewidth=0.25, alpha=0.35)
            ax.tick_params(labelsize=7, length=2)
    for ax in axes[-1, :]:
        ax.set_xlabel("time (s)", fontsize=8)
    fig.suptitle("Generated ECG samples", y=0.995, fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.972))
    fig.savefig(os.path.join(vis_dir, f"ddpm{cfg.ldm.timesteps}_epoch_{epoch:04d}.png"), dpi=180)
    fig.savefig(os.path.join(vis_dir, f"ddpm{cfg.ldm.timesteps}_latest.png"), dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=default_config_path())
    parser.add_argument("--vae-run", default=None,
                        help="VAE run dir；不传则用 config 的 vae.run_dir。")
    parser.add_argument("--run-name", default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    lead_names = list(cfg.data.lead_names)
    if args.vae_run is None:
        args.vae_run = cfg.vae.run_dir
    set_seed(cfg.seed)
    device = get_device(cfg.device)
    run_dir = prepare_run_dir(cfg, "dm", args.run_name)
    ckpt_dir = os.path.join(run_dir, "checkpoints")
    vis_dir = os.path.join(run_dir, "vis")

    vae, latent_norm = load_frozen_vae(cfg, args.vae_run, device)
    loaders, norm_stats = build_dataloaders(cfg, splits=("train",), batch_size=cfg.ldm.batch_size)
    train_loader = loaders["train"]
    classes = load_label_classes(cfg)

    denoiser = build_denoiser(cfg).to(device)
    diffusion = GaussianDiffusion(denoiser, cfg).to(device)
    ema = StateDictEMA(
        denoiser, cfg, device,
        beta=cfg.ldm.ema_decay,
        update_after_step=cfg.ldm.ema_update_after,
        update_every=cfg.ldm.ema_update_every,
    )
    opt = torch.optim.AdamW(denoiser.parameters(), lr=cfg.ldm.lr, weight_decay=cfg.ldm.weight_decay)
    total_steps = max(cfg.ldm.epochs * len(train_loader), 1)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=total_steps)

    best_loss = float("inf")
    save_interval = int(cfg.ldm.save_interval)
    sample_interval = int(cfg.ldm.sample_interval)
    sample_num = int(cfg.ldm.sample_num)
    for epoch in range(cfg.ldm.epochs):
        denoiser.train()
        diffusion.train()
        meter = AverageMeter()
        for x, labels in train_loader:
            x, labels = x.to(device), labels.to(device)
            with torch.no_grad():
                z = latent_norm.encode(vae.encode(x).sample())
            loss = diffusion.training_loss(z, labels)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(denoiser.parameters(), cfg.ldm.grad_clip)
            opt.step()
            sched.step()
            ema.update()
            meter.update(loss.item(), x.shape[0])
        if meter.avg < best_loss:
            best_loss = meter.avg
            save_checkpoint(
                os.path.join(ckpt_dir, "dm_best.pt"),
                model=denoiser, ema=ema.ema_model,
                latent_mean=latent_norm.mean.flatten().cpu().tolist(),
                latent_std=latent_norm.std.flatten().cpu().tolist(),
            )
        if epoch % save_interval == 0 or epoch == cfg.ldm.epochs - 1:
            save_checkpoint(
                os.path.join(ckpt_dir, "dm_last.pt"),
                model=denoiser, ema=ema.ema_model,
                latent_mean=latent_norm.mean.flatten().cpu().tolist(),
                latent_std=latent_norm.std.flatten().cpu().tolist(),
            )
        if (epoch + 1) % sample_interval == 0 or epoch == cfg.ldm.epochs - 1:
            save_ddpm200_visual(
                epoch, cfg, vae, ema.ema_model, latent_norm, train_loader,
                norm_stats, classes, device, vis_dir, num_samples=sample_num,
                lead_names=lead_names,
            )

    print(f"Saved weights: {ckpt_dir}", flush=True)


if __name__ == "__main__":
    main()
