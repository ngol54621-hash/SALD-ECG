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

from src.models import VAE
from src.scripts.utils import AverageMeter, get_device, load_config, save_checkpoint, set_seed
from src.scripts.utils.data import build_dataloaders
from src.scripts.utils.paths import default_config_path, prepare_run_dir


LEAD_NAMES = ["I", "V1", "V2", "V3", "V4", "V5", "V6", "aVF"]


def multi_resolution_stft_loss(y, x, fft_sizes):
    b, c, length = x.shape
    y = y.reshape(b * c, length)
    x = x.reshape(b * c, length)
    total = 0.0
    for n_fft in fft_sizes:
        hop = n_fft // 4
        win = torch.hann_window(n_fft, device=x.device)
        sx = torch.stft(x, n_fft=n_fft, hop_length=hop, win_length=n_fft,
                        window=win, return_complex=True)
        sy = torch.stft(y, n_fft=n_fft, hop_length=hop, win_length=n_fft,
                        window=win, return_complex=True)
        mx, my = sx.abs(), sy.abs()
        sc = torch.linalg.norm(my - mx) / (torch.linalg.norm(mx) + 1e-8)
        mag = torch.mean(torch.abs(torch.log(my + 1e-7) - torch.log(mx + 1e-7)))
        total = total + sc + mag
    return total / len(fft_sizes)


def vae_loss(cfg, x, x_rec, posterior, beta):
    v = cfg.vae
    l1 = torch.abs(x_rec - x).mean()
    mse = torch.mean((x_rec - x) ** 2)
    stft = multi_resolution_stft_loss(x_rec, x, v.stft_fft_sizes)
    recon = v.l1_weight * l1 + v.mse_weight * mse + v.stft_weight * stft
    kl = posterior.kl(free_bits=v.free_bits)
    loss = recon + beta * kl
    return loss, float(mse.item())


@torch.no_grad()
def compute_latent_norm_stats(vae, loader, device, max_batches=40):
    chunks = []
    for i, (x, _) in enumerate(loader):
        z = vae.encode(x.to(device)).sample()
        chunks.append(z.transpose(0, 1).reshape(z.shape[1], -1))
        if i + 1 >= max_batches:
            break
    cat = torch.cat(chunks, dim=1)
    return cat.mean(dim=1).cpu().tolist(), cat.std(dim=1).clamp_min(1e-6).cpu().tolist()


def save_reconstruction_visual(
    epoch, x, x_rec, norm_stats, vis_dir, sampling_rate_hz=100.0,
    lead_names=LEAD_NAMES, max_samples=4,
):
    os.makedirs(vis_dir, exist_ok=True)
    mean, std = norm_stats
    x = x.detach().cpu()[:max_samples]
    x_rec = x_rec.detach().cpu()[:max_samples]
    mean_t = torch.from_numpy(mean).to(dtype=x.dtype)
    std_t = torch.from_numpy(std).to(dtype=x.dtype)
    x_mv = x * std_t + mean_t
    rec_mv = x_rec * std_t + mean_t
    time = np.arange(x.shape[-1]) / float(sampling_rate_hz)

    fig, axes = plt.subplots(len(lead_names), x.shape[0], figsize=(16, 11), sharex=True)
    if x.shape[0] == 1:
        axes = axes[:, None]
    for col in range(x.shape[0]):
        mse = torch.mean((x_rec[col] - x[col]) ** 2).item()
        for row, lead in enumerate(lead_names):
            ax = axes[row, col]
            ax.plot(time, x_mv[col, row].numpy(), color="#111827", linewidth=0.7,
                    label="real" if row == 0 and col == 0 else None)
            ax.plot(time, rec_mv[col, row].numpy(), color="#dc2626", linewidth=0.65, alpha=0.9,
                    label="recon" if row == 0 and col == 0 else None)
            if col == 0:
                ax.set_ylabel(lead, rotation=0, labelpad=18, fontsize=8)
            if row == 0:
                ax.set_title(f"sample {col + 1} | norm MSE {mse:.4f}", fontsize=9)
            ax.grid(True, linewidth=0.25, alpha=0.35)
            ax.tick_params(labelsize=7, length=2)
    for ax in axes[-1, :]:
        ax.set_xlabel("time (s)", fontsize=8)
    fig.legend(loc="upper center", ncol=2, frameon=False, fontsize=9)
    fig.suptitle("VAE reconstruction", y=0.995, fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.975))
    fig.savefig(os.path.join(vis_dir, f"reconstruction_epoch_{epoch:04d}.png"), dpi=180)
    fig.savefig(os.path.join(vis_dir, "reconstruction_latest.png"), dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=default_config_path())
    parser.add_argument("--run-name", default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    lead_names = list(cfg.data.lead_names)
    set_seed(cfg.seed)
    device = get_device(cfg.device)
    run_dir = prepare_run_dir(cfg, "vae", args.run_name)
    ckpt_dir = os.path.join(run_dir, "checkpoints")
    vis_dir = os.path.join(run_dir, "vis")

    loaders, norm_stats = build_dataloaders(cfg, splits=("train", "val"), batch_size=cfg.vae.batch_size)
    train_loader, val_loader = loaders["train"], loaders["val"]

    vae = VAE(cfg).to(device)
    opt = torch.optim.Adam(vae.parameters(), lr=cfg.vae.lr, weight_decay=cfg.vae.weight_decay)
    total_steps = max(cfg.vae.epochs * len(train_loader), 1)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=total_steps)

    step, best_val = 0, float("inf")
    for epoch in range(cfg.vae.epochs):
        vae.train()
        beta = 0.0
        for x, _ in train_loader:
            x = x.to(device)
            beta = cfg.vae.beta_max * min(1.0, step / max(cfg.vae.kl_warmup_steps, 1))
            x_rec, posterior = vae(x, sample_posterior=True)
            loss, _ = vae_loss(cfg, x, x_rec, posterior, beta)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(vae.parameters(), cfg.vae.grad_clip)
            opt.step()
            sched.step()
            step += 1

        if epoch % cfg.vae.val_interval == 0 or epoch == cfg.vae.epochs - 1:
            vae.eval()
            val_mse = AverageMeter()
            vis_batch = None
            with torch.no_grad():
                for x, _ in val_loader:
                    x = x.to(device)
                    x_rec, posterior = vae(x, sample_posterior=False)
                    _, mse = vae_loss(cfg, x, x_rec, posterior, beta)
                    val_mse.update(mse, x.shape[0])
                    if vis_batch is None:
                        vis_batch = (x.detach().cpu(), x_rec.detach().cpu())
            if vis_batch is not None:
                save_reconstruction_visual(
                    epoch, vis_batch[0], vis_batch[1], norm_stats, vis_dir,
                    sampling_rate_hz=float(cfg.data.sampling_rate_hz),
                    lead_names=lead_names,
                )
            if val_mse.avg < best_val:
                best_val = val_mse.avg
                save_checkpoint(os.path.join(ckpt_dir, "vae_best.pt"), model=vae)

    vae.eval()
    lat_mean, lat_std = compute_latent_norm_stats(vae, train_loader, device)
    save_checkpoint(
        os.path.join(ckpt_dir, "vae_last.pt"),
        model=vae,
        latent_mean=lat_mean,
        latent_std=lat_std,
    )
    print(f"Saved weights: {ckpt_dir}", flush=True)


if __name__ == "__main__":
    main()
