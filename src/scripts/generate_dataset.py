import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.multiprocessing as mp

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.models import GaussianDiffusion, LatentNorm, VAE, build_denoiser
from src.scripts.utils import get_device, load_checkpoint, load_config, set_seed
from src.scripts.utils.data import (
    array_paths,
    compute_or_load_norm_stats,
    data_array_prefix,
    data_dataset_id,
    real_array_root,
)
from src.scripts.utils.paths import default_config_path, prepare_run_dir, resolve_run_dir

def _parse_device_token(token):
    token = token.strip()
    if token.isdigit():
        return torch.device(f"cuda:{token}")
    return torch.device(token)


def _normalize_device(device):
    if device.type == "cuda":
        idx = 0 if device.index is None else device.index
        return torch.device(f"cuda:{idx}")
    return device


def resolve_generation_devices(cfg_device, devices_arg=None):
    if devices_arg:
        return [_normalize_device(_parse_device_token(item))
                for item in devices_arg.split(",") if item.strip()]
    if str(cfg_device) == "cuda" and torch.cuda.is_available():
        return [torch.device(f"cuda:{idx}") for idx in range(min(torch.cuda.device_count(), 2))]
    return [_normalize_device(get_device(cfg_device))]


def output_paths(out_dir, split, array_prefix="ptbxl"):
    return (
        os.path.join(out_dir, f"{array_prefix}_{split}_data.npy"),
        os.path.join(out_dir, f"{array_prefix}_{split}_labels.npy"),
        os.path.join(out_dir, f"{array_prefix}_{split}_done.npy"),
    )


def plan_pending_indices(done, max_samples=None):
    pending = np.flatnonzero(~np.asarray(done, dtype=np.bool_))
    if max_samples is not None:
        pending = pending[:max_samples]
    return pending


def shard_pending_indices(pending, rank, world_size):
    return np.asarray(pending, dtype=np.int64)[rank::world_size]


def resolve_model_dirs(cfg, vae_run, dm_run):
    return resolve_run_dir(cfg, "vae", vae_run), resolve_run_dir(cfg, "dm", dm_run)


def load_models(cfg, vae_run, dm_run, device):
    vae_dir, dm_dir = resolve_model_dirs(cfg, vae_run, dm_run)
    vae_ckpt = load_checkpoint(os.path.join(vae_dir, "checkpoints", "vae_last.pt"), map_location=device)
    vae = VAE(cfg).to(device)
    vae.load_state_dict(vae_ckpt["model"])
    vae.eval()

    dm_ckpt = load_checkpoint(os.path.join(dm_dir, "checkpoints", "dm_last.pt"), map_location=device)
    denoiser = build_denoiser(cfg).to(device)
    denoiser.load_state_dict(dm_ckpt["ema"])
    denoiser.eval()
    diffusion = GaussianDiffusion(denoiser, cfg).to(device)
    lm = dm_ckpt.get("latent_mean")
    ls = dm_ckpt.get("latent_std")
    if lm is None or ls is None:
        lm, ls = vae_ckpt["latent_mean"], vae_ckpt["latent_std"]
    return vae, diffusion, LatentNorm(lm, ls, device=device)


def prepare_outputs(cfg, split, out_dir, overwrite=False):
    array_prefix = data_array_prefix(cfg)
    src_data_path, src_label_path = array_paths(real_array_root(cfg), split, array_prefix)
    src_data = np.load(src_data_path, mmap_mode="r")
    src_labels = np.load(src_label_path, mmap_mode="r")
    data_path, label_path, done_path = output_paths(out_dir, split, array_prefix)
    if overwrite:
        for path in (data_path, label_path, done_path):
            if os.path.exists(path):
                os.remove(path)
    shape = (src_data.shape[0], cfg.data.num_leads, cfg.data.signal_length)
    if os.path.exists(data_path):
        out_data = np.load(data_path, mmap_mode="r+")
    else:
        out_data = np.lib.format.open_memmap(data_path, mode="w+", dtype=np.float32, shape=shape)
    if os.path.exists(label_path):
        out_labels = np.load(label_path, mmap_mode="r+")
    else:
        out_labels = np.lib.format.open_memmap(label_path, mode="w+", dtype=np.float32, shape=src_labels.shape)
        out_labels[:] = src_labels[:]
        out_labels.flush()
    if os.path.exists(done_path):
        done = np.load(done_path, mmap_mode="r+")
    else:
        done = np.lib.format.open_memmap(done_path, mode="w+", dtype=np.bool_, shape=(src_data.shape[0],))
        done[:] = False
        done.flush()
    return src_labels, out_data, out_labels, done


def initialize_split_outputs(cfg, split, out_dir, overwrite=False, max_samples=None):
    src_labels, out_data, out_labels, done = prepare_outputs(cfg, split, out_dir, overwrite)
    pending = plan_pending_indices(done, max_samples)
    out_data.flush()
    out_labels.flush()
    done.flush()
    del src_labels, out_data, out_labels, done
    return pending


@torch.no_grad()
def generate_batch(cfg, vae, diffusion, labels, latent_norm, norm_stats, sampler, steps, eta, device):
    bsz = labels.shape[0]
    down = 2 ** (len(cfg.vae.ch_mult) - 1)
    latent_len = cfg.data.signal_length // down
    shape = (bsz, cfg.vae.z_channels, latent_len)
    if sampler == "ddpm":
        z = diffusion.ddpm_sample(shape, labels, device=device)
    elif sampler == "ddim":
        z = diffusion.ddim_sample(shape, labels, ddim_steps=steps, eta=eta, device=device)
    x = vae.decode(latent_norm.decode(z))
    mean, std = norm_stats
    mean_t = torch.from_numpy(mean).to(device=device, dtype=x.dtype)
    std_t = torch.from_numpy(std).to(device=device, dtype=x.dtype)
    return (x * std_t + mean_t).detach().cpu().numpy().astype(np.float32)


def run_split(cfg, split, out_arrays, args, vae, diffusion, latent_norm, norm_stats, device):
    src_labels, out_data, out_labels, done = prepare_outputs(cfg, split, out_arrays, args.overwrite)
    pending = np.flatnonzero(~done[:])
    if args.max_samples is not None:
        pending = pending[:args.max_samples]
    for start in range(0, len(pending), args.batch_size):
        idx = pending[start:start + args.batch_size]
        label_np = np.asarray(src_labels[idx], dtype=np.float32)
        labels = torch.from_numpy(label_np).to(device)
        batch = generate_batch(cfg, vae, diffusion, labels, latent_norm, norm_stats,
                               args.sampler, args.steps, args.eta, device)
        out_data[idx] = batch
        out_labels[idx] = label_np
        done[idx] = True
        out_data.flush(); out_labels.flush(); done.flush()


def run_split_worker(cfg, split, out_arrays, args, vae, diffusion, latent_norm,
                     norm_stats, device, rank, world_size, planned_pending):
    array_prefix = data_array_prefix(cfg)
    _, src_label_path = array_paths(real_array_root(cfg), split, array_prefix)
    src_labels = np.load(src_label_path, mmap_mode="r")
    data_path, label_path, done_path = output_paths(out_arrays, split, array_prefix)
    out_data = np.load(data_path, mmap_mode="r+")
    out_labels = np.load(label_path, mmap_mode="r+")
    done = np.load(done_path, mmap_mode="r+")

    pending = shard_pending_indices(planned_pending, rank, world_size)

    for start in range(0, len(pending), args.batch_size):
        idx = pending[start:start + args.batch_size]
        label_np = np.asarray(src_labels[idx], dtype=np.float32)
        labels = torch.from_numpy(label_np).to(device)
        batch = generate_batch(cfg, vae, diffusion, labels, latent_norm, norm_stats,
                               args.sampler, args.steps, args.eta, device)
        out_data[idx] = batch
        out_labels[idx] = label_np
        done[idx] = True
        out_data.flush(); out_labels.flush(); done.flush()


def worker_main(
    rank, world_size, device_names, cfg, splits, out_arrays, args, norm_stats,
    seed, pending_by_split,
):
    device = torch.device(device_names[rank])
    if device.type == "cuda":
        torch.cuda.set_device(device)
    set_seed(seed + rank * 100003)
    vae, diffusion, latent_norm = load_models(cfg, args.vae_run, args.dm_run, device)
    if args.guidance_scale is not None:
        diffusion.guidance_scale = float(args.guidance_scale)
    for split in splits:
        run_split_worker(cfg, split, out_arrays, args, vae, diffusion, latent_norm,
                         norm_stats, device, rank, world_size, pending_by_split[split])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=default_config_path())
    parser.add_argument("--vae-run", default=None,
                        help="不传则用 config 的 vae.run_dir。")
    parser.add_argument(
        "--dm-run",
        default=None,
        help="不传则用 config 的 ldm.run_dir。",
    )
    parser.add_argument("--run-name", default=None)
    parser.add_argument(
        "--resume-run",
        default=None,
        help="复用已有 generate run（名称或绝对路径）；可配合 --splits 和 --overwrite。",
    )
    parser.add_argument("--splits", default="train,val,test")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--sampler", choices=["ddpm", "ddim"], default=None)
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--eta", type=float, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--guidance-scale", type=float, default=None,
                        help="CFG 引导强度，覆盖 config 的 sample.guidance_scale。")
    parser.add_argument("--devices", default=None,
                        help="逗号分隔的生成设备，如 cuda:0,cuda:1 或 0,1；默认最多使用两张可见 CUDA 卡。")
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.vae_run is None:
        args.vae_run = cfg.vae.run_dir
    if args.dm_run is None:
        args.dm_run = cfg.ldm.run_dir
    seed = cfg.seed if args.seed is None else args.seed
    set_seed(seed)
    args.sampler = str(cfg.sample.sampler if args.sampler is None else args.sampler).lower()
    args.steps = cfg.sample.ddim_steps if args.steps is None else args.steps
    args.eta = cfg.sample.ddim_eta if args.eta is None else args.eta
    splits = [s.strip() for s in args.splits.split(",") if s.strip()]
    devices = resolve_generation_devices(cfg.device, args.devices)
    device_names = [str(device) for device in devices]
    norm_stats = compute_or_load_norm_stats(cfg)

    guidance_scale = float(args.guidance_scale if args.guidance_scale is not None
                           else cfg.sample.guidance_scale)
    if args.resume_run is None and args.run_name is None:
        sampler_tag = args.sampler
        if args.sampler == "ddim":
            sampler_tag = f"ddim{int(args.steps)}"
        guidance_tag = f"{guidance_scale:g}".replace(".", "p")
        args.run_name = (
            f"{data_dataset_id(cfg)}_ecg_sald_{sampler_tag}_"
            f"g{guidance_tag}_seed{int(seed)}"
        )

    if args.resume_run is not None:
        run_dir = resolve_run_dir(cfg, "generate", args.resume_run)
    else:
        run_dir = prepare_run_dir(cfg, "generate", args.run_name)
    out_arrays = os.path.join(run_dir, "sssd_arrays")
    os.makedirs(out_arrays, exist_ok=True)

    if len(devices) == 1:
        device = devices[0]
        if device.type == "cuda":
            torch.cuda.set_device(device)
        vae, diffusion, latent_norm = load_models(cfg, args.vae_run, args.dm_run, device)
        if args.guidance_scale is not None:
            diffusion.guidance_scale = float(args.guidance_scale)
        for split in splits:
            run_split(cfg, split, out_arrays, args, vae, diffusion, latent_norm, norm_stats, device)
    else:
        pending_by_split = {}
        for split in splits:
            planned_pending = initialize_split_outputs(
                cfg, split, out_arrays, args.overwrite, args.max_samples,
            )
            pending_by_split[split] = planned_pending
        mp.spawn(
            worker_main,
            args=(
                len(devices), device_names, cfg, splits, out_arrays, args,
                norm_stats, seed, pending_by_split,
            ),
            nprocs=len(devices),
            join=True,
        )
    print(f"Done. Output directory: {out_arrays}", flush=True)


if __name__ == "__main__":
    main()
