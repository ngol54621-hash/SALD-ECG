import json
import os

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


DEFAULT_ARRAY_PREFIX = "ptbxl"


def data_array_prefix(cfg):
    return str(cfg.data.array_prefix).strip()


def data_dataset_id(cfg):
    return str(cfg.data.dataset_id).strip()


def array_paths(root, split, array_prefix=DEFAULT_ARRAY_PREFIX):
    return (
        os.path.join(root, f"{array_prefix}_{split}_data.npy"),
        os.path.join(root, f"{array_prefix}_{split}_labels.npy"),
    )


def real_array_root(cfg):
    return os.path.join(cfg.data.root, cfg.data.array_subdir)


def _config_array_paths(cfg, split):
    return array_paths(real_array_root(cfg), split, data_array_prefix(cfg))


def norm_stats_cache_path(cfg):
    return os.path.join(
        cfg.data.cache_dir,
        f"norm_stats_{data_dataset_id(cfg)}_{cfg.data.norm_name}_{cfg.data.num_leads}lead.npz",
    )


def compute_or_load_norm_stats(cfg):
    os.makedirs(cfg.data.cache_dir, exist_ok=True)
    low = float(cfg.data.norm_low_pct)
    high = float(cfg.data.norm_high_pct)
    cache = norm_stats_cache_path(cfg)
    if os.path.exists(cache):
        with np.load(cache) as z:
            return z["mean"].astype(np.float32), z["std"].astype(np.float32)

    data = np.load(_config_array_paths(cfg, "train")[0], mmap_mode="r")
    lo = np.percentile(data, low, axis=(0, 2), keepdims=True)
    hi = np.percentile(data, high, axis=(0, 2), keepdims=True)

    total_sum = np.zeros((1, data.shape[1], 1), dtype=np.float64)
    total_count = 0
    sq_sum = 0.0
    for start in range(0, data.shape[0], 512):
        chunk = np.asarray(data[start:start + 512], dtype=np.float32)
        clipped = np.clip(chunk, lo, hi)
        total_sum += clipped.sum(axis=(0, 2), keepdims=True)
        total_count += clipped.shape[0] * clipped.shape[2]
    mean = (total_sum / max(total_count, 1)).astype(np.float32)

    for start in range(0, data.shape[0], 512):
        chunk = np.asarray(data[start:start + 512], dtype=np.float32)
        clipped = np.clip(chunk, lo, hi)
        centered = clipped - mean
        sq_sum += float(np.sum(centered.astype(np.float64) ** 2))
    global_std = np.sqrt(sq_sum / max(total_count * data.shape[1], 1))
    std = np.full_like(mean, max(float(global_std), 1e-6), dtype=np.float32)
    np.savez(
        cache,
        mean=mean,
        std=std,
    )
    return mean, std


class ECGArrayDataset(Dataset):
    def __init__(
        self, root, split, mean, std, array_prefix=DEFAULT_ARRAY_PREFIX,
    ):
        data_path, label_path = array_paths(root, split, array_prefix)
        self.data = np.load(data_path, mmap_mode="r")
        self.labels = np.load(label_path, mmap_mode="r")
        self.mean = mean.astype(np.float32).reshape(-1, 1)
        self.std = std.astype(np.float32).reshape(-1, 1)

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        x = np.asarray(self.data[idx], dtype=np.float32)
        x = (x - self.mean) / self.std
        y = np.asarray(self.labels[idx], dtype=np.float32)
        return torch.from_numpy(np.ascontiguousarray(x)), torch.from_numpy(np.array(y, copy=True))


def make_loader(
    root, split, mean, std, batch_size, shuffle, num_workers,
    drop_last=False, array_prefix=DEFAULT_ARRAY_PREFIX,
):
    dataset = ECGArrayDataset(root, split, mean, std, array_prefix)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=drop_last,
    )


def build_dataloaders(cfg, splits=("train", "val"), batch_size=None):
    mean, std = compute_or_load_norm_stats(cfg)
    loaders = {}
    bs = batch_size or cfg.vae.batch_size
    root = real_array_root(cfg)
    array_prefix = data_array_prefix(cfg)
    for split in splits:
        loaders[split] = make_loader(
            root, split, mean, std, bs,
            shuffle=(split == "train"),
            num_workers=cfg.data.num_workers,
            drop_last=(split == "train"),
            array_prefix=array_prefix,
        )
    return loaders, (mean, std)


def load_label_classes(cfg):
    with open(cfg.data.label_metadata, "r", encoding="utf-8") as f:
        meta = json.load(f)
    classes = list(meta["label_classes"][cfg.data.label_column])
    return classes
