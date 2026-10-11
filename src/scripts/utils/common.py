import os
import pickle
import random
import types
from pathlib import Path

import numpy as np
import torch
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[3]


class Config(dict):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        for key, value in list(self.items()):
            if isinstance(value, dict):
                self[key] = Config(value)

    def __getattr__(self, name):
        if name in self:
            return self[name]
        return super().__getattribute__(name)

    def __setattr__(self, name, value):
        self[name] = Config(value) if isinstance(value, dict) else value


def load_config(path):
    with open(os.path.expandvars(os.path.expanduser(str(path))), "r", encoding="utf-8") as f:
        value = yaml.safe_load(f)
    return resolve_config_paths(value)


def project_path(value):
    path = Path(os.path.expandvars(os.path.expanduser(str(value))))
    return str((path if path.is_absolute() else PROJECT_ROOT / path).resolve())


def resolve_config_paths(value):
    cfg = Config(value)
    cfg.output_root = project_path(cfg.output_root)
    for section_name, keys in (
        ("data", ("root", "label_metadata", "cache_dir")),
        ("eval", ("hubert_model_dir",)),
    ):
        section = cfg[section_name]
        for key in keys:
            section[key] = project_path(section[key])
    for stage in ("vae", "ldm"):
        section = cfg[stage]
        run = section.run_dir
        if str(run) != "latest" and ("/" in str(run) or str(run).startswith(("~", "$", "."))):
            section["run_dir"] = project_path(run)
    return cfg


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device(name):
    return torch.device(name)


def save_checkpoint(path, **objects):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    payload = {}
    for key, value in objects.items():
        payload[key] = value.state_dict() if hasattr(value, "state_dict") else value
    torch.save(payload, path)


class _CheckpointUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if name == "Config" and module in {
            "utils.common", "generation.utils.common", "scripts.utils.common",
            "src.scripts.utils.common",
        }:
            return Config
        return super().find_class(module, name)


_checkpoint_pickle = types.ModuleType("saldm_checkpoint_pickle")
_checkpoint_pickle.Unpickler = _CheckpointUnpickler
_checkpoint_pickle.load = pickle.load
_checkpoint_pickle.loads = pickle.loads


def load_checkpoint(path, map_location="cpu"):
    checkpoint = torch.load(
        project_path(path), map_location=map_location,
        pickle_module=_checkpoint_pickle, weights_only=False,
    )
    if isinstance(checkpoint, dict) and isinstance(checkpoint.get("config"), dict):
        checkpoint["config"] = Config(checkpoint["config"])
    return checkpoint


class AverageMeter:
    def __init__(self):
        self.reset()

    def reset(self):
        self.sum = 0.0
        self.count = 0

    def update(self, value, n=1):
        self.sum += float(value) * n
        self.count += n

    @property
    def avg(self):
        return self.sum / max(self.count, 1)
