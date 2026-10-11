import importlib.util
from pathlib import Path
import sys

import numpy as np
import torch

HUBERT_FEATURE_BATCH = 64
HUBERT_WINDOW_SAMPLES = 500
HUBERT_FLAT_WINDOW_SAMPLES = 12 * HUBERT_WINDOW_SAMPLES


def load_encoder(model_dir, device="cpu"):
    root = Path(model_dir).resolve()
    module_name = "_saldm_hubert_ecg"
    spec = importlib.util.spec_from_file_location(module_name, root / "hubert_ecg.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    model = module.HuBERTECG.from_pretrained(root, local_files_only=True)
    model.to(device).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


def sssd8_to_standard12(x):
    arr = np.asarray(x, dtype=np.float32)
    squeeze = arr.ndim == 2
    if squeeze:
        arr = arr[None]
    if arr.shape[1] == 8:
        arr8 = arr
    elif arr.shape[2] == 8:
        arr8 = np.transpose(arr, (0, 2, 1))

    lead_i = arr8[:, 0]
    v1, v2, v3, v4, v5, v6 = [arr8[:, idx] for idx in range(1, 7)]
    avf = arr8[:, 7]
    lead_ii = avf + 0.5 * lead_i
    lead_iii = lead_ii - lead_i
    avr = -0.5 * (lead_i + lead_ii)
    avl = lead_i - 0.5 * lead_ii
    out = np.stack([lead_i, lead_ii, lead_iii, avr, avl, avf, v1, v2, v3, v4, v5, v6], axis=1)
    return out[0] if squeeze else out


@torch.inference_mode()
def extract_embeddings(model, array, batch_size=HUBERT_FEATURE_BATCH, device="cpu"):
    model.eval()
    features = []
    for start in range(0, len(array), batch_size):
        values = np.asarray(array[start:start + batch_size], dtype=np.float32)
        values_12 = sssd8_to_standard12(values)
        lead_min = values_12.min(axis=2, keepdims=True)
        lead_max = values_12.max(axis=2, keepdims=True)
        lead_range = lead_max - lead_min
        scaled = np.zeros_like(values_12, dtype=np.float32)
        np.divide(
            2.0 * (values_12 - lead_min),
            lead_range,
            out=scaled,
            where=lead_range > np.finfo(np.float32).eps,
        )
        scaled -= 1.0
        scaled = np.where(lead_range > np.finfo(np.float32).eps, scaled, 0.0)

        first = scaled[:, :, :HUBERT_WINDOW_SAMPLES].reshape(
            len(values), HUBERT_FLAT_WINDOW_SAMPLES
        )
        second = scaled[:, :, HUBERT_WINDOW_SAMPLES:].reshape(
            len(values), HUBERT_FLAT_WINDOW_SAMPLES
        )
        batch_windows = np.ascontiguousarray(
            np.concatenate((first, second), axis=0), dtype=np.float32
        )
        outputs = model(torch.from_numpy(batch_windows).to(device, non_blocking=True))
        pooled = outputs.last_hidden_state.mean(dim=1).float()
        record_features = (pooled[:len(values)] + pooled[len(values):]) * 0.5
        features.append(record_features.cpu().numpy())

    result = np.concatenate(features, axis=0).astype(np.float32, copy=False)
    return result
