import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import resampy
from scipy.io import loadmat

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.scripts.utils.common import load_config, project_path


CONFIG_DIR = Path(__file__).resolve().parents[1] / "configs"
STANDARD_LEADS = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]
CLASS_NAMES = ["N", "AF", "I-AVB", "LBBB", "RBBB", "PAC", "PVC", "STD", "STE"]


def read_record(path, sampling_rate, signal_length, lead_names):
    signal = np.asarray(loadmat(path)["ECG"][0, 0]["data"], dtype=np.float32)
    indices = [STANDARD_LEADS.index(name) for name in lead_names]
    data = np.stack([
        resampy.resample(signal[index], 500, sampling_rate).astype(np.float32)
        for index in indices
    ])
    length = data.shape[1]
    if length > signal_length:
        start = (length - signal_length) // 2
        return data[:, start:start + signal_length]
    if length < signal_length:
        fixed = np.zeros((len(lead_names), signal_length), dtype=np.float32)
        fixed[:, :length] = data
        return fixed
    return data


def preprocess(cfg, raw_dir, output_dir, split_file):
    raw_dir, output_dir = Path(raw_dir), Path(output_dir)
    reference = pd.read_csv(raw_dir / "REFERENCE.csv").set_index("Recording")
    splits = pd.read_csv(split_file)
    mat_paths = {path.stem: path for path in raw_dir.rglob("*.mat")}
    arrays_dir = output_dir / cfg.data.array_subdir
    arrays_dir.mkdir(parents=True, exist_ok=True)
    for split in ("train", "val", "test"):
        records = reference.loc[splits.loc[splits["split"] == split, "record_name"]]
        data = np.lib.format.open_memmap(
            arrays_dir / f"{cfg.data.array_prefix}_{split}_data.npy",
            mode="w+", dtype=np.float32,
            shape=(len(records), len(cfg.data.lead_names), cfg.data.signal_length),
        )
        labels = np.lib.format.open_memmap(
            arrays_dir / f"{cfg.data.array_prefix}_{split}_labels.npy",
            mode="w+", dtype=np.float32, shape=(len(records), len(CLASS_NAMES)),
        )
        for index, (record_name, row) in enumerate(records.iterrows()):
            data[index] = read_record(
                mat_paths[record_name], cfg.data.sampling_rate_hz,
                cfg.data.signal_length, cfg.data.lead_names,
            )
            target = np.zeros(len(CLASS_NAMES), dtype=np.float32)
            for column in ("First_label", "Second_label", "Third_label"):
                if pd.notna(row[column]):
                    target[int(row[column]) - 1] = 1
            labels[index] = target
        data.flush()
        labels.flush()
    with (output_dir / "label_metadata.json").open("w", encoding="utf-8") as handle:
        json.dump({"label_classes": {"label_all": CLASS_NAMES}}, handle, indent=2)
    print(f"Saved arrays: {arrays_dir}", flush=True)


def main():
    parser = argparse.ArgumentParser(description="Prepare CPSC2018 arrays using the paper's fixed record splits.")
    parser.add_argument("--config", default=str(CONFIG_DIR / "physiodit_cpsc2018_label_all9.yaml"))
    parser.add_argument("--raw-dir", required=True, help="CPSC2018 directory containing REFERENCE.csv and the original MAT records")
    parser.add_argument("--output-dir", help="Defaults to data.root in the config")
    parser.add_argument("--split-file", default=str(CONFIG_DIR / "cpsc2018_splits.csv"))
    args = parser.parse_args()
    cfg = load_config(args.config)
    preprocess(
        cfg, project_path(args.raw_dir),
        project_path(args.output_dir) if args.output_dir else cfg.data.root,
        project_path(args.split_file),
    )


if __name__ == "__main__":
    main()
