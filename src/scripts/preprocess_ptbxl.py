import argparse
import ast
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import resampy
import wfdb

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.scripts.utils.common import load_config, project_path
from src.scripts.utils.paths import default_config_path


def read_record(path, sampling_rate, signal_length, lead_names):
    signal, header = wfdb.rdsamp(str(path))
    source_fs = int(header["fs"])
    target_length = int(len(signal) * sampling_rate / source_fs)
    data = np.zeros((target_length, len(lead_names)), dtype=np.float32)
    lead_indices = {name.lower(): index for index, name in enumerate(lead_names)}
    for index, name in enumerate(header["sig_name"]):
        if name.lower() in lead_indices:
            data[:, lead_indices[name.lower()]] = resampy.resample(
                signal[:, index], source_fs, sampling_rate,
            ).astype(np.float32)
    if len(data) != signal_length:
        fixed = np.zeros((signal_length, len(lead_names)), dtype=np.float32)
        length = min(signal_length, len(data))
        fixed[:length] = data[:length]
        data = fixed
    return data.T


def preprocess(cfg, raw_dir, output_dir):
    raw_dir, output_dir = Path(raw_dir), Path(output_dir)
    database = pd.read_csv(raw_dir / "ptbxl_database.csv", index_col="ecg_id")
    database["scp_codes"] = database["scp_codes"].apply(
        lambda value: ast.literal_eval(str(value).replace("nan", "None"))
    )
    classes = sorted({label for codes in database["scp_codes"] for label in codes})
    label_indices = {label: index for index, label in enumerate(classes)}
    database["split"] = np.where(
        database["strat_fold"] == 10, "test",
        np.where(database["strat_fold"] == 9, "val", "train"),
    )
    arrays_dir = output_dir / cfg.data.array_subdir
    arrays_dir.mkdir(parents=True, exist_ok=True)
    for split in ("train", "val", "test"):
        records = database.loc[database["split"] == split]
        data = np.lib.format.open_memmap(
            arrays_dir / f"{cfg.data.array_prefix}_{split}_data.npy",
            mode="w+", dtype=np.float32,
            shape=(len(records), len(cfg.data.lead_names), cfg.data.signal_length),
        )
        labels = np.lib.format.open_memmap(
            arrays_dir / f"{cfg.data.array_prefix}_{split}_labels.npy",
            mode="w+", dtype=np.float32, shape=(len(records), len(classes)),
        )
        for index, (_, row) in enumerate(records.iterrows()):
            data[index] = read_record(
                raw_dir / row["filename_hr"], cfg.data.sampling_rate_hz,
                cfg.data.signal_length, cfg.data.lead_names,
            )
            target = np.zeros(len(classes), dtype=np.float32)
            for label in row["scp_codes"]:
                target[label_indices[label]] = 1
            labels[index] = target
        data.flush()
        labels.flush()
    with (output_dir / "label_metadata.json").open("w", encoding="utf-8") as handle:
        json.dump({"label_classes": {"label_all": classes}}, handle, indent=2)
    print(f"Saved arrays: {arrays_dir}", flush=True)


def main():
    parser = argparse.ArgumentParser(description="Prepare the paper's PTB-XL arrays from 500 Hz WFDB records.")
    parser.add_argument("--config", default=default_config_path())
    parser.add_argument("--raw-dir", required=True, help="PTB-XL directory containing ptbxl_database.csv and records500")
    parser.add_argument("--output-dir", help="Defaults to data.root in the config")
    args = parser.parse_args()
    cfg = load_config(args.config)
    preprocess(cfg, project_path(args.raw_dir), project_path(args.output_dir) if args.output_dir else cfg.data.root)


if __name__ == "__main__":
    main()
