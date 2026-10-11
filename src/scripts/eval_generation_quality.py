import argparse
import csv
import json
from pathlib import Path
import sys

import numpy as np
from sklearn.decomposition import PCA

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.scripts.ecg_embeddings import extract_embeddings, load_encoder
from src.scripts.metrics.hubert import hubert_coverage, hubert_fid, real_manifold_radii
from src.scripts.utils import get_device, load_config, set_seed
from src.scripts.utils.common import project_path
from src.scripts.utils.data import array_paths, data_array_prefix, real_array_root
from src.scripts.utils.paths import default_config_path


def load_inputs(cfg, synthetic_root, split):
    prefix = data_array_prefix(cfg)
    real_root = real_array_root(cfg)
    train_path, _ = array_paths(real_root, "train", prefix)
    real_path, _ = array_paths(real_root, split, prefix)
    generated_path, _ = array_paths(synthetic_root, split, prefix)
    arrays = {
        "real_train": np.load(train_path, mmap_mode="r", allow_pickle=False),
        "real_evaluation": np.load(real_path, mmap_mode="r", allow_pickle=False),
        "generated": np.load(generated_path, mmap_mode="r", allow_pickle=False),
    }
    return arrays


def output_paths(output):
    path = Path(output)
    if path.suffix.lower() in {".json", ".csv", ".md"}:
        path = path.with_suffix("")
    return tuple(Path(str(path) + extension) for extension in (".json", ".csv", ".md"))


def write_outputs(report, output):
    paths = output_paths(output)
    paths[0].parent.mkdir(parents=True, exist_ok=True)
    with paths[0].open("x") as handle:
        handle.write(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    row = {"model": "SALD", "split": report["split"], "records": report["records"], **report["metrics"]}
    with paths[1].open("x", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row))
        writer.writeheader()
        writer.writerow(row)
    with paths[2].open("x") as handle:
        handle.write(
            "# SALD generation quality\n\n"
            "| Model | ECG-FID ↓ | Coverage ↑ |\n| --- | --- | --- |\n"
            f"| SALD | {row['ECG-FID']:.6f} | {row['Coverage']:.6f} |\n"
        )
    return paths


def main():
    parser = argparse.ArgumentParser(description="Evaluate generated ECGs with the paper's frozen HuBERT-ECG FID/Coverage protocol.")
    parser.add_argument("--config", default=default_config_path())
    parser.add_argument("--hubert-model-dir", help="Downloaded HuBERT-ECG Base directory; overrides eval.hubert_model_dir")
    parser.add_argument("--synthetic-root", required=True, help="Generated sssd_arrays directory containing data and labels")
    parser.add_argument("--real-split", default="test", choices=("val", "test"))
    parser.add_argument("--device", help="Evaluation device, e.g. cuda:0 or cpu; defaults to config.device")
    parser.add_argument("--embedding-batch-size", type=int, help="Overrides eval.embedding_batch_size")
    parser.add_argument("--output", help="New report basename or JSON path; relative to this checkout")
    args = parser.parse_args()

    cfg = load_config(args.config)
    model_dir = project_path(args.hubert_model_dir or cfg.eval.hubert_model_dir)
    synthetic_root = project_path(args.synthetic_root)
    output = project_path(args.output) if args.output else str(
        Path(cfg.output_root) / "evaluation" / f"hubert_{Path(synthetic_root).parent.name}_{args.real_split}.json"
    )
    batch_size = (args.embedding_batch_size if args.embedding_batch_size is not None
                  else int(cfg.eval.embedding_batch_size))
    pca_dim, pca_seed = int(cfg.eval.pca_dim), int(cfg.eval.pca_seed)
    k, metric_batch = int(cfg.eval.coverage_k), int(cfg.eval.metrics_batch_size)
    arrays = load_inputs(cfg, synthetic_root, args.real_split)
    set_seed(int(cfg.seed))
    device = get_device(args.device or cfg.device)
    encoder = load_encoder(model_dir, device)
    features = {}
    for key, array in arrays.items():
        features[key] = extract_embeddings(
            encoder, array,
            batch_size=batch_size, device=device,
        )

    pca = PCA(n_components=pca_dim, svd_solver="randomized", whiten=False, random_state=pca_seed)
    pca.fit(features["real_train"])
    real = pca.transform(features["real_evaluation"]).astype(np.float32)
    generated = pca.transform(features["generated"]).astype(np.float32)
    radii = real_manifold_radii(real, k=k, device=device)
    coverage = hubert_coverage(real, generated, radii, batch_size=metric_batch, device=device)
    metrics = {"ECG-FID": hubert_fid(real, generated), "Coverage": coverage}
    report = {
        "model": "SALD", "split": args.real_split, "records": len(real),
        "metrics": metrics,
    }
    paths = write_outputs(report, output)
    print(f"ECG-FID={metrics['ECG-FID']:.6f} Coverage={coverage:.6f}", flush=True)
    print("Saved: " + ", ".join(str(path) for path in paths), flush=True)


if __name__ == "__main__":
    main()
