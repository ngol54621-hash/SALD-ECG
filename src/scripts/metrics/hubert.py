import numpy as np
import torch
from scipy import linalg as scipy_linalg

COVERAGE_K = 5
COVERAGE_BATCH = 256


def hubert_fid(enc_real, enc_gen) -> float:
    real = np.asarray(enc_real, dtype=np.float64)
    gen = np.asarray(enc_gen, dtype=np.float64)
    mu_r, mu_g = real.mean(0), gen.mean(0)
    sig_r = np.cov(real, rowvar=False)
    sig_g = np.cov(gen, rowvar=False)
    diff = mu_r - mu_g
    covmean, _ = scipy_linalg.sqrtm(sig_r @ sig_g, disp=False)
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    fid = float(diff @ diff + np.trace(sig_r + sig_g - 2.0 * covmean))
    return max(fid, 0.0)


def real_manifold_radii(enc_real, k=COVERAGE_K, device="cpu"):
    real = torch.from_numpy(np.asarray(enc_real, dtype=np.float32)).to(device)
    distances = torch.cdist(real, real, p=2)
    distances.fill_diagonal_(float("inf"))
    radii = distances.kthvalue(k, dim=1).values
    return radii.cpu().numpy().astype(np.float32, copy=False)


def hubert_coverage(
    enc_real,
    enc_gen,
    real_radii,
    *,
    batch_size=COVERAGE_BATCH,
    device="cpu",
):
    real = torch.from_numpy(np.asarray(enc_real, dtype=np.float32)).to(device)
    generated = np.asarray(enc_gen, dtype=np.float32)
    radii = torch.from_numpy(np.asarray(real_radii, dtype=np.float32)).to(device)

    nearest_generated = torch.full(
        (len(real),), float("inf"), dtype=real.dtype, device=device
    )
    for start in range(0, len(generated), batch_size):
        batch = torch.from_numpy(
            np.ascontiguousarray(generated[start:start + batch_size])
        ).to(device)
        distances = torch.cdist(batch, real, p=2)
        nearest_generated = torch.minimum(nearest_generated, distances.min(dim=0).values)

    coverage = float((nearest_generated <= radii).float().mean().item())
    return coverage
