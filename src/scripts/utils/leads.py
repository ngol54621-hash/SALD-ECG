import torch

STANDARD_12_ORDER = ["I", "II", "III", "aVR", "aVL", "aVF",
                     "V1", "V2", "V3", "V4", "V5", "V6"]


def reconstruct_12_from_8(x8):
    I = x8[:, 0]
    V1, V2, V3, V4, V5, V6 = (x8[:, i] for i in range(1, 7))
    aVF = x8[:, 7]

    II = 0.5 * I + aVF
    III = aVF - 0.5 * I
    aVR = -0.75 * I - 0.5 * aVF
    aVL = 0.75 * I - 0.5 * aVF

    leads = [I, II, III, aVR, aVL, aVF, V1, V2, V3, V4, V5, V6]
    return torch.stack(leads, dim=1)
