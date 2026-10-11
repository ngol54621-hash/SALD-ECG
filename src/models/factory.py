def build_denoiser(cfg):
    from .physiodit import PhysioDiT
    return PhysioDiT(cfg)
