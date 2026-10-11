import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.blocks import sinusoidal_embedding, BiMamba2


def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class MSM(nn.Module):

    def __init__(self, dim, kernel_sizes=(3, 7, 15)):
        super().__init__()
        self.branches = nn.ModuleList([
            nn.Conv1d(dim, dim, k, padding=k // 2, groups=dim) for k in kernel_sizes
        ])
        self.fuse = nn.Conv1d(dim * len(kernel_sizes), dim, 1)

    def forward(self, x):
        h = x.transpose(1, 2)
        outs = [branch(h) for branch in self.branches]
        h = torch.cat(outs, dim=1)
        h = self.fuse(h)
        return h.transpose(1, 2)


class PeriodFold(nn.Module):

    def __init__(self, dim, top_k=2, p_min=8, p_max=125,
                 flatness_scale=1.0, flatness_bias=3.0):
        super().__init__()
        self.top_k = int(top_k)
        self.p_min = int(p_min)
        self.p_max = int(p_max)
        self.flatness_scale = float(flatness_scale)
        self.flatness_bias = float(flatness_bias)
        self.inter_beat = nn.Conv2d(dim, dim, kernel_size=(3, 1),
                                    padding=(1, 0), groups=dim)

    def forward(self, x):
        B, L, D = x.shape
        h = x.transpose(1, 2).contiguous()
        Xf = torch.fft.rfft(h, dim=-1)
        amp = Xf.abs().mean(dim=(0, 1))
        amp = amp.clone()
        amp[0] = 0.0

        F_bins = amp.shape[0]
        freqs = torch.arange(F_bins, device=x.device)
        with torch.no_grad():
            period_len_all = torch.where(
                freqs > 0, (L / freqs.clamp(min=1)).round().long(),
                torch.zeros_like(freqs),
            )
            valid = (freqs >= 2) & (period_len_all >= self.p_min) & (period_len_all <= self.p_max)

        if valid.sum() == 0:
            return torch.zeros_like(x)

        amp_valid = amp.masked_fill(~valid, 0.0)
        k = int(min(self.top_k, int(valid.sum().item())))
        top_amp, top_bins = torch.topk(amp_valid, k)
        weights = torch.softmax(top_amp, dim=0)

        peak_ratio = amp_valid.max() / (amp[valid].mean() + 1e-8)
        rho = torch.sigmoid(self.flatness_scale * (peak_ratio - self.flatness_bias))

        out = torch.zeros_like(h)
        for j in range(k):
            bin_j = int(top_bins[j].item())
            p = int(period_len_all[bin_j].item())
            if p < 2:
                continue
            num_p = (L + p - 1) // p
            pad = num_p * p - L
            hp = F.pad(h, (0, pad))
            folded = hp.reshape(B, D, num_p, p)
            folded = self.inter_beat(folded)
            unfolded = folded.reshape(B, D, num_p * p)[:, :, :L]
            out = out + weights[j] * unfolded

        out = rho * out
        return out.transpose(1, 2)


class MSR(nn.Module):

    def __init__(self, dim, top_k=2, p_min=8, p_max=125):
        super().__init__()
        self.fold = PeriodFold(dim, top_k=top_k, p_min=p_min, p_max=p_max)
        self.ssm = BiMamba2(dim)

    def forward(self, x):
        return self.fold(x) + self.ssm(x)


class ConditionEmbedder(nn.Module):

    def __init__(self, cfg, dim):
        super().__init__()
        m = cfg.ldm
        self.num_labels = int(cfg.data.num_labels)
        label_embed_dim = int(m.label_embed_dim)
        self.sin_dim = int(m.base_channels)

        self.time_mlp = nn.Sequential(
            nn.Linear(self.sin_dim, dim), nn.SiLU(),
            nn.Linear(dim, dim),
        )
        self.label_mlp = nn.Sequential(
            nn.Linear(self.num_labels, label_embed_dim, bias=False), nn.SiLU(),
            nn.Linear(label_embed_dim, dim, bias=False),
        )

    def forward(self, t, labels):
        time_emb = self.time_mlp(sinusoidal_embedding(t, self.sin_dim))
        c = time_emb + self.label_mlp(labels)
        return c


class PhysioDiTBlock(nn.Module):

    def __init__(self, dim, mlp_ratio=4.0, dropout=0.0,
                 top_k=2, p_min=8, p_max=125):
        super().__init__()

        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.morph = MSM(dim)

        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.rhythm = MSR(
            dim,
            top_k=top_k,
            p_min=p_min,
            p_max=p_max,
        )

        self.norm3 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden), nn.SiLU(), nn.Dropout(dropout),
            nn.Linear(hidden, dim), nn.Dropout(dropout),
        )

        self.adaLN = nn.Sequential(nn.SiLU(), nn.Linear(dim, 9 * dim))

    def forward(self, x, c):
        mods = self.adaLN(c).chunk(9, dim=-1)
        i = 0

        def take():
            nonlocal i
            s, sc, g = mods[i], mods[i + 1], mods[i + 2]
            i += 3
            return s, sc, g

        s, sc, g = take()
        x = x + g.unsqueeze(1) * self.morph(modulate(self.norm1(x), s, sc))
        s, sc, g = take()
        x = x + g.unsqueeze(1) * self.rhythm(modulate(self.norm2(x), s, sc))
        s, sc, g = take()
        x = x + g.unsqueeze(1) * self.mlp(modulate(self.norm3(x), s, sc))
        return x


class FinalLayer1D(nn.Module):
    def __init__(self, dim, out_channels):
        super().__init__()
        self.norm = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(dim, out_channels)
        self.adaLN = nn.Sequential(nn.SiLU(), nn.Linear(dim, 2 * dim))

    def forward(self, x, c):
        shift, scale = self.adaLN(c).chunk(2, dim=-1)
        return self.linear(modulate(self.norm(x), shift, scale))


class PhysioDiT(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        d = cfg.ldm.physiodit
        self.latent_channels = int(cfg.vae.z_channels)
        self.latent_len = int(cfg.data.signal_length // (2 ** (len(cfg.vae.ch_mult) - 1)))
        self.dim = int(d.dim)
        self.depth = int(d.depth)
        self.mlp_ratio = float(d.mlp_ratio)
        self.dropout = float(d.dropout)
        self.top_k = int(d.top_k)
        self.p_min = int(d.p_min)
        self.p_max = int(d.p_max)

        self.in_proj = nn.Linear(self.latent_channels, self.dim)
        self.pos_embed = nn.Parameter(torch.zeros(1, self.latent_len, self.dim))
        self.cond = ConditionEmbedder(cfg, self.dim)
        self.blocks = nn.ModuleList([
            PhysioDiTBlock(self.dim, self.mlp_ratio, self.dropout,
                           self.top_k, self.p_min, self.p_max)
            for _ in range(self.depth)
        ])
        self.final = FinalLayer1D(self.dim, self.latent_channels)
        self._init_weights()

    def _init_weights(self):
        def basic(module):
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(basic)

        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.normal_(self.in_proj.weight, std=0.02)

        for block in self.blocks:
            nn.init.constant_(block.adaLN[-1].weight, 0)
            nn.init.constant_(block.adaLN[-1].bias, 0)
        nn.init.constant_(self.final.adaLN[-1].weight, 0)
        nn.init.constant_(self.final.adaLN[-1].bias, 0)
        nn.init.constant_(self.final.linear.weight, 0)
        nn.init.constant_(self.final.linear.bias, 0)

    def _without_legacy_channel_coherence(self, state_dict):
        upgraded = state_dict.copy()
        for key in list(upgraded):
            if ".chmix." in key:
                upgraded.pop(key)

        kept_prefix = 6 * self.dim
        removed_end = 9 * self.dim
        for index, block in enumerate(self.blocks):
            for name in ("weight", "bias"):
                key = f"blocks.{index}.adaLN.1.{name}"
                value = upgraded.get(key)
                expected = block.adaLN[-1].state_dict()[name]
                if value is not None and value.shape[0] == expected.shape[0] + 3 * self.dim:
                    upgraded[key] = torch.cat((value[:kept_prefix], value[removed_end:]), dim=0)
        return upgraded

    def load_state_dict(self, state_dict, strict=True, assign=False):
        state_dict = self._without_legacy_channel_coherence(state_dict)
        return super().load_state_dict(state_dict, strict=strict, assign=assign)

    def forward(self, x, t, labels):
        h = x.transpose(1, 2)
        h = self.in_proj(h) + self.pos_embed
        c = self.cond(t, labels)
        for block in self.blocks:
            h = block(h, c)
        h = self.final(h, c)
        return h.transpose(1, 2)
