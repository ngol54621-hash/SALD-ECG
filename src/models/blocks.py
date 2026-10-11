import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def nonlinearity(x):
    return F.silu(x)


def Normalize(channels, num_groups=32):
    g = min(num_groups, channels)
    while channels % g != 0:
        g -= 1
    return nn.GroupNorm(num_groups=g, num_channels=channels, eps=1e-6, affine=True)


def sinusoidal_embedding(timesteps, dim, max_period=10000):
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(half, device=timesteps.device, dtype=torch.float32) / max(half - 1, 1)
    )
    args = timesteps.float()[:, None] * freqs[None, :]
    emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2 == 1:
        emb = F.pad(emb, (0, 1))
    return emb


class Upsample1d(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv1d(channels, channels, 3, padding=1)

    def forward(self, x):
        x = F.interpolate(x, scale_factor=2.0, mode="nearest")
        return self.conv(x)


class Downsample1d(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv1d(channels, channels, 3, stride=2, padding=1)

    def forward(self, x):
        return self.conv(x)


class ResnetBlock1d(nn.Module):

    def __init__(self, in_channels, out_channels, groups=32, dropout=0.0):
        super().__init__()

        self.norm1 = Normalize(in_channels, groups)
        self.conv1 = nn.Conv1d(in_channels, out_channels, 3, padding=1)

        self.norm2 = Normalize(out_channels, groups)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv1d(out_channels, out_channels, 3, padding=1)

        if in_channels != out_channels:
            self.skip = nn.Conv1d(in_channels, out_channels, 1)
        else:
            self.skip = nn.Identity()

    def forward(self, x):
        h = self.conv1(nonlinearity(self.norm1(x)))
        h = self.norm2(h)
        h = self.conv2(self.dropout(nonlinearity(h)))
        return h + self.skip(x)


class AttnBlock1d(nn.Module):

    def __init__(self, channels, groups=32, num_heads=8, attn_dropout=0.0):
        super().__init__()
        heads = min(num_heads, channels)
        while channels % heads != 0:
            heads -= 1
        self.num_heads = heads
        self.head_dim = channels // heads
        self.attn_dropout = attn_dropout

        self.norm = Normalize(channels, groups)
        self.q = nn.Conv1d(channels, channels, 1)
        self.k = nn.Conv1d(channels, channels, 1)
        self.v = nn.Conv1d(channels, channels, 1)
        self.proj = nn.Conv1d(channels, channels, 1)

    def _to_heads(self, x):
        b, c, l = x.shape
        return x.reshape(b, self.num_heads, self.head_dim, l).transpose(2, 3)

    def forward(self, x):
        h = self.norm(x)
        q, k, v = self._to_heads(self.q(h)), self._to_heads(self.k(h)), self._to_heads(self.v(h))

        dropout_p = self.attn_dropout if self.training else 0.0
        out = F.scaled_dot_product_attention(q, k, v, dropout_p=dropout_p)

        b, _, l, _ = out.shape
        out = out.transpose(2, 3).reshape(b, self.num_heads * self.head_dim, l)
        return x + self.proj(out)


class BiMamba2(nn.Module):

    def __init__(self, dim, d_state=64, d_conv=4, expand=2, headdim=64):
        super().__init__()
        from mamba_ssm import Mamba2
        kw = dict(d_model=dim, d_state=d_state, d_conv=d_conv,
                  expand=expand, headdim=headdim)
        self.fwd = Mamba2(**kw)
        self.bwd = Mamba2(**kw)

    def forward(self, x):
        y_fwd = self.fwd(x)
        y_bwd = self.bwd(x.flip(1)).flip(1)
        return y_fwd + y_bwd
