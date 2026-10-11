import torch
import torch.nn as nn

from .blocks import (Normalize, nonlinearity, ResnetBlock1d, AttnBlock1d,
                     Downsample1d, Upsample1d)


class LatentNorm:

    def __init__(self, mean, std, device="cpu"):
        self.mean = torch.as_tensor(mean, dtype=torch.float32).view(1, -1, 1).to(device)
        self.std = torch.clamp(
            torch.as_tensor(std, dtype=torch.float32).view(1, -1, 1), min=1e-6
        ).to(device)

    def encode(self, z):
        return (z - self.mean) / self.std

    def decode(self, z):
        return z * self.std + self.mean


class DiagonalGaussian:

    def __init__(self, parameters):
        self.mean, self.logvar = torch.chunk(parameters, 2, dim=1)
        self.logvar = torch.clamp(self.logvar, -30.0, 20.0)
        self.std = torch.exp(0.5 * self.logvar)

    def sample(self):
        return self.mean + self.std * torch.randn_like(self.mean)

    def mode(self):
        return self.mean

    def kl(self, free_bits=0.0):
        kl_elem = 0.5 * (self.mean.pow(2) + self.logvar.exp() - self.logvar - 1.0)
        if free_bits > 0.0:
            kl_per_channel = kl_elem.mean(dim=(0, 2))
            kl_per_channel = torch.clamp(kl_per_channel, min=free_bits)
            return kl_per_channel.sum()
        return kl_elem.sum(dim=1).mean()


class Encoder(nn.Module):
    def __init__(self, in_channels, base_channels, ch_mult, num_res_blocks,
                 z_channels, use_mid_attention, groups):
        super().__init__()
        self.conv_in = nn.Conv1d(in_channels, base_channels, 3, padding=1)

        self.down = nn.ModuleList()
        ch = base_channels
        for i, mult in enumerate(ch_mult):
            out_ch = base_channels * mult
            blocks = nn.ModuleList()
            for _ in range(num_res_blocks):
                blocks.append(ResnetBlock1d(ch, out_ch, groups=groups))
                ch = out_ch
            stage = nn.Module()
            stage.blocks = blocks
            stage.downsample = Downsample1d(ch) if i != len(ch_mult) - 1 else nn.Identity()
            self.down.append(stage)

        self.mid_block1 = ResnetBlock1d(ch, ch, groups=groups)
        self.mid_attn = AttnBlock1d(ch, groups) if use_mid_attention else nn.Identity()
        self.mid_block2 = ResnetBlock1d(ch, ch, groups=groups)

        self.norm_out = Normalize(ch, groups)
        self.conv_out = nn.Conv1d(ch, 2 * z_channels, 3, padding=1)

    def forward(self, x):
        h = self.conv_in(x)
        for stage in self.down:
            for blk in stage.blocks:
                h = blk(h)
            h = stage.downsample(h)
        h = self.mid_block2(self.mid_attn(self.mid_block1(h)))
        h = self.conv_out(nonlinearity(self.norm_out(h)))
        return h


class Decoder(nn.Module):
    def __init__(self, out_channels, base_channels, ch_mult, num_res_blocks,
                 z_channels, use_mid_attention, groups):
        super().__init__()
        ch = base_channels * ch_mult[-1]
        self.conv_in = nn.Conv1d(z_channels, ch, 3, padding=1)

        self.mid_block1 = ResnetBlock1d(ch, ch, groups=groups)
        self.mid_attn = AttnBlock1d(ch, groups) if use_mid_attention else nn.Identity()
        self.mid_block2 = ResnetBlock1d(ch, ch, groups=groups)

        self.up = nn.ModuleList()
        for i, mult in enumerate(reversed(ch_mult)):
            out_ch = base_channels * mult
            blocks = nn.ModuleList()
            for _ in range(num_res_blocks + 1):
                blocks.append(ResnetBlock1d(ch, out_ch, groups=groups))
                ch = out_ch
            stage = nn.Module()
            stage.blocks = blocks
            stage.upsample = Upsample1d(ch) if i != len(ch_mult) - 1 else nn.Identity()
            self.up.append(stage)

        self.norm_out = Normalize(ch, groups)
        self.conv_out = nn.Conv1d(ch, out_channels, 3, padding=1)

    def forward(self, z):
        h = self.conv_in(z)
        h = self.mid_block2(self.mid_attn(self.mid_block1(h)))
        for stage in self.up:
            for blk in stage.blocks:
                h = blk(h)
            if not isinstance(stage.upsample, nn.Identity):
                h = stage.upsample(h)
        return self.conv_out(nonlinearity(self.norm_out(h)))


class VAE(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        v = cfg.vae
        c = cfg.data
        self.encoder = Encoder(c.num_leads, v.base_channels, v.ch_mult,
                               v.num_res_blocks, v.z_channels,
                               v.use_mid_attention, v.groupnorm_groups)
        self.decoder = Decoder(c.num_leads, v.base_channels, v.ch_mult,
                               v.num_res_blocks, v.z_channels,
                               v.use_mid_attention, v.groupnorm_groups)

    def encode(self, x):
        return DiagonalGaussian(self.encoder(x))

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x, sample_posterior=True):
        posterior = self.encode(x)
        z = posterior.sample() if sample_posterior else posterior.mode()
        return self.decode(z), posterior
