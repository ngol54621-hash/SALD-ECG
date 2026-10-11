import torch
import torch.nn as nn
import torch.nn.functional as F


def cosine_alpha_bar(timesteps, s=0.008):
    steps = torch.arange(timesteps + 1, dtype=torch.float64)
    f = torch.cos(((steps / timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
    alpha_bar = f / f[0]
    betas = 1.0 - (alpha_bar[1:] / alpha_bar[:-1])
    return torch.clamp(betas, 1e-8, 0.999).float()


def _extract(arr, t, shape):
    out = arr.gather(0, t)
    return out.reshape(t.shape[0], *([1] * (len(shape) - 1)))


class GaussianDiffusion(nn.Module):
    def __init__(self, model, cfg):
        super().__init__()
        self.model = model
        self.pred_type = cfg.ldm.pred_type
        self.num_timesteps = cfg.ldm.timesteps
        self.cfg_dropout = float(cfg.ldm.cfg_dropout)
        self.fft_weight = float(cfg.ldm.fft_weight)
        self.guidance_scale = float(cfg.sample.guidance_scale)

        if cfg.ldm.beta_schedule == "cosine":
            betas = cosine_alpha_bar(self.num_timesteps)
        else:
            betas = torch.linspace(1e-4, 0.02, self.num_timesteps)
        alphas = 1.0 - betas
        acp = torch.cumprod(alphas, dim=0)

        self.register_buffer("betas", betas)
        self.register_buffer("alphas_cumprod", acp)
        self.register_buffer("sqrt_acp", torch.sqrt(acp))
        self.register_buffer("sqrt_one_minus_acp", torch.sqrt(1.0 - acp))

    def training_loss(self, x0, labels):
        b = x0.shape[0]
        t = torch.randint(0, self.num_timesteps, (b,), device=x0.device)
        noise = torch.randn_like(x0)
        sqrt_acp = _extract(self.sqrt_acp, t, x0.shape)
        sqrt_om = _extract(self.sqrt_one_minus_acp, t, x0.shape)
        x_t = sqrt_acp * x0 + sqrt_om * noise

        if self.pred_type == "v":
            target = sqrt_acp * noise - sqrt_om * x0
        else:
            target = noise

        if self.training and self.cfg_dropout > 0.0:
            drop = torch.rand(b, device=x0.device) < self.cfg_dropout
            if drop.any():
                labels = labels.clone()
                labels[drop] = 0.0

        pred = self.model(x_t, t, labels)
        loss = F.mse_loss(pred, target)

        if self.fft_weight > 0.0:
            x0_pred, _ = self._x0_eps_from_pred(x_t, t, pred)
            Xp = torch.fft.rfft(x0_pred, dim=-1).abs()
            Xt = torch.fft.rfft(x0, dim=-1).abs()
            snr_w = _extract(self.alphas_cumprod, t, x0.shape)
            fft_per = F.l1_loss(Xp, Xt, reduction="none").mean(dim=(-2, -1))
            loss = loss + self.fft_weight * (snr_w.squeeze(-1).squeeze(-1) * fft_per).mean()

        return loss

    def _x0_eps_from_pred(self, x_t, t, pred):
        sqrt_acp = _extract(self.sqrt_acp, t, x_t.shape)
        sqrt_om = _extract(self.sqrt_one_minus_acp, t, x_t.shape)
        if self.pred_type == "v":
            x0 = sqrt_acp * x_t - sqrt_om * pred
            eps = sqrt_om * x_t + sqrt_acp * pred
        else:
            eps = pred
            x0 = (x_t - sqrt_om * eps) / sqrt_acp
        return x0, eps

    def _guided_pred(self, x_t, t, labels):
        if self.guidance_scale == 1.0:
            return self.model(x_t, t, labels)
        pred_c = self.model(x_t, t, labels)
        pred_u = self.model(x_t, t, torch.zeros_like(labels))
        return pred_u + self.guidance_scale * (pred_c - pred_u)

    @torch.no_grad()
    def ddim_sample(self, shape, labels, ddim_steps=100, eta=0.0, device="cuda"):
        step = max(self.num_timesteps // ddim_steps, 1)
        ts = list(range(0, self.num_timesteps, step))[::-1]
        x_t = torch.randn(shape, device=device)

        for i, t_cur in enumerate(ts):
            t = torch.full((shape[0],), t_cur, device=device, dtype=torch.long)
            pred = self._guided_pred(x_t, t, labels)
            x0, eps = self._x0_eps_from_pred(x_t, t, pred)

            acp_t = self.alphas_cumprod[t_cur]
            t_prev = ts[i + 1] if i + 1 < len(ts) else -1
            acp_prev = self.alphas_cumprod[t_prev] if t_prev >= 0 else torch.tensor(1.0, device=device)

            sigma = eta * torch.sqrt((1 - acp_prev) / (1 - acp_t)) * torch.sqrt(1 - acp_t / acp_prev)
            dir_xt = torch.sqrt(torch.clamp(1 - acp_prev - sigma ** 2, min=0.0)) * eps
            noise = torch.randn_like(x_t) if (eta > 0 and t_prev >= 0) else 0.0
            x_t = torch.sqrt(acp_prev) * x0 + dir_xt + sigma * noise

        return x_t

    @torch.no_grad()
    def ddpm_sample(self, shape, labels, device="cuda"):
        x_t = torch.randn(shape, device=device)

        for t_cur in reversed(range(self.num_timesteps)):
            t = torch.full((shape[0],), t_cur, device=device, dtype=torch.long)
            pred = self._guided_pred(x_t, t, labels)
            x0, _ = self._x0_eps_from_pred(x_t, t, pred)

            beta_t = self.betas[t_cur]
            alpha_t = 1.0 - beta_t
            acp_t = self.alphas_cumprod[t_cur]
            acp_prev = (
                self.alphas_cumprod[t_cur - 1]
                if t_cur > 0
                else torch.tensor(1.0, device=device)
            )

            coef_x0 = torch.sqrt(acp_prev) * beta_t / (1.0 - acp_t)
            coef_xt = torch.sqrt(alpha_t) * (1.0 - acp_prev) / (1.0 - acp_t)
            mean = coef_x0 * x0 + coef_xt * x_t
            var = beta_t * (1.0 - acp_prev) / (1.0 - acp_t)
            noise = torch.randn_like(x_t) if t_cur > 0 else 0.0
            x_t = mean + torch.sqrt(torch.clamp(var, min=1e-20)) * noise

        return x_t
