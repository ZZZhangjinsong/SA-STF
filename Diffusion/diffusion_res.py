import math
import torch
from torch import device, nn, einsum
import torch.nn.functional as F
from inspect import isfunction
from functools import partial
import numpy as np
from tqdm import tqdm
from Diffusion.loss_perceptual import PerceptualLoss
from collections import namedtuple 

ModelResPrediction = namedtuple(
    'ModelResPrediction', ['pred_res', 'pred_noise', 'pred_x_start'])
def exists(x):
    return x is not None


def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d

def extract(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))

def identity(t, *args, **kwargs):
    return t

def gen_coefficients(timesteps, schedule="increased", sum_scale=1, ratio=1):
    if schedule == "increased":
        x = np.linspace(0, 1, timesteps, dtype=np.float32)
        y = x**ratio
        y = torch.from_numpy(y)
        y_sum = y.sum()
        alphas = y/y_sum
    elif schedule == "decreased":
        x = np.linspace(0, 1, timesteps, dtype=np.float32)
        y = x**ratio
        y = torch.from_numpy(y)
        y_sum = y.sum()
        y = torch.flip(y, dims=[0])
        alphas = y/y_sum
    elif schedule == "average":
        alphas = torch.full([timesteps], 1/timesteps, dtype=torch.float32)
    elif schedule == "normal":
        sigma = 1.0
        mu = 0.0
        x = np.linspace(-3+mu, 3+mu, timesteps, dtype=np.float32)
        y = np.e**(-((x-mu)**2)/(2*(sigma**2)))/(np.sqrt(2*np.pi)*(sigma**2))
        y = torch.from_numpy(y)
        alphas = y/y.sum()
    else:
        alphas = torch.full([timesteps], 1/timesteps, dtype=torch.float32)
    assert (alphas.sum()-1).abs() < 1e-6

    return alphas*sum_scale

class ResidualDiffusion(nn.Module):
    def __init__(
        self,
        model_x0,
        total_epoch,
        timesteps=100,
        sampling_steps=50,
        ddim_sampling_eta=0.,
        device= device
    ):
        super().__init__()
        self.timesteps = timesteps
        self.sampling_steps = sampling_steps
        self.model_x0 = model_x0
        self.device = device
        self.total_epoch = total_epoch
        self.warm_epoch = total_epoch // 2
        self.x0_recon_loss_func = nn.L1Loss(reduction='sum').to(self.device)
        blocks = [0, 1, 2]  # 使用VGG16的前3个卷积块
        weights = [1.0, 0.8, 0.6]  # 为每个块设置不同的权重
        self.loss_pre = PerceptualLoss(blocks, weights, device)
        self.ddim_sampling_eta = ddim_sampling_eta

        if self.sampling_steps == self.timesteps:
            convert_to_ddim = False
        else:
            convert_to_ddim = True
        self.is_ddim_sampling = convert_to_ddim
        if convert_to_ddim:
            beta_start = 0.0001
            beta_end = 0.02
            betas = torch.linspace(beta_start, beta_end, timesteps, dtype=torch.float32)
            alphas = 1.0 - betas
            alphas_cumprod = torch.cumprod(alphas, dim=0)
            alphas_cumsum = 1 - alphas_cumprod ** 0.5
            betas2_cumsum = 1 - alphas_cumprod
            alphas_cumsum_prev = F.pad(alphas_cumsum[:-1], (1, 0), value=1.)
            betas2_cumsum_prev = F.pad(betas2_cumsum[:-1], (1, 0), value=1.)
            alphas = alphas_cumsum - alphas_cumsum_prev
            alphas[0] = 0
            betas2 = betas2_cumsum - betas2_cumsum_prev
            betas2[0] = 0
        else:
            alphas = gen_coefficients(timesteps, schedule="decreased")
            betas2 = gen_coefficients(
                timesteps, schedule="increased", sum_scale=1)

            alphas_cumsum = alphas.cumsum(dim=0).clip(0, 1)
            betas2_cumsum = betas2.cumsum(dim=0).clip(0, 1)

            alphas_cumsum_prev = F.pad(alphas_cumsum[:-1], (1, 0), value=1.)
            betas2_cumsum_prev = F.pad(betas2_cumsum[:-1], (1, 0), value=1.)
        betas_cumsum = torch.sqrt(betas2_cumsum)
        posterior_variance = betas2 * betas2_cumsum_prev / betas2_cumsum
        posterior_variance[0] = 0
        def register_buffer(name, val): return self.register_buffer(
            name, val.to(torch.float32))

        register_buffer('alphas', alphas.to(self.device))
        register_buffer('alphas_cumsum', alphas_cumsum.to(self.device))
        register_buffer('one_minus_alphas_cumsum', (1 - alphas_cumsum).to(self.device))
        register_buffer('betas2', betas2.to(self.device))
        register_buffer('betas', (torch.sqrt(betas2)).to(self.device))
        register_buffer('betas2_cumsum', betas2_cumsum.to(self.device))
        register_buffer('betas_cumsum', betas_cumsum.to(self.device))
        register_buffer('posterior_mean_coef1',
                        (betas2_cumsum_prev / betas2_cumsum).to(self.device))
        register_buffer('posterior_mean_coef2', ((betas2 *
                                                  alphas_cumsum_prev - betas2_cumsum_prev * alphas) / betas2_cumsum).to(
            self.device))
        register_buffer('posterior_mean_coef3', (betas2 / betas2_cumsum).to(self.device))
        register_buffer('posterior_variance', posterior_variance.to(self.device))
        register_buffer('posterior_log_variance_clipped',
                        (torch.log(posterior_variance.clamp(min=1e-20))).to(self.device))

        self.posterior_mean_coef1[0] = 0
        self.posterior_mean_coef2[0] = 0
        self.posterior_mean_coef3[0] = 1
        self.one_minus_alphas_cumsum[-1] = 1e-6


    #采样
    def q_posterior(self, pred_res, x_start, x_t, t):
        posterior_mean = self.posterior_mean_coef1[t] * x_t + self.posterior_mean_coef2[t] * pred_res + \
                         self.posterior_mean_coef3[t] * x_start

        posterior_variance = self.posterior_variance[t]
        posterior_log_variance_clipped = self.posterior_log_variance_clipped[t]
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def get_x0_predict(self, xt, t, x_lr, x_ref0, x_refsr0, x_ref1, x_refsr1):
        batch_size = xt.shape[0]
        alphas_level = self.alphas_cumsum[t] * self.timesteps
        alphas_level = alphas_level.view(batch_size, 1).to(self.device)
        betas_level = self.betas_cumsum[t] * self.timesteps
        betas_level = betas_level.view(batch_size, 1).to(self.device)
        time = [alphas_level, betas_level]
        x0_pridict = self.model_x0(xt=xt, time=time, lrsr=x_lr, refsr0=x_refsr0, ref0=x_ref0, refsr1=x_refsr1,
                                   ref1=x_ref1)
        return x0_pridict

    def p_mean_variance(self, xt, t, x_lr, x_ref0, x_refsr0, x_ref1, x_refsr1):
        x0_predict = self.get_x0_predict(xt, t, x_lr, x_ref0, x_refsr0, x_ref1, x_refsr1)
        pred_res = x_lr - x0_predict
        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(
            pred_res=pred_res, x_start=x0_predict, x_t=xt, t=t)
        return model_mean, posterior_variance, posterior_log_variance, x0_predict

    def p_sample(self, xt, t: int, x_lr, x_ref0, x_refsr0, x_ref1, x_refsr1):
        model_mean, _, model_log_variance, x_start = self.p_mean_variance(
            xt=xt, t=t, x_lr=x_lr, x_ref0=x_ref0, x_refsr0=x_refsr0, x_ref1=x_ref1, x_refsr1=x_refsr1)
        noise = torch.randn_like(xt) if t > 0 else 0.  # no noise if t == 0
        pred_img = model_mean + (0.5 * model_log_variance).exp() * noise
        return pred_img


    def p_sample_loop(self, x_lr, x_ref0, x_refsr0, x_ref1, x_refsr1):
        shape = x_lr.shape
        img = x_lr + torch.randn(shape, device=self.device)
        for i in tqdm(reversed(range(0, self.timesteps)), desc='sampling loop time step', total=self.timesteps):
            img = self.p_sample(xt=img, t=i, x_lr=x_lr, x_ref0=x_ref0, x_refsr0=x_refsr0, x_ref1=x_ref1,
                                     x_refsr1=x_refsr1)
        return img

    def DDIM(self, x_lr, x_ref0, x_refsr0, x_ref1, x_refsr1):
        shape = x_lr.shape
        total_timesteps = self.timesteps
        sampling_steps = self.sampling_steps
        eta = self.ddim_sampling_eta
        times = torch.linspace(-1, total_timesteps - 1,
                               steps=sampling_steps + 1)
        times = list(reversed(times.int().tolist()))
        # [(T-1, T-2), (T-2, T-3), ..., (1, 0), (0, -1)]
        time_pairs = list(zip(times[:-1], times[1:]))
        img = x_lr + torch.randn(shape, device=self.device)
        for time, time_next in tqdm(time_pairs, desc='sampling loop time step'):
            time_cond = torch.full(
                (shape[0],), time, device=self.device, dtype=torch.long)
            x0_predict, lr_feature, lr_recon0, lr_recon1 = self.get_x0_predict(img, time_cond, x_lr, x_ref0, x_refsr0, x_ref1,
                                                                   x_refsr1)
            x_res_predict = x_lr - x0_predict
            if time_next < 0:
                img = x0_predict
                continue

            if eta == 0:
                noise = 0
            else:
                noise = torch.randn_like(img)
            alpha_cumsum = self.alphas_cumsum[time]
            alpha_cumsum_next = self.alphas_cumsum[time_next]
            alpha = alpha_cumsum - alpha_cumsum_next

            betas2_cumsum = self.betas2_cumsum[time]
            betas2_cumsum_next = self.betas2_cumsum[time_next]
            betas2 = betas2_cumsum - betas2_cumsum_next
            betas = betas2.sqrt()
            betas_cumsum = self.betas_cumsum[time]
            betas_cumsum_next = self.betas_cumsum[time_next]
            sigma2 = eta * (betas2 * betas2_cumsum_next / betas2_cumsum)
            sqrt_betas2_cumsum_next_minus_sigma2_divided_betas_cumsum = (
                                                                                    betas2_cumsum_next - sigma2).sqrt() / betas_cumsum
            img = sqrt_betas2_cumsum_next_minus_sigma2_divided_betas_cumsum * img + \
                  (1 - sqrt_betas2_cumsum_next_minus_sigma2_divided_betas_cumsum) * x0_predict + \
                  (
                              alpha_cumsum_next - alpha_cumsum * sqrt_betas2_cumsum_next_minus_sigma2_divided_betas_cumsum) * x_res_predict + \
                  sigma2.sqrt() * noise

        return img



    def sample(self, x_lr, x_ref0, x_refsr0, x_ref1, x_refsr1):
        if self.is_ddim_sampling:
            sample_fn = self.DDIM
        else:
            sample_fn = self.p_sample_loop

        return sample_fn(x_lr, x_ref0, x_refsr0, x_ref1, x_refsr1)



    def q_sample(self, x_start, x_res, t, noise=None):
        noise = default(noise, lambda: torch.randn_like(x_start))

        return (
            x_start+extract(self.alphas_cumsum, t, x_start.shape) * x_res +
            extract(self.betas_cumsum, t, x_start.shape) * noise
        )

    def get_loss(self, x0, x0_recon, lr_feature,  lr_feature_recon0, lr_feature_recon1, current_epoch, lambda_lr):
        [b, c, h, w] = x0.shape
        n = b * c * h * w
        loss_x0_recon = self.x0_recon_loss_func(x0, x0_recon) / n
        loss_pre = self.loss_pre(x0, x0_recon)
        lambda_warm_up = lambda_lr * 0.5 * (1 - math.cos(math.pi * min(current_epoch / self.warm_epoch, 1)))
        loss_lr_feature_recon = 0.5 * F.mse_loss(lr_feature_recon0, lr_feature) + \
                                0.5 * F.mse_loss(lr_feature_recon1, lr_feature)
        loss = loss_x0_recon + 0.01 * loss_pre + lambda_warm_up * loss_lr_feature_recon

        return loss, loss_lr_feature_recon

    def forward(self, x_in, current_epoch, noise=None):
        b = x_in['HR'].shape[0]
        t = torch.randint(0, self.timesteps, (b,), device=self.device).long()
        noise = default(noise, lambda: torch.randn_like(x_in['HR']))
        x_res = x_in['LR'] - x_in['HR']
        xt = self.q_sample(x_in['HR'], x_res, t, noise=noise).to(self.device)
        alphas_level = self.alphas_cumsum[t]*self.timesteps
        alphas_level = alphas_level.view(b, 1).to(self.device)
        betas_level = self.betas_cumsum[t] * self.timesteps
        betas_level = betas_level.view(b, 1).to(self.device)
        time = [alphas_level, betas_level]
        x0_recon, lr_feature, lr_recon0, lr_recon1 = self.model_x0(xt, time, x_in['LR'], x_in['RefSR0'], x_in['Ref0'], x_in['RefSR1'],
                                             x_in['Ref1'])
        lambda_lr = 1e-4
        loss, loss_lr_feature_recon = self.get_loss(x0=x_in['HR'], x0_recon=x0_recon,
                             lr_feature=lr_feature,  lr_feature_recon0=lr_recon0,
                             lr_feature_recon1=lr_recon1, current_epoch=current_epoch,
                             lambda_lr=lambda_lr)
        return loss, loss_lr_feature_recon

