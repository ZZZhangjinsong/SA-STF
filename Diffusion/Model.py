import math
import torch
from torch import nn
from inspect import isfunction

def exists(x):
    return x is not None


def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d

class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)

class PositionalEncoding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, noise_level):
        count = self.dim // 2
        step = torch.arange(count, dtype=noise_level.dtype,
                            device=noise_level.device) / count
        encoding = noise_level.unsqueeze(
            1) * torch.exp(-math.log(1e4) * step.unsqueeze(0))
        encoding = torch.cat(
            [torch.sin(encoding), torch.cos(encoding)], dim=-1)
        return encoding


class FeatureWiseAffine(nn.Module):
    def __init__(self, in_channels, out_channels, use_affine_level=False):
        super(FeatureWiseAffine, self).__init__()
        self.use_affine_level = use_affine_level
        self.noise_func = nn.Sequential(
            nn.Linear(in_channels, out_channels * (1 + self.use_affine_level))
        )

    def forward(self, x, noise_embed):
        batch = x.shape[0]
        if self.use_affine_level:
            gamma, beta = self.noise_func(noise_embed).view(
                batch, -1, 1, 1).chunk(2, dim=1)
            x = (1 + gamma) * x + beta
        else:
            x = x + self.noise_func(noise_embed).view(batch, -1, 1, 1)
        return x


class Downsample(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv = nn.Conv2d(dim, dim, 3, 2, 1)

    def forward(self, x):
        return self.conv(x)


class Upsample(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="nearest")
        self.conv = nn.Conv2d(dim, dim, 3, padding=1)

    def forward(self, x):
        return self.conv(self.up(x))

class Block(nn.Module):
    def __init__(self, dim, dim_out, groups, dropout=0):
        super().__init__()
        self.block = nn.Sequential(
            nn.GroupNorm(groups, dim),
            Swish(),
            nn.Dropout(dropout) if dropout != 0 else nn.Identity(),
            nn.Conv2d(dim, dim_out, 3, padding=1)
        )

    def forward(self, x):
        return self.block(x)




class ResnetBlock_noise(nn.Module):
    def __init__(self, dim, dim_out, noise_level_emb_dim=None, dropout=0, use_affine_level=False, norm_groups=32):
        super().__init__()
        self.noise_func = FeatureWiseAffine(
            noise_level_emb_dim, dim_out, use_affine_level)

        self.block1 = Block(dim, dim_out, groups=norm_groups)
        self.block2 = Block(dim_out, dim_out, groups=norm_groups, dropout=dropout)
        self.res_conv = nn.Conv2d(
            dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(self, x, time_emb):
        h = self.block1(x)
        h = self.noise_func(h, time_emb)
        h = self.block2(h)
        return h + self.res_conv(x)


class ResnetBlock_image(nn.Module):
    def __init__(self, dim, dim_out, norm_groups):
        super().__init__()
        self.block1 = Block(dim, dim_out, groups=norm_groups)
        self.res_conv = nn.Conv2d(
            dim, dim_out, 1) if dim != dim_out else nn.Identity()
    def forward(self, x):
        h = self.block1(x)

        return h + self.res_conv(x)


class SelfAttention(nn.Module):
    def __init__(self, in_channel, n_head=1, norm_groups=32):
        super().__init__()

        self.n_head = n_head

        self.norm = nn.GroupNorm(norm_groups, in_channel)
        self.qkv = nn.Conv2d(in_channel, in_channel * 3, 1, bias=False)
        self.out = nn.Conv2d(in_channel, in_channel, 1)

    def forward(self, input):
        batch, channel, height, width = input.shape
        n_head = self.n_head
        head_dim = channel // n_head

        norm = self.norm(input)
        qkv = self.qkv(norm).view(batch, n_head, head_dim * 3, height, width)
        query, key, value = qkv.chunk(3, dim=2)  # bhdyx

        attn = torch.einsum(
            "bnchw, bncyx -> bnhwyx", query, key
        ).contiguous() / math.sqrt(channel)
        attn = attn.view(batch, n_head, height, width, -1)
        attn = torch.softmax(attn, -1)
        attn = attn.view(batch, n_head, height, width, height, width)

        out = torch.einsum("bnhwyx, bncyx -> bnchw", attn, value).contiguous()
        out = self.out(out.view(batch, channel, height, width))

        return out + input


class CrossAttention(nn.Module):
    def __init__(self, in_channel, norm_groups=32):
        super(CrossAttention, self).__init__()
        self.scale = in_channel ** -0.5


        self.Wq = nn.Conv2d(in_channel, in_channel, 1, bias=False)
        self.Wk = nn.Conv2d(in_channel, in_channel, 1, bias=False)
        self.Wv = nn.Conv2d(in_channel, in_channel, 1, bias=False)
        self.norm = nn.GroupNorm(norm_groups, in_channel)
        self.proj_out = nn.Conv2d(in_channel, in_channel, kernel_size=1, stride=1, padding=0)

    def forward(self, x, y, a):
        '''

        :param x: [batch_size, c, h, w]
        :param context: [batch_szie, seq_len, emb_dim]
        :param pad_mask: [batch_size, seq_len, seq_len]
        :return:
        '''
        batch, c, h, w = x.shape
        x_norm = self.norm(x)
        y_norm = self.norm(y)
        q = self.Wq(x_norm).view(batch, c, h*w)
        k = self.Wk(y_norm).view(batch, c, h*w)
        v_a = self.Wv(a).view(batch, c, h*w)
        att_weights = torch.einsum('bcn,bcm -> bnm', q, k)
        att_weights = att_weights * self.scale
        att_weights = torch.softmax(att_weights, dim=-1)
        out_a = torch.einsum('bnm, bcn -> bcm', att_weights, v_a)
        out_a = out_a.view(batch, c, h, w)

        return out_a


class Time_Fusion(nn.Module):
    def __init__(self, innerchannel):
        super(Time_Fusion, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_a = nn.Sequential(nn.Conv2d(innerchannel * 2, innerchannel, 1, bias=True),
                                    Swish(),
                                    nn.Conv2d(innerchannel, innerchannel, 1, bias=True)
                                    )
        self.conv_b = nn.Conv2d(innerchannel, innerchannel, kernel_size=1, bias=False)
        self.conv_b_full = nn.Sequential(nn.Conv2d(innerchannel, innerchannel, 1, bias=True),
                                         Swish(),
                                         nn.Conv2d(innerchannel, innerchannel, 1, bias=True),
                                         )
        self.cross_atten = CrossAttention(in_channel=innerchannel)

    def forward(self, lr, reflr0, ref0):
        a = self.conv_a(torch.cat([lr, reflr0], dim=1))
        b = self.conv_b(ref0 - reflr0)
        b_full = self.conv_b_full(self.avg_pool(b))
        lr_recon = a * reflr0 + b_full
        a = self.cross_atten(ref0, reflr0, a)
        hr_time = a * ref0 + b_full



        return hr_time, lr_recon

class Space_Fusion(nn.Module):
    def __init__(self, innerchannel, normgroups):
        super(Space_Fusion, self).__init__()
        self.final_conv = Block(innerchannel, innerchannel, normgroups)
        self.res_conv = nn.Conv2d(innerchannel, innerchannel, 1)
        self.time_attention0 = Time_Fusion(innerchannel=innerchannel)
        self.time_attention1 = Time_Fusion(innerchannel=innerchannel)
        self.attention = SelfAttention(in_channel=innerchannel, norm_groups=normgroups)
        self.norm = nn.GroupNorm(normgroups, innerchannel)


    def forward(self, lrsr, refsr0, ref0, refsr1, ref1):

        # 自注意力明确语义信息
        lrsr = self.attention(lrsr)
        refsr0 = self.attention(refsr0)
        refsr1 = self.attention(refsr1)
        ref0 = self.attention(ref0)
        ref1 = self.attention(ref1)

        #获取时间变化结果
        hr_time0, lr_recon0 = self.time_attention0(lr=lrsr, reflr0=refsr0, ref0=ref0)
        hr_time1, lr_recon1 = self.time_attention1(lr=lrsr, reflr0=refsr1, ref0=ref1)


        #根据相似度来选取patch
        S01_lr = torch.pow((lrsr - ref0), 2)
        S12_lr = torch.pow((lrsr - ref1), 2)
        S02_hr = torch.pow((ref1 - ref0), 2)
        condition_lr = S01_lr < S12_lr
        S_min = torch.where(condition_lr, S01_lr, S12_lr)
        condition_hr = S02_hr < S_min
        condition_hr = condition_hr.contiguous()
        C = torch.where(condition_hr, 1, 0)
        S01_change = torch.pow((lrsr - hr_time0), 2)
        S12_change = torch.pow((lrsr - hr_time1), 2)
        condition = S01_change < S12_change
        condition = condition.contiguous()
        T = torch.where(condition, C * ref0 + (1-C) * hr_time0, C * ref0 + (1-C) * hr_time1)


        return self.final_conv(T) + self.res_conv(lrsr), lrsr, lr_recon0, lr_recon1

class Deep_Fuison_Block(nn.Module):
    def __init__(self, dim, dim_out,  norm_groups=32):
        super().__init__()
        self.res_block1 =ResnetBlock_image(dim, dim_out,  norm_groups=norm_groups)
        self.res_block2 = ResnetBlock_image(dim_out, dim_out,  norm_groups=norm_groups)
        self.fusion = Space_Fusion(innerchannel=dim_out, normgroups=norm_groups)

    def forward(self, lrsr, refsr0, ref0, refsr1, ref1):
        lrsr = self.res_block1(lrsr)
        refsr0 = self.res_block1(refsr0)
        ref0 = self.res_block1(ref0)
        refsr1 = self.res_block1(refsr1)
        ref1 = self.res_block1(ref1)
        x, lrsr, lr_recon0, lr_recon1 = self.fusion(lrsr, refsr0, ref0, refsr1, ref1)

        return self.res_block2(x), lrsr, lr_recon0, lr_recon1

class Middle_Block(nn.Module):
    def __init__(self, dim, dim_out, noise_level_emb_dim, dropout, norm_groups):
        super().__init__()
        self.res_block1 = ResnetBlock_noise(dim, dim_out, noise_level_emb_dim=noise_level_emb_dim,
                                     dropout=dropout, norm_groups=norm_groups)
        self.res_block2 = ResnetBlock_noise(dim, dim_out, noise_level_emb_dim=noise_level_emb_dim,
                                     dropout=dropout, norm_groups=norm_groups)
        self.norm = nn.GroupNorm(norm_groups, dim)

    def forward(self, lrsr, h, xt, alpha_t, beta_t):
        lrsr = self.norm(lrsr)
        h = self.norm(h)
        xt = self.norm(xt)
        res_t = self.norm(self.res_block1(lrsr-h, alpha_t))
        x = self.res_block2(xt-res_t, beta_t)

        return x

class Shallow_Fuison_Block(nn.Module):
    def __init__(self, dim, dim_out, norm_groups):
        super().__init__()
        self.crose_block = Block(dim*3, dim_out, groups=norm_groups)
        self.fine_block = Block(dim*2, dim_out, groups=norm_groups)
        self.fusion_conv = nn.Conv2d(dim*2, dim_out, kernel_size=1)


    def forward(self, lrsr, refsr0, ref0, refsr1, ref1):
        crose_feature = self.crose_block(torch.cat((lrsr, refsr0, refsr1), dim=1))
        fine_feature = self.fine_block(torch.cat((ref0, ref1), dim=1))
        fusion_feature = self.fusion_conv(torch.cat((crose_feature, fine_feature), dim=1))

        return fusion_feature

class UNet(nn.Module):
    def __init__(
            self,
            in_channel=6,
            out_channel=6,
            inner_channel=32,
            norm_groups=16,
            channel_mults=(1, 2, 4, 8, 8),
            dropout=0
    ):
        super().__init__()
        noise_level_channel = inner_channel
        self.noise_level_mlp = nn.Sequential(
            PositionalEncoding(inner_channel),
            nn.Linear(inner_channel, inner_channel * 4),
            Swish(),
            nn.Linear(inner_channel * 4, inner_channel)
        )
        self.res_level_mlp = nn.Sequential(
            PositionalEncoding(inner_channel),
            nn.Linear(inner_channel, inner_channel * 4),
            Swish(),
            nn.Linear(inner_channel * 4, inner_channel)
        )

        num_mults = len(channel_mults)
        pre_channel = inner_channel
        feat_channels = []
        image_downs = [nn.Conv2d(in_channel, inner_channel, kernel_size=3, padding=1)]
        noisy_downs = [nn.Conv2d(in_channel, inner_channel, kernel_size=3, padding=1)]
        image_fusion = []
        for ind in range(num_mults):
            is_last = (ind == num_mults - 1)
            channel_mult = inner_channel * channel_mults[ind]
            image_downs.append(ResnetBlock_image(pre_channel, channel_mult, norm_groups=norm_groups))
            noisy_downs.append(ResnetBlock_noise(pre_channel, channel_mult, noise_level_emb_dim=noise_level_channel,
                                                      dropout=dropout,  norm_groups=norm_groups))
            feat_channels.append(channel_mult)
            pre_channel = channel_mult
            if not is_last:
                image_fusion.append(Shallow_Fuison_Block(channel_mult, channel_mult, norm_groups=norm_groups))
                noisy_downs.append(Downsample(pre_channel))
                image_downs.append(Downsample(pre_channel))
            else:
                image_fusion.append(Deep_Fuison_Block(channel_mult, channel_mult, norm_groups))
        self.noise_downs = nn.ModuleList(noisy_downs)
        self.image_downs = nn.ModuleList(image_downs)
        self.image_fusion = nn.ModuleList(image_fusion)
        self.mid = Middle_Block(pre_channel, pre_channel, noise_level_emb_dim=noise_level_channel,
                                     dropout=dropout, norm_groups=norm_groups)
        image_ups = []
        for ind in reversed(range(num_mults)):
            is_last = (ind < 1)
            channel_mult = inner_channel * channel_mults[ind]
            image_ups.append(ResnetBlock_noise(pre_channel + feat_channels.pop(), channel_mult, noise_level_emb_dim=noise_level_channel,
                              dropout=dropout, norm_groups=norm_groups))
            pre_channel = channel_mult
            if not is_last:
                image_ups.append(Upsample(pre_channel))
        self.image_ups = nn.ModuleList(image_ups)  #
        self.final_conv = Block(pre_channel, default(out_channel, in_channel), groups=norm_groups)



    def forward(self, xt, time, lrsr, refsr0, ref0, refsr1, ref1):
        alpha_t = self.res_level_mlp(time[0])
        beta_t = self.noise_level_mlp(time[1])
        feats_hr = []
        down_num = len(self.image_downs)
        for down_idx in range(down_num):
            layer_image = self.image_downs[down_idx]
            layer_noise = self.noise_downs[down_idx]
            layer_imageFusion = self.image_fusion[down_idx//2]
            if isinstance(layer_image, ResnetBlock_image):
                lrsr = layer_image(lrsr)
                refsr0 = layer_image(refsr0)
                ref0 = layer_image(ref0)
                refsr1 = layer_image(refsr1)
                ref1 = layer_image(ref1)
                if isinstance(layer_imageFusion, Shallow_Fuison_Block):
                    hr_feature = layer_imageFusion(lrsr, refsr0, ref0, refsr1, ref1)
                else:
                    hr_feature, lr_feature, lr_recon0, lr_recon1 = layer_imageFusion(lrsr, refsr0, ref0, refsr1, ref1)
                xt = layer_noise(xt, alpha_t)#alpha_t
                feats_hr.append(hr_feature)
            else:
                lrsr = layer_image(lrsr)
                refsr0 = layer_image(refsr0)
                ref0 = layer_image(ref0)
                refsr1 = layer_image(refsr1)
                ref1 = layer_image(ref1)
                xt = layer_noise(xt)

        xt = self.mid(lrsr, hr_feature, xt, alpha_t, beta_t)#beta_t



        for layer in self.image_ups:
            if isinstance(layer, ResnetBlock_noise):
                xt = layer(torch.cat((xt, feats_hr.pop()), dim=1), beta_t)#beta_t
            else:
                xt = layer(xt)
        return self.final_conv(xt), lr_feature, lr_recon0, lr_recon1


if __name__ == '__main__':
    xt = torch.rand(2, 6, 256, 256)
    lrsr = torch.rand(2, 6, 256, 256)
    ref0 = torch.rand(2, 6, 256, 256)
    refsr0 = torch.rand(2, 6, 256, 256)
    ref1 = torch.rand(2, 6, 256, 256)
    refsr1 = torch.rand(2, 6, 256, 256)
    time0 = torch.rand(2, 1)
    time1 = torch.rand(2, 1)
    time = [time0, time1]
    net_model = UNet(in_channel=6,
            out_channel=6,
            inner_channel=64,
            norm_groups=32,
            channel_mults=(1, 2, 4, 8, 8),
            dropout=0)
    x,b,c,d = net_model(xt, time, lrsr, refsr0, ref0, refsr1, ref1)
    print(x.shape)
